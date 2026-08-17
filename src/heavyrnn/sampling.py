"""Deterministic random sampling for heavy-tailed recurrent networks.

The stable distribution used by the flagship experiment has characteristic
function ``exp(-|k|**alpha)``.  In particular, the ``alpha=2`` endpoint is a
normal distribution with variance two, rather than a standard normal.

Random streams are keyed by a complete experimental condition.  Consequently,
sampling a condition alone or as part of a differently ordered batch yields the
same seed and the same tensor (for a fixed PyTorch version, device type, and
chunking choice).
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import math
import operator

import torch


RECURRENT_WEIGHT_STREAM = "recurrent_weights"
INITIAL_STATE_STREAM = "initial_state"

_MAX_TORCH_SEED = (1 << 63) - 1
_SEED_PERSON = b"heavyrnn-seed-v1"
_DEFAULT_CMS_WORKSPACE_BYTES = 256 * 1024 * 1024
_CMS_LIVE_FLOAT64_ARRAYS = 4


def _as_int(value: int, *, name: str) -> int:
    """Return an integer-like value while rejecting booleans."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from exc


def _as_finite_float(value: float, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def _canonical_float(value: float) -> str:
    """Encode equal binary64 values identically, including signed zero."""
    return 0.0.hex() if value == 0.0 else value.hex()


def derive_seed(
    base_seed: int,
    *,
    n: int,
    alpha: float,
    gain: float,
    trial: int,
    stream: str,
) -> int:
    """Derive a deterministic PyTorch seed for one named condition stream.

    A BLAKE2b hash is computed from canonical representations of every
    condition field.  The result is independent of Python's process-randomized
    hash function and of sweep ordering or batching.

    Args:
        base_seed: Non-negative experiment-level seed.
        n: Number of recurrent units.
        alpha: Stable-law exponent in ``(0, 2]``.
        gain: Non-negative recurrent gain.
        trial: Non-negative trial index.
        stream: Non-empty stream name, for example ``"recurrent_weights"`` or
            ``"initial_state"``.

    Returns:
        An integer in the range accepted uniformly by CPU and CUDA generators.
    """
    base_seed = _as_int(base_seed, name="base_seed")
    n = _as_int(n, name="n")
    trial = _as_int(trial, name="trial")
    alpha = _as_finite_float(alpha, name="alpha")
    gain = _as_finite_float(gain, name="gain")

    if base_seed < 0:
        raise ValueError(f"base_seed must be non-negative, got {base_seed}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < alpha <= 2.0:
        raise ValueError(f"alpha must lie in (0, 2], got {alpha}")
    if gain < 0.0:
        raise ValueError(f"gain must be non-negative, got {gain}")
    if trial < 0:
        raise ValueError(f"trial must be non-negative, got {trial}")
    if not isinstance(stream, str):
        raise TypeError(f"stream must be a string, got {type(stream).__name__}")
    if not stream:
        raise ValueError("stream must be non-empty")

    fields = (
        ("base_seed", str(base_seed)),
        ("n", str(n)),
        ("alpha", _canonical_float(alpha)),
        ("gain", _canonical_float(gain)),
        ("trial", str(trial)),
        ("stream", stream),
    )
    digest = hashlib.blake2b(digest_size=8, person=_SEED_PERSON)
    for key, value in fields:
        encoded = value.encode("utf-8")
        digest.update(key.encode("ascii"))
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), byteorder="big") & _MAX_TORCH_SEED


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create a local generator without modifying PyTorch's global RNG state."""
    seed = _as_int(seed, name="seed")
    if not 0 <= seed <= _MAX_TORCH_SEED:
        raise ValueError(f"seed must lie in [0, {_MAX_TORCH_SEED}], got {seed}")
    resolved_device = torch.device(device)
    try:
        generator = torch.Generator(device=resolved_device)
    except RuntimeError as exc:
        raise RuntimeError(
            f"could not create a PyTorch generator on device {resolved_device}"
        ) from exc
    generator.manual_seed(seed)
    return generator


def _normalize_shape(shape: Sequence[int] | torch.Size) -> tuple[int, ...]:
    try:
        normalized = tuple(_as_int(size, name="shape dimension") for size in shape)
    except TypeError as exc:
        if isinstance(shape, int):
            raise TypeError("shape must be a sequence; use (size,) for a vector") from exc
        raise
    if any(size < 0 for size in normalized):
        raise ValueError(f"shape dimensions must be non-negative, got {normalized}")
    return normalized


def _validate_output_dtype(dtype: torch.dtype) -> None:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be floating point, got {dtype}")


def _assert_all_finite(values: torch.Tensor, *, context: str) -> None:
    """Fail explicitly instead of hiding an extreme or invalid stable draw."""
    if values.numel() == 0:
        return
    finite = torch.isfinite(values)
    if bool(finite.all().item()):
        return
    nonfinite_count = int((~finite).sum().item())
    raise FloatingPointError(
        f"{context} produced {nonfinite_count} non-finite value(s) out of "
        f"{values.numel()}. Heavy-tailed draws are never clipped or resampled; "
        "choose another seed or a higher-precision dynamics dtype if appropriate."
    )


def _sample_symmetric_alpha_stable_float64(
    shape: tuple[int, ...],
    alpha: float,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Draw scale-one symmetric stable variates using float64 arithmetic."""
    if alpha == 2.0:
        samples = torch.empty(shape, dtype=torch.float64, device=device)
        samples.normal_(mean=0.0, std=math.sqrt(2.0), generator=generator)
        return samples

    angles = torch.empty(shape, dtype=torch.float64, device=device)
    angles.uniform_(-math.pi / 2.0, math.pi / 2.0, generator=generator)
    if alpha == 1.0:
        return angles.tan_()

    # Symmetric (beta=0) Chambers--Mallows--Stuck construction:
    # sin(alpha U) / cos(U)^(1/alpha)
    #     * [cos((1-alpha) U) / E]^((1-alpha)/alpha), E ~ Exp(1).
    exponential = torch.empty(shape, dtype=torch.float64, device=device)
    exponential.exponential_(lambd=1.0, generator=generator)

    second_factor = angles.clone()
    second_factor.mul_(1.0 - alpha).cos_().div_(exponential)
    second_factor.pow_((1.0 - alpha) / alpha)
    del exponential

    samples = angles.clone()
    samples.mul_(alpha).sin_()
    angles.cos_().pow_(1.0 / alpha)
    samples.div_(angles).mul_(second_factor)
    return samples


def sample_symmetric_alpha_stable(
    shape: Sequence[int] | torch.Size,
    alpha: float,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample a symmetric, scale-one alpha-stable tensor.

    All random draws and CMS arithmetic are float64.  The result is checked for
    finite values before and after conversion to ``dtype``.  ``alpha=1`` uses
    the exact Cauchy construction and ``alpha=2`` uses ``sqrt(2) * N(0, 1)``.

    This lower-level function accepts a concrete seed.  Experiment code should
    normally obtain it with :func:`derive_seed`.
    """
    normalized_shape = _normalize_shape(shape)
    alpha = _as_finite_float(alpha, name="alpha")
    if not 0.0 < alpha <= 2.0:
        raise ValueError(f"alpha must lie in (0, 2], got {alpha}")
    _validate_output_dtype(dtype)
    resolved_device = torch.device(device)
    generator = make_generator(seed, resolved_device)

    samples = _sample_symmetric_alpha_stable_float64(
        normalized_shape,
        alpha,
        generator=generator,
        device=resolved_device,
    )
    context = f"symmetric alpha-stable sampling (alpha={alpha}, seed={seed})"
    _assert_all_finite(samples, context=context)
    if dtype != torch.float64:
        samples = samples.to(dtype=dtype)
        _assert_all_finite(samples, context=f"{context} after conversion to {dtype}")
    return samples


def _default_chunk_rows(n: int) -> int:
    bytes_per_row = n * torch.empty((), dtype=torch.float64).element_size()
    bytes_per_row *= _CMS_LIVE_FLOAT64_ARRAYS
    return min(n, max(1, _DEFAULT_CMS_WORKSPACE_BYTES // bytes_per_row))


def _fill_recurrent_weights(
    output: torch.Tensor,
    *,
    alpha: float,
    gain: float,
    seed: int,
    chunk_rows: int,
) -> None:
    n = output.shape[0]
    generator = make_generator(seed, output.device)
    scale = gain * math.pow(n, -1.0 / alpha)
    if not math.isfinite(scale):
        raise FloatingPointError(
            f"weight scale g*N^(-1/alpha) is non-finite for "
            f"n={n}, alpha={alpha}, gain={gain}"
        )

    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        raw = _sample_symmetric_alpha_stable_float64(
            (stop - start, n),
            alpha,
            generator=generator,
            device=output.device,
        )
        context = (
            f"recurrent-weight sampling (n={n}, alpha={alpha}, gain={gain}, "
            f"seed={seed}, rows={start}:{stop})"
        )
        _assert_all_finite(raw, context=context)
        raw.mul_(scale)
        _assert_all_finite(raw, context=f"{context} after scaling")
        converted = raw.to(dtype=output.dtype)
        _assert_all_finite(
            converted,
            context=f"{context} after conversion to {output.dtype}",
        )
        output[start:stop].copy_(converted)


def sample_recurrent_weights(
    n: int,
    alpha: float,
    gain: float,
    *,
    base_seed: int,
    trial: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """Sample one dense recurrent matrix with flagship normalization.

    The returned entries obey

    ``W_ij = gain * n**(-1/alpha) * z_ij``,

    where ``z_ij`` are independent symmetric scale-one alpha-stable draws.
    Sampling and scaling occur in float64, followed by a checked conversion to
    the requested dynamics dtype.  Matrix rows are generated in chunks to bound
    temporary CMS storage for large networks.
    """
    n = _as_int(n, name="n")
    trial = _as_int(trial, name="trial")
    alpha = _as_finite_float(alpha, name="alpha")
    gain = _as_finite_float(gain, name="gain")
    # derive_seed performs the shared condition validation.
    seed = derive_seed(
        base_seed,
        n=n,
        alpha=alpha,
        gain=gain,
        trial=trial,
        stream=RECURRENT_WEIGHT_STREAM,
    )
    _validate_output_dtype(dtype)

    if chunk_rows is None:
        resolved_chunk_rows = _default_chunk_rows(n)
    else:
        resolved_chunk_rows = _as_int(chunk_rows, name="chunk_rows")
        if resolved_chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive, got {resolved_chunk_rows}")
        resolved_chunk_rows = min(n, resolved_chunk_rows)

    output = torch.empty((n, n), dtype=dtype, device=torch.device(device))
    _fill_recurrent_weights(
        output,
        alpha=alpha,
        gain=gain,
        seed=seed,
        chunk_rows=resolved_chunk_rows,
    )
    return output


def sample_recurrent_weight_batch(
    n: int,
    alpha: float,
    gain: float,
    trials: Sequence[int],
    *,
    base_seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    chunk_rows: int | None = None,
) -> torch.Tensor:
    """Sample a batch of independent matrices for one ``(n, alpha, gain)``.

    Each matrix is filled directly into the batch output, avoiding an extra
    full-matrix copy.  Its seed depends on the trial identifier rather than the
    trial's position, so ordering and batch composition do not change draws.
    """
    normalized_trials = tuple(_as_int(trial, name="trial") for trial in trials)
    if not normalized_trials:
        raise ValueError("trials must contain at least one trial index")

    n = _as_int(n, name="n")
    alpha = _as_finite_float(alpha, name="alpha")
    gain = _as_finite_float(gain, name="gain")
    # Validate common fields before allocating a potentially large tensor.
    first_seed = derive_seed(
        base_seed,
        n=n,
        alpha=alpha,
        gain=gain,
        trial=normalized_trials[0],
        stream=RECURRENT_WEIGHT_STREAM,
    )
    _validate_output_dtype(dtype)

    if chunk_rows is None:
        resolved_chunk_rows = _default_chunk_rows(n)
    else:
        resolved_chunk_rows = _as_int(chunk_rows, name="chunk_rows")
        if resolved_chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive, got {resolved_chunk_rows}")
        resolved_chunk_rows = min(n, resolved_chunk_rows)

    output = torch.empty(
        (len(normalized_trials), n, n),
        dtype=dtype,
        device=torch.device(device),
    )
    for index, trial in enumerate(normalized_trials):
        seed = first_seed if index == 0 else derive_seed(
            base_seed,
            n=n,
            alpha=alpha,
            gain=gain,
            trial=trial,
            stream=RECURRENT_WEIGHT_STREAM,
        )
        _fill_recurrent_weights(
            output[index],
            alpha=alpha,
            gain=gain,
            seed=seed,
            chunk_rows=resolved_chunk_rows,
        )
    return output


__all__ = [
    "INITIAL_STATE_STREAM",
    "RECURRENT_WEIGHT_STREAM",
    "derive_seed",
    "make_generator",
    "sample_recurrent_weight_batch",
    "sample_recurrent_weights",
    "sample_symmetric_alpha_stable",
]
