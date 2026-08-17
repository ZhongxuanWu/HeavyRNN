"""Condition expansion, batched execution, persistence, and aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
import csv
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any
import zipfile
import zlib

import numpy as np
from omegaconf import OmegaConf
import torch

from .config import ExperimentConfig, as_dictconfig, validate_config
from .diagnostics import AggregateRecord, TransitionRecord, aggregate_trials, locate_transitions
from .sampling import (
    INITIAL_STATE_STREAM,
    RECURRENT_WEIGHT_STREAM,
    derive_seed,
    make_generator,
    sample_recurrent_weight_batch,
)
from .simulation import SimulationBatchResult, simulate_batch


ARTIFACT_SCHEMA_VERSION = 2
_NUMERICAL_SOURCE_FILES = (
    "config.py",
    "sampling.py",
    "simulation.py",
    "diagnostics.py",
    "experiment.py",
)


def _exception_summary(exc: BaseException) -> str:
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__


def _numerical_source_digest(paths: Sequence[Path] | None = None) -> str:
    """Hash the installed numerical source, including uncommitted edits."""

    if paths is None:
        package_directory = Path(__file__).resolve().parent
        paths = tuple(package_directory / name for name in _NUMERICAL_SOURCE_FILES)
    digest = hashlib.sha256()
    for path in sorted((Path(path) for path in paths), key=lambda item: item.name):
        name = path.name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


_LOADED_NUMERICAL_SOURCE_DIGEST = _numerical_source_digest()


def _implementation_identity(device: torch.device) -> dict[str, Any]:
    """Return a stable compatibility fingerprint for one numerical runtime."""

    torch_configuration = torch.__config__.show().encode("utf-8")
    compatibility: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "numerical_source_sha256": _LOADED_NUMERICAL_SOURCE_DIGEST,
        "python_implementation": sys.implementation.name,
        "python_version": list(sys.version_info[:3]),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "torch_config_sha256": hashlib.sha256(torch_configuration).hexdigest(),
        "device_type": device.type,
        "machine": platform.machine(),
    }
    if device.type == "cuda":
        compatibility["gpu_name"] = torch.cuda.get_device_name(device)
        compatibility["gpu_capability"] = list(torch.cuda.get_device_capability(device))
    canonical = json.dumps(
        compatibility, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return {
        "fingerprint": hashlib.sha256(canonical).hexdigest(),
        "compatibility": compatibility,
    }


def _batch_identity(condition_ids: Sequence[str]) -> str:
    digest = hashlib.blake2b(digest_size=12, person=b"hrnn-batch-v1")
    for condition_id in condition_ids:
        encoded = condition_id.encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ConditionSpec:
    """One independently sampled network at one point in the sweep."""

    condition_id: str
    network_size: int
    alpha: float
    gain: float
    gain_index: int
    trial: int
    weight_seed: int
    state_seed: int


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    output_directory: Path
    total_conditions: int
    computed_conditions: int
    resumed_conditions: int
    metrics_path: Path
    aggregates_path: Path
    transitions_path: Path


class OutputDirectoryError(RuntimeError):
    """Raised when an output directory cannot safely hold the requested run."""


def _condition_identity(
    network_size: int,
    alpha: float,
    gain: float,
    gain_index: int,
    trial: int,
) -> str:
    canonical = "|".join(
        (
            str(network_size),
            float(alpha).hex(),
            float(gain).hex(),
            str(gain_index),
            str(trial),
        )
    )
    suffix = hashlib.blake2b(canonical.encode("ascii"), digest_size=6).hexdigest()
    return f"n{network_size}_a{alpha:g}_g{gain_index:03d}_t{trial:03d}_{suffix}"


def expand_conditions(config: ExperimentConfig) -> list[ConditionSpec]:
    """Expand the Cartesian sweep into immutable, deterministically seeded runs."""

    validate_config(config)
    conditions: list[ConditionSpec] = []
    for network_size in config.sweep.network_sizes:
        for alpha in config.sweep.alphas:
            for gain_index, gain in enumerate(config.gain_values()):
                for trial in range(config.sweep.trials):
                    conditions.append(
                        ConditionSpec(
                            condition_id=_condition_identity(
                                network_size, alpha, gain, gain_index, trial
                            ),
                            network_size=network_size,
                            alpha=alpha,
                            gain=gain,
                            gain_index=gain_index,
                            trial=trial,
                            weight_seed=derive_seed(
                                config.runtime.base_seed,
                                n=network_size,
                                alpha=alpha,
                                gain=gain,
                                trial=trial,
                                stream=RECURRENT_WEIGHT_STREAM,
                            ),
                            state_seed=derive_seed(
                                config.runtime.base_seed,
                                n=network_size,
                                alpha=alpha,
                                gain=gain,
                                trial=trial,
                                stream=INITIAL_STATE_STREAM,
                            ),
                        )
                    )
    return conditions


def _torch_dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError as exc:  # configuration validation normally catches this
        raise ValueError(f"unsupported dynamics dtype: {name}") from exc


def estimate_condition_bytes(config: ExperimentConfig, network_size: int) -> int:
    """Conservative persistent/workspace estimate for one simulated condition."""

    element_size = torch.empty((), dtype=_torch_dtype(config.simulation.dtype)).element_size()
    n = network_size
    k = config.simulation.num_exponents
    steps = config.simulation.measurement_steps
    weight_bytes = n * n * element_size
    state_and_qr_bytes = (8 * n * k + 6 * n + 2 * k * k) * element_size
    activity_bytes = steps * n * element_size if config.simulation.record_activity else 0
    diagnostics_bytes = steps * k * 8 + k * 8 + steps * 5 * element_size
    return weight_bytes + state_and_qr_bytes + activity_bytes + diagnostics_bytes


def estimate_output_bytes(config: ExperimentConfig) -> int:
    """Estimate array payload size for all per-condition artifacts."""

    element_size = torch.empty((), dtype=_torch_dtype(config.simulation.dtype)).element_size()
    steps = config.simulation.measurement_steps
    k = config.simulation.num_exponents
    per_condition_diagnostics = 2 * k * 8 + steps * k * 8 + steps * 5 * element_size
    total = 0
    gain_count = len(config.gain_values())
    for n in config.sweep.network_sizes:
        activity = steps * n * element_size if config.simulation.record_activity else 0
        total += (
            len(config.sweep.alphas)
            * gain_count
            * config.sweep.trials
            * (activity + per_condition_diagnostics)
        )
    return total


def inspect_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Return condition, output, and device-memory estimates without writing files."""

    conditions = expand_conditions(config)
    device = torch.device(config.runtime.device)
    report: dict[str, Any] = {
        "conditions": len(conditions),
        "output_bytes": estimate_output_bytes(config),
        "largest_condition_bytes": max(
            estimate_condition_bytes(config, size) for size in config.sweep.network_sizes
        ),
        "device": str(device),
        "device_available": device.type == "cpu",
    }
    if device.type == "cuda":
        report["device_available"] = torch.cuda.is_available() and device.index is not None and (
            device.index < torch.cuda.device_count()
        )
        if report["device_available"]:
            try:
                with torch.cuda.device(device):
                    free, total = torch.cuda.mem_get_info(device)
                report["device_free_bytes"] = int(free)
                report["device_total_bytes"] = int(total)
                report["estimated_largest_auto_batch"] = _auto_batch_size(
                    config,
                    max(config.sweep.network_sizes),
                    config.sweep.trials,
                    device,
                )
            except Exception as exc:  # CUDA backends use several exception types.
                report["device_available"] = False
                report["device_error"] = f"{type(exc).__name__}: {_exception_summary(exc)}"
    return report


def _auto_batch_size(
    config: ExperimentConfig,
    network_size: int,
    pending_count: int,
    device: torch.device,
) -> int:
    if isinstance(config.runtime.batch_size, int):
        return min(config.runtime.batch_size, pending_count)
    if device.type == "cuda":
        with torch.cuda.device(device):
            free_bytes, _ = torch.cuda.mem_get_info(device)
            reclaimable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        available_bytes = int(free_bytes + reclaimable)
    else:
        try:
            available_bytes = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (AttributeError, OSError, ValueError):
            return 1
    # The sampler bounds live float64 temporaries to approximately 256 MiB.
    sampling_workspace = 256 * 1024 * 1024
    budget = int(available_bytes * config.runtime.memory_fraction) - sampling_workspace
    per_condition = estimate_condition_bytes(config, network_size)
    return max(1, min(pending_count, budget // max(1, per_condition)))


def _sample_initial_states(
    conditions: Sequence[ConditionSpec],
    *,
    device: torch.device,
    dtype: torch.dtype,
    std: float,
) -> torch.Tensor:
    states = torch.empty(
        len(conditions), conditions[0].network_size, device=device, dtype=dtype
    )
    for index, condition in enumerate(conditions):
        generator = make_generator(condition.state_seed, device)
        states[index].normal_(mean=0.0, std=std, generator=generator)
    return states


def _simulate_conditions(
    conditions: Sequence[ConditionSpec],
    config: ExperimentConfig,
    device: torch.device,
) -> SimulationBatchResult:
    first = conditions[0]
    if any(
        (condition.network_size, condition.alpha, condition.gain)
        != (first.network_size, first.alpha, first.gain)
        for condition in conditions
    ):
        raise ValueError("a simulation batch must share network size, alpha, and gain")
    if len({condition.condition_id for condition in conditions}) != len(conditions):
        raise ValueError("a simulation batch must not contain duplicate conditions")

    configured_gains = config.gain_values()
    for condition in conditions:
        if condition.network_size not in config.sweep.network_sizes:
            raise ValueError(
                f"condition {condition.condition_id} uses N={condition.network_size}, "
                "which is not in this config"
            )
        if condition.alpha not in config.sweep.alphas:
            raise ValueError(
                f"condition {condition.condition_id} uses alpha={condition.alpha}, "
                "which is not in this config"
            )
        if not 0 <= condition.gain_index < len(configured_gains):
            raise ValueError(
                f"condition {condition.condition_id} has invalid gain_index="
                f"{condition.gain_index}"
            )
        if condition.gain != configured_gains[condition.gain_index]:
            raise ValueError(
                f"condition {condition.condition_id} has gain={condition.gain}, "
                f"expected {configured_gains[condition.gain_index]} at its gain_index"
            )
        if not 0 <= condition.trial < config.sweep.trials:
            raise ValueError(
                f"condition {condition.condition_id} has invalid trial={condition.trial}"
            )
        expected_condition_id = _condition_identity(
            condition.network_size,
            condition.alpha,
            condition.gain,
            condition.gain_index,
            condition.trial,
        )
        if condition.condition_id != expected_condition_id:
            raise ValueError(
                f"condition_id {condition.condition_id!r} does not match "
                f"{expected_condition_id!r}"
            )
        expected_weight_seed = derive_seed(
            config.runtime.base_seed,
            n=condition.network_size,
            alpha=condition.alpha,
            gain=condition.gain,
            trial=condition.trial,
            stream=RECURRENT_WEIGHT_STREAM,
        )
        expected_state_seed = derive_seed(
            config.runtime.base_seed,
            n=condition.network_size,
            alpha=condition.alpha,
            gain=condition.gain,
            trial=condition.trial,
            stream=INITIAL_STATE_STREAM,
        )
        if condition.weight_seed != expected_weight_seed:
            raise ValueError(
                f"condition {condition.condition_id} has weight_seed="
                f"{condition.weight_seed}, expected {expected_weight_seed} for this config"
            )
        if condition.state_seed != expected_state_seed:
            raise ValueError(
                f"condition {condition.condition_id} has state_seed="
                f"{condition.state_seed}, expected {expected_state_seed} for this config"
            )

    dtype = _torch_dtype(config.simulation.dtype)
    weights = sample_recurrent_weight_batch(
        first.network_size,
        first.alpha,
        first.gain,
        [condition.trial for condition in conditions],
        base_seed=config.runtime.base_seed,
        device=device,
        dtype=dtype,
    )
    initial_state = _sample_initial_states(
        conditions,
        device=device,
        dtype=dtype,
        std=config.simulation.initial_state_std,
    )
    return simulate_batch(
        weights,
        initial_state,
        warmup_steps=config.simulation.warmup_steps,
        measurement_steps=config.simulation.measurement_steps,
        num_exponents=config.simulation.num_exponents,
        saturation_threshold=config.simulation.saturation_threshold,
        record_activity=config.simulation.record_activity,
    )


def simulate_conditions(
    conditions: Sequence[ConditionSpec],
    config: ExperimentConfig,
) -> SimulationBatchResult:
    """Simulate one homogeneous condition batch without writing run artifacts."""

    validate_config(config)
    if not conditions:
        raise ValueError("conditions must not be empty")
    torch.set_float32_matmul_precision(config.runtime.matmul_precision)
    device = torch.device(config.runtime.device)
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("runtime.device must select one CUDA device, for example 'cuda:0'")
        try:
            torch.cuda.set_device(device)
            torch.empty(0, device=device)
        except Exception as exc:
            raise RuntimeError(
                f"could not initialize {device}: {_exception_summary(exc)}"
            ) from exc
    return _simulate_conditions(tuple(conditions), config, device)


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _artifact_path(output_directory: Path, condition: ConditionSpec) -> Path:
    return output_directory / "conditions" / f"{condition.condition_id}.npz"


def _write_condition_artifacts(
    output_directory: Path,
    conditions: Sequence[ConditionSpec],
    result: SimulationBatchResult,
    *,
    compressed: bool,
    implementation_fingerprint: str,
) -> None:
    arrays: dict[str, np.ndarray | None] = {
        "activity": None if result.activity is None else result.activity.cpu().numpy(),
        "lyapunov_exponents": result.lyapunov_exponents.cpu().numpy(),
        "local_log_stretches": result.local_log_stretches.cpu().numpy(),
        "floored_stretch_counts": result.floored_stretch_counts.cpu().numpy(),
        "activity_mean": result.activity_mean.cpu().numpy(),
        "activity_rms": result.activity_rms.cpu().numpy(),
        "activity_std": result.activity_std.cpu().numpy(),
        "activity_step_rms": result.activity_step_rms.cpu().numpy(),
        "saturation_fraction": result.saturation_fraction.cpu().numpy(),
        "finite": result.finite.cpu().numpy(),
    }
    saver = np.savez_compressed if compressed else np.savez
    batch_condition_ids = tuple(condition.condition_id for condition in conditions)
    execution_batch_id = _batch_identity(batch_condition_ids)

    for index, condition in enumerate(conditions):
        lyapunov = arrays["lyapunov_exponents"][index]  # type: ignore[index]
        payload: dict[str, Any] = {
            "schema_version": np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int32),
            "implementation_fingerprint": np.asarray(implementation_fingerprint),
            "condition_id": np.asarray(condition.condition_id),
            "network_size": np.asarray(condition.network_size, dtype=np.int64),
            "alpha": np.asarray(condition.alpha, dtype=np.float64),
            "gain": np.asarray(condition.gain, dtype=np.float64),
            "gain_index": np.asarray(condition.gain_index, dtype=np.int64),
            "trial": np.asarray(condition.trial, dtype=np.int64),
            "weight_seed": np.asarray(condition.weight_seed, dtype=np.int64),
            "state_seed": np.asarray(condition.state_seed, dtype=np.int64),
            "execution_batch_size": np.asarray(len(conditions), dtype=np.int64),
            "batch_position": np.asarray(index, dtype=np.int64),
            "execution_batch_id": np.asarray(execution_batch_id),
            "batch_condition_ids": np.asarray(batch_condition_ids),
            "lyapunov_exponents": lyapunov,
            "local_log_stretches": arrays["local_log_stretches"][index],  # type: ignore[index]
            "floored_stretch_counts": arrays["floored_stretch_counts"][index],  # type: ignore[index]
            "floored_stretches": np.asarray(
                np.sum(arrays["floored_stretch_counts"][index]), dtype=np.int64  # type: ignore[index]
            ),
            "lambda_max": np.asarray(np.max(lyapunov), dtype=np.float64),
            "activity_mean": arrays["activity_mean"][index],  # type: ignore[index]
            "activity_rms": arrays["activity_rms"][index],  # type: ignore[index]
            "activity_std": arrays["activity_std"][index],  # type: ignore[index]
            "activity_step_rms": arrays["activity_step_rms"][index],  # type: ignore[index]
            "saturation_fraction": arrays["saturation_fraction"][index],  # type: ignore[index]
            "finite": np.asarray(arrays["finite"][index], dtype=np.bool_),  # type: ignore[index]
        }
        if arrays["activity"] is not None:
            payload["activity"] = arrays["activity"][index]

        destination = _artifact_path(output_directory, condition)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("wb") as handle:
            saver(handle, **payload)
        os.replace(temporary, destination)


def _npz_headers(
    path: Path, *, verify_payload: bool
) -> dict[str, tuple[tuple[int, ...], np.dtype[Any]]]:
    """Read NPY headers and optionally stream every ZIP member through its CRC."""

    headers: dict[str, tuple[tuple[int, ...], np.dtype[Any]]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith(".npy"):
                continue
            with archive.open(member) as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise OutputDirectoryError(
                        f"unsupported NPY header version {version} in {path}"
                    )
            headers[member.removesuffix(".npy")] = (shape, dtype)
        if verify_payload:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise OutputDirectoryError(
                    f"CRC check failed for member {corrupt_member!r} in {path}"
                )
    return headers


def _read_condition_record(
    path: Path,
    expected: ConditionSpec,
    config: ExperimentConfig,
    implementation_fingerprint: str,
    *,
    verify_payload: bool = True,
) -> dict[str, object]:
    try:
        headers = _npz_headers(path, verify_payload=verify_payload)
        steps = config.simulation.measurement_steps
        exponents = config.simulation.num_exponents
        required_shapes = {
            "schema_version": (),
            "implementation_fingerprint": (),
            "condition_id": (),
            "network_size": (),
            "alpha": (),
            "gain": (),
            "gain_index": (),
            "trial": (),
            "weight_seed": (),
            "state_seed": (),
            "execution_batch_size": (),
            "batch_position": (),
            "execution_batch_id": (),
            "lyapunov_exponents": (exponents,),
            "local_log_stretches": (steps, exponents),
            "floored_stretch_counts": (exponents,),
            "floored_stretches": (),
            "lambda_max": (),
            "activity_mean": (steps,),
            "activity_rms": (steps,),
            "activity_std": (steps,),
            "activity_step_rms": (steps,),
            "saturation_fraction": (steps,),
            "finite": (),
        }
        if config.simulation.record_activity:
            required_shapes["activity"] = (steps, expected.network_size)
        for name, required_shape in required_shapes.items():
            if name not in headers:
                raise OutputDirectoryError(f"missing array {name!r} in {path}")
            if headers[name][0] != required_shape:
                raise OutputDirectoryError(
                    f"array {name!r} in {path} has shape {headers[name][0]}, "
                    f"expected {required_shape}"
                )
        if (
            "batch_condition_ids" not in headers
            or len(headers["batch_condition_ids"][0]) != 1
        ):
            raise OutputDirectoryError(
                f"missing or invalid array 'batch_condition_ids' in {path}"
            )
        if not config.simulation.record_activity and "activity" in headers:
            raise OutputDirectoryError(f"unexpected recorded activity in {path}")

        dynamics_dtype = np.dtype(config.simulation.dtype)
        required_dtypes = {
            "schema_version": np.dtype(np.int32),
            "network_size": np.dtype(np.int64),
            "alpha": np.dtype(np.float64),
            "gain": np.dtype(np.float64),
            "gain_index": np.dtype(np.int64),
            "trial": np.dtype(np.int64),
            "weight_seed": np.dtype(np.int64),
            "state_seed": np.dtype(np.int64),
            "execution_batch_size": np.dtype(np.int64),
            "batch_position": np.dtype(np.int64),
            "lyapunov_exponents": np.dtype(np.float64),
            "local_log_stretches": np.dtype(np.float64),
            "floored_stretch_counts": np.dtype(np.int64),
            "floored_stretches": np.dtype(np.int64),
            "lambda_max": np.dtype(np.float64),
            "activity_mean": dynamics_dtype,
            "activity_rms": dynamics_dtype,
            "activity_std": dynamics_dtype,
            "activity_step_rms": dynamics_dtype,
            "saturation_fraction": dynamics_dtype,
            "finite": np.dtype(np.bool_),
        }
        if config.simulation.record_activity:
            required_dtypes["activity"] = dynamics_dtype
        for name, required_dtype in required_dtypes.items():
            if headers[name][1] != required_dtype:
                raise OutputDirectoryError(
                    f"array {name!r} in {path} has dtype {headers[name][1]}, "
                    f"expected {required_dtype}"
                )
        for name in (
            "implementation_fingerprint",
            "condition_id",
            "execution_batch_id",
            "batch_condition_ids",
        ):
            if headers[name][1].kind != "U":
                raise OutputDirectoryError(
                    f"array {name!r} in {path} must use a Unicode dtype"
                )

        with np.load(path, allow_pickle=False) as artifact:
            if int(artifact["schema_version"].item()) != ARTIFACT_SCHEMA_VERSION:
                raise OutputDirectoryError(f"unsupported artifact schema in {path}")
            if (
                str(artifact["implementation_fingerprint"].item())
                != implementation_fingerprint
            ):
                raise OutputDirectoryError(f"implementation fingerprint mismatch in {path}")
            if str(artifact["condition_id"].item()) != expected.condition_id:
                raise OutputDirectoryError(f"condition identifier mismatch in {path}")
            expected_scalars: dict[str, int | float] = {
                "network_size": expected.network_size,
                "alpha": expected.alpha,
                "gain": expected.gain,
                "gain_index": expected.gain_index,
                "trial": expected.trial,
                "weight_seed": expected.weight_seed,
                "state_seed": expected.state_seed,
            }
            for name, expected_value in expected_scalars.items():
                actual_value = artifact[name].item()
                if actual_value != expected_value:
                    raise OutputDirectoryError(
                        f"condition field {name!r} in {path} is {actual_value!r}, "
                        f"expected {expected_value!r}"
                    )
            execution_batch_size = int(artifact["execution_batch_size"].item())
            batch_position = int(artifact["batch_position"].item())
            if (
                not 1 <= execution_batch_size <= config.sweep.trials
                or not 0 <= batch_position < execution_batch_size
            ):
                raise OutputDirectoryError(f"invalid batch metadata in {path}")
            batch_condition_ids = tuple(
                str(value) for value in artifact["batch_condition_ids"].tolist()
            )
            if len(batch_condition_ids) != execution_batch_size:
                raise OutputDirectoryError(f"batch member count mismatch in {path}")
            if len(set(batch_condition_ids)) != len(batch_condition_ids):
                raise OutputDirectoryError(f"duplicate batch members in {path}")
            if batch_condition_ids[batch_position] != expected.condition_id:
                raise OutputDirectoryError(
                    f"batch position does not identify {expected.condition_id} in {path}"
                )
            valid_group_ids = {
                _condition_identity(
                    expected.network_size,
                    expected.alpha,
                    expected.gain,
                    expected.gain_index,
                    trial,
                )
                for trial in range(config.sweep.trials)
            }
            if not set(batch_condition_ids).issubset(valid_group_ids):
                raise OutputDirectoryError(f"batch contains a foreign condition in {path}")
            stored_batch_id = str(artifact["execution_batch_id"].item())
            if stored_batch_id != _batch_identity(batch_condition_ids):
                raise OutputDirectoryError(f"execution_batch_id is inconsistent in {path}")
            floor_counts = artifact["floored_stretch_counts"]
            floored_stretches = int(artifact["floored_stretches"].item())
            if np.any(floor_counts < 0) or np.any(floor_counts > steps):
                raise OutputDirectoryError(f"invalid floored-stretch counts in {path}")
            if floored_stretches != int(np.sum(floor_counts)):
                raise OutputDirectoryError(
                    f"floored_stretches is inconsistent with per-exponent counts in {path}"
                )
            lyapunov = artifact["lyapunov_exponents"]
            stored_lambda_max = float(artifact["lambda_max"].item())
            if stored_lambda_max != float(np.max(lyapunov)):
                raise OutputDirectoryError(
                    f"lambda_max is inconsistent with the spectrum in {path}"
                )
            return {
                "condition_id": expected.condition_id,
                "network_size": expected.network_size,
                "alpha": expected.alpha,
                "gain": expected.gain,
                "gain_index": expected.gain_index,
                "trial": expected.trial,
                "weight_seed": expected.weight_seed,
                "state_seed": expected.state_seed,
                "execution_batch_size": execution_batch_size,
                "batch_position": batch_position,
                "execution_batch_id": stored_batch_id,
                "lambda_max": stored_lambda_max,
                "floored_stretches": floored_stretches,
                "activity_mean": float(np.mean(artifact["activity_mean"])),
                "activity_rms": float(np.mean(artifact["activity_rms"])),
                "activity_std": float(np.mean(artifact["activity_std"])),
                "activity_step_rms": float(np.mean(artifact["activity_step_rms"])),
                "saturation_fraction": float(np.mean(artifact["saturation_fraction"])),
                "finite": bool(artifact["finite"].item()),
                "artifact": str(path),
            }
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, zlib.error) as exc:
        raise OutputDirectoryError(f"invalid or incomplete condition artifact {path}: {exc}") from exc


def _write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: Sequence[str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _environment_metadata(
    device: torch.device, implementation_identity: dict[str, Any]
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None

    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "git_revision": revision,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "implementation_fingerprint": implementation_identity["fingerprint"],
        "implementation_compatibility": implementation_identity["compatibility"],
    }
    if device.type == "cuda" and torch.cuda.is_available():
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
    return metadata


def _resume_identity(config_data: Any) -> Any:
    """Remove execution-only fields that may safely change when resuming."""

    normalized = json.loads(json.dumps(config_data))
    runtime = normalized.get("runtime", {})
    runtime.pop("batch_size", None)
    runtime.pop("memory_fraction", None)
    output = normalized.get("output", {})
    output.pop("directory", None)
    output.pop("resume", None)
    output.pop("compressed", None)
    return normalized


def _prepare_output(
    config: ExperimentConfig,
    device: torch.device,
    implementation_identity: dict[str, Any],
) -> Path:
    output_directory = Path(config.output.directory).expanduser().resolve()
    resolved_path = output_directory / "resolved_config.yaml"
    condition_directory = output_directory / "conditions"
    output_directory.mkdir(parents=True, exist_ok=True)

    current_config = OmegaConf.to_container(as_dictconfig(config), resolve=True)
    if resolved_path.exists():
        previous_config = OmegaConf.to_container(OmegaConf.load(resolved_path), resolve=True)
        if _resume_identity(previous_config) != _resume_identity(current_config):
            raise OutputDirectoryError(
                f"{output_directory} contains a different resolved configuration; "
                "choose a new output.directory"
            )
    elif any(output_directory.iterdir()):
        raise OutputDirectoryError(
            f"refusing to use non-empty output directory without resolved_config.yaml: "
            f"{output_directory}"
        )
    else:
        _atomic_text(resolved_path, OmegaConf.to_yaml(as_dictconfig(config), resolve=True))

    existing = list(condition_directory.glob("*.npz")) if condition_directory.exists() else []
    environment_path = output_directory / "environment.json"
    if environment_path.exists():
        try:
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OutputDirectoryError(f"invalid environment metadata in {environment_path}") from exc
        if not isinstance(environment, dict) or (
            environment.get("implementation_fingerprint")
            != implementation_identity["fingerprint"]
            or environment.get("implementation_compatibility")
            != implementation_identity["compatibility"]
        ):
            raise OutputDirectoryError(
                f"{output_directory} was created by a different numerical implementation; "
                "choose a new output.directory"
            )
    elif existing:
        raise OutputDirectoryError(
            f"{output_directory} has condition artifacts but no environment fingerprint; "
            "choose a new output.directory"
        )
    else:
        _atomic_json(
            environment_path, _environment_metadata(device, implementation_identity)
        )

    condition_directory.mkdir(exist_ok=True)
    if existing and not config.output.resume:
        raise OutputDirectoryError(
            f"{len(existing)} condition artifacts already exist and output.resume=false"
        )
    return output_directory


def _consolidate(
    output_directory: Path,
    conditions: Sequence[ConditionSpec],
    config: ExperimentConfig,
    implementation_fingerprint: str,
    payload_validated: set[str],
) -> tuple[Path, Path, Path, list[dict[str, object]]]:
    records = [
        _read_condition_record(
            _artifact_path(output_directory, condition),
            condition,
            config,
            implementation_fingerprint,
            verify_payload=condition.condition_id not in payload_validated,
        )
        for condition in conditions
        if _artifact_path(output_directory, condition).exists()
    ]
    if len(records) != len(conditions):
        raise OutputDirectoryError(
            f"cannot aggregate an incomplete run: found {len(records)} of "
            f"{len(conditions)} condition artifacts"
        )
    metrics_path = output_directory / "metrics.csv"
    metric_fields = [
        "condition_id",
        "network_size",
        "alpha",
        "gain",
        "gain_index",
        "trial",
        "weight_seed",
        "state_seed",
        "execution_batch_size",
        "batch_position",
        "execution_batch_id",
        "lambda_max",
        "floored_stretches",
        "activity_mean",
        "activity_rms",
        "activity_std",
        "activity_step_rms",
        "saturation_fraction",
        "finite",
        "artifact",
    ]
    _write_csv(metrics_path, records, metric_fields)

    aggregates = aggregate_trials(records)
    aggregates_path = output_directory / "aggregates.csv"
    _write_csv(
        aggregates_path,
        [asdict(row) for row in aggregates],
        list(AggregateRecord.__dataclass_fields__),
    )

    transitions = locate_transitions(aggregates)
    transitions_path = output_directory / "transitions.csv"
    _write_csv(
        transitions_path,
        [asdict(row) for row in transitions],
        list(TransitionRecord.__dataclass_fields__),
    )
    return metrics_path, aggregates_path, transitions_path, records


def run_experiment(
    config: ExperimentConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> ExperimentResult:
    """Execute a configured sweep on one device with resumable atomic artifacts."""

    validate_config(config)
    torch.set_float32_matmul_precision(config.runtime.matmul_precision)
    device = torch.device(config.runtime.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        if device.index is None:
            raise ValueError("runtime.device must select one CUDA device, for example 'cuda:0'")
        if device.index >= torch.cuda.device_count():
            raise ValueError(
                f"runtime.device {device} does not exist; found {torch.cuda.device_count()} CUDA device(s)"
            )
        try:
            torch.cuda.set_device(device)
            # Force lazy CUDA initialization before creating any output files.
            torch.empty(0, device=device)
        except Exception as exc:  # CUDA backends use several exception types.
            raise RuntimeError(
                f"could not initialize {device}: {_exception_summary(exc)}. "
                "If multiple physical GPUs are "
                "visible, select one before launch with CUDA_VISIBLE_DEVICES=<index>"
            ) from exc

    implementation_identity = _implementation_identity(device)
    implementation_fingerprint = str(implementation_identity["fingerprint"])
    conditions = expand_conditions(config)
    output_directory = _prepare_output(config, device, implementation_identity)
    manifest_path = output_directory / "manifest.json"
    artifact_exists: dict[str, bool] = {}
    payload_validated: set[str] = set()
    try:
        expected_artifact_paths = {
            _artifact_path(output_directory, condition) for condition in conditions
        }
        actual_artifact_paths = set((output_directory / "conditions").glob("*.npz"))
        if progress and actual_artifact_paths:
            progress(
                f"validating {len(actual_artifact_paths)} existing condition "
                "artifact(s), including payload CRCs"
            )
        unexpected_artifacts = sorted(actual_artifact_paths - expected_artifact_paths)
        if unexpected_artifacts:
            preview = ", ".join(path.name for path in unexpected_artifacts[:3])
            suffix = "" if len(unexpected_artifacts) <= 3 else ", ..."
            raise OutputDirectoryError(
                f"found {len(unexpected_artifacts)} unexpected condition artifact(s): "
                f"{preview}{suffix}"
            )
        for condition in conditions:
            path = _artifact_path(output_directory, condition)
            exists = path.exists()
            if exists:
                _read_condition_record(
                    path, condition, config, implementation_fingerprint
                )
                payload_validated.add(condition.condition_id)
            artifact_exists[condition.condition_id] = exists
    except Exception as exc:
        _atomic_json(
            manifest_path,
            {
                "status": "failed",
                "stage": "artifact_validation",
                "total_conditions": len(conditions),
                "error": f"{type(exc).__name__}: {_exception_summary(exc)}",
            },
        )
        raise
    resumed = sum(artifact_exists.values())
    pending = [condition for condition in conditions if not artifact_exists[condition.condition_id]]
    computed = 0

    manifest: dict[str, Any] = {
        "status": "running",
        "total_conditions": len(conditions),
        "resumed_conditions": resumed,
        "computed_conditions": 0,
        "remaining_conditions": len(pending),
        "implementation_fingerprint": implementation_fingerprint,
        "execution": {
            "device": config.runtime.device,
            "batch_size": config.runtime.batch_size,
            "memory_fraction": config.runtime.memory_fraction,
            "compressed": config.output.compressed,
        },
    }
    _atomic_json(manifest_path, manifest)
    if progress:
        progress(
            f"{len(conditions)} conditions: {resumed} resumed, {len(pending)} pending on {device}"
        )

    grouped: dict[tuple[int, float, float], list[ConditionSpec]] = defaultdict(list)
    for condition in pending:
        grouped[(condition.network_size, condition.alpha, condition.gain)].append(condition)

    try:
        for (network_size, alpha, gain), group in sorted(grouped.items()):
            position = 0
            batch_size = _auto_batch_size(config, network_size, len(group), device)
            while position < len(group):
                current_size = min(batch_size, len(group) - position)
                batch = group[position : position + current_size]
                try:
                    result = _simulate_conditions(batch, config, device)
                    if not bool(result.finite.all().item()):
                        invalid = [
                            condition.condition_id
                            for condition, is_finite in zip(
                                batch, result.finite.tolist(), strict=True
                            )
                            if not is_finite
                        ]
                        raise RuntimeError(
                            "non-finite simulation diagnostics for " + ", ".join(invalid)
                        )
                    _write_condition_artifacts(
                        output_directory,
                        batch,
                        result,
                        compressed=config.output.compressed,
                        implementation_fingerprint=implementation_fingerprint,
                    )
                except torch.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                    if current_size == 1:
                        raise RuntimeError(
                            f"condition {batch[0].condition_id} does not fit on {device}; "
                            "reduce N, exponent count, or measurement length"
                        ) from None
                    batch_size = max(1, current_size // 2)
                    if progress:
                        progress(
                            f"{device.type.upper()} OOM; retrying with "
                            f"batch_size={batch_size}"
                        )
                    continue

                position += current_size
                computed += current_size
                manifest["computed_conditions"] = computed
                manifest["remaining_conditions"] = len(pending) - computed
                _atomic_json(manifest_path, manifest)
                if progress:
                    progress(
                        f"completed N={network_size}, alpha={alpha:g}, gain={gain:.6g}, "
                        f"trials {position - current_size + 1}-{position}/{len(group)}"
                    )
                del result

        metrics_path, aggregates_path, transitions_path, records = _consolidate(
            output_directory,
            conditions,
            config,
            implementation_fingerprint,
            payload_validated,
        )
        manifest.update(
            status="complete",
            completed_conditions=len(records),
            remaining_conditions=len(conditions) - len(records),
        )
        _atomic_json(manifest_path, manifest)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _atomic_json(manifest_path, manifest)
        raise

    return ExperimentResult(
        output_directory=output_directory,
        total_conditions=len(conditions),
        computed_conditions=computed,
        resumed_conditions=resumed,
        metrics_path=metrics_path,
        aggregates_path=aggregates_path,
        transitions_path=transitions_path,
    )


__all__ = [
    "ConditionSpec",
    "ExperimentResult",
    "OutputDirectoryError",
    "estimate_condition_bytes",
    "estimate_output_bytes",
    "expand_conditions",
    "inspect_experiment",
    "run_experiment",
    "simulate_conditions",
]
