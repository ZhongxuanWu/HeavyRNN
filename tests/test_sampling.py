import math

import pytest
import torch

from heavyrnn.sampling import (
    RECURRENT_WEIGHT_STREAM,
    derive_seed,
    sample_recurrent_weight_batch,
    sample_recurrent_weights,
    sample_symmetric_alpha_stable,
)


def test_alpha_one_matches_cauchy_special_case() -> None:
    seed = 9182
    actual = sample_symmetric_alpha_stable((128,), 1.0, seed=seed)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    angles = torch.empty(128, dtype=torch.float64)
    angles.uniform_(-math.pi / 2.0, math.pi / 2.0, generator=generator)
    expected = angles.tan()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_alpha_two_matches_sqrt_two_gaussian_special_case() -> None:
    seed = 4751
    actual = sample_symmetric_alpha_stable((128,), 2.0, seed=seed)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    expected = torch.empty(128, dtype=torch.float64)
    expected.normal_(mean=0.0, std=math.sqrt(2.0), generator=generator)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("frequency", [0.5, 1.0])
def test_empirical_characteristic_function(alpha: float, frequency: float) -> None:
    samples = sample_symmetric_alpha_stable(
        (60_000,), alpha, seed=10_000 + int(10 * alpha)
    )
    empirical_real = torch.cos(frequency * samples).mean().item()
    empirical_imaginary = torch.sin(frequency * samples).mean().item()
    expected = math.exp(-(abs(frequency) ** alpha))

    assert empirical_real == pytest.approx(expected, abs=0.015)
    assert empirical_imaginary == pytest.approx(0.0, abs=0.015)


def test_recurrent_weights_apply_flagship_scaling() -> None:
    n = 7
    alpha = 1.5
    gain = 0.8
    base_seed = 314
    trial = 2
    seed = derive_seed(
        base_seed,
        n=n,
        alpha=alpha,
        gain=gain,
        trial=trial,
        stream=RECURRENT_WEIGHT_STREAM,
    )
    raw = sample_symmetric_alpha_stable((n, n), alpha, seed=seed)
    expected = (raw * (gain * n ** (-1.0 / alpha))).to(torch.float32)

    actual = sample_recurrent_weights(
        n,
        alpha,
        gain,
        base_seed=base_seed,
        trial=trial,
        dtype=torch.float32,
        chunk_rows=n,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_seeded_sampling_is_deterministic_and_seed_sensitive() -> None:
    first = sample_symmetric_alpha_stable((256,), 1.5, seed=123)
    repeated = sample_symmetric_alpha_stable((256,), 1.5, seed=123)
    different = sample_symmetric_alpha_stable((256,), 1.5, seed=124)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


def test_weight_batch_matches_single_trials_independent_of_order() -> None:
    kwargs = dict(
        n=6,
        alpha=1.5,
        gain=0.7,
        base_seed=99,
        dtype=torch.float64,
        chunk_rows=6,
    )
    batch = sample_recurrent_weight_batch(trials=[3, 0], **kwargs)
    trial_three = sample_recurrent_weights(trial=3, **kwargs)
    trial_zero = sample_recurrent_weights(trial=0, **kwargs)

    torch.testing.assert_close(batch[0], trial_three, rtol=0.0, atol=0.0)
    torch.testing.assert_close(batch[1], trial_zero, rtol=0.0, atol=0.0)
