import pytest
import torch

from heavyrnn.simulation import simulate_batch, stable_sech_squared


def test_stable_sech_squared_matches_direct_formula_and_handles_extremes() -> None:
    moderate = torch.linspace(-20.0, 20.0, 81, dtype=torch.float64)
    expected = torch.cosh(moderate).reciprocal().square()
    torch.testing.assert_close(
        stable_sech_squared(moderate), expected, rtol=2e-15, atol=0.0
    )

    extreme = torch.tensor([-1000.0, 0.0, 1000.0], dtype=torch.float64)
    actual = stable_sech_squared(extreme)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    )


def _explicit_reference(
    weights: torch.Tensor,
    initial_state: torch.Tensor,
    *,
    warmup_steps: int,
    measurement_steps: int,
    num_exponents: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Small-system reference that materializes every tangent Jacobian."""
    h = initial_state.clone()
    for _ in range(warmup_steps):
        h = torch.tanh(weights @ h)

    q = torch.eye(weights.shape[0], num_exponents, dtype=weights.dtype)
    activity = []
    log_stretches = []
    for _ in range(measurement_steps):
        preactivation = weights @ h
        h = torch.tanh(preactivation)
        jacobian = torch.diag(stable_sech_squared(preactivation)) @ weights
        q, r = torch.linalg.qr(jacobian @ q, mode="reduced")

        diagonal = torch.diagonal(r)
        signs = torch.where(
            diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal)
        )
        q = q * signs.unsqueeze(0)
        log_stretches.append(torch.log(diagonal * signs))
        activity.append(h.clone())

    return torch.stack(activity), torch.stack(log_stretches)


def test_simulation_matches_manual_recurrence_and_explicit_jacobian() -> None:
    weights = torch.tensor(
        [[0.35, -0.20, 0.10], [0.15, 0.25, -0.30], [-0.05, 0.20, 0.40]],
        dtype=torch.float64,
    )
    initial_state = torch.tensor([0.4, -0.2, 0.1], dtype=torch.float64)
    expected_activity, expected_stretches = _explicit_reference(
        weights,
        initial_state,
        warmup_steps=2,
        measurement_steps=5,
        num_exponents=2,
    )

    result = simulate_batch(
        weights.unsqueeze(0),
        initial_state.unsqueeze(0),
        warmup_steps=2,
        measurement_steps=5,
        num_exponents=2,
    )

    torch.testing.assert_close(
        result.activity[0], expected_activity, rtol=1e-14, atol=1e-14
    )
    torch.testing.assert_close(
        result.local_log_stretches[0], expected_stretches, rtol=1e-13, atol=1e-13
    )
    torch.testing.assert_close(
        result.lyapunov_exponents[0],
        expected_stretches.mean(dim=0),
        rtol=1e-13,
        atol=1e-13,
    )


def test_diagonal_zero_state_has_analytic_qr_exponents() -> None:
    diagonal = torch.tensor([0.5, 1.2, -0.8], dtype=torch.float64)
    weights = torch.diag(diagonal).unsqueeze(0)
    result = simulate_batch(
        weights,
        torch.zeros((1, 3), dtype=torch.float64),
        warmup_steps=3,
        measurement_steps=7,
        num_exponents=3,
    )

    expected = torch.log(diagonal.abs())
    torch.testing.assert_close(result.activity, torch.zeros_like(result.activity))
    torch.testing.assert_close(
        result.local_log_stretches[0], expected.expand(7, -1), rtol=1e-14, atol=1e-14
    )
    torch.testing.assert_close(
        result.lyapunov_exponents[0], expected, rtol=1e-14, atol=1e-14
    )
    assert result.finite.tolist() == [True]


def test_batched_simulation_matches_individual_runs() -> None:
    weights = torch.tensor(
        [
            [[0.30, -0.10, 0.05], [0.08, 0.25, -0.06], [-0.04, 0.12, 0.20]],
            [[0.18, 0.07, -0.11], [-0.09, 0.32, 0.04], [0.06, -0.03, 0.27]],
        ],
        dtype=torch.float64,
    )
    initial_state = torch.tensor(
        [[0.2, -0.3, 0.1], [-0.15, 0.05, 0.25]], dtype=torch.float64
    )
    batched = simulate_batch(
        weights,
        initial_state,
        warmup_steps=2,
        measurement_steps=4,
        num_exponents=2,
    )

    tensor_fields = (
        "activity",
        "lyapunov_exponents",
        "local_log_stretches",
        "floored_stretch_counts",
        "activity_mean",
        "activity_rms",
        "activity_std",
        "activity_step_rms",
        "saturation_fraction",
        "finite",
    )
    for index in range(2):
        individual = simulate_batch(
            weights[index : index + 1],
            initial_state[index : index + 1],
            warmup_steps=2,
            measurement_steps=4,
            num_exponents=2,
        )
        for field in tensor_fields:
            torch.testing.assert_close(
                getattr(batched, field)[index],
                getattr(individual, field)[0],
                rtol=1e-13,
                atol=1e-13,
            )


def test_activity_recording_can_be_disabled_without_losing_diagnostics() -> None:
    result = simulate_batch(
        torch.eye(3, dtype=torch.float64).unsqueeze(0) * 0.5,
        torch.ones((1, 3), dtype=torch.float64),
        warmup_steps=1,
        measurement_steps=2,
        num_exponents=2,
        record_activity=False,
    )

    assert result.activity is None
    assert result.activity_rms.shape == (1, 2)
    assert result.lyapunov_exponents.shape == (1, 2)
    assert result.finite.tolist() == [True]


def test_zero_qr_stretches_are_floored_and_reported() -> None:
    result = simulate_batch(
        torch.diag(torch.tensor([0.0, 0.5], dtype=torch.float32)).unsqueeze(0),
        torch.zeros((1, 2), dtype=torch.float32),
        warmup_steps=0,
        measurement_steps=3,
        num_exponents=2,
    )

    assert result.floored_stretch_counts.tolist() == [[3, 0]]
    assert result.lyapunov_exponents[0, 0].item() == pytest.approx(
        torch.log(torch.tensor(torch.finfo(torch.float32).tiny)).item()
    )
    assert result.finite.tolist() == [True]


def test_tangent_update_matches_finite_difference_at_current_transition() -> None:
    weights = torch.tensor(
        [[0.4, -0.2], [0.1, 0.3]], dtype=torch.float64
    ).unsqueeze(0)
    state = torch.tensor([[0.25, -0.4]], dtype=torch.float64)
    epsilon = 1.0e-6
    direction = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    plus = torch.tanh((weights @ (state + epsilon * direction).unsqueeze(-1)).squeeze(-1))
    minus = torch.tanh((weights @ (state - epsilon * direction).unsqueeze(-1)).squeeze(-1))
    finite_difference_norm = torch.linalg.vector_norm((plus - minus) / (2.0 * epsilon))
    result = simulate_batch(
        weights,
        state,
        warmup_steps=0,
        measurement_steps=1,
        num_exponents=1,
    )

    assert result.local_log_stretches[0, 0, 0].item() == pytest.approx(
        torch.log(finite_difference_norm).item(), rel=1.0e-10, abs=1.0e-10
    )


@pytest.mark.cuda
def test_cuda_simulation_smoke() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    try:
        device = torch.device("cuda:0")
        weights = (torch.eye(4, device=device) * 0.8).unsqueeze(0)
    except Exception as exc:  # CUDA discovery can fail with backend-specific errors.
        pytest.skip(f"CUDA could not initialize in this process: {exc}")

    result = simulate_batch(
        weights,
        torch.zeros((1, 4), device=device),
        warmup_steps=1,
        measurement_steps=2,
        num_exponents=2,
    )
    assert result.activity is not None
    assert result.activity.is_cuda
    assert result.finite.tolist() == [True]


@pytest.mark.parametrize(
    ("warmup_steps", "measurement_steps"),
    [(-1, 1), (0, 0)],
)
def test_simulation_rejects_invalid_step_counts(
    warmup_steps: int, measurement_steps: int
) -> None:
    weights = torch.eye(2, dtype=torch.float64).unsqueeze(0)
    state = torch.zeros((1, 2), dtype=torch.float64)
    with pytest.raises(ValueError):
        simulate_batch(
            weights,
            state,
            warmup_steps=warmup_steps,
            measurement_steps=measurement_steps,
            num_exponents=1,
        )
