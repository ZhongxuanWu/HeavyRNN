"""Internal batched heavy-tailed RNN simulation and Lyapunov analysis."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class SimulationBatchResult:
    """Outputs for a batch of independent recurrent networks.

    Tensors remain on the simulation device.  The experiment runner is responsible
    for transferring them to CPU and persisting them.
    """

    activity: Tensor | None
    lyapunov_exponents: Tensor
    local_log_stretches: Tensor
    floored_stretch_counts: Tensor
    activity_mean: Tensor
    activity_rms: Tensor
    activity_std: Tensor
    activity_step_rms: Tensor
    saturation_fraction: Tensor
    finite: Tensor


class NumericalSimulationError(RuntimeError):
    """Raised when a batched simulation cannot produce finite diagnostics."""


def stable_sech_squared(x: Tensor) -> Tensor:
    """Return sech(x)^2 without overflowing cosh for large preactivations."""

    exp_term = torch.exp(-2.0 * torch.abs(x))
    return 4.0 * exp_term / torch.square(1.0 + exp_term)


def _validate_inputs(weights: Tensor, initial_state: Tensor, num_exponents: int) -> None:
    if weights.ndim != 3:
        raise ValueError(f"weights must have shape [batch, N, N], got {tuple(weights.shape)}")
    batch, rows, cols = weights.shape
    if rows != cols:
        raise ValueError("recurrent weight matrices must be square")
    if initial_state.shape != (batch, rows):
        raise ValueError(
            f"initial_state must have shape {(batch, rows)}, got {tuple(initial_state.shape)}"
        )
    if initial_state.device != weights.device:
        raise ValueError("weights and initial_state must be on the same device")
    if initial_state.dtype != weights.dtype:
        raise ValueError("weights and initial_state must have the same dtype")
    if not 1 <= num_exponents <= rows:
        raise ValueError(f"num_exponents must be in [1, N], got {num_exponents} for N={rows}")


@torch.inference_mode()
def simulate_batch(
    weights: Tensor,
    initial_state: Tensor,
    *,
    warmup_steps: int,
    measurement_steps: int,
    num_exponents: int,
    saturation_threshold: float = 0.99,
    record_activity: bool = True,
) -> SimulationBatchResult:
    """Run independent autonomous tanh RNNs and estimate leading exponents.

    The matrices are quenched within each trajectory.  During measurement the
    canonical tangent map ``diag(sech(preactivation)^2) @ W`` is applied to the
    first ``num_exponents`` coordinate directions, followed by a reduced QR at
    every time step.
    """

    _validate_inputs(weights, initial_state, num_exponents)
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if measurement_steps <= 0:
        raise ValueError("measurement_steps must be positive")
    if not 0.0 < saturation_threshold < 1.0:
        raise ValueError("saturation_threshold must be strictly between 0 and 1")

    batch, hidden_size, _ = weights.shape
    device = weights.device
    dynamics_dtype = weights.dtype
    h = initial_state.clone()

    for _ in range(warmup_steps):
        h = torch.tanh(torch.bmm(weights, h.unsqueeze(-1)).squeeze(-1))

    if not bool(torch.isfinite(h).all().item()):
        raise NumericalSimulationError("non-finite hidden state after warmup")

    basis = torch.eye(
        hidden_size,
        num_exponents,
        dtype=dynamics_dtype,
        device=device,
    )
    q = basis.unsqueeze(0).expand(batch, -1, -1).clone()

    activity = (
        torch.empty(
            batch,
            measurement_steps,
            hidden_size,
            dtype=dynamics_dtype,
            device=device,
        )
        if record_activity
        else None
    )
    local_log_stretches = torch.empty(
        batch,
        measurement_steps,
        num_exponents,
        dtype=torch.float64,
        device=device,
    )
    activity_mean = torch.empty(batch, measurement_steps, dtype=dynamics_dtype, device=device)
    activity_rms = torch.empty_like(activity_mean)
    activity_std = torch.empty_like(activity_mean)
    activity_step_rms = torch.empty_like(activity_mean)
    saturation_fraction = torch.empty_like(activity_mean)
    floored_stretch_counts = torch.zeros(
        batch, num_exponents, dtype=torch.int64, device=device
    )

    min_positive = torch.finfo(dynamics_dtype).tiny
    for step in range(measurement_steps):
        previous_h = h
        preactivation = torch.bmm(weights, previous_h.unsqueeze(-1)).squeeze(-1)
        h = torch.tanh(preactivation)

        tangent = torch.bmm(weights, q)
        tangent.mul_(stable_sech_squared(preactivation).unsqueeze(-1))
        q, r = torch.linalg.qr(tangent, mode="reduced")

        diagonal = torch.diagonal(r, dim1=-2, dim2=-1)
        signs = torch.where(diagonal < 0, -torch.ones_like(diagonal), torch.ones_like(diagonal))
        q.mul_(signs.unsqueeze(-2))
        positive_diagonal = diagonal * signs
        floored_stretch_counts.add_((positive_diagonal < min_positive).to(torch.int64))
        local_log_stretches[:, step] = torch.log(
            positive_diagonal.clamp_min(min_positive).to(torch.float64)
        )

        if activity is not None:
            activity[:, step] = h
        mean = h.mean(dim=-1)
        mean_square = torch.square(h).mean(dim=-1)
        activity_mean[:, step] = mean
        activity_rms[:, step] = torch.sqrt(mean_square)
        activity_std[:, step] = torch.sqrt(torch.clamp_min(mean_square - torch.square(mean), 0.0))
        activity_step_rms[:, step] = torch.sqrt(torch.square(h - previous_h).mean(dim=-1))
        saturation_fraction[:, step] = (torch.abs(h) >= saturation_threshold).to(dynamics_dtype).mean(
            dim=-1
        )

    lyapunov_exponents = local_log_stretches.mean(dim=1)
    finite = torch.isfinite(lyapunov_exponents).all(dim=1)
    if activity is not None:
        finite &= torch.isfinite(activity).all(dim=(1, 2))
    else:
        finite &= torch.isfinite(h).all(dim=1)

    return SimulationBatchResult(
        activity=activity,
        lyapunov_exponents=lyapunov_exponents,
        local_log_stretches=local_log_stretches,
        floored_stretch_counts=floored_stretch_counts,
        activity_mean=activity_mean,
        activity_rms=activity_rms,
        activity_std=activity_std,
        activity_step_rms=activity_step_rms,
        saturation_fraction=saturation_fraction,
        finite=finite,
    )
