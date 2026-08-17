"""Structured configuration for HeavyRNN experiments.

Configuration is represented by ordinary dataclasses at runtime.  OmegaConf is
used only at the loading boundary so YAML files and command-line dot-list
overrides remain convenient while unknown keys and incompatible value types are
rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Union

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException


BatchSize = Union[int, str]


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is internally inconsistent."""


@dataclass
class GainSpec:
    """Gain grid definition.

    ``mode="logspace"`` uses ``start``, ``stop``, and ``num``.  In
    ``mode="explicit"``, ``values`` must contain a strictly increasing list of
    positive gains.  The unused fields are deliberately retained in both modes
    so switching modes with a dot-list override does not require rebuilding the
    entire nested object.
    """

    mode: str = "logspace"
    start: float = 1.0e-2
    stop: float = 1.0e1
    num: int = 50
    values: list[float] = field(default_factory=list)


@dataclass
class SweepConfig:
    """Network conditions included in an experiment sweep."""

    network_sizes: list[int] = field(
        default_factory=lambda: [1000, 3000, 10000]
    )
    alphas: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])
    gains: GainSpec = field(default_factory=GainSpec)
    trials: int = 10


@dataclass
class SimulationConfig:
    """Numerical dynamics and measurement settings."""

    warmup_steps: int = 2900
    measurement_steps: int = 100
    num_exponents: int = 100
    dtype: str = "float32"
    initial_state_std: float = 1.0
    record_activity: bool = True
    saturation_threshold: float = 0.99


@dataclass
class RuntimeConfig:
    """Execution settings that do not change the modeled system."""

    device: str = "cuda:0"
    batch_size: BatchSize = "auto"
    base_seed: int = 40
    matmul_precision: str = "highest"
    memory_fraction: float = 0.7


@dataclass
class OutputConfig:
    """Artifact and restart behavior."""

    directory: str = "runs/flagship"
    resume: bool = True
    compressed: bool = False


@dataclass
class ExperimentConfig:
    """Complete configuration for a HeavyRNN experiment."""

    sweep: SweepConfig = field(default_factory=SweepConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def gain_values(self) -> tuple[float, ...]:
        """Return the validated gain grid as immutable floating-point values."""

        return expand_gains(self.sweep.gains)


def expand_gains(spec: GainSpec) -> tuple[float, ...]:
    """Expand a gain specification without requiring NumPy.

    Validation is performed first, and configured log-space endpoints are copied
    exactly into the result to avoid tiny endpoint drift from log/exp roundoff.
    """

    _validate_gain_spec(spec)
    if spec.mode == "explicit":
        return tuple(float(value) for value in spec.values)
    if spec.num == 1:
        return (float(spec.start),)

    log_start = math.log(spec.start)
    log_stop = math.log(spec.stop)
    step = (log_stop - log_start) / (spec.num - 1)
    gains = [math.exp(log_start + index * step) for index in range(spec.num)]
    gains[0] = float(spec.start)
    gains[-1] = float(spec.stop)
    return tuple(gains)


def validate_config(config: ExperimentConfig) -> None:
    """Validate cross-field invariants for an experiment configuration."""

    sweep = config.sweep
    simulation = config.simulation
    runtime = config.runtime
    output = config.output

    if not sweep.network_sizes:
        raise ConfigurationError("sweep.network_sizes must not be empty")
    for index, size in enumerate(sweep.network_sizes):
        _require_int(f"sweep.network_sizes[{index}]", size, minimum=1)
    if len(set(sweep.network_sizes)) != len(sweep.network_sizes):
        raise ConfigurationError("sweep.network_sizes must not contain duplicates")

    if not sweep.alphas:
        raise ConfigurationError("sweep.alphas must not be empty")
    for index, alpha in enumerate(sweep.alphas):
        _require_finite(f"sweep.alphas[{index}]", alpha)
        if not 0.0 < alpha <= 2.0:
            raise ConfigurationError(
                f"sweep.alphas[{index}] must satisfy 0 < alpha <= 2"
            )
    if len(set(sweep.alphas)) != len(sweep.alphas):
        raise ConfigurationError("sweep.alphas must not contain duplicates")

    _validate_gain_spec(sweep.gains)
    _require_int("sweep.trials", sweep.trials, minimum=1)

    _require_int("simulation.warmup_steps", simulation.warmup_steps, minimum=0)
    _require_int(
        "simulation.measurement_steps", simulation.measurement_steps, minimum=1
    )
    _require_int("simulation.num_exponents", simulation.num_exponents, minimum=1)
    if simulation.num_exponents > min(sweep.network_sizes):
        raise ConfigurationError(
            "simulation.num_exponents cannot exceed the smallest network size"
        )
    if simulation.dtype not in {"float32", "float64"}:
        raise ConfigurationError(
            "simulation.dtype must be either 'float32' or 'float64'"
        )
    _require_finite("simulation.initial_state_std", simulation.initial_state_std)
    if simulation.initial_state_std <= 0.0:
        raise ConfigurationError("simulation.initial_state_std must be positive")
    if not isinstance(simulation.record_activity, bool):
        raise ConfigurationError("simulation.record_activity must be a boolean")
    _require_finite(
        "simulation.saturation_threshold", simulation.saturation_threshold
    )
    if not 0.0 < simulation.saturation_threshold < 1.0:
        raise ConfigurationError(
            "simulation.saturation_threshold must satisfy 0 < threshold < 1"
        )

    if not isinstance(runtime.device, str) or not runtime.device.strip():
        raise ConfigurationError("runtime.device must be a non-empty string")
    if runtime.batch_size != "auto":
        _require_int("runtime.batch_size", runtime.batch_size, minimum=1)
    _require_int("runtime.base_seed", runtime.base_seed, minimum=0)
    if runtime.base_seed >= 2**63:
        raise ConfigurationError("runtime.base_seed must be smaller than 2**63")
    if runtime.matmul_precision not in {"highest", "high", "medium"}:
        raise ConfigurationError(
            "runtime.matmul_precision must be 'highest', 'high', or 'medium'"
        )
    _require_finite("runtime.memory_fraction", runtime.memory_fraction)
    if not 0.0 < runtime.memory_fraction <= 0.9:
        raise ConfigurationError(
            "runtime.memory_fraction must satisfy 0 < value <= 0.9"
        )

    if not isinstance(output.directory, str) or not output.directory.strip():
        raise ConfigurationError("output.directory must be a non-empty string")
    if not isinstance(output.resume, bool):
        raise ConfigurationError("output.resume must be a boolean")
    if not isinstance(output.compressed, bool):
        raise ConfigurationError("output.compressed must be a boolean")


def load_config(
    path: str | Path | None = None,
    overrides: Sequence[str] | str | None = None,
) -> ExperimentConfig:
    """Load, merge, resolve, and validate an experiment configuration.

    Merge precedence is structured flagship defaults, then ``path``, then
    OmegaConf dot-list ``overrides``.  Unknown keys are rejected by the
    structured schema.  Passing ``None`` as ``path`` returns validated flagship
    defaults.
    """

    override_items = (
        [overrides] if isinstance(overrides, str) else list(overrides or ())
    )
    schema = OmegaConf.structured(ExperimentConfig)

    try:
        file_config: DictConfig = (
            OmegaConf.load(Path(path)) if path is not None else OmegaConf.create()
        )
        dotlist_config = OmegaConf.from_dotlist(override_items)
        merged = OmegaConf.merge(schema, file_config, dotlist_config)
        OmegaConf.resolve(merged)
        config = OmegaConf.to_object(merged)
    except OmegaConfBaseException as exc:
        source = f" from {path}" if path is not None else ""
        raise ConfigurationError(f"could not load configuration{source}: {exc}") from exc

    if not isinstance(config, ExperimentConfig):  # defensive; schema guarantees this
        raise ConfigurationError("resolved configuration has an unexpected root type")
    validate_config(config)
    return config


def as_dictconfig(config: ExperimentConfig) -> DictConfig:
    """Convert a validated config to a resolved OmegaConf object for serialization."""

    validate_config(config)
    result = OmegaConf.structured(config)
    OmegaConf.resolve(result)
    return result


def _validate_gain_spec(spec: GainSpec) -> None:
    if spec.mode not in {"logspace", "explicit"}:
        raise ConfigurationError("sweep.gains.mode must be 'logspace' or 'explicit'")

    if spec.mode == "logspace":
        _require_finite("sweep.gains.start", spec.start)
        _require_finite("sweep.gains.stop", spec.stop)
        _require_int("sweep.gains.num", spec.num, minimum=1)
        if spec.start <= 0.0 or spec.stop <= 0.0:
            raise ConfigurationError("log-spaced gains must be positive")
        if spec.stop < spec.start:
            raise ConfigurationError("sweep.gains.stop must be >= sweep.gains.start")
        if spec.num == 1 and spec.start != spec.stop:
            raise ConfigurationError(
                "a one-point log-space grid requires identical start and stop"
            )
        if spec.num > 1 and spec.stop == spec.start:
            raise ConfigurationError(
                "a multi-point log-space grid requires stop greater than start"
            )
        return

    if not spec.values:
        raise ConfigurationError(
            "sweep.gains.values must not be empty when mode is 'explicit'"
        )
    previous = -math.inf
    for index, value in enumerate(spec.values):
        _require_finite(f"sweep.gains.values[{index}]", value)
        if value <= 0.0:
            raise ConfigurationError("explicit gains must be positive")
        if value <= previous:
            raise ConfigurationError(
                "sweep.gains.values must be strictly increasing without duplicates"
            )
        previous = value


def _require_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise ConfigurationError(f"{name} must be a finite number")


def _require_int(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if minimum == 0 else f">= {minimum}"
        raise ConfigurationError(f"{name} must be an integer {qualifier}")


__all__ = [
    "BatchSize",
    "ConfigurationError",
    "ExperimentConfig",
    "GainSpec",
    "OutputConfig",
    "RuntimeConfig",
    "SimulationConfig",
    "SweepConfig",
    "as_dictconfig",
    "expand_gains",
    "load_config",
    "validate_config",
]
