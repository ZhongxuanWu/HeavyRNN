"""Lightweight simulation tools for heavy-tailed recurrent neural networks."""

from .config import ExperimentConfig, load_config
from .experiment import (
    ConditionSpec,
    ExperimentResult,
    expand_conditions,
    run_experiment,
    simulate_conditions,
)
from .simulation import SimulationBatchResult, simulate_batch

__all__ = [
    "ConditionSpec",
    "ExperimentConfig",
    "ExperimentResult",
    "SimulationBatchResult",
    "expand_conditions",
    "load_config",
    "run_experiment",
    "simulate_batch",
    "simulate_conditions",
]

__version__ = "0.1.0"
