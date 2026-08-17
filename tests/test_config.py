from pathlib import Path

import pytest

from config import ConfigurationError, GainSpec, expand_gains, load_config
from experiment import expand_conditions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLAGSHIP_CONFIG = PROJECT_ROOT / "configs" / "flagship.yaml"
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"


def test_flagship_configuration_expands_to_4500_conditions() -> None:
    config = load_config(FLAGSHIP_CONFIG)
    conditions = expand_conditions(config)

    assert len(conditions) == 3 * 3 * 50 * 10 == 4500
    assert config.gain_values()[0] == 0.01
    assert config.gain_values()[-1] == 10.0
    assert config.runtime.base_seed == 40
    assert len({condition.condition_id for condition in conditions}) == 4500


def test_dotlist_overrides_replace_nested_values() -> None:
    config = load_config(
        SMOKE_CONFIG,
        [
            "sweep.network_sizes=[5]",
            "sweep.alphas=[1.5]",
            "sweep.gains.mode=explicit",
            "sweep.gains.values=[0.25,0.75]",
            "sweep.trials=3",
            "simulation.num_exponents=2",
            "runtime.device=cpu",
            "runtime.batch_size=1",
            "runtime.base_seed=17",
            "output.resume=false",
        ],
    )

    assert config.sweep.network_sizes == [5]
    assert config.sweep.alphas == [1.5]
    assert config.gain_values() == (0.25, 0.75)
    assert config.sweep.trials == 3
    assert config.simulation.num_exponents == 2
    assert config.runtime.batch_size == 1
    assert config.runtime.base_seed == 17
    assert config.output.resume is False


@pytest.mark.parametrize(
    "override",
    [
        "unknown.option=1",
        "sweep.alphas=[0.0]",
        "sweep.network_sizes=[4,4]",
        "sweep.gains.values=[1.0,0.5]",
        "simulation.num_exponents=33",
        "runtime.batch_size=0",
        "runtime.memory_fraction=1.0",
    ],
)
def test_invalid_configuration_is_rejected(override: str) -> None:
    with pytest.raises(ConfigurationError):
        load_config(SMOKE_CONFIG, override)


def test_one_point_logspace_requires_equal_endpoints() -> None:
    with pytest.raises(ConfigurationError, match="identical start and stop"):
        expand_gains(GainSpec(mode="logspace", start=0.5, stop=1.0, num=1))


def test_multi_point_logspace_rejects_duplicate_values() -> None:
    with pytest.raises(ConfigurationError, match="stop greater than start"):
        expand_gains(GainSpec(mode="logspace", start=0.5, stop=0.5, num=2))
