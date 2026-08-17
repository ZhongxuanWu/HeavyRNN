import json
from pathlib import Path

from heavyrnn.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"


def test_inspect_cli_reports_estimates_without_writing(tmp_path: Path, capsys) -> None:
    output_directory = tmp_path / "inspect-only"
    status = main(
        [
            "inspect",
            "--config",
            str(SMOKE_CONFIG),
            "sweep.network_sizes=[4]",
            "sweep.alphas=[2.0]",
            "sweep.gains.values=[0.5]",
            "sweep.trials=2",
            "simulation.num_exponents=2",
            "runtime.device=cpu",
            f"output.directory={output_directory}",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert status == 0
    assert captured.err == ""
    assert report["conditions"] == 2
    assert report["device"] == "cpu"
    assert report["device_available"] is True
    assert report["output_bytes"] > 0
    assert report["largest_condition_bytes"] > 0
    assert report["output_human"]
    assert report["largest_condition_human"]
    assert not output_directory.exists()


def test_run_cli_executes_tiny_cpu_sweep(tmp_path: Path, capsys) -> None:
    output_directory = tmp_path / "cli-run"
    status = main(
        [
            "run",
            "--config",
            str(SMOKE_CONFIG),
            "sweep.network_sizes=[3]",
            "sweep.alphas=[2.0]",
            "sweep.gains.values=[0.7]",
            "sweep.trials=1",
            "simulation.warmup_steps=1",
            "simulation.measurement_steps=2",
            "simulation.num_exponents=1",
            "runtime.device=cpu",
            "runtime.batch_size=1",
            f"output.directory={output_directory}",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert captured.err == ""
    assert "complete: 1 conditions (1 computed, 0 resumed)" in captured.out
    assert f"results: {output_directory.resolve()}" in captured.out
    assert len(list((output_directory / "conditions").glob("*.npz"))) == 1


def test_cli_returns_error_status_for_invalid_override(capsys) -> None:
    status = main(
        [
            "inspect",
            "--config",
            str(SMOKE_CONFIG),
            "simulation.num_exponents=1000",
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert "heavyrnn: error:" in captured.err


def test_cli_reports_sampling_floating_point_error_without_traceback(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    def fail_run(*args, **kwargs):
        raise FloatingPointError("extreme stable draw was non-finite")

    monkeypatch.setattr("heavyrnn.cli.run_experiment", fail_run)
    status = main(
        [
            "run",
            "--config",
            str(SMOKE_CONFIG),
            f"output.directory={tmp_path / 'unused'}",
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert captured.err == "heavyrnn: error: extreme stable draw was non-finite\n"
