import json
from pathlib import Path
import subprocess
import sys

import run_experiment as runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "run_experiment.py"
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"


def _run_script(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUN_SCRIPT), *arguments],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def test_inspect_script_reports_estimates_without_writing(tmp_path: Path) -> None:
    output_directory = tmp_path / "inspect-only"
    completed = _run_script(
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
        cwd=tmp_path,
    )

    report = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert report["conditions"] == 2
    assert report["device"] == "cpu"
    assert report["device_available"] is True
    assert report["output_bytes"] > 0
    assert report["largest_condition_bytes"] > 0
    assert report["output_human"]
    assert report["largest_condition_human"]
    assert not output_directory.exists()


def test_run_script_executes_and_resumes_tiny_cpu_sweep(tmp_path: Path) -> None:
    output_directory = tmp_path / "script-run"
    arguments = (
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
    )
    first = _run_script(*arguments, cwd=tmp_path)

    assert first.returncode == 0
    assert first.stderr == ""
    assert "complete: 1 conditions (1 computed, 0 resumed)" in first.stdout
    assert f"results: {output_directory.resolve()}" in first.stdout
    artifact_path = next((output_directory / "conditions").glob("*.npz"))
    modification_time = artifact_path.stat().st_mtime_ns

    resumed = _run_script(*arguments, cwd=tmp_path)
    assert resumed.returncode == 0
    assert resumed.stderr == ""
    assert "complete: 1 conditions (0 computed, 1 resumed)" in resumed.stdout
    assert artifact_path.stat().st_mtime_ns == modification_time


def test_script_returns_error_status_for_invalid_override(tmp_path: Path) -> None:
    completed = _run_script(
        "inspect",
        "--config",
        str(SMOKE_CONFIG),
        "simulation.num_exponents=1000",
        cwd=tmp_path,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "run_experiment.py: error:" in completed.stderr


def test_main_reports_sampling_floating_point_error_without_traceback(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    def fail_run(*args, **kwargs):
        raise FloatingPointError("extreme stable draw was non-finite")

    monkeypatch.setattr(runner, "run_experiment", fail_run)
    status = runner.main(
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
    assert captured.err == "run_experiment.py: error: extreme stable draw was non-finite\n"
