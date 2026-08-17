import csv
from dataclasses import replace
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from config import load_config
from experiment import (
    OutputDirectoryError,
    _numerical_source_digest,
    _simulate_conditions,
    expand_conditions,
    run_experiment,
)
from sampling import INITIAL_STATE_STREAM, RECURRENT_WEIGHT_STREAM, derive_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"


def _tiny_config(output_directory: Path):
    return load_config(
        SMOKE_CONFIG,
        [
            "sweep.network_sizes=[4]",
            "sweep.alphas=[2.0]",
            "sweep.gains.mode=explicit",
            "sweep.gains.values=[0.5,1.25]",
            "sweep.trials=2",
            "simulation.warmup_steps=2",
            "simulation.measurement_steps=3",
            "simulation.num_exponents=2",
            "simulation.dtype=float64",
            "runtime.device=cpu",
            "runtime.batch_size=2",
            "runtime.base_seed=1234",
            f"output.directory={output_directory}",
            "output.resume=true",
            "output.compressed=false",
        ],
    )


def test_condition_seeds_are_order_independent_and_stream_specific(tmp_path: Path) -> None:
    common = [
        "sweep.gains.mode=explicit",
        "sweep.gains.values=[0.75]",
        "sweep.trials=2",
        "simulation.num_exponents=2",
        "runtime.base_seed=42",
        f"output.directory={tmp_path / 'unused'}",
    ]
    forward = load_config(
        SMOKE_CONFIG,
        ["sweep.network_sizes=[3,4]", "sweep.alphas=[1.0,2.0]", *common],
    )
    reversed_order = load_config(
        SMOKE_CONFIG,
        ["sweep.network_sizes=[4,3]", "sweep.alphas=[2.0,1.0]", *common],
    )

    def indexed(config):
        return {
            (condition.network_size, condition.alpha, condition.gain, condition.trial): condition
            for condition in expand_conditions(config)
        }

    first = indexed(forward)
    second = indexed(reversed_order)
    assert first.keys() == second.keys()
    for key, condition in first.items():
        reordered = second[key]
        assert condition.condition_id == reordered.condition_id
        assert condition.weight_seed == reordered.weight_seed
        assert condition.state_seed == reordered.state_seed
        assert condition.weight_seed != condition.state_seed
        assert condition.weight_seed == derive_seed(
            42,
            n=condition.network_size,
            alpha=condition.alpha,
            gain=condition.gain,
            trial=condition.trial,
            stream=RECURRENT_WEIGHT_STREAM,
        )
        assert condition.state_seed == derive_seed(
            42,
            n=condition.network_size,
            alpha=condition.alpha,
            gain=condition.gain,
            trial=condition.trial,
            stream=INITIAL_STATE_STREAM,
        )


def test_numerical_source_digest_tracks_raw_source_bytes(tmp_path: Path) -> None:
    first_path = tmp_path / "sampling.py"
    second_path = tmp_path / "simulation.py"
    first_path.write_bytes(b"sampling-v1\n")
    second_path.write_bytes(b"simulation-v1\n")
    before = _numerical_source_digest([second_path, first_path])

    second_path.write_bytes(b"simulation-v2\n")
    after = _numerical_source_digest([first_path, second_path])

    assert before != after


def test_internal_simulation_rejects_stale_identity_and_seeds(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path / "unused")
    condition = expand_conditions(config)[0]
    device = torch.device(config.runtime.device)

    with pytest.raises(ValueError, match="condition_id"):
        _simulate_conditions(
            [replace(condition, condition_id="stale-condition")], config, device
        )

    with pytest.raises(ValueError, match="weight_seed"):
        _simulate_conditions(
            [replace(condition, weight_seed=condition.weight_seed + 1)], config, device
        )

    with pytest.raises(ValueError, match="state_seed"):
        _simulate_conditions(
            [replace(condition, state_seed=condition.state_seed + 1)], config, device
        )


def test_tiny_cpu_experiment_writes_complete_artifacts_and_resumes(tmp_path: Path) -> None:
    output_directory = tmp_path / "tiny-run"
    config = _tiny_config(output_directory)

    first = run_experiment(config)

    assert first.total_conditions == 4
    assert first.computed_conditions == 4
    assert first.resumed_conditions == 0
    assert first.output_directory == output_directory.resolve()
    assert (output_directory / "resolved_config.yaml").is_file()
    environment_path = output_directory / "environment.json"
    assert environment_path.is_file()
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    fingerprint = environment["implementation_fingerprint"]
    assert len(fingerprint) == 64
    assert environment["implementation_compatibility"]["numerical_source_sha256"]

    artifact_paths = sorted((output_directory / "conditions").glob("*.npz"))
    assert len(artifact_paths) == 4
    for path in artifact_paths:
        with np.load(path, allow_pickle=False) as artifact:
            assert artifact["activity"].shape == (3, 4)
            assert artifact["lyapunov_exponents"].shape == (2,)
            assert artifact["local_log_stretches"].shape == (3, 2)
            assert artifact["floored_stretch_counts"].shape == (2,)
            assert artifact["floored_stretches"].shape == ()
            assert artifact["execution_batch_size"].item() == 2
            assert artifact["batch_position"].item() in (0, 1)
            assert artifact["implementation_fingerprint"].item() == fingerprint
            assert len(artifact["execution_batch_id"].item()) == 24
            assert artifact["batch_condition_ids"].shape == (2,)
            assert (
                artifact["batch_condition_ids"][artifact["batch_position"].item()]
                == artifact["condition_id"].item()
            )
            for metric in (
                "activity_mean",
                "activity_rms",
                "activity_std",
                "activity_step_rms",
                "saturation_fraction",
            ):
                assert artifact[metric].shape == (3,)
            assert "weights" not in artifact.files
            assert bool(artifact["finite"].item())

    with first.metrics_path.open(newline="", encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    with first.aggregates_path.open(newline="", encoding="utf-8") as handle:
        aggregates = list(csv.DictReader(handle))
    with first.transitions_path.open(newline="", encoding="utf-8") as handle:
        transitions = list(csv.DictReader(handle))
    assert len(metrics) == 4
    assert len(aggregates) == 2
    assert len(transitions) == 1
    assert {int(row["trial"]) for row in metrics} == {0, 1}
    assert {int(row["gain_index"]) for row in metrics} == {0, 1}

    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["completed_conditions"] == 4
    assert manifest["remaining_conditions"] == 0

    modification_times = {path.name: path.stat().st_mtime_ns for path in artifact_paths}
    resumed = run_experiment(config)
    assert resumed.total_conditions == 4
    assert resumed.computed_conditions == 0
    assert resumed.resumed_conditions == 4
    assert modification_times == {
        path.name: path.stat().st_mtime_ns for path in artifact_paths
    }


def test_resume_rejects_stale_or_corrupt_condition_artifact(tmp_path: Path) -> None:
    output_directory = tmp_path / "corrupt-run"
    config = _tiny_config(output_directory)
    run_experiment(config)
    artifact_path = next((output_directory / "conditions").glob("*.npz"))

    with np.load(artifact_path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["network_size"] = np.asarray(999, dtype=np.int64)
    payload["lyapunov_exponents"] = np.zeros((99,), dtype=np.float64)
    with artifact_path.open("wb") as handle:
        np.savez(handle, **payload)

    with pytest.raises(OutputDirectoryError, match="lyapunov_exponents.*shape"):
        run_experiment(config)
    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["stage"] == "artifact_validation"


def test_resume_rejects_inconsistent_floor_metadata(tmp_path: Path) -> None:
    output_directory = tmp_path / "floor-corrupt-run"
    config = _tiny_config(output_directory)
    run_experiment(config)
    artifact_path = next((output_directory / "conditions").glob("*.npz"))

    with np.load(artifact_path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["floored_stretches"] = np.asarray(123, dtype=np.int64)
    with artifact_path.open("wb") as handle:
        np.savez(handle, **payload)

    with pytest.raises(OutputDirectoryError, match="floored_stretches is inconsistent"):
        run_experiment(config)


def test_resume_rejects_different_runtime_fingerprint(tmp_path: Path) -> None:
    output_directory = tmp_path / "fingerprint-run"
    config = _tiny_config(output_directory)
    run_experiment(config)
    environment_path = output_directory / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    environment["implementation_fingerprint"] = "0" * 64
    environment_path.write_text(json.dumps(environment), encoding="utf-8")

    with pytest.raises(OutputDirectoryError, match="different numerical implementation"):
        run_experiment(config)


def test_resume_requires_fingerprint_metadata_when_artifacts_exist(tmp_path: Path) -> None:
    output_directory = tmp_path / "missing-environment-run"
    config = _tiny_config(output_directory)
    run_experiment(config)
    (output_directory / "environment.json").unlink()

    with pytest.raises(OutputDirectoryError, match="no environment fingerprint"):
        run_experiment(config)


def test_resume_rejects_artifact_fingerprint_mismatch(tmp_path: Path) -> None:
    output_directory = tmp_path / "artifact-fingerprint-run"
    config = _tiny_config(output_directory)
    run_experiment(config)
    artifact_path = next((output_directory / "conditions").glob("*.npz"))
    with np.load(artifact_path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["implementation_fingerprint"] = np.asarray("0" * 64)
    with artifact_path.open("wb") as handle:
        np.savez(handle, **payload)

    with pytest.raises(OutputDirectoryError, match="implementation fingerprint mismatch"):
        run_experiment(config)


def test_resume_rejects_wrong_artifact_dtype(tmp_path: Path) -> None:
    output_directory = tmp_path / "dtype-corrupt-run"
    config = _tiny_config(output_directory)
    run_experiment(config)
    artifact_path = next((output_directory / "conditions").glob("*.npz"))
    with np.load(artifact_path, allow_pickle=False) as original:
        payload = {name: original[name] for name in original.files}
    payload["activity"] = payload["activity"].astype(np.float32)
    with artifact_path.open("wb") as handle:
        np.savez(handle, **payload)

    with pytest.raises(OutputDirectoryError, match="activity.*dtype"):
        run_experiment(config)


def test_resume_rejects_late_npz_payload_corruption(tmp_path: Path) -> None:
    output_directory = tmp_path / "payload-corrupt-run"
    config = _tiny_config(output_directory)
    run_experiment(config)
    artifact_path = next((output_directory / "conditions").glob("*.npz"))

    with zipfile.ZipFile(artifact_path) as archive:
        member = archive.getinfo("activity.npy")
    with artifact_path.open("r+b") as handle:
        handle.seek(member.header_offset + 26)
        name_length = int.from_bytes(handle.read(2), "little")
        extra_length = int.from_bytes(handle.read(2), "little")
        payload_offset = member.header_offset + 30 + name_length + extra_length
        handle.seek(payload_offset + member.file_size - 1)
        original_byte = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([original_byte[0] ^ 0xFF]))

    with pytest.raises(OutputDirectoryError, match="CRC.*activity.npy"):
        run_experiment(config)


def test_resume_normalizes_compressed_payload_corruption(tmp_path: Path) -> None:
    output_directory = tmp_path / "compressed-corrupt-run"
    config = _tiny_config(output_directory)
    config.output.compressed = True
    run_experiment(config)
    artifact_path = next((output_directory / "conditions").glob("*.npz"))

    with zipfile.ZipFile(artifact_path) as archive:
        member = archive.getinfo("activity.npy")
        assert member.compress_type == zipfile.ZIP_DEFLATED
    with artifact_path.open("r+b") as handle:
        handle.seek(member.header_offset + 26)
        name_length = int.from_bytes(handle.read(2), "little")
        extra_length = int.from_bytes(handle.read(2), "little")
        payload_offset = member.header_offset + 30 + name_length + extra_length
        handle.seek(payload_offset + 1)
        original_byte = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([original_byte[0] ^ 0xFF]))

    with pytest.raises(OutputDirectoryError, match="invalid or incomplete condition artifact"):
        run_experiment(config)


def test_resume_rejects_unexpected_condition_artifacts(tmp_path: Path) -> None:
    output_directory = tmp_path / "foreign-artifact-run"
    config = _tiny_config(output_directory)
    run_experiment(config)
    source = next((output_directory / "conditions").glob("*.npz"))
    (output_directory / "conditions" / "foreign.npz").write_bytes(source.read_bytes())

    with pytest.raises(OutputDirectoryError, match="unexpected condition artifact"):
        run_experiment(config)
    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["stage"] == "artifact_validation"


@pytest.mark.cuda
def test_tiny_cuda_experiment_exercises_sampling_and_artifacts(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    try:
        torch.empty(0, device="cuda:0")
    except Exception as exc:  # CUDA discovery can fail with backend-specific errors.
        pytest.skip(f"CUDA could not initialize in this process: {exc}")

    config = load_config(
        SMOKE_CONFIG,
        [
            "sweep.network_sizes=[4]",
            "sweep.alphas=[1.5]",
            "sweep.gains.values=[0.8]",
            "sweep.trials=2",
            "simulation.warmup_steps=1",
            "simulation.measurement_steps=2",
            "simulation.num_exponents=2",
            "runtime.device=cuda:0",
            "runtime.batch_size=2",
            f"output.directory={tmp_path / 'cuda-run'}",
        ],
    )
    result = run_experiment(config)

    assert result.computed_conditions == 2
    for path in (result.output_directory / "conditions").glob("*.npz"):
        with np.load(path, allow_pickle=False) as artifact:
            assert artifact["activity"].shape == (2, 4)
            assert artifact["execution_batch_size"].item() == 2
            assert bool(artifact["finite"].item())
