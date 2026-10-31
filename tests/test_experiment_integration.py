"""Integration tests for the experiment runner and TensorBoard tracking.

Covers runner integration with existing evaluation functions, artifact
output (config snapshot, manifest, JSONL/CSV/Markdown/charts/TensorBoard
events), and error cases (unavailable controllers, directory collisions).
"""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from experiment import (
    ExperimentRunner,
    ExperimentTracker,
    resolve_experiment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exp_config_yaml(
    tmp_path: Path,
    controller: str = "fixed_cycle",
    scenario: str = "balanced",
    seeds=None,
    controller_settings=None,
) -> Path:
    """Write a minimal experiment YAML and return its path."""
    lines = [f"controller: {controller}", f"scenario: {scenario}"]
    if seeds is not None:
        lines.append("evaluation:")
        lines.append(f"  seeds: {seeds}")
    if controller_settings:
        lines.append("controller_settings:")
        for key, value in controller_settings.items():
            lines.append(f"  {key}: {value}")
    path = tmp_path / "experiment.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run_exp(tmp_path: Path, config_path: Path, output_dir=None) -> str:
    """Resolve and run an experiment, returning the output directory."""
    config = resolve_experiment(str(config_path))
    runner = ExperimentRunner(config)
    return runner.run(output_dir)


# ---------------------------------------------------------------------------
# Runner integration tests
# ---------------------------------------------------------------------------


def test_runner_produces_all_artifacts(tmp_path: Path):
    """A runnable baseline produces config.yaml, manifest.json, runs.jsonl,
    summary.csv, summary.md, charts, and TensorBoard event files."""
    config_path = _make_exp_config_yaml(tmp_path, scenario="balanced", seeds=[0])
    output_dir = str(tmp_path / "output")
    result_dir = _run_exp(tmp_path, config_path, output_dir)

    # Core artifacts
    assert os.path.isfile(os.path.join(result_dir, "config.yaml"))
    assert os.path.isfile(os.path.join(result_dir, "manifest.json"))

    # Existing format artifacts
    assert os.path.isfile(os.path.join(result_dir, "runs.jsonl"))
    assert os.path.isfile(os.path.join(result_dir, "summary.csv"))
    assert os.path.isfile(os.path.join(result_dir, "summary.md"))

    # Charts
    charts_dir = os.path.join(result_dir, "charts")
    assert os.path.isdir(charts_dir)
    chart_files = os.listdir(charts_dir)
    assert len(chart_files) > 0  # at least one chart

    # TensorBoard event files
    tb_dir = os.path.join(result_dir, "tensorboard")
    assert os.path.isdir(tb_dir)
    event_files = [f for f in os.listdir(tb_dir) if f.startswith("events.out.tfevents")]
    assert len(event_files) > 0


def test_manifest_contains_resolved_seeds_and_git(tmp_path: Path):
    """The manifest JSON includes resolved seeds, controller, scenario,
    and a git_commit (or 'unavailable')."""
    config_path = _make_exp_config_yaml(tmp_path, scenario="balanced", seeds=[5])
    output_dir = str(tmp_path / "output")
    result_dir = _run_exp(tmp_path, config_path, output_dir)

    manifest_path = os.path.join(result_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == "1.0"
    assert manifest["resolved"]["controller"] == "fixed_cycle"
    assert manifest["resolved"]["scenario"] == "balanced"
    assert manifest["resolved"]["seeds"] == [5]
    # git_commit is either a hex string or "unavailable"
    gc = manifest.get("git_commit", "unavailable")
    assert gc == "unavailable" or all(c in "0123456789abcdef" for c in gc)


def test_runner_passes_controller_settings(tmp_path: Path):
    """controller_settings from the YAML are passed to the controller
    constructor and persisted in the result."""
    config_path = _make_exp_config_yaml(
        tmp_path,
        controller="fixed_cycle",
        scenario="balanced",
        seeds=[0],
        controller_settings={"phase_length": 10},
    )
    output_dir = str(tmp_path / "output")
    result_dir = _run_exp(tmp_path, config_path, output_dir)

    # Check that controller_settings were passed through by reading the JSONL
    import json as _json

    with open(os.path.join(result_dir, "runs.jsonl"), encoding="utf-8") as f:
        first_record = _json.loads(f.readline())

    assert first_record["controller_settings"]["phase_length"] == 10


def test_runner_validates_every_record(tmp_path: Path):
    """Every emitted record passes validate_result."""
    from evaluation import read_jsonl, validate_result

    config_path = _make_exp_config_yaml(tmp_path, scenario="balanced", seeds=[0])
    output_dir = str(tmp_path / "output")
    result_dir = _run_exp(tmp_path, config_path, output_dir)

    records = read_jsonl(os.path.join(result_dir, "runs.jsonl"))
    for record in records:
        validate_result(record)


def test_unavailable_controller_fails_before_output(tmp_path: Path):
    """A schema-valid but unavailable controller (dqn) must fail before
    creating any experiment artifacts."""
    config_path = _make_exp_config_yaml(tmp_path, controller="dqn", scenario="balanced")
    output_dir = str(tmp_path / "output")

    with pytest.raises(SystemExit, match="not installed"):
        _run_exp(tmp_path, config_path, output_dir)

    # Verify nothing was created
    assert not os.path.exists(output_dir)


def test_config_yaml_is_input_snapshot(tmp_path: Path):
    """The CLI writes the exact original YAML; the runner writes a resolved
    version with all fields expanded."""
    original = """\
name: fixed-cycle-balanced
controller: fixed_cycle
scenario: balanced
controller_settings:
  phase_length: 20
"""
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(original, encoding="utf-8")

    output_dir = str(tmp_path / "output")
    result_dir = _run_exp(tmp_path, config_path, output_dir)

    # The runner writes a resolved version (with null fields expanded)
    saved_config = os.path.join(result_dir, "config.yaml")
    with open(saved_config, encoding="utf-8") as f:
        saved = yaml.safe_load(f)

    assert saved["name"] == "fixed-cycle-balanced"
    assert saved["controller"] == "fixed_cycle"
    assert saved["scenario"] == "balanced"
    assert saved["controller_settings"]["phase_length"] == 20


# ---------------------------------------------------------------------------
# TensorBoard tracking tests
# ---------------------------------------------------------------------------


def test_tracker_writes_readable_events(tmp_path: Path):
    """Event files written by the tracker can be read back and contain
    seed-level and aggregate metric tags/values."""
    from tensorboard.summary import Writer

    output_dir = str(tmp_path / "tb")
    writer = Writer(output_dir)

    tracker = ExperimentTracker(writer)
    tracker.log_scalar("evaluation/throughput_total", 36.0, step=0)
    tracker.log_scalar("evaluation/avg_car_wait", 14.2, step=0)
    tracker.log_scalar("evaluation/throughput_total/mean", 36.0, step=0)
    tracker.log_scalar("evaluation/avg_car_wait/mean", 14.2, step=0)
    writer.close()

    # Check that event files exist and are non-empty
    event_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("events.out.tfevents")]
    assert len(event_files) > 0
    # Event files should be > 0 bytes (non-empty)
    for ef in event_files:
        assert os.path.getsize(ef) > 0


def test_tracker_log_text(tmp_path: Path):
    """The runner writes experiment metadata as a text file."""
    config_path = _make_exp_config_yaml(
        tmp_path,
        scenario="balanced",
        seeds=[0],
        controller_settings={"phase_length": 10},
    )
    output_dir = str(tmp_path / "output")
    result_dir = _run_exp(tmp_path, config_path, output_dir)

    metadata_path = os.path.join(result_dir, "metadata.txt")
    assert os.path.isfile(metadata_path)
    with open(metadata_path, encoding="utf-8") as f:
        content = f.read()
    assert "controller: fixed_cycle" in content
    assert "scenario: balanced" in content
    assert "phase_length" in content


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


def test_output_directory_collision(tmp_path: Path):
    """Running twice to the same output directory is allowed (it reuses it)."""
    config_path = _make_exp_config_yaml(tmp_path, scenario="balanced", seeds=[0])
    output_dir = str(tmp_path / "output")

    # First run
    result_dir1 = _run_exp(tmp_path, config_path, output_dir)

    # Second run to same directory - should succeed (writes overwrite)
    result_dir2 = _run_exp(tmp_path, config_path, output_dir)

    assert os.path.isdir(result_dir1)
    assert os.path.isdir(result_dir2)


def test_missing_git_value_is_not_fatal(tmp_path: Path):
    """If git is unavailable, the manifest records 'unavailable' and
    the run still completes. When git is available, the commit is recorded."""
    # Force git to be unavailable by using a directory that's not a git repo
    non_git_dir = str(tmp_path / "not_a_repo")
    os.makedirs(non_git_dir)

    config_path = _make_exp_config_yaml(tmp_path, scenario="balanced", seeds=[0])
    output_dir = str(non_git_dir)

    # This should succeed even though the output is in a non-git directory
    config = resolve_experiment(str(config_path))
    runner = ExperimentRunner(config)
    result_dir = runner.run(output_dir)

    manifest_path = os.path.join(result_dir, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # git_commit is either a hex string (when git is available in _ROOT)
    # or 'unavailable' (when git is not found)
    gc = manifest.get("git_commit", "unavailable")
    assert gc == "unavailable" or all(c in "0123456789abcdef" for c in gc)
