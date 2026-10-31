"""Unit tests for experiment YAML loading and validation.

Covers schema validation, seed defaulting/overriding, and rejection of
malformed input (missing fields, unknown scenarios, invalid action_space,
duplicate/non-integer seeds).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from experiment import (
    load_experiment_config,
    resolve_experiment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tmp_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# YAML loading and validation
# ---------------------------------------------------------------------------


def test_load_minimal_baseline_yaml(tmp_path: Path):
    """A minimal baseline YAML with controller + scenario is valid."""
    path = _write_tmp_yaml(
        tmp_path,
        "controller: fixed_cycle\nscenario: balanced\n",
    )
    config = load_experiment_config(str(path))
    assert config.controller == "fixed_cycle"
    assert config.scenario == "balanced"


def test_load_yaml_with_all_fields(tmp_path: Path):
    """A YAML with all planned fields loads without error."""
    path = _write_tmp_yaml(
        tmp_path,
        """\
name: test-exp
controller: dqn
action_space: multi_binary_masked
state: full
reward: weighted
network:
  layers: [64, 64]
  activation: relu
optimizer:
  type: adam
  lr: 0.001
training:
  episodes: 100
  batch_size: 32
scenario: balanced
evaluation:
  seeds: [10, 20]
""",
    )
    config = load_experiment_config(str(path))
    assert config.controller == "dqn"
    assert config.action_space == "multi_binary_masked"
    assert config.state == "full"
    assert config.reward == "weighted"
    assert config.network == {"layers": [64, 64], "activation": "relu"}
    assert config.optimizer == {"type": "adam", "lr": 0.001}
    assert config.training == {"episodes": 100, "batch_size": 32}
    assert config.scenario == "balanced"
    assert config.evaluation.seeds == (10, 20)


def test_scenario_seed_defaulting(tmp_path: Path):
    """When evaluation.seeds is omitted, the scenario's seeds are used."""
    path = _write_tmp_yaml(
        tmp_path,
        "controller: fixed_cycle\nscenario: balanced\n",
    )
    config = resolve_experiment(str(path))
    assert config.evaluation.seeds == (0, 1, 2, 3, 4)  # scenario defaults


def test_scenario_seed_override(tmp_path: Path):
    """Explicit evaluation.seeds overrides the scenario's seeds."""
    path = _write_tmp_yaml(
        tmp_path,
        "controller: fixed_cycle\nscenario: balanced\nevaluation:\n  seeds: [42, 43]\n",
    )
    config = resolve_experiment(str(path))
    assert config.evaluation.seeds == (42, 43)


def test_normalized_preserves_ml_fields(tmp_path: Path):
    """ML dials from the supplied example are preserved in the resolved config."""
    path = _write_tmp_yaml(
        tmp_path,
        """\
name: dqn-balanced
controller: dqn
action_space: multi_binary_masked
state: full
reward: weighted
network:
  layers: [128, 64]
  activation: relu
optimizer:
  type: adam
  lr: 0.0003
training:
  episodes: 500
  batch_size: 64
  learning_rate: 0.0003
scenario: balanced
""",
    )
    config = resolve_experiment(str(path))
    assert config.controller == "dqn"
    assert config.action_space == "multi_binary_masked"
    assert config.state == "full"
    assert config.reward == "weighted"
    assert config.network == {"layers": [128, 64], "activation": "relu"}
    assert config.optimizer == {"type": "adam", "lr": 0.0003}
    assert config.training == {"episodes": 500, "batch_size": 64, "learning_rate": 0.0003}


def test_rejects_non_mapping_yaml(tmp_path: Path):
    """A YAML that is not a mapping (e.g. a list) is rejected."""
    path = _write_tmp_yaml(tmp_path, "- item1\n- item2\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_experiment_config(str(path))


def test_rejects_missing_controller(tmp_path: Path):
    """A YAML without controller is rejected."""
    path = _write_tmp_yaml(tmp_path, "scenario: balanced\n")
    with pytest.raises(ValueError, match="controller"):
        load_experiment_config(str(path))


def test_rejects_missing_scenario(tmp_path: Path):
    """A YAML without scenario is rejected."""
    path = _write_tmp_yaml(tmp_path, "controller: fixed_cycle\n")
    with pytest.raises(ValueError, match="scenario"):
        load_experiment_config(str(path))


def test_rejects_unknown_scenario(tmp_path: Path):
    """A YAML referencing a non-existent scenario is rejected."""
    path = _write_tmp_yaml(
        tmp_path,
        "controller: fixed_cycle\nscenario: nonexistent\n",
    )
    with pytest.raises(KeyError, match="nonexistent"):
        resolve_experiment(str(path))


def test_rejects_invalid_action_space(tmp_path: Path):
    """An unknown action_space value is rejected."""
    path = _write_tmp_yaml(
        tmp_path,
        "controller: fixed_cycle\nscenario: balanced\naction_space: invalid\n",
    )
    with pytest.raises(ValueError, match="action_space"):
        load_experiment_config(str(path))


def test_rejects_duplicate_seeds(tmp_path: Path):
    """Duplicate seeds in evaluation.seeds are rejected."""
    path = _write_tmp_yaml(
        tmp_path,
        "controller: fixed_cycle\nscenario: balanced\nevaluation:\n  seeds: [1, 1, 2]\n",
    )
    with pytest.raises(ValueError, match="seeds"):
        load_experiment_config(str(path))


def test_rejects_non_integer_seeds(tmp_path: Path):
    """Non-integer seeds in evaluation.seeds are rejected."""
    path = _write_tmp_yaml(
        tmp_path,
        "controller: fixed_cycle\nscenario: balanced\nevaluation:\n  seeds: [1, 1.5, 2]\n",
    )
    with pytest.raises(ValueError, match="seeds"):
        load_experiment_config(str(path))
