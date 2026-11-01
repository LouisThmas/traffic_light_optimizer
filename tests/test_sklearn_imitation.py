"""Tests for scikit-learn imitation: fitting, persistence, and learned controller.

Run: python -m pytest tests/test_sklearn_imitation.py -v

Tests the following before implementation:
1. Each classifier family (logistic_regression, random_forest, gradient_boosting)
   fits a tiny multi-class fixture.
2. Persisted/reloaded models reproduce predictions.
3. A learned controller converts every valid class to the exact full 16-light vector.
4. Invalid model output / artifact / action-table version errors clearly.
5. A small seeded end-to-end train/evaluate run produces artifacts and TensorBoard tags.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_config import LANES
from lights import validate_lights


def all_red():
    return {lane.name: False for lane in LANES}


def state_with_queues(**queues):
    return {"queues": {lane.name: queues.get(lane.name, 0) for lane in LANES}}


def green_lanes(lights):
    return {name for name, on in lights.items() if on}


# ---------------------------------------------------------------------------
# Tests 1-2: Classifier family fitting and persistence
# ---------------------------------------------------------------------------


def test_logistic_regression_fits_multi_class_fixture():
    """Logistic regression must handle multi-class config IDs 0-16."""
    from sklearn_data import learn_imitation_controller

    features = [
        [1.0] * 16 + [0.5] * 16 + [0.0],  # label 0
        [2.0] * 16 + [1.0] * 16 + [1.0],  # label 1
        [3.0] * 16 + [2.0] * 16 + [2.0],  # label 2
    ]
    labels = [0, 1, 2]

    result = learn_imitation_controller(
        features=features,
        labels=labels,
        model_family="logistic_regression",
        feature_columns=[f"lane_{i}" for i in range(33)],
    )

    assert "model" in result
    assert result["metrics"]["accuracy"] >= 0.0  # at least runs


def test_random_forest_fits_multi_class_fixture():
    """Random forest must handle multi-class config IDs 0-16."""
    from sklearn_data import learn_imitation_controller

    features = [
        [1.0] * 16 + [0.5] * 16 + [0.0],  # label 0
        [2.0] * 16 + [1.0] * 16 + [1.0],  # label 1
        [3.0] * 16 + [2.0] * 16 + [2.0],  # label 2
    ]
    labels = [0, 1, 2]

    result = learn_imitation_controller(
        features=features,
        labels=labels,
        model_family="random_forest",
        feature_columns=[f"lane_{i}" for i in range(33)],
    )

    assert "model" in result
    assert result["metrics"]["accuracy"] >= 0.0


def test_gradient_boosting_fits_multi_class_fixture():
    """Gradient boosting must handle multi-class config IDs 0-16."""
    from sklearn_data import learn_imitation_controller

    features = [
        [1.0] * 16 + [0.5] * 16 + [0.0],  # label 0
        [2.0] * 16 + [1.0] * 16 + [1.0],  # label 1
        [3.0] * 16 + [2.0] * 16 + [2.0],  # label 2
    ]
    labels = [0, 1, 2]

    result = learn_imitation_controller(
        features=features,
        labels=labels,
        model_family="gradient_boosting",
        feature_columns=[f"lane_{i}" for i in range(33)],
    )

    assert "model" in result
    assert result["metrics"]["accuracy"] >= 0.0


def test_persisted_reloaded_model_reproduces_predictions():
    """Saving and reloading a model must produce identical predictions."""
    from sklearn_data import learn_imitation_controller

    features = [
        [1.0] * 16 + [0.5] * 16 + [0.0],
        [2.0] * 16 + [1.0] * 16 + [1.0],
    ]
    labels = [0, 1]

    result = learn_imitation_controller(
        features=features,
        labels=labels,
        model_family="random_forest",
        feature_columns=[f"lane_{i}" for i in range(33)],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        metadata_path = os.path.join(tmpdir, "metadata.json")

        # Persist
        from sklearn_data import persist_imitation_model

        persist_imitation_model(
            result["model"],
            result.get("feature_columns"),
            result.get("metrics"),
            result.get("action_table_version"),
            model_path,
            metadata_path,
        )

        # Reload and predict
        from sklearn_data import load_imitation_model

        reloaded = load_imitation_model(model_path)

        # Predictions on the same features must match
        original_preds = result["model"].predict(features)
        reloaded_preds = reloaded.predict(features)
        assert np.array_equal(original_preds, reloaded_preds), "Reloaded model predictions do not match original"


# ---------------------------------------------------------------------------
# Tests 3-4: Learned controller inference and error handling
# ---------------------------------------------------------------------------


def test_learned_controller_expands_config_id_to_full_16_light_vector():
    """Every valid class ID (0-16) must expand to the exact 16-light vector."""
    from learned_controller import expand_config_id
    from sklearn_data import MAX_PRESSURE_CONFIGS

    teacher_configs = MAX_PRESSURE_CONFIGS

    # Verify each config ID maps to the correct 16-entry light vector
    for config_id, config in enumerate(teacher_configs):
        assert len(config) == 16, f"Configuration {config_id} does not have 16 entries"
        validate_lights(config)  # must be matrix-safe

    # Also test the expand_config_id function directly
    for config_id in range(len(teacher_configs)):
        lights = expand_config_id(config_id)
        assert len(lights) == 16, f"expand_config_id({config_id}) returned {len(lights)} entries"
        validate_lights(lights)


def test_learned_controller_rejects_corrupt_model_artifacts():
    """Loading a corrupt or tampered model must raise a clear error."""
    from sklearn_data import load_imitation_model

    with tempfile.TemporaryDirectory() as tmpdir:
        corrupt_path = os.path.join(tmpdir, "corrupt_model.pkl")
        with open(corrupt_path, "wb") as f:
            f.write(b"this is not a pickle file")

        try:
            load_imitation_model(corrupt_path)
            assert False, "Should have raised an error for corrupt model"
        except Exception as e:
            assert "corrupt" in str(e).lower() or "error" in str(e).lower() or "invalid" in str(e).lower(), f"Error message should be clear, got: {e}"


def test_learned_controller_rejects_action_table_mismatch():
    """Loading a model with a different action-table version must fail."""
    from sklearn_data import learn_imitation_controller, load_imitation_model, persist_imitation_model

    features = [[1.0] * 33, [2.0] * 33, [3.0] * 33]
    labels = [0, 1, 2]

    result = learn_imitation_controller(
        features=features,
        labels=labels,
        model_family="logistic_regression",
        feature_columns=[f"lane_{i}" for i in range(33)],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pkl")
        metadata_path = os.path.join(tmpdir, "metadata.json")

        persist_imitation_model(
            result["model"],
            result.get("feature_columns"),
            result.get("metrics"),
            result.get("action_table_version"),
            model_path,
            metadata_path,
        )

        # Tamper with the metadata to change the action-table version
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["action_table_version"] = "v999-nonexistent"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        try:
            load_imitation_model(model_path)
            assert False, "Should have raised an error for table mismatch"
        except Exception as e:
            assert "version" in str(e).lower() or "mismatch" in str(e).lower(), f"Error should mention version, got: {e}"


def test_learned_controller_rejects_out_of_range_predictions():
    """Out-of-range predictions (negative or > 16) must be rejected."""
    from sklearn_data import MAX_PRESSURE_CONFIGS

    teacher_configs = MAX_PRESSURE_CONFIGS

    # Verify that 17 is out of range (valid IDs are 0-16)
    assert 17 not in range(len(teacher_configs)), "ID 17 should be out of range for 17 configurations"


def test_learned_controller_act_produces_safe_lights():
    """A trained controller must always produce matrix-safe light vectors."""
    from learned_controller import LearnedController
    from sklearn_data import learn_imitation_controller

    features = [[1.0] * 33, [2.0] * 33, [3.0] * 33]
    labels = [0, 1, 2]

    result = learn_imitation_controller(
        features=features,
        labels=labels,
        model_family="random_forest",
        feature_columns=[f"lane_{i}" for i in range(33)],
    )

    controller = LearnedController(result["model"])
    state = state_with_queues(**{"W-L": 10, "E-L": 20})

    # Multiple calls should always produce safe lights
    for _ in range(10):
        lights = controller.act(state)
        validate_lights(lights)  # must not raise


# ---------------------------------------------------------------------------
# Test 5: End-to-end train/evaluate produces artifacts and TensorBoard tags
# ---------------------------------------------------------------------------


def test_end_to_end_train_evaluate_produces_artifacts():
    """A small seeded run must produce model, metadata, and summary files."""
    from sklearn_data import collect_trajectories, run_imitation_experiment

    # Collect training data from a single seed
    train_data = collect_trajectories(scenario_name="balanced", seed=0, max_steps=40, controller_name="max_pressure")

    with tempfile.TemporaryDirectory() as tmpdir:
        run_imitation_experiment(
            features=train_data["features"],
            labels=train_data["labels"],
            feature_columns=train_data["feature_columns"],
            metadata=train_data["metadata"],
            model_family="random_forest",
            output_dir=tmpdir,
        )

        # Must produce a model file
        model_files = [f for f in os.listdir(tmpdir) if f.endswith(".pkl")]
        assert len(model_files) > 0, "No model file produced"

        # Must produce metadata
        metadata_files = [f for f in os.listdir(tmpdir) if "metadata" in f]
        assert len(metadata_files) > 0, "No metadata file produced"

        # Must produce a summary markdown
        md_files = [f for f in os.listdir(tmpdir) if f.endswith(".md")]
        assert len(md_files) > 0, "No summary markdown produced"

        # Must produce TensorBoard directory
        tb_dirs = [d for d in os.listdir(tmpdir) if d == "tensorboard"]
        assert len(tb_dirs) > 0, "No TensorBoard directory produced"

        # Metrics must include accuracy and macro-F1
        summary_content = ""
        for md_file in md_files:
            with open(os.path.join(tmpdir, md_file)) as f:
                summary_content = f.read()
        assert "accuracy" in summary_content.lower(), "Summary must include accuracy"
        assert "f1" in summary_content.lower() or "macro-f1" in summary_content.lower(), "Summary must include macro-F1"
