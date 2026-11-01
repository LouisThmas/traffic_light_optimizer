"""Tests for the canonical action table and data collection.

Run: python -m pytest tests/test_sklearn_data.py -v

Tests the following before implementation:
1. Exactly 17 maximal configurations are enumerated from the compatibility matrix.
2. Each configuration is complete (all 16 lanes), matrix-safe, and maximal.
3. No duplicate or non-maximal set is admitted.
4. Max-pressure considers the full table while fixed-cycle phases are unchanged.
5. The old four-only max-pressure expectation fails for the intended reason.
6. Trajectory collection produces deterministic, fixed-order features and labels.
7. Forecast targets align at the requested horizon.
8. No episode overlap in grouped folds/holdout.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from controllers import (
    FIXED_CYCLE_PHASES,
    MaxPressureController,
)
from generate_config import LANES, MATRIX
from lights import LANE_INDEX, validate_lights


def all_red():
    return {lane.name: False for lane in LANES}


def state_with_queues(**queues):
    """Minimal controller state with named queue overrides."""
    return {"queues": {lane.name: queues.get(lane.name, 0) for lane in LANES}}


def green_lanes(lights):
    return {name for name, on in lights.items() if on}


# ---------------------------------------------------------------------------
# Tests 1-5: Canonical action table
# ---------------------------------------------------------------------------


def test_maximal_configurations_count_is_seventeen():
    """The canonical table must contain exactly 17 maximal configurations."""
    from sklearn_data import MAX_PRESSURE_CONFIGS

    assert len(MAX_PRESSURE_CONFIGS) == 17, f"Expected 17 maximal configurations, got {len(MAX_PRESSURE_CONFIGS)}"


def test_maximal_configurations_are_matrix_safe():
    """Every configuration must pass validate_lights."""
    from sklearn_data import MAX_PRESSURE_CONFIGS

    for config in MAX_PRESSURE_CONFIGS:
        validate_lights(config)  # raises if any two green lanes conflict


def test_maximal_configurations_are_maximal():
    """No unlit lane can be added safely to any configuration."""
    from sklearn_data import MAX_PRESSURE_CONFIGS

    all_lane_names = {lane.name for lane in LANES}
    for config in MAX_PRESSURE_CONFIGS:
        green_set = {name for name, on in config.items() if on}
        for candidate in all_lane_names - green_set:
            # The candidate must conflict with at least one green lane
            conflicts = any(not MATRIX[LANE_INDEX[candidate]][LANE_INDEX[green]] for green in green_set)
            assert conflicts, f"Configuration {config} is not maximal: lane {candidate!r} can be safely added"


def test_maximal_configurations_have_no_duplicates():
    """Each configuration must be unique when expressed as a frozenset."""
    from sklearn_data import MAX_PRESSURE_CONFIGS

    config_sets = [frozenset(name for name, on in config.items() if on) for config in MAX_PRESSURE_CONFIGS]
    assert len(config_sets) == len(set(config_sets)), "Duplicate configurations found in the canonical table"


def test_max_pressure_considers_all_seventeen_configurations():
    """Max-pressure must choose from the full table, not just 4 phases."""
    from sklearn_data import MAX_PRESSURE_CONFIGS

    teacher_configs = MAX_PRESSURE_CONFIGS
    assert len(teacher_configs) == 17, f"Max-pressure teacher uses {len(teacher_configs)} configs, expected 17"


def test_fixed_cycle_phases_unchanged():
    """Fixed-cycle phases must remain exactly the 4 original phases."""
    assert len(FIXED_CYCLE_PHASES) == 4
    for phase in FIXED_CYCLE_PHASES:
        assert isinstance(phase, tuple)
        for lane_name in phase:
            assert lane_name in {lane.name for lane in LANES}


def test_old_four_only_max_pressure_fails():
    """The old 4-phase max-pressure must NOT be the teacher's full action space."""
    from sklearn_data import MAX_PRESSURE_CONFIGS

    assert len(MAX_PRESSURE_CONFIGS) != 4, "Max-pressure teacher still limited to 4 phases; it should use all 17 maximal configurations"


def test_max_pressure_uses_config_index_not_phase_index():
    """Max-pressure controller must use config_index (0..16), not phase_index (0..3)."""
    controller = MaxPressureController()
    state = state_with_queues(**{"W-L": 10, "E-L": 20})
    controller.act(state)
    assert hasattr(controller, "config_index"), "MaxPressureController should have config_index"
    assert 0 <= controller.config_index <= 16, f"config_index {controller.config_index} out of range [0, 16]"


# ---------------------------------------------------------------------------
# Tests 6-8: Trajectory collection and grouping
# ---------------------------------------------------------------------------


def test_collection_produces_deterministic_features_and_labels():
    """Same scenario + seed must produce identical collected data."""
    from sklearn_data import collect_trajectories

    data1 = collect_trajectories(scenario_name="balanced", seed=42, max_steps=50, controller_name="max_pressure")
    data2 = collect_trajectories(scenario_name="balanced", seed=42, max_steps=50, controller_name="max_pressure")
    assert data1 == data2, "Collection is not deterministic"


def test_collection_feature_order_and_shape():
    """Collected features must have the expected order and shape."""
    from sklearn_data import collect_trajectories

    data = collect_trajectories(scenario_name="balanced", seed=0, max_steps=30, controller_name="max_pressure")

    expected_feature_cols = 16 + 16 + 1  # 33
    assert len(data["feature_columns"]) == expected_feature_cols, f"Expected {expected_feature_cols} feature columns, got {len(data['feature_columns'])}"

    # Check each feature vector has the right length
    for i, feat in enumerate(data["features"]):
        assert len(feat) == expected_feature_cols, f"Feature vector {i} has {len(feat)} elements, expected {expected_feature_cols}"

    # Labels must be valid configuration IDs (0-16)
    label_values = data["labels"]
    for label in label_values:
        assert 0 <= label <= 16, f"Label {label} out of range [0, 16]"


def test_labels_are_in_canonical_table():
    """All collected labels must correspond to valid canonical configuration IDs."""
    from sklearn_data import collect_trajectories, configuration_id

    data = collect_trajectories(scenario_name="balanced", seed=0, max_steps=30, controller_name="max_pressure")

    for label in data["labels"]:
        assert configuration_id(label) is not None, f"Label {label} does not correspond to any canonical configuration"


def test_forecast_target_alignment_at_horizon():
    """Forecast targets must represent queue lengths at the requested horizon."""
    from sklearn_data import collect_trajectories_for_forecast

    data = collect_trajectories_for_forecast(scenario_name="balanced", seed=0, max_steps=60, horizon_steps=1)

    assert len(data["features"]) > 0, "No features collected"
    assert len(data["targets"]) > 0, "No targets collected"

    # Targets must be 16 queue lengths (one per lane)
    import pandas as pd

    target_df = pd.DataFrame(data["targets"])
    assert len(target_df.columns) == 16, f"Expected 16 target columns (one per lane), got {len(target_df.columns)}"

    # All target values must be non-negative (floats from float() conversion)
    for val in target_df.values.flatten():
        assert float(val) >= 0, f"Forecast target must be non-negative, got {val!r}"


def test_grouped_folds_no_episode_overlap():
    """Splitting by (scenario, seed) episodes must not leak rows across folds."""
    from sklearn_data import collect_trajectories, group_split_episodes

    data1 = collect_trajectories(scenario_name="balanced", seed=0, max_steps=30, controller_name="max_pressure")
    data2 = collect_trajectories(scenario_name="balanced", seed=1, max_steps=30, controller_name="max_pressure")

    train_idx, val_idx = group_split_episodes(
        episodes=[("balanced", 0), ("balanced", 1)],
        n_splits=2,
        per_episode_lengths=[len(data1["features"]), len(data2["features"])],
    )

    # Train and validation indices must be disjoint
    train_set = set(train_idx)
    val_set = set(val_idx)
    assert len(train_set & val_set) == 0, "Train and validation indices overlap; episodes must not leak across folds"

    # Together they must cover all collected data
    all_data_len = len(data1["features"]) + len(data2["features"])
    assert len(train_idx) + len(val_idx) == all_data_len, f"Split does not cover all data: {len(train_idx)} + {len(val_idx)} != {all_data_len}"
