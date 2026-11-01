"""Tests for queue-forecasting benchmark: baseline, multi-output, grouping.

Run: python -m pytest tests/test_sklearn_forecast.py -v

Tests the following before implementation:
1. Persistence baseline values (persistence prediction: next queue = current queue).
2. Multi-output target shape (16 lanes).
3. Deterministic grouped evaluation.
4. Finite MAE/RMSE including empty queues.
5. Per-lane and aggregate report fields.
6. Invalid config cases: unknown family/task, too few episode groups, seed overlap.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_config import LANES


def state_with_queues(**queues):
    return {"queues": {lane.name: queues.get(lane.name, 0) for lane in LANES}}


# ---------------------------------------------------------------------------
# Tests 1-5: Forecast baseline, multi-output, grouping, metrics
# ---------------------------------------------------------------------------


def test_persistence_baseline_returns_current_queues():
    """The persistence baseline predicts next queue = current queue."""
    from sklearn_data import persistence_baseline_predict

    current_queues = {
        "N-L": 5,
        "N-F1": 3,
        "N-F2": 0,
        "N-R": 2,
        "S-L": 0,
        "S-F1": 7,
        "S-F2": 1,
        "S-R": 4,
        "E-L": 0,
        "E-F1": 6,
        "E-F2": 0,
        "E-R": 3,
        "W-L": 1,
        "W-F1": 0,
        "W-F2": 8,
        "W-R": 2,
    }

    predictions = persistence_baseline_predict(current_queues)

    for lane in LANES:
        assert predictions[lane.name] == current_queues[lane.name], f"Persistence baseline for {lane.name}: expected {current_queues[lane.name]}, got {predictions[lane.name]}"


def test_multi_output_target_shape_is_sixteen():
    """Multi-output forecast targets must have shape (n_samples, 16)."""
    from sklearn_data import collect_trajectories_for_forecast

    data = collect_trajectories_for_forecast(scenario_name="balanced", seed=0, max_steps=60, horizon_steps=1)

    import pandas as pd

    target_df = pd.DataFrame(data["targets"])
    assert target_df.shape[1] == 16, f"Expected 16 output columns, got {target_df.shape[1]}"
    assert target_df.shape[0] > 0, "No forecast samples collected"


def test_grouped_forecast_evaluation_is_deterministic():
    """Running the same forecast evaluation twice must produce identical results."""
    from sklearn_data import collect_trajectories_for_forecast, evaluate_forecast

    data1 = collect_trajectories_for_forecast(scenario_name="balanced", seed=0, max_steps=60, horizon_steps=1)
    data2 = collect_trajectories_for_forecast(scenario_name="balanced", seed=0, max_steps=60, horizon_steps=1)

    result1 = evaluate_forecast(
        features=data1["features"],
        targets=data1["targets"],
        feature_columns=data1["feature_columns"],
        target_columns=data1["target_columns"],
    )
    result2 = evaluate_forecast(
        features=data2["features"],
        targets=data2["targets"],
        feature_columns=data2["feature_columns"],
        target_columns=data2["target_columns"],
    )

    assert result1 == result2, "Forecast evaluation is not deterministic"


def test_forecast_mae_rmse_are_finite_even_with_empty_queues():
    """MAE and RMSE must be finite numbers, even when some queues are zero."""
    from sklearn_data import collect_trajectories_for_forecast, evaluate_forecast

    data = collect_trajectories_for_forecast(scenario_name="balanced", seed=0, max_steps=100, horizon_steps=1)

    result = evaluate_forecast(
        features=data["features"],
        targets=data["targets"],
        feature_columns=data["feature_columns"],
        target_columns=data["target_columns"],
    )

    assert math.isfinite(result["aggregate_mae"]), f"Aggregate MAE must be finite, got {result['aggregate_mae']}"
    assert math.isfinite(result["aggregate_rmse"]), f"Aggregate RMSE must be finite, got {result['aggregate_rmse']}"

    # Per-lane metrics must also be finite
    for lane in LANES:
        lane_key = lane.name
        assert math.isfinite(result["per_lane_mae"][lane_key]), f"Per-lane MAE for {lane_key} must be finite"
        assert math.isfinite(result["per_lane_rmse"][lane_key]), f"Per-lane RMSE for {lane_key} must be finite"


def test_forecast_report_contains_per_lane_and_aggregate_fields():
    """Forecast report must include per-lane MAE/RMSE and aggregate metrics."""
    from sklearn_data import collect_trajectories_for_forecast, evaluate_forecast

    data = collect_trajectories_for_forecast(scenario_name="balanced", seed=0, max_steps=60, horizon_steps=1)

    result = evaluate_forecast(
        features=data["features"],
        targets=data["targets"],
        feature_columns=data["feature_columns"],
        target_columns=data["target_columns"],
    )

    # Must have aggregate metrics
    assert "aggregate_mae" in result
    assert "aggregate_rmse" in result

    # Must have per-lane metrics for all 16 lanes
    assert len(result["per_lane_mae"]) == 16
    assert len(result["per_lane_rmse"]) == 16


# ---------------------------------------------------------------------------
# Tests 6: Invalid config cases
# ---------------------------------------------------------------------------


def test_unknown_model_family_raises():
    """An unknown model family must raise a clear ValueError."""
    from sklearn_data import collect_trajectories_for_forecast, evaluate_forecast

    data = collect_trajectories_for_forecast(scenario_name="balanced", seed=0, max_steps=30, horizon_steps=1)

    try:
        evaluate_forecast(
            features=data["features"],
            targets=data["targets"],
            feature_columns=data["feature_columns"],
            target_columns=data["target_columns"],
            model_family="unknown_model",
        )
        assert False, "Should have raised ValueError for unknown family"
    except ValueError as e:
        assert "unknown" in str(e).lower() or "family" in str(e).lower(), f"Error should mention unknown family, got: {e}"


def test_too_few_episode_groups_raises():
    """Requesting more CV folds than available episodes must raise."""
    from sklearn_data import group_split_episodes

    try:
        group_split_episodes(
            episodes=[("balanced", 0)],  # only 1 episode
            n_splits=3,  # need at least 3
        )
        assert False, "Should have raised ValueError for too few episodes"
    except ValueError as e:
        assert "episode" in str(e).lower() or "fold" in str(e).lower() or "split" in str(e).lower(), f"Error should mention episodes or splits, got: {e}"


def test_train_evaluation_seed_overlap_raises():
    """Training and evaluation seeds must be disjoint."""
    from sklearn_data import collect_trajectories, collect_trajectories_for_forecast, validate_no_seed_overlap

    # Collect training and evaluation data with overlapping seeds
    collect_trajectories(scenario_name="balanced", seed=0, max_steps=30, controller_name="max_pressure")

    collect_trajectories_for_forecast(scenario_name="balanced", seed=0, max_steps=30, horizon_steps=1)

    # The evaluation harness should detect this overlap
    try:
        validate_no_seed_overlap(
            train_episodes=[("balanced", 0)],
            eval_episodes=[("balanced", 0)],
        )
        assert False, "Should have raised ValueError for seed overlap"
    except ValueError as e:
        assert "overlap" in str(e).lower() or "disjoint" in str(e).lower(), f"Error should mention overlap, got: {e}"
