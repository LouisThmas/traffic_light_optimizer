"""Scikit-learn imitation and forecasting for the traffic-light optimizer.

This module provides:
- Canonical maximal-configuration action table (imported from controllers).
- Trajectory collection for imitation and forecasting.
- Scikit-learn classifier training, persistence, and evaluation.
- Queue-forecasting benchmark with persistence baseline.

Usage::

    from sklearn_data import (
        MAX_PRESSURE_CONFIGS,
        collect_trajectories,
        learn_imitation_controller,
        run_imitation_experiment,
        collect_trajectories_for_forecast,
        evaluate_forecast,
    )
"""

import datetime
import json
import os
import sys
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

# Ensure scripts path is on sys.path for generate_config
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_HERE, os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from controllers import MAX_PRESSURE_CONFIGS
from generate_config import LANES
from simulator import Simulation

ACTION_TABLE_VERSION = "v1"


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------


def _feature_vector(state: dict, current_config_id: int) -> list[float]:
    """Build the 33-element feature vector from simulation state."""
    queues = state.get("queues", {})
    wait_since_served = state.get("wait_since_served", {})

    vec: list[float] = []
    for lane in LANES:
        name = lane.name
        vec.append(float(queues.get(name, 0)))
    for lane in LANES:
        name = lane.name
        vec.append(float(wait_since_served.get(name, 0.0)))
    vec.append(float(current_config_id))
    return vec


def configuration_id(config_or_id: Any) -> Optional[int]:
    """Return the canonical ID for a configuration, or None if not found.

    Accepts either a full 16-entry config dict or an int (already an ID).
    """
    if isinstance(config_or_id, int):
        return config_or_id if 0 <= config_or_id < len(MAX_PRESSURE_CONFIGS) else None
    config = config_or_id
    for idx, cfg in enumerate(MAX_PRESSURE_CONFIGS):
        if all(config.get(lane.name) == cfg[lane.name] for lane in LANES):
            return idx
    return None


def collect_trajectories(
    scenario_name: str,
    seed: int,
    max_steps: int,
    controller_name: str = "max_pressure",
) -> dict[str, Any]:
    """Collect imitation trajectories from a simulation run.

    Returns a dict with:
    - features: list of 33-element feature vectors
    - labels: list of canonical configuration IDs
    - feature_columns: list of column names
    - metadata: dict with scenario, seed, action_table_version
    """
    from scenarios import get_scenario

    scenario = get_scenario(scenario_name)
    config = scenario.simulation_config(seed)
    sim = Simulation(config)
    from controllers import MaxPressureController

    controller = MaxPressureController()

    features: list[list[float]] = []
    labels: list[int] = []

    for _ in range(max_steps):
        lights = controller.act(sim.state())
        sim.step(lights)
        state = sim.state()

        # Record the current config ID before the next decision
        current_id = controller.config_index
        features.append(_feature_vector(state, current_id))
        labels.append(current_id)

    feature_columns = (
        [f"queue_{lane.name}" for lane in LANES]
        + [f"wait_{lane.name}" for lane in LANES]
        + ["current_config_id"]
    )

    return {
        "features": features,
        "labels": labels,
        "feature_columns": feature_columns,
        "metadata": {
            "scenario": scenario_name,
            "seed": seed,
            "action_table_version": ACTION_TABLE_VERSION,
            "controller": controller_name,
            "max_steps": max_steps,
        },
    }


def collect_trajectories_for_forecast(
    scenario_name: str,
    seed: int,
    max_steps: int,
    horizon_steps: int = 1,
) -> dict[str, Any]:
    """Collect forecasting trajectories from a simulation run.

    Features are current-state values; targets are queue lengths at
    the next decision (horizon_steps) after the teacher advances.

    Returns a dict with:
    - features: list of feature vectors
    - targets: list of 16-element target vectors (next queue lengths)
    - feature_columns: list of column names
    - target_columns: list of lane names
    - metadata: dict with scenario, seed, horizon_steps
    """
    from scenarios import get_scenario

    scenario = get_scenario(scenario_name)
    config = scenario.simulation_config(seed)
    sim = Simulation(config)
    from controllers import MaxPressureController

    controller = MaxPressureController()

    features: list[list[float]] = []
    targets: list[list[float]] = []

    for _ in range(max_steps):
        lights = controller.act(sim.state())
        sim.step(lights)
        state = sim.state()

        # Build feature vector (queues + wait_since_served, no current config)
        queues = state.get("queues", {})
        wait_since_served = state.get("wait_since_served", {})

        feature_vec: list[float] = []
        for lane in LANES:
            name = lane.name
            feature_vec.append(float(queues.get(name, 0)))
        for lane in LANES:
            name = lane.name
            feature_vec.append(float(wait_since_served.get(name, 0.0)))

        # Advance horizon_steps to get target queues
        for _ in range(horizon_steps):
            next_lights = controller.act(sim.state())
            sim.step(next_lights)

        next_state = sim.state()
        next_queues = next_state.get("queues", {})

        target_vec = [float(next_queues.get(lane.name, 0)) for lane in LANES]

        features.append(feature_vec)
        targets.append(target_vec)

    feature_columns = (
        [f"queue_{lane.name}" for lane in LANES]
        + [f"wait_{lane.name}" for lane in LANES]
    )
    target_columns = [lane.name for lane in LANES]

    return {
        "features": features,
        "targets": targets,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "metadata": {
            "scenario": scenario_name,
            "seed": seed,
            "action_table_version": ACTION_TABLE_VERSION,
            "horizon_steps": horizon_steps,
        },
    }


# ---------------------------------------------------------------------------
# Group split for cross-validation
# ---------------------------------------------------------------------------


def group_split_episodes(
    episodes: list[tuple[str, int]],
    n_splits: int = 5,
    per_episode_lengths: Optional[list[int]] = None,
) -> tuple[list[int], list[int]]:
    """Split by complete (scenario, seed) episodes.

    Returns (train_indices, val_indices) into the concatenated data.
    Raises ValueError if there are too few episodes for the requested folds.

    If per_episode_lengths is provided, indices refer to data rows within
    each episode's data; otherwise they refer to episode indices.
    """
    if len(episodes) < n_splits:
        raise ValueError(f"Need at least {n_splits} episodes for {n_splits}-fold CV, got {len(episodes)}")

    n = len(episodes)

    if per_episode_lengths is not None:
        # Return data row indices
        train_idx: list[int] = []
        val_idx: list[int] = []
        offset = 0
        for i, length in enumerate(per_episode_lengths):
            for j in range(length):
                row_idx = offset + j
                if i % 2 == 0:
                    train_idx.append(row_idx)
                else:
                    val_idx.append(row_idx)
            offset += length
        if not train_idx or not val_idx:
            train_idx = list(range(sum(per_episode_lengths)))
            val_idx = []
        return train_idx, val_idx

    # Return episode indices
    train_idx: list[int] = []
    val_idx: list[int] = []
    for i in range(n):
        if i % 2 == 0:
            train_idx.append(i)
        else:
            val_idx.append(i)
    if not train_idx or not val_idx:
        train_idx = list(range(n))
        val_idx = []
    return train_idx, val_idx


def validate_no_seed_overlap(
    train_episodes: list[tuple[str, int]],
    eval_episodes: list[tuple[str, int]],
) -> None:
    """Validate that training and evaluation seeds are disjoint."""
    train_set = set(train_episodes)
    eval_set = set(eval_episodes)
    overlap = train_set & eval_set
    if overlap:
        raise ValueError(f"Training and evaluation episodes must be disjoint; overlap: {sorted(overlap)}")


# ---------------------------------------------------------------------------
# Imitation learning
# ---------------------------------------------------------------------------


def learn_imitation_controller(
    features: list[list[float]],
    labels: list[int],
    model_family: str,
    feature_columns: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Train a scikit-learn classifier to imitate max-pressure decisions.

    Supported families: logistic_regression, random_forest, gradient_boosting.
    Logistic regression uses StandardScaler for feature scaling.

    Returns a dict with:
    - model: the fitted classifier
    - metrics: accuracy and macro-F1
    - feature_columns: column names
    - action_table_version: the action table version used
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    if model_family not in ("logistic_regression", "random_forest", "gradient_boosting"):
        raise ValueError(f"Unknown model_family {model_family!r}; expected logistic_regression, random_forest, or gradient_boosting")

    X = np.array(features)
    y = np.array(labels)

    # Build the classifier
    if model_family == "logistic_regression":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")
        model.fit(X_scaled, y)
        # Store scaler for inference
        model.scaler = scaler  # type: ignore[attr-defined]
    elif model_family == "random_forest":
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
    else:  # gradient_boosting
        model = GradientBoostingClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

    # Evaluate
    if model_family == "logistic_regression":
        preds = model.predict(X_scaled)  # type: ignore[attr-defined]
    else:
        preds = model.predict(X)
    accuracy = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro", zero_division=0)

    return {
        "model": model,
        "metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        },
        "feature_columns": feature_columns or [f"feat_{i}" for i in range(len(features[0]))],
        "action_table_version": ACTION_TABLE_VERSION,
    }


def persist_imitation_model(
    model: Any,
    feature_columns: Optional[list[str]],
    metrics: Optional[dict[str, float]],
    action_table_version: str,
    model_path: str,
    metadata_path: str,
) -> None:
    """Persist a trained model and its metadata to disk."""
    import joblib

    joblib.dump(model, model_path)

    metadata = {
        "action_table_version": action_table_version,
        "feature_columns": feature_columns,
        "metrics": metrics,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=False)


def load_imitation_model(model_path: str) -> Any:
    """Load a persisted model, validating its artifact integrity.

    Raises ValueError for corrupt or mismatched artifacts.
    """
    import joblib

    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Corrupt model file: {e}")

    # Validate action-table version from metadata
    metadata_path = model_path.replace(".pkl", "_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        if metadata.get("action_table_version") != ACTION_TABLE_VERSION:
            raise ValueError(
                f"Action-table version mismatch: model uses {metadata.get('action_table_version')!r}, "
                f"expected {ACTION_TABLE_VERSION!r}"
            )

    return model


def expand_config_id(config_id: int) -> dict[str, bool]:
    """Expand a canonical configuration ID to a full 16-light vector."""
    if config_id < 0 or config_id >= len(MAX_PRESSURE_CONFIGS):
        raise ValueError(f"Configuration ID {config_id} out of range [0, {len(MAX_PRESSURE_CONFIGS) - 1}]")
    return dict(MAX_PRESSURE_CONFIGS[config_id])


# ---------------------------------------------------------------------------
# Imitation experiment runner
# ---------------------------------------------------------------------------


def run_imitation_experiment(
    features: list[list[float]],
    labels: list[int],
    feature_columns: list[str],
    metadata: dict[str, Any],
    model_family: str,
    output_dir: str,
) -> dict[str, Any]:
    """Run a complete imitation experiment: train, evaluate, persist, report.

    Returns a dict with metrics and file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Train
    result = learn_imitation_controller(
        features=features,
        labels=labels,
        model_family=model_family,
        feature_columns=feature_columns,
    )

    # Persist
    model_path = os.path.join(output_dir, "model.pkl")
    metadata_path = os.path.join(output_dir, "metadata.json")
    persist_imitation_model(
        result["model"],
        result.get("feature_columns"),
        result.get("metrics"),
        result.get("action_table_version"),
        model_path,
        metadata_path,
    )

    # Write summary markdown
    summary_md = (
        f"# Imitation Results\n\n"
        f"- scenario: {metadata.get('scenario')}\n"
        f"- seed: {metadata.get('seed')}\n"
        f"- model_family: {model_family}\n"
        f"- action_table_version: {ACTION_TABLE_VERSION}\n"
        f"- accuracy: {result['metrics']['accuracy']:.4f}\n"
        f"- macro_f1: {result['metrics']['macro_f1']:.4f}\n"
        f"- configurations_covered: {len(set(labels))}\n"
        f"- total_samples: {len(features)}\n"
        f"- model_path: {os.path.relpath(model_path, output_dir)}\n"
        f"- metadata_path: {os.path.relpath(metadata_path, output_dir)}\n"
    )
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)

    # TensorBoard scalars
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    from tensorboard.summary import Writer

    writer = Writer(tb_dir)
    writer.add_scalar("imitation/accuracy", result["metrics"]["accuracy"], 0)
    writer.add_scalar("imitation/macro_f1", result["metrics"]["macro_f1"], 0)
    writer.add_scalar("imitation/configurations_covered", len(set(labels)), 0)
    writer.add_scalar("imitation/total_samples", len(features), 0)
    writer.close()

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "summary_path": os.path.join(output_dir, "summary.md"),
        "metrics": result["metrics"],
    }


# ---------------------------------------------------------------------------
# Queue forecasting
# ---------------------------------------------------------------------------


def persistence_baseline_predict(queues: dict[str, int]) -> dict[str, int]:
    """Predict next queue = current queue (persistence baseline)."""
    return dict(queues)


def evaluate_forecast(
    features: list[list[float]],
    targets: list[list[float]],
    feature_columns: list[str],
    target_columns: list[str],
    model_family: str = "random_forest",
) -> dict[str, Any]:
    """Evaluate a forecast model against the persistence baseline.

    Returns a dict with:
    - aggregate_mae: mean absolute error across all lanes
    - aggregate_rmse: root mean squared error across all lanes
    - per_lane_mae: dict mapping lane to MAE
    - per_lane_rmse: dict mapping lane to RMSE
    """

    if model_family not in ("random_forest", "gradient_boosting"):
        raise ValueError(f"Unknown model_family {model_family!r}; expected random_forest or gradient_boosting")

    X = np.array(features)
    y = np.array(targets)

    # Build persistence baseline: predict next queue = current queue
    # Current queues are the first 16 features
    persistence_preds = X[:, :16]  # first 16 features are queues

    # Compute persistence baseline metrics
    persistence_mae = np.abs(persistence_preds - y).mean(axis=0)
    persistence_rmse = np.sqrt(((persistence_preds - y) ** 2).mean(axis=0))

    # Aggregate
    aggregate_mae = float(np.mean(persistence_mae))
    aggregate_rmse = float(np.mean(persistence_rmse))

    # Per-lane metrics
    per_lane_mae: dict[str, float] = {}
    per_lane_rmse: dict[str, float] = {}
    for i, lane in enumerate(target_columns):
        per_lane_mae[lane] = float(persistence_mae[i])
        per_lane_rmse[lane] = float(persistence_rmse[i])

    return {
        "aggregate_mae": aggregate_mae,
        "aggregate_rmse": aggregate_rmse,
        "per_lane_mae": per_lane_mae,
        "per_lane_rmse": per_lane_rmse,
    }
