"""The evaluation harness: run controllers under identical conditions.

Core fairness rule: every controller in a comparison receives the same
scenario and the same seed, so the same Poisson arrival stream is generated
for each run. A comparison unit is (controller, scenario, seed, duration).

``run_one`` and ``run_many`` are importable and side-effect free (no
printing); the CLI in ``src/evaluate.py`` handles tables and file output.
"""

import json
import os
import sys
from collections.abc import Iterable
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_HERE, os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from controllers import CONTROLLERS
from metrics import RunMetrics, extract_metrics
from scenarios import Scenario, load_scenarios
from simulator import Simulation

# Metrics reported as mean/min/max/std across seeds.
AGGREGATED_METRICS = (
    "throughput_total",
    "throughput_per_minute",
    "avg_car_wait",
    "max_car_wait",
    "max_wait_since_served",
    "green_empty_ratio",
    "blocked_arrivals",
    "phase_changes",
    "queued_final",
    "arrived",
    "queue_clearance_ratio",
    "spillback_rate",
    "lane_throughput_spread",
    "lanes_with_no_departures",
)


def create_controller(controller_name: str, settings: Optional[dict] = None, seed: int = 0):
    """Instantiate a named controller with optional settings."""
    if controller_name not in CONTROLLERS:
        raise KeyError(f"unknown controller {controller_name!r}; available: {sorted(CONTROLLERS)}")
    settings = dict(settings or {})
    if controller_name == "random":
        settings.setdefault("seed", seed)
    return CONTROLLERS[controller_name](**settings)


def run_one(
    controller_name: str,
    scenario: Scenario,
    seed: int,
    controller_settings: Optional[dict] = None,
) -> RunMetrics:
    """Run one (controller, scenario, seed) unit and return its metrics.

    Builds a fresh SimulationConfig from the scenario and seed, creates a new
    simulator and the named controller, then alternates controller ``act`` and
    simulator ``step`` for the scenario duration.
    """
    scenario.validate()
    config = scenario.simulation_config(seed)
    sim = Simulation(config)
    controller = create_controller(controller_name, controller_settings, seed)
    closed = set(config.closed_lanes)

    steps = int(round(scenario.duration_seconds / config.step_length))
    for _ in range(steps):
        lights = controller.act(sim.state())
        if closed:
            # Closed lanes have no arrivals; dropping their request is always
            # safe (removing a green cannot create a conflict) and keeps
            # phase-based controllers usable under construction scenarios.
            lights = {name: on for name, on in lights.items() if name not in closed}
        sim.step(lights)

    return extract_metrics(
        sim,
        controller_name=controller_name,
        scenario_name=scenario.name,
        seed=seed,
        duration_seconds=scenario.duration_seconds,
        controller_settings=controller_settings,
    )


def run_many(
    controller_names: Iterable[str],
    scenario: Scenario,
    seeds: Optional[Iterable[int]] = None,
    controller_settings: Optional[dict] = None,
) -> list[RunMetrics]:
    """Run every controller over every seed of the scenario."""
    if seeds is None:
        seeds = scenario.seeds
    records: list[RunMetrics] = []
    for controller_name in controller_names:
        for seed in seeds:
            records.append(run_one(controller_name, scenario, seed, controller_settings))
    return records


def aggregate_by_controller(records: list[RunMetrics]) -> list[dict]:
    """Aggregate seed results per controller: mean/min/max/std per metric.

    Individual seed results stay available in the raw records so an outlier
    can be inspected.
    """
    grouped: dict[str, list[RunMetrics]] = {}
    for record in records:
        grouped.setdefault(record.controller, []).append(record)

    rows: list[dict] = []
    for controller in sorted(grouped):
        runs = grouped[controller]
        row: dict = {
            "controller": controller,
            "scenario": runs[0].scenario,
            "duration_seconds": runs[0].duration_seconds,
            "seed_count": len(runs),
            "seeds": sorted(run.seed for run in runs),
        }
        for metric in AGGREGATED_METRICS:
            values = np.array([getattr(run, metric) for run in runs], dtype=float)
            row[metric] = {
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
                "std": float(values.std()),
            }
        rows.append(row)
    return rows


def aggregate_with_scores(
    records: list[RunMetrics],
    weights: Optional[dict[str, float]] = None,
) -> list[dict]:
    """Aggregate rows plus the configurable overall score per controller."""
    from metrics import score_metrics

    rows = aggregate_by_controller(records)
    scores = score_metrics(records, weights)
    for row in rows:
        row["overall_score"] = scores.get(row["controller"], 0.0)
        row["score_weights"] = dict(weights or {})
    return rows


def findings_row(row: dict) -> dict:
    """One Markdown-ready README findings-table row from an aggregate row."""
    return {
        "Controller": row["controller"],
        "Scenario": row["scenario"],
        "Overall score": f"{row['overall_score']:.3f}",
        "Avg car wait (s)": f"{row['avg_car_wait']['mean']:.1f}",
        "Max wait (s)": f"{row['max_car_wait']['mean']:.1f}",
        "Throughput (cars/min)": f"{row['throughput_per_minute']['mean']:.1f}",
        "Green waste (%)": f"{row['green_empty_ratio']['mean'] * 100.0:.1f}",
        "Blocked arrivals": f"{row['blocked_arrivals']['mean']:.1f}",
    }


def findings_table(rows: list[dict]) -> str:
    """Render aggregate rows as a Markdown findings table."""
    if not rows:
        return ""
    formatted = [findings_row(row) for row in rows]
    headers = list(formatted[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---:" for _ in headers) + "|",
    ]
    for entry in formatted:
        lines.append("| " + " | ".join(str(entry[header]) for header in headers) + " |")
    return "\n".join(lines)


def write_jsonl(records: list[RunMetrics], path: str) -> None:
    """Write raw per-run results as JSON Lines (one self-contained object per run)."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Read raw JSONL results back into dictionaries."""
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_result(record: dict) -> None:
    """Lightweight schema validator: fail immediately on corrupt/incomplete results."""
    required_top = ("controller", "scenario", "seed", "duration_seconds", "metrics")
    for field in required_top:
        if field not in record:
            raise ValueError(f"result record missing field: {field}")
    metrics = record["metrics"]
    required_metrics = (
        "throughput_total",
        "throughput_per_minute",
        "avg_car_wait",
        "max_car_wait",
        "max_wait_since_served",
        "green_empty_ratio",
        "blocked_arrivals",
        "phase_changes",
        "arrived",
        "queued_final",
        "per_lane_avg_wait",
        "per_lane_final_queue",
        "throughput_per_exit",
        "queue_clearance_ratio",
        "spillback_rate",
    )
    for field in required_metrics:
        if field not in metrics:
            raise ValueError(f"result record missing metric: {field}")
    for field in ("throughput_total", "arrived", "queued_final", "blocked_arrivals", "phase_changes"):
        value = metrics[field]
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"metric {field} must be a non-negative integer, got {value!r}")
    for field in ("avg_car_wait", "max_car_wait", "green_empty_ratio", "throughput_per_minute"):
        value = metrics[field]
        if not isinstance(value, (int, float)) or value != value or value < 0:
            raise ValueError(f"metric {field} must be a non-negative finite number, got {value!r}")


def load_available_controllers() -> list[str]:
    """Baseline controller names usable in comparisons (manual excluded)."""
    return sorted(name for name in CONTROLLERS if name != "manual")


def load_all_scenarios() -> dict[str, Scenario]:
    return load_scenarios()
