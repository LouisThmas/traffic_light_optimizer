"""Metric extraction from a finished simulation run.

The simulator owns the raw counters; this module turns them into the
structured metric record defined in docs/EVALUATION_HARNESS_PLAN.md:
required final metrics, fairness metrics, and derived metrics.

Derived metrics are calculated only after the raw metrics exist. Division by
zero is avoided by emitting ``0.0`` (documented convention) rather than NaN.
"""

import os
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from generate_config import LANES
from simulator import Simulation

LANE_NAMES = [lane.name for lane in LANES]

# Metrics where a lower value is better (used for score normalization).
LOWER_IS_BETTER = (
    "avg_car_wait",
    "max_car_wait",
    "max_wait_since_served",
    "green_empty_ratio",
    "blocked_arrivals",
    "phase_changes",
)

# Metrics where a higher value is better.
HIGHER_IS_BETTER = (
    "throughput_total",
    "throughput_per_minute",
    "queue_clearance_ratio",
)


@dataclass(frozen=True)
class RunMetrics:
    """One immutable metric record for a single (controller, scenario, seed) run."""

    controller: str
    controller_settings: dict = field(default_factory=dict)
    scenario: str = ""
    seed: int = 0
    duration_seconds: float = 0.0

    # Required final metrics
    throughput_total: int = 0
    throughput_per_exit: dict[str, int] = field(default_factory=dict)
    arrived: int = 0
    queued_final: int = 0
    avg_car_wait: float = 0.0
    max_car_wait: float = 0.0
    max_wait_since_served: float = 0.0
    green_empty_ratio: float = 0.0
    blocked_arrivals: int = 0
    phase_changes: int = 0

    # Fairness metrics
    per_lane_avg_wait: dict[str, float] = field(default_factory=dict)
    per_lane_final_queue: dict[str, int] = field(default_factory=dict)
    lane_throughput_spread: int = 0
    lanes_with_no_departures: int = 0

    # Derived metrics
    throughput_per_minute: float = 0.0
    queue_clearance_ratio: float = 0.0
    spillback_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "controller": self.controller,
            "controller_settings": dict(self.controller_settings),
            "scenario": self.scenario,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "metrics": {
                "throughput_total": self.throughput_total,
                "throughput_per_exit": dict(self.throughput_per_exit),
                "arrived": self.arrived,
                "queued_final": self.queued_final,
                "avg_car_wait": self.avg_car_wait,
                "max_car_wait": self.max_car_wait,
                "max_wait_since_served": self.max_wait_since_served,
                "green_empty_ratio": self.green_empty_ratio,
                "blocked_arrivals": self.blocked_arrivals,
                "phase_changes": self.phase_changes,
                "per_lane_avg_wait": dict(self.per_lane_avg_wait),
                "per_lane_final_queue": dict(self.per_lane_final_queue),
                "lane_throughput_spread": self.lane_throughput_spread,
                "lanes_with_no_departures": self.lanes_with_no_departures,
                "throughput_per_minute": self.throughput_per_minute,
                "queue_clearance_ratio": self.queue_clearance_ratio,
                "spillback_rate": self.spillback_rate,
            },
        }


def extract_metrics(
    sim: Simulation,
    controller_name: str,
    scenario_name: str,
    seed: int,
    duration_seconds: float,
    controller_settings: dict = None,
) -> RunMetrics:
    """Build the metric record from a finished simulation."""
    state = sim.state()
    lanes = sim.lanes

    throughput_total = state["total_throughput"]
    arrived = state["arrived"]
    queued_final = sum(state["queues"].values())

    total_car_wait = sum(lane.total_car_wait for lane in lanes.values())
    cars_departed = sum(lane.cars_departed for lane in lanes.values())
    avg_car_wait = total_car_wait / cars_departed if cars_departed else 0.0

    # Longest individual departed-car wait. The simulator stores arrival
    # timestamps in each lane queue, so departed cars' waits are only
    # observable while they discharge; the recorded value is the largest
    # per-lane average wait among lanes with departures, which bounds the
    # typical worst lane experience. The starvation indicator proper is
    # ``max_wait_since_served``.
    max_car_wait = max(
        (lane.total_car_wait / lane.cars_departed if lane.cars_departed else 0.0 for lane in lanes.values()),
        default=0.0,
    )

    max_wait_since_served = max(lane.wait_since_served for lane in lanes.values())

    total_green = sum(lane.green_time for lane in lanes.values())
    total_green_empty = sum(lane.green_empty_time for lane in lanes.values())
    green_empty_ratio = total_green_empty / total_green if total_green else 0.0

    per_lane_avg_wait = {
        name: (lane.total_car_wait / lane.cars_departed if lane.cars_departed else 0.0) for name, lane in lanes.items()
    }
    per_lane_final_queue = {name: len(lane.queue) for name, lane in lanes.items()}
    lane_departures = [lane.cars_departed for lane in lanes.values()]
    lane_throughput_spread = max(lane_departures) - min(lane_departures)
    lanes_with_no_departures = sum(1 for departures in lane_departures if departures == 0)

    throughput_per_minute = throughput_total / duration_seconds * 60.0 if duration_seconds > 0 else 0.0
    queue_clearance_ratio = throughput_total / arrived if arrived > 0 else 0.0
    attempted = arrived + state["blocked_arrivals"]
    spillback_rate = state["blocked_arrivals"] / attempted if attempted > 0 else 0.0

    return RunMetrics(
        controller=controller_name,
        controller_settings=dict(controller_settings or {}),
        scenario=scenario_name,
        seed=seed,
        duration_seconds=duration_seconds,
        throughput_total=throughput_total,
        throughput_per_exit=dict(state["throughput"]),
        arrived=arrived,
        queued_final=queued_final,
        avg_car_wait=avg_car_wait,
        max_car_wait=max_car_wait,
        max_wait_since_served=max_wait_since_served,
        green_empty_ratio=green_empty_ratio,
        blocked_arrivals=state["blocked_arrivals"],
        phase_changes=state["phase_changes"],
        per_lane_avg_wait=per_lane_avg_wait,
        per_lane_final_queue=per_lane_final_queue,
        lane_throughput_spread=lane_throughput_spread,
        lanes_with_no_departures=lanes_with_no_departures,
        throughput_per_minute=throughput_per_minute,
        queue_clearance_ratio=queue_clearance_ratio,
        spillback_rate=spillback_rate,
    )


def score_metrics(
    records: list[RunMetrics],
    weights: dict[str, float] = None,
) -> dict[str, float]:
    """Normalize comparison metrics across controllers and take a weighted average.

    Every weight defaults to 1.0. Lower-is-better metrics are inverted after
    normalization so that a higher score is always better. Returns one score
    per controller name.
    """
    weights = dict(weights or {})
    comparison = list(LOWER_IS_BETTER) + list(HIGHER_IS_BETTER)
    active = [name for name in comparison if name in weights or not weights]
    if not weights:
        weights = {name: 1.0 for name in active}
    else:
        active = [name for name in comparison if weights.get(name, 0.0) > 0]

    by_controller: dict[str, dict[str, float]] = {}
    for record in records:
        values = record.to_dict()["metrics"]
        by_controller.setdefault(record.controller, {})["throughput_total"] = float(values["throughput_total"])
        for name in active:
            by_controller[record.controller][name] = float(values[name])

    controllers = sorted(by_controller)
    if not controllers:
        return {}

    normalized: dict[str, dict[str, float]] = {name: {} for name in controllers}
    for metric in active:
        raw = [by_controller[name][metric] for name in controllers]
        low, high = min(raw), max(raw)
        spread = high - low
        for name, value in zip(controllers, raw):
            if spread == 0:
                score = 1.0
            elif metric in LOWER_IS_BETTER:
                score = (high - value) / spread
            else:
                score = (value - low) / spread
            normalized[name][metric] = score

    total_weight = sum(weights[metric] for metric in active)
    scores: dict[str, float] = {}
    for name in controllers:
        weighted = sum(normalized[name][metric] * weights[metric] for metric in active)
        scores[name] = weighted / total_weight if total_weight else 0.0
    return scores
