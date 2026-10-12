"""Named traffic scenarios: reproducible environment facts for comparisons.

A scenario contains only traffic/environment facts (arrival rates, closures,
capacity, timing rules) — never controller choices. Controllers and seeds are
supplied by the evaluation harness, so every controller in a comparison sees
identical conditions.

Scenarios are loaded from ``config/scenarios.yaml`` and validated. The
``SimulationConfig`` built from a scenario is identical for every controller
and seed except the seed itself, which is what makes arrival streams fair.
"""

import os
import sys
from dataclasses import dataclass, field, replace
from typing import Optional

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_HERE, os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from arrivals import TurnSplit
from generate_config import LANES
from simulator import SimulationConfig

# Validated decisions (docs/EVALUATION_HARNESS_PLAN.md):
#   quick evaluation is 1800s, full comparison is 3600s.
QUICK_DURATION_SECONDS = 1800.0
FULL_DURATION_SECONDS = 3600.0

# Five seeds for everyday comparisons, twenty for final claims.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
FINAL_CLAIM_SEEDS: tuple[int, ...] = tuple(range(20))

LANE_NAMES = frozenset(lane.name for lane in LANES)
APPROACHES = ("north", "south", "east", "west")
APPROACH_CODES = {"north": "N", "south": "S", "east": "E", "west": "W"}


@dataclass(frozen=True)
class Scenario:
    """A reproducible traffic environment. No controller choices live here."""

    name: str
    duration_seconds: float = FULL_DURATION_SECONDS
    step_length: float = 1.0
    discharge_rate: float = 0.5
    min_green: float = 5.0
    yellow_duration: float = 2.0
    lane_capacity: Optional[int] = None
    arrival_rate: float = 0.2
    approach_overrides: dict[str, float] = field(default_factory=dict)
    lane_overrides: dict[str, float] = field(default_factory=dict)
    closed_lanes: tuple[str, ...] = ()
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    def validate(self) -> None:
        """Raise ValueError on any impossible or inconsistent scenario."""
        if not self.name or not self.name.strip():
            raise ValueError("scenario name must be a non-empty string")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.step_length <= 0:
            raise ValueError("step_length must be positive")
        if self.discharge_rate <= 0:
            raise ValueError("discharge_rate must be positive")
        if self.min_green < 0:
            raise ValueError("min_green cannot be negative")
        if self.yellow_duration < 0:
            raise ValueError("yellow_duration cannot be negative")
        if self.lane_capacity is not None and self.lane_capacity < 1:
            raise ValueError("lane_capacity must be at least 1 when set")
        if self.arrival_rate < 0:
            raise ValueError("arrival_rate cannot be negative")
        for approach, rate in self.approach_overrides.items():
            if approach not in APPROACHES:
                raise ValueError(f"unknown approach override: {approach!r}")
            if rate < 0:
                raise ValueError(f"approach override for {approach} is negative")
        for lane_name, rate in self.lane_overrides.items():
            if lane_name not in LANE_NAMES:
                raise ValueError(f"unknown lane override: {lane_name!r}")
            if rate < 0:
                raise ValueError(f"lane override for {lane_name} is negative")
        unknown_closed = set(self.closed_lanes) - LANE_NAMES
        if unknown_closed:
            raise ValueError(f"unknown closed lane(s): {sorted(unknown_closed)}")
        if not self.seeds:
            raise ValueError("a scenario needs at least one seed")

    def simulation_config(self, seed: int) -> SimulationConfig:
        """Build the simulator config for one seed.

        Approach overrides use the 20/60/20 turn split; direct lane overrides
        win over approach overrides when both name the same lane.
        """
        self.validate()
        split = TurnSplit()
        rates: dict[str, float] = {}
        for lane in LANES:
            approach = lane.direction
            base = self.approach_overrides.get(approach, self.arrival_rate)
            rates[lane.name] = base * split.share(lane.movement)
        rates.update(self.lane_overrides)

        closed = set(self.closed_lanes)
        # Redirect closed-lane demand exactly like lane_arrival_rates does,
        # but on top of the override-resolved rates.
        for closed_lane in sorted(closed):
            redirected = rates[closed_lane]
            rates[closed_lane] = 0.0
            direction_code = closed_lane[0]
            open_same_approach = [name for name in rates if name not in closed and name[0] == direction_code]
            forward_lanes = [name for name in open_same_approach if "-F" in name]
            recipient = (forward_lanes or open_same_approach)[0]
            rates[recipient] += redirected

        return SimulationConfig(
            step_length=self.step_length,
            discharge_rate=self.discharge_rate,
            arrival_rate=self.arrival_rate,
            turn_split=split,
            min_green=self.min_green,
            yellow_duration=self.yellow_duration,
            closed_lanes=tuple(sorted(closed)),
            lane_capacity=self.lane_capacity,
            lane_rates=rates,
            seed=seed,
        )

    def with_seeds(self, seeds: list[int]) -> "Scenario":
        """A copy of this scenario using different seeds."""
        return replace(self, seeds=tuple(seeds))


def _scenario_from_mapping(name: str, data: dict) -> Scenario:
    simulation = data.get("simulation", {}) or {}
    return Scenario(
        name=name,
        duration_seconds=float(data.get("duration_seconds", FULL_DURATION_SECONDS)),
        step_length=float(simulation.get("step_length", 1.0)),
        discharge_rate=float(simulation.get("discharge_rate", 0.5)),
        min_green=float(simulation.get("min_green", 5.0)),
        yellow_duration=float(simulation.get("yellow_duration", 2.0)),
        lane_capacity=(None if simulation.get("lane_capacity") is None else int(simulation["lane_capacity"])),
        arrival_rate=float(data.get("arrival_rates", {}).get("default", 0.2)),
        approach_overrides={
            str(approach): float(rate) for approach, rate in (data.get("approach_overrides", {}) or {}).items()
        },
        lane_overrides={str(lane): float(rate) for lane, rate in (data.get("lane_overrides", {}) or {}).items()},
        closed_lanes=tuple(str(lane) for lane in (data.get("closed_lanes", []) or [])),
        seeds=tuple(int(seed) for seed in (data.get("seeds", list(DEFAULT_SEEDS)))),
    )


def load_scenarios(path: Optional[str] = None) -> dict[str, Scenario]:
    """Load and validate every named scenario from the YAML file."""
    if path is None:
        path = os.path.join(_ROOT, "config", "scenarios.yaml")
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load_all(handle)
        documents = [document for document in raw if document]
    scenarios: dict[str, Scenario] = {}
    for data in documents:
        name = str(data.get("name", "")).strip()
        if not name:
            raise ValueError("every scenario in scenarios.yaml needs a name")
        scenario = _scenario_from_mapping(name, data)
        scenario.validate()
        scenarios[scenario.name] = scenario
    return scenarios


def get_scenario(name: str, path: Optional[str] = None) -> Scenario:
    scenarios = load_scenarios(path)
    if name not in scenarios:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(scenarios)}")
    return scenarios[name]
