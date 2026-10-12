"""The traffic simulator: arrivals, queues, discharge, and metrics.

One step = one fixed slice of time (default 1 second). Each step:
1. Arrivals join lane queues (Poisson per lane).
2. Cars discharge from green lanes (fractional-credit rate).
3. Metrics update: wait times, throughput, green-empty tracking.

The simulator never decides which lights are green — a controller does.
It validates the handed-in light vector against the compatibility matrix
and enforces minimum-green and yellow transitions.
"""

# Allow running this file directly (python src/simulator.py) as well as
# importing it as part of a package: make sibling modules and the config
# generator importable regardless of how Python was invoked.
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_HERE, os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from arrivals import TurnSplit, draw_arrivals, lane_arrival_rates
from generate_config import LANES
from lane_queue import LaneQueue
from lights import validate_lights

GREEN = "green"
YELLOW = "yellow"
RED = "red"


@dataclass
class LaneState:
    """Mutable per-lane bookkeeping."""

    queue: LaneQueue = field(default_factory=LaneQueue)
    signal: str = RED
    # seconds green so far in the current green period (for min_green)
    green_elapsed: float = 0.0
    # seconds yellow so far in the current yellow period
    yellow_elapsed: float = 0.0
    # fractional discharge credit accumulated while green
    served_credit: float = 0.0
    # seconds since this lane last discharged a car (priority signal)
    wait_since_served: float = 0.0
    # per-car wait accounting
    total_car_wait: float = 0.0
    cars_departed: int = 0
    # green-waste accounting
    green_time: float = 0.0
    green_empty_time: float = 0.0
    # effective signal phase changes (red/yellow -> green transitions)
    phase_changes: int = 0
    # lifetime counters
    arrived_total: int = 0


@dataclass
class SimulationConfig:
    step_length: float = 1.0
    discharge_rate: float = 0.5  # cars per second per green lane
    arrival_rate: float = 0.2  # cars per second per lane (base)
    turn_split: TurnSplit = field(default_factory=TurnSplit)
    min_green: float = 5.0  # seconds a lane stays green once on
    yellow_duration: float = 2.0  # seconds of yellow after green ends
    # Construction closures: arrivals are redirected to an open lane
    closed_lanes: tuple = ()
    # Physical storage limit before spillback. Arrival attempts beyond this
    # cap are recorded as blocked demand instead of exceeding the queue.
    lane_capacity: Optional[int] = None
    # Explicit per-lane Poisson rates (cars/second). When set, these replace
    # the arrival_rate * turn_split computation; the evaluation harness uses
    # this to apply scenario approach/lane overrides.
    lane_rates: Optional[dict] = None
    seed: int = 0


class Simulation:
    """The intersection simulation. Call step(lights) repeatedly."""

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        self.config = config or SimulationConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.lanes: dict[str, LaneState] = {lane.name: LaneState() for lane in LANES}
        if self.config.lane_capacity is not None and self.config.lane_capacity < 1:
            raise ValueError("lane_capacity must be at least 1")
        self.closed_lanes = frozenset(self.config.closed_lanes)
        unknown_closed = self.closed_lanes - {lane.name for lane in LANES}
        if unknown_closed:
            raise ValueError(f"unknown closed lane(s): {sorted(unknown_closed)}")
        self.arrival_rates = (
            dict(self.config.lane_rates)
            if self.config.lane_rates is not None
            else lane_arrival_rates(self.config.arrival_rate, self.config.turn_split, self.closed_lanes)
        )
        self.time = 0.0
        self.arrived = 0
        self.blocked_arrivals = 0
        # throughput per exit direction and total (all movements count)
        self.throughput: dict[str, int] = {}
        self.total_throughput = 0

    def step(self, lights: dict[str, bool]) -> None:
        """Advance one step with the given requested light vector."""
        closed_green = self.closed_lanes & {lane_name for lane_name, requested in lights.items() if requested}
        if closed_green:
            raise ValueError(f"closed lane(s) cannot be green: {sorted(closed_green)}")
        validate_lights(lights)
        self._advance_signals(lights)
        self._generate_arrivals()
        self._update_wait_times()
        self._discharge()
        self.time += self.config.step_length

    # -- signal state machine -------------------------------------------

    def _advance_signals(self, lights: dict[str, bool]) -> None:
        dt = self.config.step_length

        # First progress active signals. Red lanes are considered only after
        # all existing greens/yellows have progressed, so a new green is
        # always checked against the effective—not merely requested—state.
        for lane_name, lane in self.lanes.items():
            requested = lights.get(lane_name, False)
            if lane.signal == GREEN:
                lane.green_elapsed += dt
                lane.green_time += dt
                if len(lane.queue) == 0:
                    lane.green_empty_time += dt
                if not requested and lane.green_elapsed >= self.config.min_green:
                    lane.signal = YELLOW
                    lane.yellow_elapsed = 0.0
            elif lane.signal == YELLOW:
                lane.yellow_elapsed += dt
                if lane.yellow_elapsed >= self.config.yellow_duration:
                    # A retained request is considered in the red-lane pass
                    # below. This prevents it from bypassing a forced green.
                    lane.signal = RED

        green_names = {name for name, lane in self.lanes.items() if lane.signal == GREEN}
        for lane_name, lane in self.lanes.items():
            if lane.signal != RED or not lights.get(lane_name, False):
                continue
            if not self._is_compatible_with_greens(lane_name, green_names):
                continue
            lane.signal = GREEN
            lane.green_elapsed = 0.0
            lane.served_credit = 0.0
            lane.phase_changes += 1
            lane.green_time += dt
            if len(lane.queue) == 0:
                lane.green_empty_time += dt
            green_names.add(lane_name)

        # This is a defensive assertion for state-machine maintenance.
        validate_lights({name: name in green_names for name in self.lanes})

    @staticmethod
    def _is_compatible_with_greens(lane_name: str, green_names: set[str]) -> bool:
        """Whether `lane_name` can join the current effective green set."""
        from generate_config import MATRIX
        from lights import LANE_INDEX

        return all(MATRIX[LANE_INDEX[lane_name]][LANE_INDEX[other]] for other in green_names)

    def is_green(self, lane_name: str) -> bool:
        return self.lanes[lane_name].signal == GREEN

    def forced_green(self) -> list[str]:
        """Lanes in minimum-green: the controller cannot switch them off."""
        return [
            name
            for name, lane in self.lanes.items()
            if lane.signal == GREEN and lane.green_elapsed < self.config.min_green
        ]

    # -- arrivals --------------------------------------------------------

    def _generate_arrivals(self) -> None:
        counts = draw_arrivals(self.arrival_rates, self.config.step_length, self.rng)
        for lane_name, count in counts.items():
            if count:
                lane = self.lanes[lane_name]
                if self.config.lane_capacity is None:
                    accepted = count
                else:
                    available_space = self.config.lane_capacity - len(lane.queue)
                    accepted = min(count, max(0, available_space))
                blocked = count - accepted
                if accepted:
                    lane.queue.add(self.time, accepted)
                    lane.arrived_total += accepted
                    self.arrived += accepted
                self.blocked_arrivals += blocked

    # -- discharge -------------------------------------------------------

    def _discharge(self) -> None:
        dt = self.config.step_length
        for lane_name, lane in self.lanes.items():
            if lane.signal != GREEN:
                continue
            lane.served_credit += self.config.discharge_rate * dt
            departure_time = self.time + dt  # cars leave at the end of the step
            while lane.served_credit >= 1.0 and not lane.queue.is_empty:
                lane.served_credit -= 1.0
                arrival_time = lane.queue._cars.popleft()
                wait = departure_time - arrival_time
                lane.total_car_wait += wait
                lane.cars_departed += 1
                lane.wait_since_served = 0.0
                exit_direction = _exit_direction(lane_name)
                self.throughput[exit_direction] = self.throughput.get(exit_direction, 0) + 1
                self.total_throughput += 1
            if lane.queue.is_empty:
                lane.served_credit = min(lane.served_credit, 1.0)

    # -- metrics ---------------------------------------------------------

    def _update_wait_times(self) -> None:
        for lane in self.lanes.values():
            lane.wait_since_served += self.config.step_length

    def state(self) -> dict:
        """Everything a controller or UI needs, as plain data (pure read)."""
        return {
            "queues": {name: len(lane.queue) for name, lane in self.lanes.items()},
            "closed_lanes": sorted(self.closed_lanes),
            "wait_since_served": {name: lane.wait_since_served for name, lane in self.lanes.items()},
            "avg_car_wait": {
                name: (lane.total_car_wait / lane.cars_departed if lane.cars_departed else 0.0)
                for name, lane in self.lanes.items()
            },
            "green_time": {name: lane.green_time for name, lane in self.lanes.items()},
            "green_empty_time": {name: lane.green_empty_time for name, lane in self.lanes.items()},
            "phase_changes": sum(lane.phase_changes for lane in self.lanes.values()),
            "departed": {name: lane.cars_departed for name, lane in self.lanes.items()},
            "total_car_wait": {name: lane.total_car_wait for name, lane in self.lanes.items()},
            "signals": {name: lane.signal for name, lane in self.lanes.items()},
            "lights": {name: lane.signal == GREEN for name, lane in self.lanes.items()},
            "forced_green": self.forced_green(),
            "throughput": dict(self.throughput),
            "total_throughput": self.total_throughput,
            "arrived": self.arrived,
            "blocked_arrivals": self.blocked_arrivals,
            "lane_capacity": self.config.lane_capacity,
            "time": self.time,
        }


def _exit_direction(lane_name: str) -> str:
    """The road a car exits to, from the lane's movement."""
    from generate_config import EXIT_DIRECTIONS

    direction = lane_name[0]
    movement_code = lane_name.split("-")[1][0]
    movement = {"L": "left", "F": "forward", "R": "right"}[movement_code]
    return EXIT_DIRECTIONS[
        (  # type: ignore[index]
            {"N": "north", "S": "south", "E": "east", "W": "west"}[direction],
            movement,
        )
    ]


if __name__ == "__main__":
    # Demo: 60 seconds of alternating north-south / east-west phases.
    sim = Simulation(SimulationConfig(seed=0))
    ns = ["N-F1", "N-F2", "S-F1", "S-F2", "N-R", "S-R"]
    ew = ["E-F1", "E-F2", "W-F1", "W-F2", "E-R", "W-R"]
    for step in range(60):
        lights = {lane.name: False for lane in LANES}
        phase = ns if (step // 15) % 2 == 0 else ew
        for lane_name in phase:
            lights[lane_name] = True
        sim.step(lights)
    state = sim.state()
    print(f"Simulated {state['time']:.0f}s")
    print(f"Arrived:    {state['arrived']}")
    print(f"Throughput: {state['total_throughput']}")
    print(f"Still queued: {sum(state['queues'].values())}")
    waits = [w for w in state["avg_car_wait"].values() if w > 0]
    if waits:
        print(f"Avg per-car wait (lanes with departures): {sum(waits) / len(waits):.1f}s")
