"""Construction lane-closure and spillback-protection tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from arrivals import TurnSplit, lane_arrival_rates
from controllers import FIXED_CYCLE_PHASES, MaxPressureController
from generate_config import LANES
from simulator import Simulation, SimulationConfig


def lights_for(phase):
    return {lane.name: lane.name in phase for lane in LANES}


def test_closed_lane_arrival_rate_is_redirected_to_forward_lane():
    rates = lane_arrival_rates(
        base_rate=1.0,
        split=TurnSplit(),
        closed_lanes=frozenset(("N-L", "N-R", "N-F2")),
    )
    # The closed north lanes have no arrivals. Their 0.2 + 0.2 + 0.3 demand
    # is redirected to N-F1, which keeps its own 0.3 rate: total 1.0.
    assert rates["N-L"] == 0.0
    assert rates["N-R"] == 0.0
    assert rates["N-F2"] == 0.0
    assert rates["N-F1"] == 1.0


def test_closed_lane_never_receives_arrivals():
    sim = Simulation(
        SimulationConfig(
            arrival_rate=2.0,
            closed_lanes=("N-L", "N-R", "N-F2"),
            seed=7,
        )
    )
    all_red = {lane.name: False for lane in LANES}
    for _ in range(100):
        sim.step(all_red)
    assert len(sim.lanes["N-L"].queue) == 0
    assert len(sim.lanes["N-R"].queue) == 0
    assert len(sim.lanes["N-F2"].queue) == 0
    assert len(sim.lanes["N-F1"].queue) > 0


def test_closed_lane_cannot_be_given_green():
    sim = Simulation(SimulationConfig(closed_lanes=("N-L",)))
    lights = {lane.name: False for lane in LANES}
    lights["N-L"] = True
    try:
        sim.step(lights)
        raise AssertionError("closed lane was allowed to turn green")
    except ValueError as error:
        assert "closed" in str(error)


def test_simulator_never_exceeds_lane_capacity_and_records_spillback():
    sim = Simulation(
        SimulationConfig(
            arrival_rate=20.0,
            lane_capacity=3,
            closed_lanes=("N-L", "N-R", "N-F2"),
            seed=4,
        )
    )
    all_red = {lane.name: False for lane in LANES}
    for _ in range(20):
        sim.step(all_red)
    state = sim.state()
    assert state["queues"]["N-F1"] == 3
    assert all(queue <= 3 for queue in state["queues"].values())
    assert state["blocked_arrivals"] > 0


def test_overflow_priority_serves_the_single_remaining_construction_lane():
    controller = MaxPressureController(lane_capacity=100, overflow_threshold=0.9)
    queues = {lane.name: 0 for lane in LANES}
    # Construction leaves N-F1 as the only usable north lane. Its queue is
    # at 95%, while east has a larger aggregate total.
    queues.update({"N-F1": 95, "E-L": 60, "E-F1": 60, "E-F2": 60})
    lights = controller.act({"queues": queues})
    assert lights["N-F1"] is True
    assert {name for name, enabled in lights.items() if enabled} == set(FIXED_CYCLE_PHASES[2])
