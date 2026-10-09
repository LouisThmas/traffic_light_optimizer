"""Simulator tests: safety, conservation, statistics, and signal semantics.

Run:  python3 -m pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from arrivals import TurnSplit, lane_arrival_rates
from generate_config import LANES, MATRIX
from lights import LANE_INDEX
from simulator import RED, GREEN, YELLOW, Simulation, SimulationConfig


def all_lights() -> dict:
    return {lane.name: False for lane in LANES}


def make_sim(**overrides) -> Simulation:
    config = SimulationConfig(**overrides)
    return Simulation(config)


def test_unsafe_lights_are_rejected():
    sim = make_sim()
    lights = all_lights()
    lights["N-F1"] = True
    lights["E-F1"] = True  # perpendicular through movements conflict
    try:
        sim.step(lights)
        raise AssertionError("conflicting lights were accepted")
    except ValueError:
        pass


def test_car_conservation():
    sim = make_sim(seed=0)
    rng = np.random.default_rng(7)
    for _ in range(500):
        lights = all_lights()
        for lane in LANES:
            lights[lane.name] = bool(rng.random() < 0.3)
        # repair to a safe set: drop lanes that conflict with an earlier green
        green = []
        for lane in LANES:
            if lights[lane.name] and all(
                MATRIX[LANE_INDEX[lane.name]][LANE_INDEX[other]] for other in green
            ):
                green.append(lane.name)
        lights = {lane.name: lane.name in green for lane in LANES}
        sim.step(lights)
        state = sim.state()
        assert state["arrived"] == sum(state["queues"].values()) + state["total_throughput"]


def test_poisson_arrival_statistics():
    # With no discharge (all red), arrivals per lane over a long horizon
    # should match the Poisson rate within a few standard deviations.
    horizon = 20000
    base_rate = 0.2
    sim = make_sim(seed=42, arrival_rate=base_rate)
    lights = all_lights()
    for _ in range(horizon):
        sim.step(lights)
    rates = lane_arrival_rates(base_rate, TurnSplit())
    for lane_name, expected_rate in rates.items():
        expected = expected_rate * horizon
        std = expected ** 0.5  # Poisson std
        actual = sim.lanes[lane_name].arrived_total
        assert abs(actual - expected) < 4 * std, (
            f"{lane_name}: {actual} vs expected {expected:.1f} (std {std:.1f})"
        )


def test_turn_split_statistics():
    # Aggregate arrivals per movement across all approaches and check the
    # 20/60/20 split (forward split evenly across F1/F2).
    horizon = 20000
    base_rate = 0.2
    sim = make_sim(seed=42, arrival_rate=base_rate)
    lights = all_lights()
    for _ in range(horizon):
        sim.step(lights)
    per_movement = {"left": 0, "forward": 0, "right": 0}
    for lane in LANES:
        count = sim.lanes[lane.name].arrived_total
        if lane.movement == "forward":
            per_movement["forward"] += count
        else:
            per_movement[lane.movement] += count
    total = sum(per_movement.values())
    shares = {m: c / total for m, c in per_movement.items()}
    assert abs(shares["left"] - 0.2) < 0.02, shares
    assert abs(shares["forward"] - 0.6) < 0.02, shares
    assert abs(shares["right"] - 0.2) < 0.02, shares


def test_wait_time_semantics():
    # One car arrives at t=0; the lane goes green and the car departs when
    # discharge credit reaches 1 (end of the t=2 step). Per-car wait is 2.0.
    sim = make_sim(seed=1, arrival_rate=0.0)
    sim.lanes["N-F1"].queue.add(0.0, 1)
    lights = all_lights()
    lights["N-F1"] = True
    for _ in range(3):
        sim.step(lights)
    state = sim.state()
    assert state["total_throughput"] == 1
    assert state["avg_car_wait"]["N-F1"] == 2.0
    # wait_since_served reset at departure (t=2), one step has passed since
    assert state["wait_since_served"]["N-F1"] == 1.0


def test_wait_since_served_grows_when_idle():
    sim = make_sim(seed=2, arrival_rate=0.0)
    lights = all_lights()
    for _ in range(5):
        sim.step(lights)
    state = sim.state()
    assert state["wait_since_served"]["N-F1"] == 5.0


def test_discharge_rate():
    # A permanently green lane with a long queue serves at discharge_rate.
    sim = make_sim(seed=3, arrival_rate=0.0)
    sim.lanes["N-F1"].queue.add(0.0, 20)
    lights = all_lights()
    lights["N-F1"] = True
    for _ in range(40):
        sim.step(lights)
    # 0.5 cars/s * 40s = 20 cars (credit accumulates while the queue lasts)
    assert sim.total_throughput == 20


def test_minimum_green():
    sim = make_sim(seed=4, arrival_rate=0.0)
    lights = all_lights()
    lights["N-F1"] = True
    sim.step(lights)
    lights["N-F1"] = False  # request off after 1s; min_green is 5s
    for _ in range(4):
        sim.step(lights)
        assert sim.is_green("N-F1"), "lane switched off before min_green"
        assert "N-F1" in sim.state()["forced_green"]
    sim.step(lights)  # 5s elapsed -> yellow
    assert sim.state()["signals"]["N-F1"] == YELLOW


def test_yellow_transition():
    sim = make_sim(seed=5, arrival_rate=0.0)
    lights = all_lights()
    lights["N-F1"] = True
    for _ in range(6):  # past min_green
        sim.step(lights)
    lights["N-F1"] = False
    sim.step(lights)
    assert sim.state()["signals"]["N-F1"] == YELLOW
    sim.step(lights)
    sim.step(lights)  # yellow_duration = 2s
    assert sim.state()["signals"]["N-F1"] == RED
    # no cars discharge during yellow
    sim.lanes["N-F1"].queue.add(sim.time, 5)
    for _ in range(3):
        sim.step(lights)
    assert sim.total_throughput == 0


def test_throughput_counts_all_movements():
    sim = make_sim(seed=6, arrival_rate=0.0)
    sim.lanes["N-L"].queue.add(0.0, 2)
    sim.lanes["N-R"].queue.add(0.0, 2)
    lights = all_lights()
    lights["N-L"] = True
    lights["N-R"] = True  # same approach: compatible
    for _ in range(10):
        sim.step(lights)
    state = sim.state()
    assert state["total_throughput"] == 4
    assert state["throughput"]["west"] == 2   # N-L exits west
    assert state["throughput"]["east"] == 2   # N-R exits east


def test_green_empty_tracking():
    sim = make_sim(seed=7, arrival_rate=0.0)
    lights = all_lights()
    lights["N-F1"] = True
    for _ in range(6):
        sim.step(lights)
    state = sim.state()
    assert state["green_time"]["N-F1"] == 6.0
    assert state["green_empty_time"]["N-F1"] == 6.0  # never any cars


def test_reproducibility():
    def run(seed: int) -> dict:
        sim = make_sim(seed=seed)
        rng = np.random.default_rng(seed)
        lights = all_lights()
        lights["N-F1"] = True
        lights["S-F1"] = True
        for _ in range(200):
            sim.step(lights)
            if rng.random() < 0.05:
                lights = all_lights()
                lights["E-F1"] = True
                lights["W-F1"] = True
        return sim.state()

    first = run(11)
    second = run(11)
    assert first == second


def test_safety_invariant_random_controller():
    # 100k steps with a random safe controller: the simulator's validation
    # must never fire, and no conflicting lights may ever be green.
    sim = make_sim(seed=8)
    rng = np.random.default_rng(8)
    for _ in range(100_000):
        lights = all_lights()
        green = []
        for lane in LANES:
            if rng.random() < 0.3 and all(
                MATRIX[LANE_INDEX[lane.name]][LANE_INDEX[other]] for other in green
            ):
                green.append(lane.name)
        lights = {lane.name: lane.name in green for lane in LANES}
        sim.step(lights)  # raises if the simulator ever sees an unsafe state
    assert sim.total_throughput > 0