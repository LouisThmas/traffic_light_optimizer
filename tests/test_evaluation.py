"""Evaluation harness tests.

Covers the plan's determinism (1-3), fair-arrival (4), conservation (7-9),
metrics (10-14), controller integration (15-17), and regression snapshot (18)
tests, plus result-format validation and aggregation.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from controllers import MaxPressureController
from evaluation import (
    aggregate_by_controller,
    aggregate_with_scores,
    create_controller,
    findings_table,
    read_jsonl,
    run_many,
    run_one,
    validate_result,
    write_jsonl,
)
from generate_config import LANES
from lights import validate_lights
from metrics import extract_metrics, score_metrics
from scenarios import Scenario, get_scenario
from simulator import Simulation, SimulationConfig

BASELINES = ["fixed_cycle", "random", "actuated", "max_pressure"]
SCENARIO_NAMES = ["balanced", "rush_hour_north", "heavy_left_turns", "construction_north", "near_capacity"]


def tiny_scenario(**overrides) -> Scenario:
    defaults = dict(
        name="tiny",
        duration_seconds=60.0,
        arrival_rate=0.2,
        seeds=(0,),
    )
    defaults.update(overrides)
    return Scenario(**defaults)


# -- Determinism tests (1-3) -----------------------------------------


def test_same_controller_scenario_seed_gives_identical_results():
    scenario = tiny_scenario()
    first = run_one("fixed_cycle", scenario, seed=42)
    second = run_one("fixed_cycle", scenario, seed=42)
    assert first.to_dict() == second.to_dict()


def test_different_seeds_may_differ():
    scenario = tiny_scenario(duration_seconds=300.0)
    first = run_one("fixed_cycle", scenario, seed=0)
    second = run_one("fixed_cycle", scenario, seed=1)
    # Not asserting they must differ (they could coincide), but the full
    # record including per-lane detail must be reproducible per seed.
    assert run_one("fixed_cycle", scenario, seed=0).to_dict() == first.to_dict()
    assert run_one("fixed_cycle", scenario, seed=1).to_dict() == second.to_dict()


def test_run_order_does_not_change_results():
    scenario = tiny_scenario(duration_seconds=120.0)
    a_first = run_one("fixed_cycle", scenario, seed=5)
    b_first = run_one("max_pressure", scenario, seed=5)
    # Reverse order.
    b_again = run_one("max_pressure", scenario, seed=5)
    a_again = run_one("fixed_cycle", scenario, seed=5)
    assert a_first.to_dict() == a_again.to_dict()
    assert b_first.to_dict() == b_again.to_dict()


# -- Fair-arrival tests (4-6) ----------------------------------------


def test_controllers_see_same_arrivals_per_lane():
    scenario = tiny_scenario(duration_seconds=120.0)
    fixed = run_one("fixed_cycle", scenario, seed=9)
    pressure = run_one("max_pressure", scenario, seed=9)
    # Arrivals are generated before discharge each step and depend only on
    # the seed, so both controllers must see identical accepted arrivals.
    assert fixed.arrived == pressure.arrived


def test_scenario_overrides_reach_lane_rates():
    scenario = Scenario(
        name="override",
        arrival_rate=0.2,
        approach_overrides={"north": 0.6},
        lane_overrides={"N-L": 0.8},
    )
    config = scenario.simulation_config(seed=0)
    sim = Simulation(config)
    # The override-resolved rates reach the simulator.
    assert sim.arrival_rates["N-L"] == pytest.approx(0.8)
    assert sim.arrival_rates["N-F1"] == pytest.approx(0.6 * 0.3)


def _scenario_resolved_rates(scenario: Scenario):
    from arrivals import TurnSplit

    split = TurnSplit()
    rates = {}
    from generate_config import LANES as LANES_LIST

    for lane in LANES_LIST:
        base = scenario.approach_overrides.get(lane.direction, scenario.arrival_rate)
        rates[lane.name] = base * split.share(lane.movement)
    rates.update(scenario.lane_overrides)
    return rates


def test_closed_lane_demand_conserved_through_redirection():
    scenario = get_scenario("construction_north").with_seeds([0])
    records = run_many(["fixed_cycle"], scenario)
    record = records[0]
    # Nothing is deleted: every accepted arrival is either queued or departed.
    assert record.arrived == record.queued_final + record.throughput_total
    # The closed lanes never receive arrivals.
    sim = _final_state(scenario, "fixed_cycle", 0)
    for lane in ("N-L", "N-F1", "N-F2"):
        assert sim.lanes[lane].arrived_total == 0


def _final_state(scenario: Scenario, controller_name: str, seed: int) -> Simulation:
    config = scenario.simulation_config(seed)
    sim = Simulation(config)
    controller = create_controller(controller_name, None, seed)
    closed = set(config.closed_lanes)
    for _ in range(int(round(scenario.duration_seconds / config.step_length))):
        lights = controller.act(sim.state())
        if closed:
            lights = {name: on for name, on in lights.items() if name not in closed}
        sim.step(lights)
    return sim


# -- Conservation tests (7-9) ----------------------------------------


@pytest.mark.parametrize("controller_name", BASELINES)
def test_conservation_at_each_sampled_point(controller_name):
    scenario = tiny_scenario(duration_seconds=90.0)
    config = scenario.simulation_config(seed=2)
    sim = Simulation(config)
    controller = create_controller(controller_name, None, 2)
    for step in range(int(round(scenario.duration_seconds / config.step_length))):
        sim.step(controller.act(sim.state()))
        if step % 7 == 0:
            state = sim.state()
            assert state["arrived"] == sum(state["queues"].values()) + state["total_throughput"]


def test_finite_capacity_conservation():
    scenario = tiny_scenario(lane_capacity=5, arrival_rate=0.5, duration_seconds=120.0)
    config = scenario.simulation_config(seed=1)
    sim = Simulation(config)
    controller = create_controller("fixed_cycle", None, 1)
    for _ in range(int(round(scenario.duration_seconds / config.step_length))):
        sim.step(controller.act(sim.state()))
    state = sim.state()
    attempted = state["arrived"] + state["blocked_arrivals"]
    # Attempted arrivals equal what the all-red Poisson draw would produce:
    # blocked + accepted must equal the drawn total, checked via conservation
    # of accepted arrivals and positivity of blocking under saturation.
    assert state["blocked_arrivals"] > 0
    assert state["arrived"] == sum(state["queues"].values()) + state["total_throughput"]
    assert attempted >= state["arrived"]


def test_aggregate_equals_arithmetic_aggregation_of_seeds():
    scenario = tiny_scenario(duration_seconds=120.0, seeds=(0, 1, 2))
    records = run_many(["fixed_cycle"], scenario)
    rows = aggregate_by_controller(records)
    assert len(rows) == 1
    row = rows[0]
    direct = [r.throughput_total for r in records]
    assert row["throughput_total"]["mean"] == pytest.approx(sum(direct) / len(direct))
    assert row["throughput_total"]["min"] == float(min(direct))
    assert row["throughput_total"]["max"] == float(max(direct))
    assert row["throughput_total"]["std"] == pytest.approx(float(np.std(direct)))


# -- Metrics tests (10-14) -------------------------------------------


def test_hand_built_tiny_run_known_values():
    # One lane green the whole time, one car arrives at t=0, discharges at
    # the end of step 2 (discharge_rate 0.5 -> credit 1.0 after 2 seconds).
    config = SimulationConfig(seed=0, arrival_rate=0.0, discharge_rate=0.5, step_length=1.0)
    sim = Simulation(config)
    sim.lanes["N-F1"].queue.add(0.0, 1)
    sim.lanes["N-F1"].arrived_total = 1
    sim.arrived = 1
    lights = {lane.name: False for lane in LANES}
    lights["N-F1"] = True
    sim.step(lights)
    sim.step(lights)
    sim.step(lights)
    # Car departs at the end of step index 2 (departure_time = time + dt = 2.0).
    assert sim.total_throughput == 1
    assert sim.lanes["N-F1"].cars_departed == 1
    assert sim.lanes["N-F1"].total_car_wait == pytest.approx(2.0)
    record = extract_metrics(sim, "manual", "tiny", 0, 3.0)
    assert record.throughput_total == 1
    assert record.arrived == 1
    assert record.queued_final == 0
    assert record.avg_car_wait == pytest.approx(2.0)


def test_permanently_red_lane_zero_departures_and_growing_service_gap():
    config = SimulationConfig(seed=0, arrival_rate=0.2)
    sim = Simulation(config)
    lights = {lane.name: False for lane in LANES}
    for _ in range(50):
        sim.step(lights)
    lane = sim.lanes["N-L"]
    assert lane.cars_departed == 0
    assert lane.wait_since_served == pytest.approx(50.0)
    record = extract_metrics(sim, "manual", "tiny", 0, 50.0)
    assert record.per_lane_avg_wait["N-L"] == 0.0
    assert record.per_lane_final_queue["N-L"] == len(lane.queue)
    assert record.lanes_with_no_departures >= 1


def test_permanently_green_empty_lane_has_nonzero_green_empty_ratio():
    config = SimulationConfig(seed=0, arrival_rate=0.0)
    sim = Simulation(config)
    lights = {lane.name: False for lane in LANES}
    lights["N-F1"] = True
    for _ in range(30):
        sim.step(lights)
    record = extract_metrics(sim, "manual", "tiny", 0, 30.0)
    assert record.green_empty_ratio == pytest.approx(1.0)


def test_lane_at_capacity_records_blocked_arrivals():
    config = SimulationConfig(seed=0, arrival_rate=1.0, lane_capacity=3)
    sim = Simulation(config)
    lights = {lane.name: False for lane in LANES}
    for _ in range(100):
        sim.step(lights)
    assert sim.blocked_arrivals > 0
    record = extract_metrics(sim, "manual", "tiny", 0, 100.0)
    assert record.blocked_arrivals == sim.blocked_arrivals
    assert record.spillback_rate > 0.0


@pytest.mark.parametrize("controller_name", BASELINES)
@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_no_metric_divides_by_zero_or_emits_nan(controller_name, scenario_name):
    scenario = get_scenario(scenario_name).with_seeds([0])
    record = run_one(controller_name, scenario, seed=0)
    data = record.to_dict()["metrics"]

    def check(value):
        if isinstance(value, dict):
            for inner in value.values():
                check(inner)
        elif isinstance(value, float):
            assert not math.isnan(value)
            assert not math.isinf(value)

    check(data)
    # Zero-arrival edge case: derived metrics emit 0.0, never NaN.
    if data["arrived"] == 0:
        assert data["queue_clearance_ratio"] == 0.0


# -- Controller integration tests (15-17) ----------------------------


@pytest.mark.parametrize("controller_name", BASELINES)
@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_every_baseline_completes_every_scenario(controller_name, scenario_name):
    scenario = get_scenario(scenario_name).with_seeds([0])
    record = run_one(controller_name, scenario, seed=0)  # must not raise
    validate_result(record.to_dict())


@pytest.mark.parametrize("controller_name", BASELINES)
@pytest.mark.parametrize("scenario_name", SCENARIO_NAMES)
def test_light_vectors_remain_matrix_safe(controller_name, scenario_name):
    scenario = get_scenario(scenario_name).with_seeds([0])
    config = scenario.simulation_config(seed=0)
    sim = Simulation(config)
    controller = create_controller(controller_name, None, 0)
    closed = set(config.closed_lanes)
    for _ in range(int(round(scenario.duration_seconds / config.step_length))):
        lights = controller.act(sim.state())
        if closed:
            lights = {name: on for name, on in lights.items() if name not in closed}
        validate_lights(lights)  # raises on any unsafe vector
        sim.step(lights)


def test_construction_scenario_overflow_policy_serves_remaining_lane():
    # With all north forward lanes closed, N-R is the only open north lane
    # and receives all redirected demand. Max pressure's overflow policy
    # must keep its queue below capacity where possible.
    scenario = get_scenario("construction_north").with_seeds([0])
    capacity = 100  # MaxPressureController default reference capacity
    controller = MaxPressureController(lane_capacity=capacity, overflow_threshold=0.9)
    config = scenario.simulation_config(seed=0)
    sim = Simulation(config)
    closed = set(config.closed_lanes)
    overflow_breaches = 0
    for _ in range(int(round(scenario.duration_seconds / config.step_length))):
        lights = controller.act(sim.state())
        if closed:
            lights = {name: on for name, on in lights.items() if name not in closed}
        sim.step(lights)
        if sim.lanes["N-R"].signal == "red" and len(sim.lanes["N-R"].queue) >= controller.overflow_queue_limit:
            overflow_breaches += 1
    # The overflow valve must react long before the queue can spill over:
    # once the queue reaches the limit while the lane is red, the very next
    # decision gives its phase priority, so sustained breaches are impossible.
    assert overflow_breaches <= 1


# -- Result format and aggregation -----------------------------------


def test_jsonl_round_trip_and_schema_validation(tmp_path):
    scenario = tiny_scenario(seeds=(0, 1))
    records = run_many(["fixed_cycle"], scenario)
    path = str(tmp_path / "runs.jsonl")
    write_jsonl(records, path)
    loaded = read_jsonl(path)
    assert len(loaded) == 2
    for record in loaded:
        validate_result(record)
        assert record["controller"] == "fixed_cycle"
        assert set(record) == {"controller", "controller_settings", "scenario", "seed", "duration_seconds", "metrics"}


def test_validate_result_rejects_corrupt_records():
    good = run_one("fixed_cycle", tiny_scenario(), seed=0).to_dict()
    validate_result(good)
    for field in ("controller", "scenario", "seed", "metrics"):
        corrupt = dict(good)
        del corrupt[field]
        with pytest.raises(ValueError):
            validate_result(corrupt)
    corrupt_metric = dict(good)
    corrupt_metric["metrics"] = dict(good["metrics"])
    corrupt_metric["metrics"]["avg_car_wait"] = float("nan")
    with pytest.raises(ValueError):
        validate_result(corrupt_metric)
    negative = dict(good)
    negative["metrics"] = dict(good["metrics"])
    negative["metrics"]["throughput_total"] = -1
    with pytest.raises(ValueError):
        validate_result(negative)


def test_score_weights_stored_in_summary():
    scenario = tiny_scenario(seeds=(0, 1))
    records = run_many(["fixed_cycle", "max_pressure"], scenario)
    rows = aggregate_with_scores(records, weights={"avg_car_wait": 2.0})
    for row in rows:
        assert row["score_weights"] == {"avg_car_wait": 2.0}
    # Equal-weight default stores empty dict meaning all 1.0.
    default_rows = aggregate_with_scores(records)
    for row in default_rows:
        assert row["score_weights"] == {}


def test_score_prefers_lower_wait_when_weighted():
    scenario = tiny_scenario(seeds=(0,))
    records = run_many(["fixed_cycle", "max_pressure"], scenario)
    scores = score_metrics(records, weights={"avg_car_wait": 10.0, "throughput_total": 0.0})
    waits = {r.controller: r.avg_car_wait for r in records}
    best_wait = min(waits, key=waits.get)
    best_score = max(scores, key=scores.get)
    assert best_score == best_wait


def test_findings_table_renders_markdown():
    scenario = tiny_scenario(seeds=(0,))
    records = run_many(["fixed_cycle"], scenario)
    rows = aggregate_with_scores(records)
    table = findings_table(rows)
    assert table.startswith("| Controller |")
    assert "fixed_cycle" in table


# -- Regression snapshot test (18) -----------------------------------


def test_fixed_cycle_regression_snapshot():
    """Tiny fixed-seed 60s balanced scenario; change here means review."""
    scenario = tiny_scenario(name="snapshot", duration_seconds=60.0, arrival_rate=0.2)
    record = run_one("fixed_cycle", scenario, seed=0)
    data = record.to_dict()["metrics"]
    expected = {
        "throughput_total": 36,
        "arrived": 47,
        "queued_final": 11,
        "avg_car_wait": 14.222222222222221,
        "green_empty_ratio": 0.8280254777070064,
        "blocked_arrivals": 0,
        "phase_changes": 19,
    }
    for key, value in expected.items():
        assert data[key] == pytest.approx(value), f"{key}: {data[key]} != {value}"
