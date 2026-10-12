"""Scenario tests: loading, validation, and override resolution.

Covers the plan's fair-arrival tests 5 and 6 plus scenario validation.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from scenarios import (
    FULL_DURATION_SECONDS,
    QUICK_DURATION_SECONDS,
    Scenario,
    get_scenario,
    load_scenarios,
)
from simulator import Simulation, SimulationConfig


def test_all_five_scenarios_load_and_validate():
    scenarios = load_scenarios()
    expected = {"balanced", "rush_hour_north", "heavy_left_turns", "construction_north", "near_capacity"}
    assert expected <= set(scenarios)
    for scenario in scenarios.values():
        scenario.validate()  # must not raise


def test_scenario_defaults_match_validated_decisions():
    balanced = get_scenario("balanced")
    assert balanced.duration_seconds == FULL_DURATION_SECONDS == 3600.0
    assert balanced.seeds == (0, 1, 2, 3, 4)
    assert QUICK_DURATION_SECONDS == 1800.0


def test_scenario_rejects_unknown_lane_override():
    scenario = Scenario(name="bad", lane_overrides={"X-ZZ": 1.0})
    with pytest.raises(ValueError):
        scenario.validate()


def test_scenario_rejects_unknown_closed_lane():
    scenario = Scenario(name="bad", closed_lanes=("N-ZZ",))
    with pytest.raises(ValueError):
        scenario.validate()


def test_scenario_rejects_bad_capacity():
    scenario = Scenario(name="bad", lane_capacity=0)
    with pytest.raises(ValueError):
        scenario.validate()


def test_scenario_rejects_unknown_approach():
    scenario = Scenario(name="bad", approach_overrides={"northeast": 1.0})
    with pytest.raises(ValueError):
        scenario.validate()


def test_scenario_rejects_nonpositive_duration():
    scenario = Scenario(name="bad", duration_seconds=0)
    with pytest.raises(ValueError):
        scenario.validate()


def test_scenario_rejects_empty_seeds():
    scenario = Scenario(name="bad", seeds=())
    with pytest.raises(ValueError):
        scenario.validate()


def test_lane_overrides_win_over_approach_overrides():
    scenario = Scenario(
        name="event",
        arrival_rate=0.2,
        approach_overrides={"north": 0.6},
        lane_overrides={"N-L": 0.8},
    )
    config = scenario.simulation_config(seed=0)
    sim = Simulation(config)
    # N-L: direct override wins over the approach override's left share.
    assert sim.arrival_rates["N-L"] == pytest.approx(0.8)
    # N-F1: approach override with the 20/60/20 split (forward halved).
    assert sim.arrival_rates["N-F1"] == pytest.approx(0.6 * 0.3)
    # Unrelated approach uses the default rate.
    assert sim.arrival_rates["S-L"] == pytest.approx(0.2 * 0.2)


def test_approach_override_uses_turn_split():
    scenario = Scenario(name="rush", arrival_rate=0.2, approach_overrides={"north": 0.6})
    config = scenario.simulation_config(seed=0)
    sim = Simulation(config)
    assert sim.arrival_rates["N-L"] == pytest.approx(0.6 * 0.2)
    assert sim.arrival_rates["N-F1"] == pytest.approx(0.6 * 0.3)
    assert sim.arrival_rates["N-R"] == pytest.approx(0.6 * 0.2)
    assert sim.arrival_rates["S-F1"] == pytest.approx(0.2 * 0.3)


def test_closed_lane_demand_is_redirected_and_conserved():
    scenario = get_scenario("construction_north")
    config = scenario.simulation_config(seed=0)
    assert set(config.closed_lanes) == {"N-L", "N-F1", "N-F2"}
    sim = Simulation(config)
    rates = sim.arrival_rates
    # Closed lanes have zero rate.
    for lane in ("N-L", "N-F1", "N-F2"):
        assert rates[lane] == 0.0
    # Total demand equals the un-closed total: nothing is deleted. Compare
    # against the scenario's own resolved rates with no closures.
    unclosed = Scenario(
        name="construction_open",
        arrival_rate=scenario.arrival_rate,
        approach_overrides=dict(scenario.approach_overrides),
    )
    open_rates = unclosed.simulation_config(0).lane_rates
    assert sum(rates.values()) == pytest.approx(sum(open_rates.values()))
    # All redirected north demand lands on the remaining open north lane N-R.
    assert rates["N-R"] == pytest.approx(
        open_rates["N-R"] + open_rates["N-L"] + open_rates["N-F1"] + open_rates["N-F2"]
    )


def test_closed_lane_redirection_falls_back_to_turning_lane():
    # Close both forward lanes: the first available turning lane (N-L)
    # receives all redirected forward demand.
    scenario = Scenario(name="construction", closed_lanes=("N-F1", "N-F2"))
    config = scenario.simulation_config(seed=0)
    sim = Simulation(config)
    rates = sim.arrival_rates
    unclosed = Scenario(name="construction_open", arrival_rate=scenario.arrival_rate)
    open_rates = unclosed.simulation_config(0).lane_rates
    assert rates["N-L"] == pytest.approx(open_rates["N-L"] + open_rates["N-F1"] + open_rates["N-F2"])
    assert rates["N-R"] == pytest.approx(open_rates["N-R"])
    assert rates["N-F1"] == 0.0 and rates["N-F2"] == 0.0


def test_scenario_config_reaches_simulator():
    scenario = Scenario(name="cap", lane_capacity=25, arrival_rate=0.3)
    config = scenario.simulation_config(seed=3)
    assert isinstance(config, SimulationConfig)
    assert config.lane_capacity == 25
    assert config.seed == 3
    sim = Simulation(config)
    assert sim.config.lane_capacity == 25


def test_with_seeds_returns_modified_copy():
    scenario = get_scenario("balanced")
    modified = scenario.with_seeds([7, 8])
    assert modified.seeds == (7, 8)
    assert scenario.seeds == (0, 1, 2, 3, 4)  # original untouched
