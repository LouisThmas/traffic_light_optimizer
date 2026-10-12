"""Exhaustive deterministic tests for manual and baseline controllers.

Run: python -m pytest tests/test_controllers.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from controllers import (
    FIXED_CYCLE_PHASES,
    ActuatedController,
    FixedCycleController,
    ManualController,
    MaxPressureController,
    RandomController,
)
from generate_config import LANES, MATRIX
from lights import LANE_INDEX, validate_lights
from simulator import Simulation, SimulationConfig


def all_red():
    return {lane.name: False for lane in LANES}


def state_with_queues(**queues):
    """Minimal controller state with named queue overrides."""
    return {"queues": {lane.name: queues.get(lane.name, 0) for lane in LANES}}


def green_lanes(lights):
    return {name for name, enabled in lights.items() if enabled}


def test_manual_controller_rejects_conflicting_request():
    controller = ManualController()
    assert controller.toggle("N-F1").changed
    result = controller.toggle("E-F1")
    assert not result.changed
    assert "conflicts" in result.reason


def test_manual_controller_cannot_switch_off_forced_green():
    sim = Simulation(SimulationConfig(arrival_rate=0.0))
    controller = ManualController()
    assert controller.toggle("N-F1").changed
    sim.step(controller.act(sim.state()))
    result = controller.toggle("N-F1", sim.state())
    assert not result.changed
    assert result.reason == "minimum green is still active"


def test_manual_controller_uses_actual_green_for_eligibility():
    sim = Simulation(SimulationConfig(arrival_rate=0.0))
    lights = all_red()
    lights["N-F1"] = True
    sim.step(lights)
    controller = ManualController()
    assert not controller.can_turn_on("E-F1", sim.state())


def test_random_controller_is_reproducible_with_seed():
    first = RandomController(seed=41, green_probability=0.45)
    second = RandomController(seed=41, green_probability=0.45)
    state = state_with_queues()
    assert [first.act(state) for _ in range(30)] == [second.act(state) for _ in range(30)]


def test_random_controller_always_returns_a_safe_configuration():
    controller = RandomController(seed=9, green_probability=0.8)
    for _ in range(500):
        lights = controller.act(state_with_queues())
        validate_lights(lights)
        green = green_lanes(lights)
        for first in green:
            for second in green:
                assert MATRIX[LANE_INDEX[first]][LANE_INDEX[second]]


def test_fixed_cycle_uses_the_exact_documented_phase_order():
    controller = FixedCycleController(phase_length=1)
    actual = [green_lanes(controller.act(state_with_queues())) for _ in FIXED_CYCLE_PHASES]
    assert actual == [set(phase) for phase in FIXED_CYCLE_PHASES]


def test_fixed_cycle_holds_each_phase_for_its_configured_length():
    controller = FixedCycleController(phase_length=3)
    actual = [green_lanes(controller.act(state_with_queues())) for _ in range(6)]
    assert actual[:3] == [set(FIXED_CYCLE_PHASES[0])] * 3
    assert actual[3:] == [set(FIXED_CYCLE_PHASES[1])] * 3


def test_fixed_cycle_wraps_back_to_its_first_phase():
    controller = FixedCycleController(phase_length=1)
    for _ in FIXED_CYCLE_PHASES:
        controller.act(state_with_queues())
    assert green_lanes(controller.act(state_with_queues())) == set(FIXED_CYCLE_PHASES[0])


def test_fixed_cycle_serves_every_lane_once_per_cycle():
    served = set().union(*(set(phase) for phase in FIXED_CYCLE_PHASES))
    assert served == {lane.name for lane in LANES}


def test_every_fixed_cycle_phase_is_safe_and_maximal():
    """No conflict is lit, and no unlit lane can be added safely."""
    all_lanes = {lane.name for lane in LANES}
    for phase in FIXED_CYCLE_PHASES:
        phase_set = set(phase)
        validate_lights({lane.name: lane.name in phase_set for lane in LANES})
        for candidate in all_lanes - phase_set:
            assert any(not MATRIX[LANE_INDEX[candidate]][LANE_INDEX[active]] for active in phase_set), (
                f"{candidate} can be added to non-maximal phase {phase}"
            )


def test_actuated_switches_when_an_inactive_lane_reaches_threshold():
    controller = ActuatedController(queue_threshold=20)
    # E-L is served by phase 1, but not the initial phase 0.
    lights = controller.act(state_with_queues(**{"E-L": 20}))
    assert green_lanes(lights) == set(FIXED_CYCLE_PHASES[1])


def test_actuated_does_not_switch_when_active_phase_has_threshold_queue():
    controller = ActuatedController(queue_threshold=20)
    lights = controller.act(state_with_queues(**{"W-L": 20, "E-L": 30}))
    assert green_lanes(lights) == set(FIXED_CYCLE_PHASES[0])


def test_actuated_resolves_simultaneous_threshold_ties_by_phase_order():
    controller = ActuatedController(queue_threshold=20)
    # E-L is phase 1; N-L is phase 2. Phase 1 wins the deterministic tie.
    lights = controller.act(state_with_queues(**{"N-L": 20, "E-L": 20}))
    assert green_lanes(lights) == set(FIXED_CYCLE_PHASES[1])


def test_actuated_does_not_switch_below_threshold():
    controller = ActuatedController(queue_threshold=20)
    lights = controller.act(state_with_queues(**{"E-L": 19}))
    assert green_lanes(lights) == set(FIXED_CYCLE_PHASES[0])


def test_max_pressure_switches_to_a_substantially_busier_phase():
    controller = MaxPressureController(switch_ratio=1.4)
    lights = controller.act(state_with_queues(**{"W-L": 10, "E-L": 20}))
    assert green_lanes(lights) == set(FIXED_CYCLE_PHASES[1])


def test_max_pressure_does_not_switch_for_less_than_40_percent_advantage():
    controller = MaxPressureController(switch_ratio=1.4, lane_capacity=1000)
    lights = controller.act(state_with_queues(**{"W-L": 100, "E-L": 139}))
    assert green_lanes(lights) == set(FIXED_CYCLE_PHASES[0])


def test_max_pressure_switches_at_exactly_40_percent_advantage():
    controller = MaxPressureController(switch_ratio=1.4, lane_capacity=1000)
    lights = controller.act(state_with_queues(**{"W-L": 100, "E-L": 140}))
    assert green_lanes(lights) == set(FIXED_CYCLE_PHASES[1])


def test_max_pressure_considers_total_phase_volume_not_just_one_lane():
    controller = MaxPressureController(switch_ratio=1.4)
    lights = controller.act(state_with_queues(**{"W-L": 50, "E-F1": 40, "E-F2": 40}))
    assert green_lanes(lights) == set(FIXED_CYCLE_PHASES[1])


def test_max_pressure_tie_keeps_current_phase_and_avoids_churn():
    controller = MaxPressureController(switch_ratio=1.4)
    lights = controller.act(state_with_queues(**{"W-L": 25, "E-L": 25}))
    assert green_lanes(lights) == set(FIXED_CYCLE_PHASES[0])


def test_max_pressure_overflow_lane_beats_larger_other_phase_total():
    controller = MaxPressureController(switch_ratio=1.4, lane_capacity=100, overflow_threshold=0.9)
    # W-L is the only usable west lane during construction and is at 90%.
    # Phase 1 has a larger total, but the phase serving W-L must win.
    lights = controller.act(state_with_queues(**{"W-L": 90, "E-L": 50, "E-F1": 50}))
    assert green_lanes(lights) == set(FIXED_CYCLE_PHASES[0])
