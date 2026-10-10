"""Controller interaction tests independent of a pygame display."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from controllers import ALTERNATING_PHASES, AlternatingController, ManualController
from generate_config import LANES
from lights import validate_lights
from simulator import Simulation, SimulationConfig


def all_red() -> dict[str, bool]:
    return {lane.name: False for lane in LANES}


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


def test_alternating_controller_serves_every_lane_with_safe_phases():
    controller = AlternatingController(phase_length=1)
    seen = set()
    for _ in range(len(ALTERNATING_PHASES)):
        lights = controller.act({})
        validate_lights(lights)
        seen.update(name for name, enabled in lights.items() if enabled)
    assert seen == {lane.name for lane in LANES}
    assert "W-L" in seen