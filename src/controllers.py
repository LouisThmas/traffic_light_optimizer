"""Controllers: decide which lanes are green each step.

A controller receives the simulation state and returns a light vector
(lane name -> bool). The simulator validates safety, so a controller can
never create a conflicting green.
"""
from dataclasses import dataclass
from typing import Dict

from generate_config import LANES

# Approaches and their lane groups, used by phase-based controllers.
NS_LEFT = ["N-L", "S-L"]
NS_THROUGH = ["N-F1", "N-F2", "S-F1", "S-F2", "N-R", "S-R"]
EW_LEFT = ["E-L", "W-L"]
EW_THROUGH = ["E-F1", "E-F2", "W-F1", "W-F2", "E-R", "W-R"]
ALTERNATING_PHASES = (NS_LEFT, NS_THROUGH, EW_LEFT, EW_THROUGH)


def all_red() -> Dict[str, bool]:
    return {lane.name: False for lane in LANES}


@dataclass(frozen=True)
class ToggleResult:
    """Result of a requested manual signal change."""

    changed: bool
    reason: str = ""


class ManualController:
    """The user is the controller: lights come from UI clicks.

    The user's desired lights are filtered against the compatibility
    matrix: a click that would conflict with an already-green lane is
    ignored (the UI also grays out unclickable lanes). The simulator
    still enforces min-green/yellow, so clicks may be deferred.
    """

    def __init__(self) -> None:
        self.desired: Dict[str, bool] = all_red()

    def _is_allowed(self, lane_name: str, state: Dict | None = None) -> bool:
        """True if turning this lane green conflicts with nothing already green."""
        from lights import LANE_INDEX
        from generate_config import MATRIX

        actual_green = set()
        if state is not None:
            actual_green = {
                other for other, signal in state.get("signals", {}).items()
                if signal == "green"
            }

        return all(
            (not self.desired[other] and other not in actual_green)
            or MATRIX[LANE_INDEX[lane_name]][LANE_INDEX[other]]
            for other in self.desired
        )

    def can_turn_on(self, lane_name: str, state: Dict | None = None) -> bool:
        """Whether a red lane may safely be requested on right now."""
        return not self.desired[lane_name] and self._is_allowed(lane_name, state)

    def toggle(self, lane_name: str, state: Dict | None = None) -> ToggleResult:
        """Request a toggle, returning why a blocked request was rejected."""
        if lane_name not in self.desired:
            return ToggleResult(False, "unknown lane")
        if state is not None and lane_name in state.get("forced_green", []):
            return ToggleResult(False, "minimum green is still active")
        if self.desired[lane_name]:
            self.desired[lane_name] = False  # switching off is always safe
            return ToggleResult(True)
        if self._is_allowed(lane_name, state):
            self.desired[lane_name] = True
            return ToggleResult(True)
        return ToggleResult(False, "conflicts with an active requested signal")

    def set(self, lights: Dict[str, bool]) -> None:
        self.desired = dict(lights)

    def act(self, state: Dict) -> Dict[str, bool]:
        return dict(self.desired)


class AlternatingController:
    """Fixed four-phase cycle that serves every one of the 16 lanes.

    Protected opposing left turns are followed by the corresponding
    through/right phase, first north-south and then east-west.
    """

    def __init__(self, phase_length: int = 15) -> None:
        self.phase_length = phase_length
        self.step_count = 0

    def act(self, state: Dict) -> Dict[str, bool]:
        lights = all_red()
        phase_index = (self.step_count // self.phase_length) % len(ALTERNATING_PHASES)
        phase = ALTERNATING_PHASES[phase_index]
        for lane_name in phase:
            lights[lane_name] = True
        self.step_count += 1
        return lights


class RandomController:
    """Picks a random safe light vector each step (the floor to beat)."""

    def __init__(self, seed: int = 0, green_probability: float = 0.3) -> None:
        import numpy as np

        self.rng = np.random.default_rng(seed)
        self.green_probability = green_probability

    def act(self, state: Dict) -> Dict[str, bool]:
        from lights import LANE_INDEX, validate_lights
        from generate_config import MATRIX

        green = []
        for lane in LANES:
            if self.rng.random() < self.green_probability and all(
                MATRIX[LANE_INDEX[lane.name]][LANE_INDEX[other]] for other in green
            ):
                green.append(lane.name)
        lights = all_red()
        for lane_name in green:
            lights[lane_name] = True
        validate_lights(lights)  # never trip the simulator's check
        return lights


CONTROLLERS = {
    "manual": ManualController,
    "alternating": AlternatingController,
    "random": RandomController,
}