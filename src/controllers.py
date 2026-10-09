"""Controllers: decide which lanes are green each step.

A controller receives the simulation state and returns a light vector
(lane name -> bool). The simulator validates safety, so a controller can
never create a conflicting green.
"""
from typing import Dict

from generate_config import LANES

# Approaches and their lane groups, used by phase-based controllers.
NS_THROUGH = ["N-F1", "N-F2", "S-F1", "S-F2", "N-R", "S-R"]
EW_THROUGH = ["E-F1", "E-F2", "W-F1", "W-F2", "E-R", "W-R"]


def all_red() -> Dict[str, bool]:
    return {lane.name: False for lane in LANES}


class ManualController:
    """The user is the controller: lights come from UI clicks.

    The user's desired lights are filtered against the compatibility
    matrix: a click that would conflict with an already-green lane is
    ignored (the UI also grays out unclickable lanes). The simulator
    still enforces min-green/yellow, so clicks may be deferred.
    """

    def __init__(self) -> None:
        self.desired: Dict[str, bool] = all_red()

    def _is_allowed(self, lane_name: str) -> bool:
        """True if turning this lane green conflicts with nothing already green."""
        from lights import LANE_INDEX
        from generate_config import MATRIX

        return all(
            not self.desired[other]
            or MATRIX[LANE_INDEX[lane_name]][LANE_INDEX[other]]
            for other in self.desired
        )

    def toggle(self, lane_name: str) -> None:
        if self.desired[lane_name]:
            self.desired[lane_name] = False  # switching off is always safe
        elif self._is_allowed(lane_name):
            self.desired[lane_name] = True
        # else: conflicting click ignored

    def set(self, lights: Dict[str, bool]) -> None:
        self.desired = dict(lights)

    def act(self, state: Dict) -> Dict[str, bool]:
        return dict(self.desired)


class AlternatingController:
    """Fixed cycle: north-south for `phase_length` steps, then east-west."""

    def __init__(self, phase_length: int = 15) -> None:
        self.phase_length = phase_length
        self.step_count = 0

    def act(self, state: Dict) -> Dict[str, bool]:
        lights = all_red()
        phase = NS_THROUGH if (self.step_count // self.phase_length) % 2 == 0 else EW_THROUGH
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