"""Light-state validation against the compatibility matrix.

The simulator refuses any light vector that turns on two conflicting lanes.
Controllers are responsible for safety; a violation here is a bug, not
something to silently repair.
"""

from generate_config import LANES, MATRIX

LANE_INDEX = {lane.name: index for index, lane in enumerate(LANES)}


def validate_lights(lights: dict[str, bool]) -> None:
    """Raise ValueError if any two green lanes conflict."""
    green = [name for name, on in lights.items() if on]
    for first_index, first_name in enumerate(green):
        for second_name in green[first_index + 1 :]:
            if not MATRIX[LANE_INDEX[first_name]][LANE_INDEX[second_name]]:
                raise ValueError(f"Unsafe light state: {first_name} and {second_name} conflict but are both green")
