"""Geometric ground-truth tests for the compatibility matrix.

These tests do NOT read expectations from the matrix itself. They iterate over
lane pairs grouped by geometric relationship (same approach, opposing
approaches, perpendicular approaches) and assert what road rules say should
happen. If someone edits the matrix or the conflict rules by accident, these
tests catch it.

Run:  python3 -m pytest tests/ -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_config import EXIT_DIRECTIONS, LANES, MATRIX, Lane, _compatible

DIRECTIONS = ("north", "south", "east", "west")
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
PERPENDICULAR = {
    "north": ("east", "west"),
    "south": ("east", "west"),
    "east": ("north", "south"),
    "west": ("north", "south"),
}


def lane(direction: str, movement: str, lane_number: int = 1) -> Lane:
    return Lane(direction=direction, movement=movement, lane_number=lane_number)


def matrix_entry(first: Lane, second: Lane) -> bool:
    """The matrix value for a lane pair, looked up by lane name."""
    first_index = [lane_.name for lane_ in LANES].index(first.name)
    second_index = [lane_.name for lane_ in LANES].index(second.name)
    return MATRIX[first_index][second_index]


def test_matrix_is_symmetric_with_true_diagonal():
    for first_index, first_lane in enumerate(LANES):
        assert matrix_entry(first_lane, first_lane) is True
        for second_index in range(first_index + 1, len(LANES)):
            second_lane = LANES[second_index]
            assert matrix_entry(first_lane, second_lane) == matrix_entry(second_lane, first_lane)


def test_same_approach_lanes_are_always_compatible():
    # Parallel lanes from the same road never conflict, whatever the movements.
    for direction in DIRECTIONS:
        for first_movement in ("left", "forward", "right"):
            for second_movement in ("left", "forward", "right"):
                for first_number in (1, 2):
                    for second_number in (1, 2):
                        first = lane(direction, first_movement, first_number)
                        second = lane(direction, second_movement, second_number)
                        assert matrix_entry(first, second) is True, f"{first.name} vs {second.name}"


def test_opposing_left_turns_are_compatible():
    # The intersection is assumed large enough for opposing lefts to turn
    # side by side without crossing.
    for direction in DIRECTIONS:
        assert matrix_entry(lane(direction, "left"), lane(OPPOSITE[direction], "left")) is True, (
            f"{direction} left vs {OPPOSITE[direction]} left"
        )


def test_opposing_pairs():
    # For every opposing direction pair, check each movement combination
    # against what the geometry allows.
    for direction in DIRECTIONS:
        opposing_direction = OPPOSITE[direction]
        for first_movement in ("left", "forward", "right"):
            for second_movement in ("left", "forward", "right"):
                first = lane(direction, first_movement)
                second = lane(opposing_direction, second_movement)
                expected = _expected_opposing(direction, first_movement, opposing_direction, second_movement)
                assert matrix_entry(first, second) is expected, f"{first.name} vs {second.name}: expected {expected}"


def _expected_opposing(first_direction: str, first_movement: str, second_direction: str, second_movement: str) -> bool:
    """Geometric expectation for two lanes from opposing approaches.

    Exit-based, like the perpendicular rule: two lanes conflict when they
    exit onto the same road (merge) or when a left turn crosses oncoming
    through traffic. For an opposing pair the exits per movement are
    (north vs south): left->west, forward->south, right->east versus
    left->east, forward->north, right->west, so:

    - left + left: exits west vs east, paths turn side by side -> allowed
    - left + forward: exits differ, but the left crosses oncoming
      through traffic -> illegal
    - left + right: both exit west (the same side road) -> merge -> illegal
    - forward + forward: parallel paths straight through -> allowed
    - forward + right: exits south vs west -> allowed
    - right + right: exits east vs west, different roads -> allowed
    """
    first_exit = EXIT_DIRECTIONS[(first_direction, first_movement)]
    second_exit = EXIT_DIRECTIONS[(second_direction, second_movement)]
    if first_exit == second_exit:
        return False  # merge into the same exit road
    if {first_movement, second_movement} == {"left", "forward"}:
        return False  # left turn crosses oncoming through traffic
    return True


def test_perpendicular_pairs():
    # For every perpendicular direction pair, check each movement combination.
    for direction in DIRECTIONS:
        for perpendicular_direction in PERPENDICULAR[direction]:
            for first_movement in ("left", "forward", "right"):
                for second_movement in ("left", "forward", "right"):
                    first = lane(direction, first_movement)
                    second = lane(perpendicular_direction, second_movement)
                    expected = _expected_perpendicular(
                        direction, first_movement, perpendicular_direction, second_movement
                    )
                    assert matrix_entry(first, second) is expected, (
                        f"{first.name} vs {second.name}: expected {expected}"
                    )


def _expected_perpendicular(
    first_direction: str, first_movement: str, second_direction: str, second_movement: str
) -> bool:
    """Geometric expectation for two lanes from perpendicular approaches.

    Conflicts, verified against explicit 2D path geometry:

    - forward + forward: paths cross at the center -> illegal
    - left + left: both sweep the center, paths cross -> illegal
    - left + forward where the forward approach's road IS the left's
      exit road: the sweep crosses that road's through lanes -> illegal
      (e.g. N-L exits east and crosses E-F1 westbound)
    - everything else (including left + right with different exits):
      paths do not cross -> allowed
    """
    first_exit = EXIT_DIRECTIONS[(first_direction, first_movement)]
    second_exit = EXIT_DIRECTIONS[(second_direction, second_movement)]
    if first_exit == second_exit:
        return False  # merge into the same exit road
    if first_movement == "forward" and second_movement == "forward":
        return False  # paths cross at the center
    if first_movement == "left" and second_movement == "left":
        return False  # both sweep the center
    if first_movement == "left" and second_movement == "forward":
        if second_direction == first_exit:
            return False  # sweep crosses the exit road's through lanes
    if second_movement == "left" and first_movement == "forward":
        if first_direction == second_exit:
            return False
    return True


def test_same_exit_from_different_approaches_is_illegal():
    # Any two lanes from different approaches that exit onto the same road
    # merge into one stream -> illegal, regardless of movement type.
    for first in LANES:
        for second in LANES:
            if first.direction != second.direction and first.exit_direction == second.exit_direction:
                assert matrix_entry(first, second) is False, (
                    f"{first.name} and {second.name} both exit {first.exit_direction}"
                )


def test_matrix_matches_compatibility_function():
    # The generated matrix must agree with the _compatible function for
    # every lane pair (guards against the matrix drifting from the rules).
    for first_index, first_lane in enumerate(LANES):
        for second_index, second_lane in enumerate(LANES):
            assert matrix_entry(first_lane, second_lane) == _compatible(first_lane, second_lane)
