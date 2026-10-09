"""Generate config/intersections.yaml from the lane/movement definitions below.

Run:  python3 scripts/generate_config.py
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

# Lane naming: <Direction>-<Movement><Lane#>
#   Directions: N (north), S (south), E (east), W (west)
#   Movements:  L (left), F (forward), R (right)
#   Lane numbers start at 1. Only forward movements have more than one lane,
#   so the number is part of the name for forward lanes only (N-F1, N-F2);
#   single-lane movements omit it (N-L, N-R).
DIRECTION_CODES = {"north": "N", "south": "S", "east": "E", "west": "W"}
MOVEMENT_CODES = {"left": "L", "forward": "F", "right": "R"}

# Where a vehicle ends up for each (approach direction, movement) pair.
# An approach is named by where the vehicle COMES FROM: a car on the north
# approach is heading south. In the driver's frame, heading south:
# forward = south, left = east, right = west.
EXIT_DIRECTIONS = {
    ("north", "left"): "east", ("north", "forward"): "south", ("north", "right"): "west",
    ("south", "left"): "west", ("south", "forward"): "north", ("south", "right"): "east",
    ("east", "left"): "south", ("east", "forward"): "west", ("east", "right"): "north",
    ("west", "left"): "north", ("west", "forward"): "east", ("west", "right"): "south",
}


@dataclass(frozen=True)
class Lane:
    """One physical lane feeding the intersection.

    direction:    which road the lane approaches from ("north", "south", ...)
    movement:     what vehicles in the lane do ("left", "forward", "right")
    lane_number:  1-based index among lanes sharing the same direction and
                  movement (1 by default; only forward has a lane 2)
    """

    direction: str
    movement: str
    lane_number: int = 1

    @property
    def name(self) -> str:
        number_suffix = str(self.lane_number) if self.movement == "forward" else ""
        return (
            f"{DIRECTION_CODES[self.direction]}-"
            f"{MOVEMENT_CODES[self.movement]}{number_suffix}"
        )

    @property
    def exit_direction(self) -> str:
        return EXIT_DIRECTIONS[(self.direction, self.movement)]


def _build_lanes() -> list[Lane]:
    lanes: list[Lane] = []
    for direction in ("north", "south", "east", "west"):
        lanes.append(Lane(direction=direction, movement="left"))
        lanes.append(Lane(direction=direction, movement="forward", lane_number=1))
        lanes.append(Lane(direction=direction, movement="forward", lane_number=2))
        lanes.append(Lane(direction=direction, movement="right"))
    return lanes


LANES = _build_lanes()

# Compatibility rules:
#   True  = movements may run simultaneously (no crossing conflict)
#   False = potential collision; must not be green at the same time
#
# Conflict rules (applied symmetrically), verified against explicit 2D
# path geometry for all 120 cross-approach lane pairs:
#   - Same exit road from different approaches -> merge conflict (streams
#     converge). Lanes from the SAME approach are parallel and compatible.
#   - Left turn vs opposing through -> the left crosses oncoming traffic
#     (e.g. N-L vs S-F1).
#   - Left turn vs perpendicular through traffic ON ITS EXIT ROAD -> the
#     left's sweep crosses that road's through lanes (e.g. N-L exits east
#     and crosses E-F1 westbound on the east road).
#   - Perpendicular left turns -> both sweep the center, paths cross
#     (e.g. N-L vs W-L).
#   - Perpendicular through movements -> paths cross at the center
#     (e.g. N-F1 vs E-F1).
#   - Opposing left turns are ALLOWED: the intersection is assumed large
#     enough for the two lefts to turn side by side without crossing.
#   - Left turn vs perpendicular RIGHT turn is allowed when exits differ:
#     the right turn hugs the corner and clears before the sweep arrives.
#   - Everything else is compatible (e.g. N-F vs S-F).
OPPOSITE_DIRECTIONS = {"north": "south", "south": "north", "east": "west", "west": "east"}
PERPENDICULAR_DIRECTIONS = {
    "north": {"east", "west"},
    "south": {"east", "west"},
    "east": {"north", "south"},
    "west": {"north", "south"},
}


def _conflicts(first_lane: Lane, second_lane: Lane) -> bool:
    if first_lane.direction == second_lane.direction:
        return False  # same approach: parallel lanes, no conflict
    if first_lane.exit_direction == second_lane.exit_direction:
        return True  # merge into the same exit road
    if first_lane.direction == OPPOSITE_DIRECTIONS[second_lane.direction]:
        if {first_lane.movement, second_lane.movement} == {"left", "forward"}:
            return True  # left turn crosses opposing through traffic
        return False  # opposing lefts allowed; other opposing pairs fine
    # Perpendicular approaches:
    if first_lane.movement == "forward" and second_lane.movement == "forward":
        return True  # through movements cross at the center
    if first_lane.movement == "left" and second_lane.movement == "left":
        return True  # both sweep the center, paths cross
    # A left turn's sweep crosses the through lanes of the road it turns
    # into: conflict with the perpendicular approach's forward traffic.
    for turning, other in ((first_lane, second_lane), (second_lane, first_lane)):
        if (
            turning.movement == "left"
            and other.movement == "forward"
            and other.direction == turning.exit_direction
        ):
            return True
    return False


def _compatible(first_lane: Lane, second_lane: Lane) -> bool:
    if first_lane == second_lane:
        return True
    return not _conflicts(first_lane, second_lane)


MATRIX = [[_compatible(first_lane, second_lane) for second_lane in LANES] for first_lane in LANES]


def main() -> None:
    config = {
        "intersection": {
            "name": "default-4-way",
            "grid_size": 16,
        },
        "lanes": [lane.name for lane in LANES],
        "movements": {lane.name: lane.exit_direction for lane in LANES},
        # compatibility[i][j] == True  -> lanes i and j may be green together
        # compatibility[i][j] == False -> potential collision, never green together
        "compatibility": MATRIX,
    }
    out = Path(__file__).resolve().parent.parent / "config" / "intersections.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
