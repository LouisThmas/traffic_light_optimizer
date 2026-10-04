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
# A vehicle entering from the north exits south (forward), west (left), east (right).
EXIT_DIRECTIONS = {
    ("north", "left"): "west", ("north", "forward"): "south", ("north", "right"): "east",
    ("south", "left"): "east", ("south", "forward"): "north", ("south", "right"): "west",
    ("east", "left"): "north", ("east", "forward"): "west", ("east", "right"): "south",
    ("west", "left"): "south", ("west", "forward"): "east", ("west", "right"): "north",
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
# Conflict rules (applied symmetrically):
#   - Same exit from different approaches -> merge conflict (streams converge
#     into the same exit road). Lanes from the SAME approach are parallel and
#     stay compatible (e.g. N-F1 vs N-F2).
#   - Left turn vs opposing through -> the left crosses oncoming traffic
#     (e.g. N-L vs S-F1).
#   - Perpendicular through movements -> paths cross at the center
#     (e.g. N-F1 vs E-F1).
#   - Opposing left turns are ALLOWED: the intersection is assumed large
#     enough for the two lefts to turn side by side without crossing.
#   - Everything else is compatible (e.g. N-F vs S-F, N-F vs S-R, N-L vs E-L).
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
    if (
        first_lane.movement == "forward"
        and second_lane.movement == "forward"
        and second_lane.direction in PERPENDICULAR_DIRECTIONS[first_lane.direction]
    ):
        return True  # perpendicular through movements cross
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
