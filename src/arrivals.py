"""Poisson arrivals with a turn split, per lane.

Each lane is movement-pure: the queue for N-L only ever contains cars that
will turn left. The turn split (left/forward/right shares) is applied when
deciding how many cars arrive at each lane: a lane's effective rate is the
approach's base rate times its movement share, with forward split evenly
between the two forward lanes.
"""

from dataclasses import dataclass

import numpy as np

from generate_config import LANES

MOVEMENTS = ("left", "forward", "right")


@dataclass(frozen=True)
class TurnSplit:
    left: float = 0.2
    forward: float = 0.6
    right: float = 0.2

    def share(self, movement: str) -> float:
        """Fraction of an approach's traffic using this movement.

        Forward is split evenly between the two forward lanes (F1, F2),
        so each forward lane carries half the forward share.
        """
        shares = {"left": self.left, "forward": self.forward / 2, "right": self.right}
        return shares[movement]


def lane_arrival_rates(base_rate: float, split: TurnSplit, closed_lanes=frozenset()) -> dict[str, float]:
    """Effective Poisson rate for every open lane, redirecting closed traffic.

    Closing a lane does not delete demand. Its arrival rate is redirected to
    an open lane on the same approach, preferring a forward lane, then any
    other open lane. This makes construction scenarios conserve all demand.
    """
    rates = {lane.name: base_rate * split.share(lane.movement) for lane in LANES}
    for closed_lane in closed_lanes:
        redirected_rate = rates[closed_lane]
        rates[closed_lane] = 0.0
        direction = closed_lane[0]
        open_same_approach = [
            lane.name for lane in LANES if lane.name not in closed_lanes and lane.direction[0].upper() == direction
        ]
        forward_lanes = [name for name in open_same_approach if "-F" in name]
        recipient = (forward_lanes or open_same_approach)[0]
        rates[recipient] += redirected_rate
    return rates


def draw_arrivals(rates: dict[str, float], step_length: float, rng: np.random.Generator) -> dict[str, int]:
    """Draw the number of arriving cars per lane for one step."""
    return {lane_name: int(rng.poisson(lam * step_length)) for lane_name, lam in rates.items()}
