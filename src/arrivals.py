"""Poisson arrivals with a turn split, per lane.

Each lane is movement-pure: the queue for N-L only ever contains cars that
will turn left. The turn split (left/forward/right shares) is applied when
deciding how many cars arrive at each lane: a lane's effective rate is the
approach's base rate times its movement share, with forward split evenly
between the two forward lanes.
"""
from dataclasses import dataclass
from typing import Dict

import numpy as np

from generate_config import LANES, Lane

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


def lane_arrival_rates(base_rate: float, split: TurnSplit) -> Dict[str, float]:
    """Effective Poisson rate (cars/second) for each lane name."""
    return {
        lane.name: base_rate * split.share(lane.movement) for lane in LANES
    }


def draw_arrivals(
    rates: Dict[str, float], step_length: float, rng: np.random.Generator
) -> Dict[str, int]:
    """Draw the number of arriving cars per lane for one step."""
    return {
        lane_name: int(rng.poisson(lam * step_length))
        for lane_name, lam in rates.items()
    }