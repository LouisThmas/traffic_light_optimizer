"""Lane queue: FIFO cars with arrival timestamps for per-car wait tracking."""
from collections import deque
from typing import Optional, Tuple


class LaneQueue:
    """FIFO queue of cars for one lane.

    Each car is stored with the simulation time it arrived, so per-car wait
    (departure minus arrival) can be computed when it discharges.
    """

    def __init__(self) -> None:
        self._cars: deque = deque()

    def add(self, arrival_time: float, count: int = 1) -> None:
        for _ in range(count):
            self._cars.append(arrival_time)

    def pop(self) -> Optional[Tuple[float, float]]:
        """Remove the front car; return (arrival_time, wait) or None if empty."""
        if not self._cars:
            return None
        arrival_time = self._cars.popleft()
        return arrival_time, 0.0  # wait filled in by caller with departure time

    def __len__(self) -> int:
        return len(self._cars)

    @property
    def is_empty(self) -> bool:
        return not self._cars