"""Controllers: decide which lanes are green each step.

A controller receives the simulation state and returns a light vector
(lane name -> bool). The simulator validates safety, so a controller can
never create a conflicting green.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from generate_config import LANES, MATRIX

# The exact four-vector, safe matrix cover. Every lane appears at least once.
# It replaces the earlier six-phase hand-written cycle because fewer phase
# changes reduce clearance loss while retaining full movement coverage.
FIXED_CYCLE_PHASES = (
    ("N-R", "W-L", "W-F1", "W-F2", "W-R"),
    ("S-R", "E-L", "E-F1", "E-F2", "E-R"),
    ("N-L", "N-F1", "N-F2", "N-R", "E-R"),
    ("S-L", "S-F1", "S-F2", "S-R", "W-R"),
)

# Publicly useful for tests and future controller implementations.
PHASES = FIXED_CYCLE_PHASES


def all_red() -> dict[str, bool]:
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
        self.desired: dict[str, bool] = all_red()

    def _is_allowed(self, lane_name: str, state: Optional[dict] = None) -> bool:
        """True if turning this lane green conflicts with nothing already green."""
        from lights import LANE_INDEX

        actual_green = set()
        if state is not None:
            actual_green = {other for other, signal in state.get("signals", {}).items() if signal == "green"}

        return all(
            (not self.desired[other] and other not in actual_green) or MATRIX[LANE_INDEX[lane_name]][LANE_INDEX[other]]
            for other in self.desired
        )

    def can_turn_on(self, lane_name: str, state: Optional[dict] = None) -> bool:
        """Whether a red lane may safely be requested on right now."""
        return not self.desired[lane_name] and self._is_allowed(lane_name, state)

    def toggle(self, lane_name: str, state: Optional[dict] = None) -> ToggleResult:
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

    def set(self, lights: dict[str, bool]) -> None:
        self.desired = dict(lights)

    def act(self, state: dict) -> dict[str, bool]:
        return dict(self.desired)


def lights_for_phase(phase: tuple) -> dict[str, bool]:
    """Build a complete light vector for a fixed-cycle phase."""
    lights = all_red()
    for lane_name in phase:
        lights[lane_name] = True
    return lights


class FixedCycleController:
    """Serve the four matrix-optimized phases in this exact order.

    Each phase lasts ``phase_length`` controller decisions. The order is:

    1. N-R, W-L, W-F1, W-F2, W-R
    2. S-R, E-L, E-F1, E-F2, E-R
    3. N-L, N-F1, N-F2, N-R, E-R
    4. S-L, S-F1, S-F2, S-R, W-R

    This is the saved output of the exact matrix-cover exercise: it covers
    every lane in four vectors (rather than the old six), so it needs fewer
    phase transitions and less yellow-clearance time. Every phase is maximal:
    no additional lane can be enabled without conflicting with a lit lane.
    """

    def __init__(self, phase_length: int = 15) -> None:
        if phase_length < 1:
            raise ValueError("phase_length must be at least 1")
        self.phase_length = phase_length
        self.step_count = 0

    def current_phase_index(self) -> int:
        return (self.step_count // self.phase_length) % len(FIXED_CYCLE_PHASES)

    def act(self, state: dict) -> dict[str, bool]:
        phase = FIXED_CYCLE_PHASES[self.current_phase_index()]
        self.step_count += 1
        return lights_for_phase(phase)


# Kept as an alias so existing UI selections continue to work.
AlternatingController = FixedCycleController


class RandomController:
    """Picks a random safe light vector each step (the floor to beat)."""

    def __init__(self, seed: int = 0, green_probability: float = 0.3) -> None:
        self.rng = np.random.default_rng(seed)
        self.green_probability = green_probability

    def act(self, state: dict) -> dict[str, bool]:
        from lights import LANE_INDEX, validate_lights

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


class ActuatedController:
    """Keep the active phase unless another phase has a queue at threshold.

    When any inactive lane reaches ``queue_threshold`` cars, select the
    first fixed-cycle phase containing one of the threshold-reaching lanes.
    Ties are intentionally deterministic (phase order); either winner is
    valid in the traffic-policy sense. An active phase with a threshold-
    reaching lane never switches away merely because another lane does too.
    """

    def __init__(self, queue_threshold: int = 20) -> None:
        if queue_threshold < 1:
            raise ValueError("queue_threshold must be at least 1")
        self.queue_threshold = queue_threshold
        self.phase_index = 0

    def act(self, state: dict) -> dict[str, bool]:
        queues = state.get("queues", {})
        active = FIXED_CYCLE_PHASES[self.phase_index]
        if not any(queues.get(lane, 0) >= self.queue_threshold for lane in active):
            for index, phase in enumerate(FIXED_CYCLE_PHASES):
                if any(queues.get(lane, 0) >= self.queue_threshold for lane in phase):
                    self.phase_index = index
                    break
        return lights_for_phase(FIXED_CYCLE_PHASES[self.phase_index])


# ---------------------------------------------------------------------------
# Canonical maximal-configuration table (17 configs from the compatibility matrix)
# ---------------------------------------------------------------------------


def _build_maximal_safety_configs() -> list[dict[str, bool]]:
    """Enumerate all maximal safe configurations from the compatibility matrix.

    A configuration is maximal when no additional lane can be safely added.
    Returns a sorted list (by 16-lane bit vector) of configurations, each a
    dict mapping lane name to bool.  Exactly 17 configurations are returned.
    """
    all_lane_names = [lane.name for lane in LANES]
    n = len(all_lane_names)  # 16

    # Enumerate all 2^16 subsets; keep only maximal cliques in the
    # compatibility graph. With 16 lanes this is tractable (65536 subsets).
    maximal_configs: list[set] = []

    for mask in range(1, 1 << n):
        config_lanes = {i for i in range(n) if mask & (1 << i)}
        # Check: all pairs in the config must be compatible
        is_safe = all(MATRIX[i][j] for i in config_lanes for j in config_lanes if i < j)
        if not is_safe:
            continue
        # Check: no outside lane is compatible with all lanes in the config
        is_maximal = all(
            any(not MATRIX[outside][inside] for inside in config_lanes)
            for outside in range(n)
            if outside not in config_lanes
        )
        if is_maximal:
            maximal_configs.append(set(config_lanes))

    # Convert to light vectors and sort by bit vector for determinism
    configs: list[dict[str, bool]] = []
    for config_lanes in maximal_configs:
        lights = {all_lane_names[i]: (i in config_lanes) for i in range(n)}
        configs.append(lights)

    def _bit_key(cfg: dict[str, bool]) -> int:
        key = 0
        for i, lane in enumerate(all_lane_names):
            if cfg[lane]:
                key |= 1 << i
        return key

    configs.sort(key=_bit_key)
    return configs


# The teacher's action table: 17 maximal safe configurations
MAX_PRESSURE_CONFIGS: list[dict[str, bool]] = _build_maximal_safety_configs()


class MaxPressureController:
    """Choose the busiest maximal-safe configuration with overflow protection.

    Scores every one of the 17 maximal configurations by total queued cars
    in the lit lanes. The controller changes configuration only when the
    best is at least ``switch_ratio`` (default 40%) above the current one,
    avoiding flicker.

    ``overflow_threshold`` adds a per-lane safety valve. If any lane reaches
    that fraction of ``lane_capacity`` (default 90%), its containing config
    wins immediately, even when the total queue on another config is larger.
    This models a construction diversion where one remaining lane must be
    drained before its queue spills into upstream traffic.
    """

    def __init__(
        self,
        switch_ratio: float = 1.4,
        lane_capacity: int = 100,
        overflow_threshold: float = 0.9,
    ) -> None:
        if switch_ratio < 1.0:
            raise ValueError("switch_ratio must be at least 1.0")
        if lane_capacity < 1:
            raise ValueError("lane_capacity must be at least 1")
        if not 0.0 < overflow_threshold <= 1.0:
            raise ValueError("overflow_threshold must be between 0 and 1")
        self.switch_ratio = switch_ratio
        self.lane_capacity = lane_capacity
        self.overflow_threshold = overflow_threshold
        self.config_index = 0  # current configuration ID (0..16)

    @property
    def overflow_queue_limit(self) -> int:
        """Cars at which a single lane gets emergency priority."""
        return int(self.lane_capacity * self.overflow_threshold)

    def act(self, state: dict) -> dict[str, bool]:
        queues = state.get("queues", {})
        overflow_lanes = {lane for lane, queue in queues.items() if queue >= self.overflow_queue_limit}

        if overflow_lanes:
            # Emergency: pick the config with the greatest urgent-lane queue
            candidates = [
                idx
                for idx, cfg in enumerate(MAX_PRESSURE_CONFIGS)
                if any(lane in cfg and cfg[lane] for lane in overflow_lanes)
            ]
            self.config_index = max(
                candidates,
                key=lambda idx: (
                    max(
                        (queues.get(lane, 0) for lane, on in MAX_PRESSURE_CONFIGS[idx].items()
                         if on and lane in overflow_lanes),
                        default=0,
                    ),
                    -idx,
                ),
            )
            return dict(MAX_PRESSURE_CONFIGS[self.config_index])

        pressures = [sum(queues.get(lane, 0) for lane, on in cfg.items() if on) for cfg in MAX_PRESSURE_CONFIGS]
        best_index = max(range(len(pressures)), key=lambda idx: pressures[idx])
        current_pressure = pressures[self.config_index]
        best_pressure = pressures[best_index]
        if best_index != self.config_index and (
            current_pressure == 0 or best_pressure >= current_pressure * self.switch_ratio
        ):
            self.config_index = best_index
        return dict(MAX_PRESSURE_CONFIGS[self.config_index])


CONTROLLERS = {
    "manual": ManualController,
    "fixed_cycle": FixedCycleController,
    "alternating": FixedCycleController,
    "random": RandomController,
    "actuated": ActuatedController,
    "max_pressure": MaxPressureController,
}
