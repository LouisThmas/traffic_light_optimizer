# Baseline Controllers

These controllers are the non-ML baselines. A future ML controller must beat
these under the same simulation seed, duration, and metrics.

## Safety contract

Every controller returns a complete 16-lane light vector. The simulator
validates it against `config/intersections.yaml`; an unsafe vector raises an
error. The fixed phases below are all **maximal**: no additional lane can be
lit without causing a conflict.

## Random

`RandomController` considers lanes in the configured lane order, randomly
tries to add each one, and accepts it only if it is compatible with every
already-selected lane.

- It is a reproducible floor: provide a seed.
- It never emits an unsafe state.
- It does not promise fairness or useful timing.

## Fixed cycle

`FixedCycleController` holds each phase for `phase_length` decisions
(default: 15). It then moves through this exact four-phase order and wraps to
phase 1. Every lane occurs in at least one phase.

This was saved from the exact matrix-cover exercise: it is the minimum number
of maximal-safe vectors that covers all 16 lanes. Four phases mean fewer green
to yellow to red transitions than the earlier six-phase hand-written cycle.

| Phase | Lit lanes | Visual reading |
|---|---|---|
| 1 | `N-R W-L W-F1 W-F2 W-R` | West movement group plus the north right turn. |
| 2 | `S-R E-L E-F1 E-F2 E-R` | East movement group plus the south right turn. |
| 3 | `N-L N-F1 N-F2 N-R E-R` | North movement group plus the east right turn. |
| 4 | `S-L S-F1 S-F2 S-R W-R` | South movement group plus the west right turn. |

The code defines this sequence as `FIXED_CYCLE_PHASES` in
`src/controllers.py`. Tests assert the order exactly, assert phase duration,
assert wrapping, assert all 16 lanes are served, and assert each phase is safe
and maximal.

## Saved matrix-search results

The one-time exact search reviewed all $2^{16} = 65,536$ lane subsets. The
five highest-scoring maximal-safe configurations were:

1. **6 lights:** `E-F1 E-F2 E-R W-F1 W-F2 W-R`
2. **6 lights:** `N-F1 N-F2 N-R S-F1 S-F2 S-R`
3. **5 lights:** `N-F1 N-F2 N-R S-R E-R`
4. **5 lights:** `N-L N-F1 N-F2 N-R E-R`
5. **5 lights:** `N-R E-R W-F1 W-F2 W-R`

The globally largest 6-light configurations alone cannot cover the left-turn
lanes, which is why the saved four-phase cover uses both 5- and 6-light
maximal configurations.

## Actuated

`ActuatedController(queue_threshold=20)` starts on fixed-cycle phase 1 and
selects among the same four matrix-optimized phases.

1. If any active lane has at least 20 queued cars, keep the current phase.
2. Otherwise, find the first phase (in fixed-cycle order) containing an
   inactive lane with at least 20 queued cars, and switch to it.
3. If two queues reach 20 simultaneously, phase order breaks the tie. Either
   could be a reasonable traffic policy; deterministic order makes runs
   reproducible.
4. If no lane reaches 20, keep the current phase.

## Max pressure and spillback protection

`MaxPressureController(switch_ratio=1.4)` computes each four-phase
configuration's normal pressure as:

$$
P(phase) = \sum_{lane \in phase} queue(lane)
$$

It chooses the highest-pressure phase, but only switches away from the
currently active phase when:

$$
P(best) \geq 1.4 \times P(current)
$$

That 40% hysteresis prevents phase flicker caused by a lane being only one car
more crowded than another. Ties retain the current phase.

A single remaining lane can be more urgent than a larger total on another
side, particularly during construction. `lane_capacity` and
`overflow_threshold` provide a per-lane override:

$$
queue(lane) \geq laneCapacity \times overflowThreshold
$$

With controller defaults (`lane_capacity=100`, `overflow_threshold=0.9`),
any lane at 90 cars immediately selects a phase that serves it, even when
another phase has a larger total pressure. This is the anti-spillback policy.

The simulator itself remains unlimited by default so existing baseline runs
are unchanged. Set `SimulationConfig(lane_capacity=...)` to impose a physical
queue limit; arrivals beyond it are recorded as `blocked_arrivals` instead of
being added to the queue.

## Construction lane closures

`SimulationConfig(closed_lanes=(...))` closes named lanes. Demand is not
lost: each closed lane's Poisson arrival rate is redirected to an open lane on
the same approach, preferring a forward lane. Closed lanes cannot receive a
green request. This enables construction and lane-reduction scenarios.

## Test coverage

Run the baseline-controller tests with:

```bash
python -m pytest tests/test_controllers.py -v
```

The tests cover seeded random reproducibility/safety; every fixed-cycle
invariant; threshold, hold, and tie behavior for actuated control; and
40%-boundary, anti-churn, and aggregate-pressure behavior for max pressure.