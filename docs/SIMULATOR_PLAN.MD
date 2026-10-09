# Module 2: Simulator — Detailed Plan

This is the implementation plan for the traffic simulator. Nothing here is
implemented yet; validate the decisions first, then we build it.

## What the simulator does

One simulation step = one fixed slice of time (default 1 second). Each step:

1. **Arrivals** — new cars join lane queues (Poisson process).
2. **Discharge** — cars leave queues whose lane light is green.
3. **Bookkeeping** — wait times, throughput counters, controller state.

The simulator owns *state and physics only*. It never decides which lights
are green — a controller does that (module 4). The simulator just applies
whatever light state it is handed and refuses unsafe ones.

## Arrival process (Poisson)

- Each of the 16 lanes has its own independent Poisson arrival process with
  rate `lambda` (cars per second).
- Per step of length `dt`, arrivals for a lane are drawn as
  `Poisson(lambda * dt)` — with `dt = 1s` this is just `Poisson(lambda)`.
- Default `lambda = 0.2` cars/second/lane (one car every 5 seconds per lane),
  configurable per lane or per approach so we can build asymmetric scenarios
  (rush hour, one-direction heavy) later.
- Implementation: `numpy.random.poisson(lam, size=...)` per lane per step.
  All randomness flows through one seeded `numpy.random.Generator` so runs
  are reproducible.

## Turn split

When a car is created it is assigned a movement immediately, because each
lane *is* a movement (the queue for `N-L` only contains cars that will turn
left). The split is applied at arrival time:

| Movement | Share |
|----------|-------|
| Right    | 20%   |
| Left     | 20%   |
| Forward  | 60%   |

- Forward cars are split evenly between the two forward lanes (`N-F1` and
  `N-F2` get 30% each).
- The split is a config dial (`turn_split: {left: 0.2, forward: 0.6, right: 0.2}`),
  not a hardcoded constant.
- Two equivalent implementations; we use (a):
  - (a) Draw the *number* of arrivals per lane from Poisson, where each
    lane's effective rate is `lambda * share` (e.g. `N-L` gets
    `0.2 * 0.2 = 0.04` cars/s). Simple, and queue membership is implicit.
  - (b) Draw total arrivals per approach from Poisson, then route each car
    to a lane by sampling the split. Keeps per-car objects natural if we
    later want car IDs, routes, or travel-time traces.

## Queues

- One FIFO queue per lane (16 total). A queue is just a count plus a
  per-car arrival-time list (needed for wait-time accounting).
- Cars join the back on arrival and leave from the front when discharging.
- No lane capacity limit in v1 (queues can grow unboundedly); a capacity
  dial can be added later if we want spillback.

## Discharge (served cars per green step)

- When a lane is green, up to `discharge_rate * dt` cars leave its queue
  per step. Default `discharge_rate = 0.5` cars/second (one car every 2
  seconds — saturation flow for a single lane is ~0.5 veh/s, i.e. 1800
  veh/h).
- With `dt = 1s` this means a green lane serves at most 1 car every 2 steps.
  We track fractional accumulation so the rate is honored exactly over time
  (a `served_credit` float per lane: add `discharge_rate * dt` each green
  step, pop one car whenever credit >= 1).
- Discharge order across lanes within a step does not matter for the
  metrics (queues are independent), so we simply iterate lanes.

## Wait time per lane

The metric the model will eventually optimize. Per lane we track:

- `wait_time[lane]` — seconds since this lane last discharged a car, reset
  to 0 whenever a car leaves the queue. If the lane has been idle (red, or
  green but empty), this grows by `dt` each step. So "hasn't been activated
  for 5 seconds" reads directly as `wait_time == 5.0`.
- `total_wait[lane]` — cumulative sum of `wait_time` over the run, for
  averaging.
- `max_wait[lane]` — running maximum, to detect starvation.

Definition note: this is *time since last served*, not per-car waiting
time. It is cheap, monotone, and exactly the priority signal the plan
calls for. Per-car wait (arrival-to-departure) can be added later from the
per-car arrival timestamps we already keep in the queue.

## Light state and safety

- The controller hands the simulator a 16-bit light vector (one bool per
  lane) each step (or every k steps).
- Before applying it, the simulator checks the vector against the
  compatibility matrix from `config/intersections.yaml`. If any two green
  lanes conflict, the simulator **raises** — controllers are responsible
  for safety, and a violation is a bug, not something to silently repair.
  (The masked-action-space variant in module 8 will make violations
  impossible by construction; the strict check stays as a backstop.)
- Minimum green / yellow / all-red clearance: v1 has none — the controller
  can switch every step. This is listed under "things to watch" in the main
  plan; clearance time becomes a dial when we add it, because without it
  agents flicker lights.

## State the simulator exposes

A plain dict (later a dataclass) with everything a controller or the UI
needs:

```python
{
  "queues":        {lane_name: int},      # cars waiting per lane
  "wait_times":    {lane_name: float},    # seconds since last served
  "max_waits":     {lane_name: float},
  "lights":        {lane_name: bool},     # current green lanes
  "throughput":    {exit_direction: int}, # cars exited per edge
  "total_throughput": int,
  "arrived":       int,                   # total cars created
  "time":          float,                 # simulated seconds elapsed
}
```

## Config (one YAML, extends the existing one)

```yaml
simulation:
  step_length: 1.0          # seconds per step
  discharge_rate: 0.5       # cars per second per green lane
  arrival_rate: 0.2         # cars per second per lane (base)
  turn_split: {left: 0.2, forward: 0.6, right: 0.2}
  min_green: 5.0            # seconds a lane stays green once switched on
  yellow_duration: 2.0      # seconds of yellow after green ends
  seed: 0
```

Per-lane or per-approach arrival-rate overrides live under
`arrival_rates:` when scenarios need them.

## Module layout

```
src/
  simulator.py      # Simulation class: step(), state, counters
  arrivals.py       # Poisson arrival generation + turn split
  queue.py          # LaneQueue: count, pop, arrival timestamps
  lights.py         # light-vector validation against the matrix
config/
  intersections.yaml   # existing: lanes, movements, compatibility
  simulation.yaml      # new: rates, splits, seed
tests/
  test_simulator.py
```

## Tests (the verification the main plan asks for)

1. **Safety invariant** — run 100k steps with a random controller that
   picks a random valid configuration; assert no conflicting lights were
   ever green together (the simulator's check never fired).
2. **Conservation** — at every step, `arrived == sum(queues) + total_throughput`.
3. **Poisson statistics** — with a fixed seed and a large horizon, the
   empirical arrival count per lane is within a few standard deviations of
   `lambda * share * horizon`, and the turn split across an approach's
   arrivals matches 20/60/20 within tolerance.
4. **Wait-time semantics** — script a tiny scenario: lane never green for
   5 steps → `wait_since_served == 5.0`; discharge one car → resets to 0.
   Per-car wait: a car arriving at t=3 and departing at t=10 contributes
   7.0 to its lane's average.
5. **Discharge rate** — a permanently green lane with a long queue serves
   cars at `discharge_rate` (±1 car boundary effects).
6. **Reproducibility** — same seed, same controller → identical final
   state and metrics.
7. **Minimum green** — request a switch-off 1 step after green starts →
   the lane stays green until `min_green` elapses; `forced_green` reports
   it.
8. **Yellow** — after a switch-off, the lane is yellow (not green, no
   discharge) for `yellow_duration`, then red.
9. **Throughput counts all movements** — cars leaving left/right/forward
   lanes all appear in the exit-edge and total counters.
10. **Green-empty tracking** — a green lane with an empty queue
    accumulates `green_empty_time` while `green_time` grows.

## Validated decisions (resolved)

1. **Wait-time definition** — track BOTH: `wait_since_served` (seconds since
   the lane last discharged a car; cheap, monotone, the priority signal)
   AND **per-car wait** (departure time minus arrival time, averaged per
   lane and globally). Per-car wait is what drivers feel and is the
   headline metric; the queues already store arrival timestamps so it is
   free.
2. **Turn split at arrival** — confirmed: each lane queue is movement-pure;
   a car's movement is decided by which queue it joins at arrival.
3. **Discharge model** — fractional credit (smooth, exact rate).
4. **Unsafe light handling** — raise. Controllers own safety; masking
   hides bugs.
5. **Step length** — 1 second.

## Additional requirements (added during validation)

- **Throughput counts every movement.** Cars departing from left, right,
  and forward lanes all increment their exit-edge counter and the global
  total, so throughputs between systems compare directly.
- **Green-but-empty penalty metric.** Per lane, track `green_time` (seconds
  green) and `green_empty_time` (seconds green with no car available to
  discharge). The ratio `green_empty_time / green_time` is the waste the
  reward will penalize; the simulator only records it.
- **Minimum green: 5 seconds.** A lane that turns green cannot be switched
  off before `min_green` elapses. The simulator enforces this by keeping
  the lane green (the controller's "off" request is deferred, not an
  error). The state exposes `forced_green` so controllers know which
  lanes they cannot switch.
- **Yellow light: 2 seconds.** When a green lane is switched off (after its
  minimum green), it enters yellow for `yellow_duration`; no cars discharge
  during yellow, and the lane is not "green" for conflict purposes. After
  yellow it becomes red.