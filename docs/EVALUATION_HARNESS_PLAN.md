# Evaluation Harness Plan

This plan defines how to compare manual, baseline, and future ML controllers
fairly. It does not implement the harness yet.

## Goal

Given a controller and a traffic scenario, run repeatable simulations and
produce comparable metrics. The harness answers:

> Under the same arrivals, construction state, timing rules, and seeds, which
> controller moves traffic most safely and fairly?

It must make it difficult to accidentally give one controller easier traffic
than another.

## Core fairness rule

Every controller in a comparison receives the same scenario and the same
random seed. That means the same arrival stream is generated for each run.

A comparison unit is:

```text
(controller, scenario, seed, duration)
```

For example, if `fixed_cycle`, `actuated`, and `max_pressure` run the
`rush_hour_north` scenario with seed `42`, all three should see identical
Poisson arrival draws.

## Scope

### Inputs

- A scenario definition
- One controller name and optional controller settings
- A seed
- A run duration in simulated seconds

### Outputs

- One structured result per run
- Aggregate summary across seeds
- A Markdown-ready findings row for the README table
- Raw JSONL/CSV data for later inspection
- A graph suite for every evaluation command: side-by-side controller charts
  and per-controller charts, saved beside the raw results

### Not in this module

- Pygame rendering
- Training ML models
- Changing signal safety rules
- Picking a controller from the UI

The harness calls the existing simulator and controllers; it does not duplicate
their logic.

## Suggested module layout

```text
src/
  evaluation.py        # run_one(), run_many(), aggregation helpers
  scenarios.py         # named Scenario definitions and validation
  metrics.py           # extracting final / time-series metrics
config/
  scenarios.yaml       # named reproducible scenarios
tests/
  test_evaluation.py
  test_scenarios.py
results/               # gitignored generated run output
```

`results/` should be ignored by Git because it is generated experiment data.
A small checked-in example result is optional, but not required.

## Scenario model

A scenario is a data object, ideally a dataclass loaded from YAML. It should
contain only traffic/environment facts, never controller choices.

```yaml
name: rush_hour_north
duration_seconds: 3600
simulation:
  step_length: 1.0
  discharge_rate: 0.5
  min_green: 5.0
  yellow_duration: 2.0
  lane_capacity: null
arrival_rates:
  default: 0.2
approach_overrides:
  north: 0.6
lane_overrides:
  N-F1: 0.8
  N-F2: 0.8
closed_lanes: []
seeds: [0, 1, 2, 3, 4]
```

The scenario layer should support:

- A default arrival rate plus per-approach and per-lane overrides
- Per-approach overrides use the 20/60/20 split; direct lane overrides win
  when a scenario needs a specific movement pattern (for example, an event
  that causes most north-approach demand to turn left)
- `closed_lanes` for construction cases
- Optional finite `lane_capacity` for spillback cases
- Duration and timing rules: **1800 seconds (30 minutes)** for quick runs,
  **3600 seconds (60 minutes)** for full comparisons
- Five seeds for everyday comparisons and twenty for final claims

### First scenario set

Start with five named scenarios:

| Scenario | Purpose |
|---|---|
| `balanced` | Baseline: equal demand on all approaches. |
| `rush_hour_north` | High north forward demand; tests direction prioritization. |
| `heavy_left_turns` | Elevated left-turn demand; tests protected-turn handling. |
| `construction_north` | Close three north lanes and redirect demand to the remaining forward lane. |
| `near_capacity` | Finite queue capacity and high demand; tests spillback protection. |

## Run lifecycle

`run_one(controller_name, scenario, seed)` should:

1. Build a fresh `SimulationConfig` from the scenario and seed.
2. Create a new simulator and the named controller.
3. Run for the scenario duration, calling controller `act(state)` then
   simulator `step(lights)` once per step.
4. Capture metrics at every step or at a configurable sampling interval.
5. Return one immutable result record; do not print from the core runner.

Use a small wrapper CLI or script to print tables and save files. Keep the core
runner importable for tests.

## Metrics to record

### Required final metrics

| Metric | Definition | Why it matters |
|---|---|---|
| `throughput_total` | Cars that departed all lanes | Primary amount of traffic moved. |
| `throughput_per_exit` | Departures by exit direction | Detects controllers that favor one exit. |
| `arrived` | Accepted arrivals | Conservation check. |
| `queued_final` | Cars still queued at end | Shows unfinished backlog. |
| `avg_car_wait` | Mean wait of departed cars | Driver experience. |
| `max_car_wait` | Longest individual departed-car wait | Starvation indicator. |
| `max_wait_since_served` | Largest lane service gap | Detects ignored lanes. |
| `green_empty_ratio` | Empty-green time / total-green time | Wasted green time. |
| `blocked_arrivals` | Arrivals rejected due to finite capacity | Spillback failure indicator. |
| `phase_changes` | Requested/effective signal phase changes | Switching/clearance cost. |

### Fairness metrics

- Per-lane average car wait
- Per-lane final queue length
- Difference between highest and lowest served-lane throughput
- Number of lanes with no departures

Do not only report a global average. A controller can have a good average while
starving a low-volume lane.

### Derived metrics

Calculate only after the raw metrics exist:

```text
throughput_per_minute = throughput_total / duration_seconds * 60
queue_clearance_ratio = throughput_total / arrived
spillback_rate = blocked_arrivals / (arrived + blocked_arrivals)
```

Avoid dividing by zero; emit `0` or `null` consistently and document which.

### Overall score

Every metric still appears individually in results and charts. The harness
also calculates one configurable overall score to rank controllers.

1. Normalize each comparison metric across the controllers in the same
   scenario. For values where lower is better (waits, green waste, blocked
   arrivals), invert the normalized score. For throughput, higher is better.
2. Take a weighted average of the normalized metric scores.
3. Default every weight to `1.0` (equal importance).
4. Store the exact weights in every result summary so a later discussion can
   change priorities without obscuring which values were used.

For example, future weights may favor lower wait over higher throughput. The
score is a decision aid, not a replacement for the individual metrics.

## Aggregation across seeds

A single seed is not evidence because Poisson arrivals vary. For each
controller/scenario pair:

1. Run all configured seeds.
2. Report mean, minimum, maximum, and standard deviation for the primary
   metrics.
3. Keep individual seed results available so an outlier can be inspected.

The initial README findings table can use means:

| Controller | Scenario | Overall score | Avg car wait (s) | Max wait (s) | Throughput (cars/min) | Green waste (%) | Blocked arrivals |
|---|---:|---:|---:|---:|---:|---:|---:|

## Graphs and visual reports

The evaluation code must include plotting from the first implementation.
Use **pandas** to load/aggregate results and **seaborn** (with matplotlib) to
create image files. Plotting belongs in `src/plots.py` or `src/reporting.py`,
not inside the core simulation runner.

For every evaluation command, write an output folder such as:

```text
results/balanced-2026-10-12/
  runs.jsonl
  summary.csv
  summary.md
  charts/
    overall-score.png
    throughput-per-minute.png
    average-car-wait.png
    maximum-car-wait.png
    green-waste.png
    blocked-arrivals.png
    throughput-by-exit.png
    per-lane-average-wait.png
    per-lane-final-queue.png
    green-waste-share.png
```

### Required chart behavior

- If $x$ controllers are evaluated, each side-by-side bar chart has exactly
  $x$ bars per metric/scenario group, one per controller.
- Use controller name consistently for color and legend across every chart.
- Display seed-aggregate mean as bar height and standard deviation as error
  bars where more than one seed is used.
- Include the scenario name, duration, seed count, and score weights in the
  chart title or subtitle.
- Save PNG files; do not require a graphical display so CI/headless runs work.

### Required charts

| File | Chart | Purpose |
|---|---|---|
| `overall-score.png` | grouped bar | Quick overall ranking under configured equal/custom weights. |
| `throughput-per-minute.png` | grouped bar | Traffic moved per controller. |
| `average-car-wait.png` | grouped bar | Typical driver experience. |
| `maximum-car-wait.png` | grouped bar | Starvation risk. |
| `green-waste.png` | grouped bar | Wasted green time. |
| `blocked-arrivals.png` | grouped bar | Spillback/construction failure. |
| `throughput-by-exit.png` | grouped/stacked bar | Whether exits are favored unevenly. |
| `per-lane-average-wait.png` | heatmap or grouped bar | Identify specific starved movements. |
| `per-lane-final-queue.png` | heatmap or grouped bar | Show accumulated backlog by lane. |
| `green-waste-share.png` | pie chart | Percentage of total green time that was wasted, per controller. |

The pie chart is supplementary; the grouped bar chart remains the preferred
comparison for exact values.

### Plotting tests

- A tiny synthetic aggregate DataFrame creates every expected PNG filename.
- Each chart file is non-empty.
- A run with $x$ controller names creates an aggregate table containing $x$
  rows and chart data containing those same $x$ controller labels.
- The plotting layer handles zero blocked arrivals and one controller without
  division errors.

## Result format

Use JSON Lines for raw results: one self-contained JSON object per run. It is
simple to append, diff, parse in Python, and later load into pandas.

Suggested fields:

```json
{
  "controller": "max_pressure",
  "controller_settings": {"switch_ratio": 1.4},
  "scenario": "construction_north",
  "seed": 2,
  "duration_seconds": 3600,
  "metrics": {
    "throughput_total": 0,
    "throughput_per_minute": 0.0,
    "avg_car_wait": 0.0,
    "max_car_wait": 0.0,
    "green_empty_ratio": 0.0,
    "blocked_arrivals": 0
  }
}
```

Include a lightweight schema validator so a corrupt or incomplete result fails
immediately rather than silently appearing in a report.

## CLI shape

Keep commands small and explicit:

```bash
# one controller, one scenario, all its configured seeds
python -m src.evaluate --controller fixed_cycle --scenario balanced

# several controllers under identical conditions
python -m src.evaluate --controllers fixed_cycle,actuated,max_pressure --scenario construction_north

# write raw results and a summary table
python -m src.evaluate --controllers all --scenario balanced --output results/balanced.jsonl
```

The exact command/module name may change, but the implementation should
separate argument parsing from `run_one()` and aggregation functions.

## Testing plan

Tests are required before trusting controller comparisons.

### Determinism tests

1. Same controller + scenario + seed gives identical complete results.
2. Different seeds are permitted to differ.
3. Running controller A before B versus B before A does not change either
   result (no shared mutable random generator).

### Fair-arrival tests

4. Two different controllers with the same scenario and seed receive the
   same arrival count per lane when their signal choices do not affect
   arrivals.
5. Scenario overrides reach the intended lane rates.
6. Closed-lane demand remains conserved through redirection.

### Conservation tests

7. At each sampled point: `accepted_arrived == queued + throughput`.
8. With finite capacity: `attempted_arrivals == accepted_arrived + blocked_arrivals`.
9. All aggregate values equal the arithmetic aggregation of individual seed
   results.

### Metrics tests

10. A hand-built tiny run produces known throughput and wait values.
11. A permanently red lane has zero departures and increasing service-gap
    wait.
12. A permanently green empty lane has nonzero green-empty ratio.
13. A known lane at capacity records blocked arrivals.
14. No metric divides by zero or emits NaN.

### Controller integration tests

15. Every baseline controller can complete each initial scenario without a
    safety exception.
16. All returned light vectors remain matrix-safe through the run.
17. The construction scenario causes the max-pressure overflow policy to
    serve the remaining lane before its queue exceeds configured capacity,
    where possible.

### Regression snapshot test

18. Pick one tiny, fixed-seed, 60-second balanced scenario. Store expected
    results for `fixed_cycle` as a small assertion fixture. If a simulator or
    controller change alters it, the test forces an intentional review.

## Validated decisions

- **Duration:** quick evaluation is `1800s` (30 minutes); full comparison is
  `3600s` (60 minutes).
- **Seeds:** use 5 seeds for everyday comparisons and 20 for final claims.
- **Arrival overrides:** support both approach-level and direct lane-level
  overrides. Direct lane overrides win. This supports ordinary rush hour as
  well as event patterns where, for example, most cars turn left.
- **Construction redirection:** redistribute closed-lane demand evenly across
  the two open forward lanes. If only one forward lane is open, it receives
  all redirected demand. If neither forward lane is open, use an available
  turning lane as the fallback.
- **Capacity:** keep the current simple blocked-arrivals model for now. It is
  enough to identify an overloaded lane without introducing a separate
  upstream-road simulation.
- **Winner:** calculate a configurable equal-weight overall score by default,
  while retaining and graphing every individual metric. Weights can change
  later as priorities become clearer.
- **Phase changes:** record effective phase changes as a descriptive metric,
  but do not include them in the default overall score. Minimum green and
  yellow already prevent flicker/frustrating short cycles.

## Completion criteria

The harness is done when:

- all five initial scenarios load and validate;
- every baseline controller runs under each scenario and seed;
- raw per-run JSONL and a per-controller aggregate table are produced;
- determinism, arrival fairness, conservation, metric, and integration tests
  pass;
- a README findings-table row can be generated without manual arithmetic.
