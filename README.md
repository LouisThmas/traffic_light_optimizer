# Traffic Light Optimizer

Exploring how machine learning can optimize traffic flow through a
signalized intersection, compared against classic controllers.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ruff is installed with the project dependencies
```

## Useful commands

```bash
# run all tests
python -m pytest tests/ -v

# lint (checks errors, imports, and modern Python syntax)
ruff check .

# format every Python file
ruff format .

# lint and apply safe automatic fixes, then format
ruff check . --fix && ruff format .

# regenerate the intersection config (lanes, movements, compatibility matrix)
python scripts/generate_config.py

# run the simulator demo (60s of alternating phases)
python src/simulator.py

# launch the interactive pygame view
python src/pygame_ui.py

# launch two lock-step simulations for controller comparison
python src/pygame_ui.py --compare

# evaluate one controller on one scenario (all its configured seeds)
python -m src.evaluate --controller fixed_cycle --scenario balanced

# compare several controllers under identical conditions
python -m src.evaluate --controllers fixed_cycle,actuated,max_pressure --scenario construction_north

# evaluate every baseline controller, with custom score weights
python -m src.evaluate --controllers all --scenario balanced --weights avg_car_wait=2.0

# write raw results to a specific JSONL file
python -m src.evaluate --controllers all --scenario near_capacity --output results/near_capacity.jsonl
```

## Pygame simulator

The interactive view renders the 16 lane signals, movement arrows, queue
counts, queue dots, and visual-only animated dots for recorded departures.
The animation does not affect the traffic simulation.

Select **Manual**, **Random**, **Fixed Cycle**, **Actuated**, or **Max
Pressure** above an intersection. In Manual mode, click a signal to request
it on or off. Signals held by the five-second minimum green are muted and
cannot be switched off; conflicting red signals are disabled. A valid off
request changes green to yellow before the signal becomes red. See
[baseline-controller details](docs/BASELINE_CONTROLLERS.md) for the exact
fixed-cycle order and controller policies.

Use the global buttons (or `Space`, Right Arrow, and `R`) to play/pause,
advance one simulation tick, change speed, and reset with the original seed.
`--compare` provides two independently controlled panels that advance
lock-step from identical arrival seeds, allowing fair controller comparison.

## Project layout

```
config/     intersection, simulation, and scenario settings (YAML)
scripts/    config generator
src/        simulator, controllers, and evaluation harness
tests/      pytest tests
docs/       plans and notes
results/    generated evaluation output (gitignored)
```

## Evaluation harness

Compare controllers fairly: every controller in a run receives the same
scenario and seed, so the same Poisson arrival stream is generated for each.
Each evaluation command writes raw per-run results (`runs.jsonl`), an
aggregate table (`summary.csv`, `summary.md`), and a chart suite
(`charts/*.png`) under `results/<scenario>-<date>/`.

## Findings

Comparison of controllers for discharging traffic. To be filled in as
experiments are run.

| Controller | Avg car wait (s) | Max wait (s) | Throughput (cars/min) | Green waste (%) |
|------------|------------------|--------------|-----------------------|-----------------|
| (none yet) |                  |              |                       |                 |