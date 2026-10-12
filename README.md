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

# run only the exhaustive baseline-controller tests
python -m pytest tests/test_controllers.py -v

# inspect the saved matrix-optimized four-phase cycle
python -m pytest tests/test_controllers.py -v
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
config/     intersection + simulation settings (YAML)
scripts/    config generator
src/        simulator code
tests/      pytest tests
docs/       plans and notes
```

## Findings

Comparison of controllers for discharging traffic. To be filled in as
experiments are run.

| Controller | Avg car wait (s) | Max wait (s) | Throughput (cars/min) | Green waste (%) |
|------------|------------------|--------------|-----------------------|-----------------|
| (none yet) |                  |              |                       |                 |