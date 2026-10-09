# Traffic Light Optimizer

Exploring how machine learning can optimize traffic flow through a
signalized intersection, compared against classic controllers.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Useful commands

```bash
# run all tests
python -m pytest tests/ -v

# regenerate the intersection config (lanes, movements, compatibility matrix)
python scripts/generate_config.py

# run the simulator demo (60s of alternating phases)
python src/simulator.py
```

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