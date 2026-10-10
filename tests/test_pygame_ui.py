"""Display-independent checks for pygame UI state and metric helpers."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

pytest.importorskip("pygame")

from pygame_ui import metric_values, percentage_delta
from simulator import Simulation, SimulationConfig


def test_metric_values_are_aggregated_from_public_state():
    sim = Simulation(SimulationConfig(arrival_rate=0.0))
    state = sim.state()
    metrics = metric_values(state)
    assert metrics["queued"] == 0
    assert metrics["throughput"] == 0
    assert metrics["avg_wait"] == 0.0
    assert metrics["green_waste"] == 0.0


def test_percentage_delta_handles_zero_baseline():
    assert percentage_delta(10, 0) == "—"
    assert percentage_delta(15, 10) == "+50%"
    assert percentage_delta(5, 10) == "-50%"