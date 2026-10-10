"""Geometry checks for right-hand-traffic pygame lane placement."""
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pygame

from generate_config import LANES
from pygame_ui import SimulationPanel


def test_east_is_above_and_west_is_below_the_horizontal_median():
    pygame.init()
    panel = SimulationPanel(pygame.Rect(0, 0, 700, 700), seed=0)
    panel.intersection_rect = pygame.Rect(0, 0, 700, 700)
    center_y = panel.intersection_rect.centery
    east_lane = next(lane for lane in LANES if lane.name == "E-L")
    west_lane = next(lane for lane in LANES if lane.name == "W-L")

    _, east_stop, _ = panel.layout_for_lane(east_lane)
    _, west_stop, _ = panel.layout_for_lane(west_lane)

    assert east_stop.y < center_y
    assert west_stop.y > center_y
    pygame.quit()


def test_right_turn_path_has_correct_entry_and_exit_tangents():
    pygame.init()
    panel = SimulationPanel(pygame.Rect(0, 0, 700, 700), seed=0)
    panel.intersection_rect = pygame.Rect(0, 0, 700, 700)
    north_right = next(lane for lane in LANES if lane.name == "N-R")

    path = panel.movement_path(north_right)

    # N-R initially travels south, then ends travelling west toward its exit.
    assert path[1].y > path[0].y
    assert path[-1].x < path[-2].x
    pygame.quit()