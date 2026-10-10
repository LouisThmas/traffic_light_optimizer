"""Interactive pygame view for the traffic-light simulator.

Run ``python src/pygame_ui.py`` for one intersection or add ``--compare``
for two lock-step simulations with identical arrival streams.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable

import pygame

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_HERE, os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from controllers import AlternatingController, ManualController, RandomController, all_red
from generate_config import LANES
from simulator import GREEN, RED, YELLOW, Simulation, SimulationConfig

BACKGROUND = (22, 27, 34)
PANEL = (35, 43, 53)
ROAD = (91, 98, 106)
ROAD_EDGE = (145, 151, 158)
TEXT = (235, 239, 242)
MUTED = (150, 160, 170)
RED_COLOR = (225, 78, 72)
YELLOW_COLOR = (239, 190, 65)
GREEN_COLOR = (74, 202, 124)
FORCED_GREEN = (100, 145, 113)
DISABLED = (76, 82, 90)
WHITE = (250, 250, 250)
BLUE = (72, 147, 220)

DIRECTIONS = ("north", "south", "east", "west")
SPEEDS = (0.5, 1.0, 2.0, 4.0)


def metric_values(state: Dict) -> Dict[str, float | int]:
    """Aggregate display metrics from the public simulation state."""
    departed = sum(state["departed"].values())
    total_wait = sum(state["total_car_wait"].values())
    green_time = sum(state["green_time"].values())
    empty_green = sum(state["green_empty_time"].values())
    return {
        "throughput": state["total_throughput"],
        "queued": sum(state["queues"].values()),
        "arrived": state["arrived"],
        "avg_wait": total_wait / departed if departed else 0.0,
        "max_wait": max(state["wait_since_served"].values(), default=0.0),
        "green_waste": 100 * empty_green / green_time if green_time else 0.0,
    }


def percentage_delta(value: float | int, other: float | int) -> str:
    """Format a signed percentage difference, avoiding a zero baseline."""
    if other == 0:
        return "—"
    return f"{(value - other) / other:+.0%}"


def direction_vector(direction: str) -> pygame.Vector2:
    return {
        "north": pygame.Vector2(0, 1),
        "south": pygame.Vector2(0, -1),
        "east": pygame.Vector2(-1, 0),
        "west": pygame.Vector2(1, 0),
    }[direction]


def turn_direction(direction: str, movement: str) -> str:
    """Return the authoritative exit direction for a visual movement."""
    return next(
        lane.exit_direction
        for lane in LANES
        if lane.direction == direction and lane.movement == movement
    )


@dataclass
class Vehicle:
    points: list[pygame.Vector2]
    progress: float = 0.0
    duration: float = 0.85

    def advance(self, seconds: float) -> bool:
        self.progress += seconds / self.duration
        return self.progress < 1.0

    def position(self) -> pygame.Vector2:
        """Interpolate smoothly across every segment in the visual path."""
        segment_count = len(self.points) - 1
        scaled = min(self.progress, 1.0) * segment_count
        segment = min(int(scaled), segment_count - 1)
        return self.points[segment].lerp(
            self.points[segment + 1], scaled - segment
        )


@dataclass
class SimulationPanel:
    rect: pygame.Rect
    seed: int
    controller_name: str = "manual"
    sim: Simulation = field(init=False)
    controller: object = field(init=False)
    signal_rects: Dict[str, pygame.Rect] = field(default_factory=dict)
    selector_rects: Dict[str, pygame.Rect] = field(default_factory=dict)
    vehicles: list[Vehicle] = field(default_factory=list)
    previous_departed: Dict[str, int] = field(default_factory=dict)
    feedback: str = ""
    intersection_rect: pygame.Rect | None = None

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sim = Simulation(SimulationConfig(seed=self.seed))
        self.set_controller(self.controller_name)
        self.previous_departed = {lane.name: 0 for lane in LANES}
        self.vehicles.clear()
        self.feedback = ""

    def set_controller(self, name: str) -> None:
        self.controller_name = name
        if name == "manual":
            self.controller = ManualController()
        elif name == "random":
            self.controller = RandomController(seed=self.seed)
        else:
            self.controller = AlternatingController()

    def step(self) -> None:
        state = self.sim.state()
        requested = self.controller.act(state)  # type: ignore[union-attr]
        self.sim.step(requested)
        new_state = self.sim.state()
        for lane in LANES:
            count = new_state["departed"][lane.name]
            new_departures = count - self.previous_departed[lane.name]
            for _ in range(max(new_departures, 0)):
                if len(self.vehicles) < 80:
                    self.vehicles.append(Vehicle(self.movement_path(lane)))
            self.previous_departed[lane.name] = count

    def update_animation(self, seconds: float) -> None:
        self.vehicles = [vehicle for vehicle in self.vehicles if vehicle.advance(seconds)]

    def layout_for_lane(self, lane) -> tuple[pygame.Vector2, pygame.Vector2, pygame.Vector2]:
        """Return queue, stop-line, and signal locations for one lane.

        Each approach has its own carriageway. Opposing traffic is separated
        by the centre median rather than sharing the same four lane strip.
        """
        rect = self.intersection_rect or self.rect
        center = pygame.Vector2(rect.centerx, rect.centery)
        scale = min(rect.width, rect.height) * 0.86
        lane_gap = max(24, scale * 0.052)
        median_half = max(10, scale * 0.025)
        stop_offset = scale * 0.205
        # In right-hand traffic, left-turn lanes touch the median and right
        # turns sit at the outside curb. The north/south positions are
        # mirrored so each incoming carriageway is on its correct road side.
        index = {"left": 0, "forward": lane.lane_number, "right": 3}[lane.movement]
        lateral = median_half + (index + 0.5) * lane_gap
        if lane.direction == "north":
            stop = center + pygame.Vector2(-lateral, -stop_offset)
        elif lane.direction == "south":
            stop = center + pygame.Vector2(lateral, stop_offset)
        elif lane.direction == "east":
            stop = center + pygame.Vector2(stop_offset, -lateral)
        else:
            stop = center + pygame.Vector2(-stop_offset, lateral)
        incoming = direction_vector(lane.direction)
        queue = stop - incoming * 28
        # Put each signal on the outside of its lane so the eight signal dots
        # around the junction remain visibly distinct.
        perpendicular = pygame.Vector2(-incoming.y, incoming.x)
        signal = stop - incoming * 11 + perpendicular * (lane_gap * 0.42)
        return queue, stop, signal

    def movement_path(self, lane) -> list[pygame.Vector2]:
        _, stop, _ = self.layout_for_lane(lane)
        rect = self.intersection_rect or self.rect
        center = pygame.Vector2(rect.centerx, rect.centery)
        exit_direction = lane.exit_direction
        outgoing = -direction_vector(exit_direction)
        scale = min(rect.width, rect.height)
        # Every exit uses the right-hand outbound carriageway, giving dots a
        # continuous route from their incoming lane onto an actual exit lane.
        exit_lane_offset = {
            "north": pygame.Vector2(scale * 0.105, 0),
            "south": pygame.Vector2(-scale * 0.105, 0),
            "east": pygame.Vector2(0, scale * 0.105),
            "west": pygame.Vector2(0, -scale * 0.105),
        }[exit_direction]
        end = center + outgoing * (scale * 0.43) + exit_lane_offset
        if lane.movement == "forward":
            return [stop, end]
        incoming = direction_vector(lane.direction)
        # A cubic Bézier turn has an incoming tangent that follows the lane
        # and an outgoing tangent that follows the exit lane. Sampling its
        # curve produces continuous movement instead of diagonal jumps.
        turn_length = scale * (0.25 if lane.movement == "left" else 0.16)
        return self._cubic_bezier(
            stop,
            stop + incoming * turn_length,
            end - outgoing * turn_length,
            end,
        )

    @staticmethod
    def _cubic_bezier(
        start: pygame.Vector2,
        first_control: pygame.Vector2,
        second_control: pygame.Vector2,
        end: pygame.Vector2,
        segments: int = 18,
    ) -> list[pygame.Vector2]:
        """Return evenly sampled points for a smooth vehicle turn."""
        points = []
        for index in range(segments + 1):
            t = index / segments
            inverse = 1 - t
            points.append(
                inverse ** 3 * start
                + 3 * inverse ** 2 * t * first_control
                + 3 * inverse * t ** 2 * second_control
                + t ** 3 * end
            )
        return points

    def can_activate(self, lane_name: str, state: Dict) -> bool:
        if self.controller_name != "manual":
            return False
        controller = self.controller
        assert isinstance(controller, ManualController)
        return controller.can_turn_on(lane_name, state)

    def click(self, position: tuple[int, int]) -> bool:
        for name, rect in self.selector_rects.items():
            if rect.collidepoint(position):
                self.set_controller(name)
                self.feedback = f"Controller: {name.title()}"
                return True
        if self.controller_name != "manual":
            return False
        state = self.sim.state()
        for lane_name, rect in self.signal_rects.items():
            if rect.collidepoint(position):
                controller = self.controller
                assert isinstance(controller, ManualController)
                result = controller.toggle(lane_name, state)
                self.feedback = "" if result.changed else result.reason
                return True
        return False


class TrafficApplication:
    """Owns pygame controls and one or two synchronized simulation panels."""

    def __init__(self, compare: bool) -> None:
        pygame.init()
        pygame.display.set_caption("Traffic Light Optimizer")
        self.compare = compare
        self.screen = pygame.display.set_mode((1500 if compare else 980, 850))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 16)
        self.small_font = pygame.font.SysFont("segoeui", 13)
        self.title_font = pygame.font.SysFont("segoeui", 22, bold=True)
        self.playing = True
        self.speed_index = 1
        self.accumulator = 0.0
        self.seed = 0
        self.control_rects: Dict[str, pygame.Rect] = {}
        panel_width = 715 if compare else 940
        panel_height = 760
        left = 20
        self.panels = [SimulationPanel(pygame.Rect(left, 72, panel_width, panel_height), self.seed)]
        if compare:
            self.panels.append(SimulationPanel(pygame.Rect(left + panel_width + 25, 72, panel_width, panel_height), self.seed))

    def reset(self) -> None:
        for panel in self.panels:
            panel.seed = self.seed
            panel.reset()

    def run(self) -> None:
        running = True
        while running:
            elapsed = min(self.clock.tick(60) / 1000, 0.15)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_RIGHT:
                        self.tick_once()
                    elif event.key == pygame.K_r:
                        self.reset()
            if self.playing:
                self.accumulator += elapsed * SPEEDS[self.speed_index]
                ticks = 0
                while self.accumulator >= 1.0 and ticks < 5:
                    self.tick_once()
                    self.accumulator -= 1.0
                    ticks += 1
            for panel in self.panels:
                panel.update_animation(elapsed)
            self.draw()
        pygame.quit()

    def tick_once(self) -> None:
        for panel in self.panels:
            panel.step()

    def handle_click(self, position: tuple[int, int]) -> None:
        for name, rect in self.control_rects.items():
            if rect.collidepoint(position):
                if name == "play":
                    self.playing = not self.playing
                elif name == "step":
                    self.tick_once()
                elif name == "speed":
                    self.speed_index = (self.speed_index + 1) % len(SPEEDS)
                elif name == "reset":
                    self.reset()
                return
        for panel in self.panels:
            if panel.rect.collidepoint(position) and panel.click(position):
                return

    def text(self, content: str, pos: tuple[float, float], color: tuple[int, int, int] = TEXT, *, small: bool = False) -> None:
        rendered = (self.small_font if small else self.font).render(content, True, color)
        self.screen.blit(rendered, pos)

    def button(self, label: str, rect: pygame.Rect, active: bool = False) -> None:
        pygame.draw.rect(self.screen, BLUE if active else PANEL, rect, border_radius=6)
        pygame.draw.rect(self.screen, (110, 125, 140), rect, 1, border_radius=6)
        text = self.small_font.render(label, True, TEXT)
        self.screen.blit(text, text.get_rect(center=rect.center))

    def draw(self) -> None:
        self.screen.fill(BACKGROUND)
        self.draw_top_bar()
        states = [panel.sim.state() for panel in self.panels]
        for index, panel in enumerate(self.panels):
            other = states[1 - index] if len(states) == 2 else None
            self.draw_panel(panel, states[index], other)
        pygame.display.flip()

    def draw_top_bar(self) -> None:
        self.screen.blit(self.title_font.render("Traffic Light Optimizer", True, TEXT), (20, 20))
        start = 325
        definitions = [
            ("play", "Pause" if self.playing else "Play"),
            ("step", "Step"),
            ("speed", f"Speed {SPEEDS[self.speed_index]:g}×"),
            ("reset", "Reset"),
        ]
        self.control_rects = {}
        for index, (name, label) in enumerate(definitions):
            rect = pygame.Rect(start + index * 116, 18, 106, 34)
            self.control_rects[name] = rect
            self.button(label, rect, name == "play" and self.playing)
        if self.compare:
            self.text("Space: play/pause   →: step   R: reset", (805, 29), MUTED, small=True)

    def draw_panel(self, panel: SimulationPanel, state: Dict, other: Dict | None) -> None:
        pygame.draw.rect(self.screen, PANEL, panel.rect, border_radius=10)
        self.draw_controller_selector(panel)
        intersection = pygame.Rect(panel.rect.x + 12, panel.rect.y + 48, int(panel.rect.width * 0.69), panel.rect.height - 60)
        stats = pygame.Rect(intersection.right + 8, intersection.y, panel.rect.right - intersection.right - 20, intersection.height)
        self.draw_intersection(panel, intersection, state)
        self.draw_stats(stats, panel, state, other)

    def draw_controller_selector(self, panel: SimulationPanel) -> None:
        self.text("Controller", (panel.rect.x + 14, panel.rect.y + 16), MUTED, small=True)
        panel.selector_rects = {}
        for index, name in enumerate(("manual", "random", "alternating")):
            rect = pygame.Rect(panel.rect.x + 92 + index * 92, panel.rect.y + 9, 86, 28)
            panel.selector_rects[name] = rect
            self.button(name.title(), rect, panel.controller_name == name)
        if panel.feedback:
            self.text(panel.feedback, (panel.rect.x + 380, panel.rect.y + 16), YELLOW_COLOR, small=True)

    def draw_intersection(self, panel: SimulationPanel, rect: pygame.Rect, state: Dict) -> None:
        panel.intersection_rect = rect
        center = pygame.Vector2(rect.centerx, rect.centery)
        size = min(rect.width, rect.height) * 0.86
        lane_gap = max(24, size * 0.052)
        median_width = max(20, size * 0.05)
        road_width = lane_gap * 8 + median_width
        vertical = pygame.Rect(int(center.x - road_width / 2), int(center.y - size / 2), int(road_width), int(size))
        horizontal = pygame.Rect(int(center.x - size / 2), int(center.y - road_width / 2), int(size), int(road_width))
        pygame.draw.rect(self.screen, ROAD, vertical)
        pygame.draw.rect(self.screen, ROAD, horizontal)
        pygame.draw.rect(self.screen, ROAD_EDGE, vertical, 2)
        pygame.draw.rect(self.screen, ROAD_EDGE, horizontal, 2)

        # Medians separate the approaches but deliberately stop at the open
        # junction. Nothing green is drawn through the conflict area.
        median_color = (61, 76, 65)
        junction_half = road_width / 2
        median_rects = (
            pygame.Rect(round(center.x - median_width / 2), vertical.top, round(median_width), round(center.y - junction_half - vertical.top)),
            pygame.Rect(round(center.x - median_width / 2), round(center.y + junction_half), round(median_width), round(vertical.bottom - (center.y + junction_half))),
            pygame.Rect(horizontal.left, round(center.y - median_width / 2), round(center.x - junction_half - horizontal.left), round(median_width)),
            pygame.Rect(round(center.x + junction_half), round(center.y - median_width / 2), round(horizontal.right - (center.x + junction_half)), round(median_width)),
        )
        for median in median_rects:
            pygame.draw.rect(self.screen, median_color, median)
            pygame.draw.rect(self.screen, (105, 125, 105), median, 1)
            for offset in range(18, median.height if median.height > median.width else median.width, 46):
                plant = (median.centerx, median.top + offset) if median.height > median.width else (median.left + offset, median.centery)
                pygame.draw.circle(self.screen, (81, 112, 77), plant, 4)

        # Dividers are drawn in each carriageway separately, keeping opposing
        # lanes visibly on opposite sides of the median.
        divider = (180, 184, 188)
        for index in range(5):
            positive = median_width / 2 + index * lane_gap
            negative = -positive
            for x in (center.x + positive, center.x + negative):
                pygame.draw.line(self.screen, divider, (x, vertical.top), (x, center.y - road_width / 2), 1)
                pygame.draw.line(self.screen, divider, (x, center.y + road_width / 2), (x, vertical.bottom), 1)
            for y in (center.y + positive, center.y + negative):
                pygame.draw.line(self.screen, divider, (horizontal.left, y), (center.x - road_width / 2, y), 1)
                pygame.draw.line(self.screen, divider, (center.x + road_width / 2, y), (horizontal.right, y), 1)

        panel.signal_rects = {}
        for lane in LANES:
            queue_pos, stop, signal_pos = panel.layout_for_lane(lane)
            signal_rect = pygame.Rect(0, 0, 18, 18)
            signal_rect.center = (round(signal_pos.x), round(signal_pos.y))
            panel.signal_rects[lane.name] = signal_rect.inflate(8, 8)
            signal = state["signals"][lane.name]
            forced = lane.name in state["forced_green"]
            if signal == GREEN:
                color = FORCED_GREEN if forced else GREEN_COLOR
            elif signal == YELLOW:
                color = YELLOW_COLOR
            elif panel.controller_name == "manual" and not panel.can_activate(lane.name, state):
                color = DISABLED
            else:
                color = RED_COLOR
            pygame.draw.circle(self.screen, color, signal_rect.center, 6)
            pygame.draw.circle(self.screen, TEXT if signal != RED else (40, 40, 40), signal_rect.center, 7, 1)
            self.draw_arrow(stop, lane.direction, lane.movement, color)
            self.draw_lane_label(rect, lane, queue_pos, state["queues"][lane.name])
            for index in range(min(state["queues"][lane.name], 5)):
                incoming = direction_vector(lane.direction)
                dot = queue_pos - incoming * index * 8
                pygame.draw.circle(self.screen, WHITE, (round(dot.x), round(dot.y)), 3)

        # Exit-direction totals use the authoritative simulator result.
        exits = {"north": (center.x - 22, rect.top + 8), "south": (center.x - 22, rect.bottom - 22), "east": (rect.right - 54, center.y - 8), "west": (rect.left + 8, center.y - 8)}
        for direction, position in exits.items():
            self.text(f"{direction[:1].upper()}: {state['throughput'].get(direction, 0)}", position, MUTED, small=True)
        for vehicle in panel.vehicles:
            point = vehicle.position()
            pygame.draw.circle(self.screen, WHITE, (round(point.x), round(point.y)), 5)

    def draw_lane_label(self, rect: pygame.Rect, lane, queue_pos: pygame.Vector2, queue_count: int) -> None:
        """Place centered lane labels and counts outside their queue dots."""
        if lane.direction == "north":
            anchor = (queue_pos.x, rect.top + 10)
        elif lane.direction == "south":
            anchor = (queue_pos.x, rect.bottom - 26)
        elif lane.direction == "east":
            anchor = (rect.right - 21, queue_pos.y - 8)
        else:
            anchor = (rect.left + 21, queue_pos.y - 8)
        label = self.small_font.render(lane.name, True, TEXT)
        count = self.small_font.render(str(queue_count), True, WHITE)
        label_rect = label.get_rect(center=(round(anchor[0]), round(anchor[1])))
        count_rect = count.get_rect(center=(round(anchor[0]), round(anchor[1] + 14)))
        self.screen.blit(label, label_rect)
        self.screen.blit(count, count_rect)

    def draw_arrow(self, start: pygame.Vector2, direction: str, movement: str, color: tuple[int, int, int]) -> None:
        incoming = direction_vector(direction)
        if movement == "forward":
            end = start + incoming * 18
            pygame.draw.line(self.screen, color, start, end, 2)
            self.draw_arrow_head(end, incoming, color)
            return
        exit_vector = -direction_vector(turn_direction(direction, movement))
        middle = start + incoming * 8
        end = middle + exit_vector * 11
        pygame.draw.lines(self.screen, color, False, [start, middle, end], 2)
        self.draw_arrow_head(end, exit_vector, color)

    def draw_arrow_head(self, end: pygame.Vector2, direction: pygame.Vector2, color: tuple[int, int, int]) -> None:
        perpendicular = pygame.Vector2(-direction.y, direction.x)
        pygame.draw.polygon(self.screen, color, [end, end - direction * 6 + perpendicular * 4, end - direction * 6 - perpendicular * 4])

    def draw_stats(self, rect: pygame.Rect, panel: SimulationPanel, state: Dict, other: Dict | None) -> None:
        self.text("Statistics", (rect.x, rect.y), TEXT)
        self.text(f"Time  {state['time']:.0f}s", (rect.x, rect.y + 27), MUTED, small=True)
        metrics = metric_values(state)
        other_metrics = metric_values(other) if other else None
        labels = [
            ("throughput", "Throughput", "{:.0f}"),
            ("queued", "Queued", "{:.0f}"),
            ("arrived", "Arrived", "{:.0f}"),
            ("avg_wait", "Avg wait", "{:.1f}s"),
            ("max_wait", "Max wait", "{:.1f}s"),
            ("green_waste", "Green waste", "{:.1f}%"),
        ]
        for index, (key, label, fmt) in enumerate(labels):
            y = rect.y + 58 + index * 47
            self.text(label, (rect.x, y), MUTED, small=True)
            self.text(fmt.format(metrics[key]), (rect.x, y + 17), TEXT, small=True)
            if other_metrics is not None:
                self.text(percentage_delta(metrics[key], other_metrics[key]), (rect.right - 42, y + 17), MUTED, small=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pygame traffic-light simulator")
    parser.add_argument("--compare", action="store_true", help="show two synchronized intersections")
    return parser.parse_args()


def main() -> None:
    TrafficApplication(parse_args().compare).run()


if __name__ == "__main__":
    main()