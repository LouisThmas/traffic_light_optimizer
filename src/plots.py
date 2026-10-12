"""Charts for evaluation results: side-by-side controller comparisons.

Uses pandas to load/aggregate results and seaborn (with matplotlib) to create
PNG image files. Plotting lives here, not inside the core simulation runner,
and requires no graphical display (Agg backend) so CI/headless runs work.

Required chart behavior:
- exactly one bar per controller per metric/scenario group;
- controller name used consistently for color and legend;
- seed-aggregate mean as bar height, standard deviation as error bars;
- scenario name, duration, seed count, and score weights in the subtitle;
- PNG output, headless-safe.
"""

import os
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_HERE,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

sns.set_theme(style="whitegrid")

# Chart file name -> (aggregate metric key, human title).
BAR_CHARTS = {
    "overall-score": ("overall_score", "Overall score"),
    "throughput-per-minute": ("throughput_per_minute", "Throughput (cars/min)"),
    "average-car-wait": ("avg_car_wait", "Average car wait (s)"),
    "maximum-car-wait": ("max_car_wait", "Maximum car wait (s)"),
    "green-waste": ("green_empty_ratio", "Green waste (empty-green ratio)"),
    "blocked-arrivals": ("blocked_arrivals", "Blocked arrivals"),
}


def _controller_palette(controllers: list[str]) -> dict[str, tuple]:
    palette = sns.color_palette("colorblind", n_colors=max(len(controllers), 1))
    return {name: palette[index] for index, name in enumerate(controllers)}


def _subtitle(scenario_name: str, duration_seconds: float, seed_count: int, weights: Optional[dict]) -> str:
    weight_text = "equal weights" if not weights else ", ".join(f"{k}={v:g}" for k, v in sorted(weights.items()))
    return (
        f"scenario: {scenario_name} | duration: {duration_seconds:g}s | seeds: {seed_count}"
        f" | score weights: {weight_text}"
    )


def _grouped_bar(
    data: pd.DataFrame,
    metric: str,
    title: str,
    subtitle: str,
    path: str,
    palette: dict[str, tuple],
    ylabel: Optional[str] = None,
) -> None:
    """One bar per controller, mean height, std error bars."""
    controllers = list(data["controller"])
    means = data[metric].apply(lambda cell: cell["mean"] if isinstance(cell, dict) else float(cell)).astype(float)
    stds = data[metric].apply(lambda cell: cell["std"] if isinstance(cell, dict) else 0.0).astype(float)

    fig, axis = plt.subplots(figsize=(min(8, 3.0 + 1.5 * len(controllers)), 5))
    colors = [palette.get(name, "gray") for name in controllers]
    bars = axis.bar(controllers, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)
    axis.set_title(f"{title}\n{subtitle}", fontsize=9, wrap=True)
    axis.set_ylabel(ylabel or title)
    axis.set_xlabel("Controller")
    for bar, mean in zip(bars, means):
        axis.annotate(
            f"{mean:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _throughput_by_exit(records: list[dict], subtitle: str, path: str, palette: dict[str, tuple]) -> None:
    """Grouped bar of departures by exit direction, one group per exit."""
    rows = []
    for record in records:
        for exit_direction, count in record["metrics"]["throughput_per_exit"].items():
            rows.append({"controller": record["controller"], "exit": exit_direction, "cars": count})
    frame = pd.DataFrame(rows)
    fig, axis = plt.subplots(figsize=(9, 5))
    if frame.empty:
        axis.text(0.5, 0.5, "no departures recorded", ha="center", va="center")
        axis.set_axis_off()
    else:
        summary = frame.groupby(["exit", "controller"], as_index=False)["cars"].mean()
        exits = sorted(summary["exit"].unique())
        controllers = sorted(summary["controller"].unique())
        width = 0.8 / max(len(controllers), 1)
        for index, controller in enumerate(controllers):
            subset = summary[summary["controller"] == controller].set_index("exit")["cars"]
            values = [subset.get(exit_direction, 0.0) for exit_direction in exits]
            positions = [i + index * width for i in range(len(exits))]
            axis.bar(
                positions,
                values,
                width=width,
                label=controller,
                color=palette.get(controller, "gray"),
                edgecolor="black",
                linewidth=0.5,
            )
        axis.set_xticks([i + width * (len(controllers) - 1) / 2 for i in range(len(exits))])
        axis.set_xticklabels(exits)
        axis.legend(title="Controller")
    axis.set_title(f"Throughput by exit direction\n{subtitle}", fontsize=10)
    axis.set_ylabel("Mean cars departed")
    axis.set_xlabel("Exit direction")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _per_lane_chart(
    records: list[dict],
    metric_key: str,
    title: str,
    subtitle: str,
    path: str,
) -> None:
    """Heatmap of a per-lane metric: controllers as rows, lanes as columns."""
    rows = []
    for record in records:
        for lane_name, value in record["metrics"][metric_key].items():
            rows.append({"controller": record["controller"], "lane": lane_name, "value": value})
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["controller", "lane", "value"])
    pivot = frame.pivot_table(index="controller", columns="lane", values="value", aggfunc="mean")
    pivot = pivot.reindex(sorted(pivot.index))
    fig, axis = plt.subplots(figsize=(12, 4 + 0.4 * len(pivot.index)))
    # Light blue (low) to dark blue (high): darker always means more, like a
    # population-density map. A single-hue ramp keeps the reading intuitive.
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Blues", ax=axis, cbar_kws={"label": title})
    axis.set_title(f"{title}\n{subtitle}", fontsize=10)
    axis.set_xlabel("Lane")
    axis.set_ylabel("Controller")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_charts(
    records: list[dict],
    aggregate_rows: list[dict],
    output_dir: str,
    weights: Optional[dict] = None,
) -> list[str]:
    """Write every required chart PNG into output_dir/charts and return paths."""
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    controllers = sorted({row["controller"] for row in aggregate_rows})
    palette = _controller_palette(controllers)
    scenario_name = aggregate_rows[0]["scenario"] if aggregate_rows else "unknown"
    duration = aggregate_rows[0].get("duration_seconds", 0) if aggregate_rows else 0
    seed_count = aggregate_rows[0]["seed_count"] if aggregate_rows else 0
    subtitle = _subtitle(scenario_name, duration, seed_count, weights)

    written: list[str] = []
    data = pd.DataFrame(aggregate_rows)
    for chart_name, (metric, title) in BAR_CHARTS.items():
        path = os.path.join(charts_dir, f"{chart_name}.png")
        _grouped_bar(data, metric, title, subtitle, path, palette)
        written.append(path)

    path = os.path.join(charts_dir, "throughput-by-exit.png")
    _throughput_by_exit(records, subtitle, path, palette)
    written.append(path)

    path = os.path.join(charts_dir, "per-lane-average-wait.png")
    _per_lane_chart(records, "per_lane_avg_wait", "Per-lane average car wait (s)", subtitle, path)
    written.append(path)

    path = os.path.join(charts_dir, "per-lane-final-queue.png")
    _per_lane_chart(records, "per_lane_final_queue", "Per-lane final queue length", subtitle, path)
    written.append(path)

    return written
