"""CLI for the evaluation harness.

Commands stay small and explicit; argument parsing is separated from
``run_one()`` and the aggregation helpers in ``src/evaluation.py``.

Examples:
    python -m src.evaluate --controller fixed_cycle --scenario balanced
    python -m src.evaluate --controllers fixed_cycle,actuated,max_pressure --scenario construction_north
    python -m src.evaluate --controllers all --scenario balanced --output results/balanced.jsonl
"""

import argparse
import os
import sys
import time
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_HERE,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from evaluation import (
    aggregate_with_scores,
    findings_table,
    load_available_controllers,
    run_many,
    validate_result,
    write_jsonl,
)
from evaluation import load_all_scenarios as load_all_scenarios
from plots import generate_charts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluate",
        description="Run controllers under identical scenario conditions and report comparable metrics.",
    )
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--controller", help="a single controller name")
    targets.add_argument(
        "--controllers",
        help="comma-separated controller names, or 'all' for every baseline controller",
    )
    parser.add_argument("--scenario", required=True, help="named scenario from config/scenarios.yaml")
    parser.add_argument("--seeds", help="comma-separated seeds overriding the scenario's configured seeds")
    parser.add_argument("--output", help="path for raw JSONL results (default: results/<scenario>-<date>/runs.jsonl)")
    parser.add_argument("--weights", help="overall-score weights as metric=value pairs, e.g. avg_car_wait=2.0")
    parser.add_argument("--no-charts", action="store_true", help="skip chart generation")
    return parser


def parse_weights(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    weights = {}
    for pair in raw.split(","):
        name, _, value = pair.partition("=")
        weights[name.strip()] = float(value)
    return weights


def resolve_controllers(raw: Optional[str]) -> list[str]:
    if raw is None:
        return []
    if raw.strip().lower() == "all":
        return load_available_controllers()
    names = [name.strip() for name in raw.split(",") if name.strip()]
    available = load_available_controllers()
    for name in names:
        if name not in available:
            raise SystemExit(f"unknown controller {name!r}; available: {available}")
    return names


def default_output_dir(scenario_name: str) -> str:
    # Unix timestamp keeps every run distinct, even two runs of the same
    # scenario on the same day with different controllers or weights.
    stamp = int(time.time())
    return os.path.join(_ROOT, "results", f"{scenario_name}-{stamp}")


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    controllers = resolve_controllers(args.controllers or args.controller)
    scenario = load_all_scenarios()[args.scenario]
    if args.seeds:
        scenario = scenario.with_seeds([int(seed) for seed in args.seeds.split(",")])

    weights = parse_weights(args.weights)
    records = run_many(controllers, scenario)
    for record in records:
        validate_result(record.to_dict())

    rows = aggregate_with_scores(records, weights)

    output_dir = args.output and os.path.dirname(args.output) or default_output_dir(scenario.name)
    jsonl_path = args.output or os.path.join(output_dir, "runs.jsonl")
    if not os.path.isabs(jsonl_path):
        jsonl_path = os.path.join(_ROOT, jsonl_path)
    output_dir = os.path.dirname(jsonl_path)
    os.makedirs(output_dir, exist_ok=True)

    write_jsonl(records, jsonl_path)

    import pandas as pd

    summary_frame = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "summary.csv")
    summary_frame.to_csv(summary_path, index=False)

    header = (
        f"# Evaluation: {scenario.name}\n\n"
        f"- duration: {scenario.duration_seconds:g}s\n"
        f"- seeds: {list(scenario.seeds)}\n"
        f"- score weights: {weights or 'equal (all 1.0)'}\n\n"
    )
    summary_md = header + findings_table(rows) + "\n"
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write(summary_md)

    chart_paths: list[str] = []
    if not args.no_charts:
        chart_paths = generate_charts(
            [record.to_dict() for record in records],
            rows,
            output_dir,
            weights,
        )

    print(summary_md)
    print(f"Raw results: {os.path.relpath(jsonl_path, _ROOT)}")
    print(f"Summary:     {os.path.relpath(summary_path, _ROOT)}")
    for path in chart_paths:
        print(f"Chart:       {os.path.relpath(path, _ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
