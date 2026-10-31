"""Experiment configuration and tracking.

Loads a versioned YAML experiment definition, validates it, executes the
experiment using the existing evaluation harness, and writes a self-
contained result directory with provenance artifacts and TensorBoard
scalar events.

Usage::

    python -m src.experiment --config config/experiments/fixed-cycle-balanced.yaml

See ``config/experiments/`` for example YAML files.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import yaml

from evaluation import (
    aggregate_with_scores,
    load_all_scenarios,
    load_available_controllers,
    run_many,
    validate_result,
    write_jsonl,
)
from plots import generate_charts
from tracker import ExperimentTracker

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_HERE, os.path.join(_ROOT, "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)


@dataclass(frozen=True)
class EvaluationSpec:
    """Seed list for evaluation."""

    seeds: tuple[int, ...] = ()


@dataclass(frozen=True)
class ExperimentConfig:
    """Versioned experiment configuration.

    Recognized top-level fields (some future, some baseline):
    - controller (required): controller name
    - action_space: configs | multi_binary_masked
    - state: observation representation
    - reward: reward function name
    - network: neural network architecture (future)
    - optimizer: optimizer config (future)
    - training: training hyperparameters (future)
    - scenario (required): named scenario from scenarios.yaml
    - name: optional experiment identifier
    - controller_settings: optional controller constructor kwargs
    - evaluation.seeds: optional seed override list
    """

    name: Optional[str] = None
    controller: str = ""
    action_space: Optional[str] = None
    state: Optional[str] = None
    reward: Optional[str] = None
    network: Optional[dict] = None
    optimizer: Optional[dict] = None
    training: Optional[dict] = None
    scenario: str = ""
    controller_settings: Optional[dict] = None
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load and validate an experiment YAML file.

    Raises ValueError on malformed input before any output is created.
    """
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("experiment YAML must be a mapping")

    # Required fields
    if "controller" not in data or not str(data.get("controller", "")).strip():
        raise ValueError("experiment must specify a non-empty controller")
    if "scenario" not in data or not str(data.get("scenario", "")).strip():
        raise ValueError("experiment must specify a non-empty scenario")

    raw_seeds = data.get("evaluation", {}).get("seeds", []) if isinstance(data.get("evaluation"), dict) else []
    if raw_seeds:
        int_seeds: list[int] = []
        for s in raw_seeds:
            if not isinstance(s, int) or isinstance(s, bool):
                raise ValueError(f"evaluation seeds must be integers, got {s!r}")
            if s < 0:
                raise ValueError(f"evaluation seeds must be non-negative, got {s}")
            int_seeds.append(s)
        if len(int_seeds) != len(set(int_seeds)):
            raise ValueError(f"evaluation seeds must be unique, got {raw_seeds}")
        seeds = tuple(int_seeds)
    else:
        seeds = ()

    # Validate action_space if provided
    if data.get("action_space") and data.get("action_space") not in (
        "configs",
        "multi_binary_masked",
    ):
        raise ValueError(
            f"invalid action_space {data.get('action_space')!r}; expected 'configs' or 'multi_binary_masked'"
        )

    return ExperimentConfig(
        name=data.get("name"),
        controller=str(data["controller"]),
        action_space=data.get("action_space"),
        state=data.get("state"),
        reward=data.get("reward"),
        network=data.get("network"),
        optimizer=data.get("optimizer"),
        training=data.get("training"),
        scenario=str(data["scenario"]),
        controller_settings=data.get("controller_settings"),
        evaluation=EvaluationSpec(seeds=seeds),
    )


def resolve_experiment(path: str) -> ExperimentConfig:
    """Load, validate, and resolve an experiment config.

    Resolves the named scenario and fills in evaluation seeds from the
    scenario defaults when not explicitly provided.
    """
    config = load_experiment_config(path)
    scenarios = load_all_scenarios()
    if config.scenario not in scenarios:
        raise KeyError(f"unknown scenario {config.scenario!r}; available: {sorted(scenarios)}")
    scenario = scenarios[config.scenario]

    # Fill seeds from scenario defaults if not explicitly provided
    if config.evaluation.seeds:
        resolved_seeds = tuple(config.evaluation.seeds)
    else:
        resolved_seeds = scenario.seeds

    return ExperimentConfig(
        name=config.name,
        controller=config.controller,
        action_space=config.action_space,
        state=config.state,
        reward=config.reward,
        network=config.network,
        optimizer=config.optimizer,
        training=config.training,
        scenario=config.scenario,
        controller_settings=config.controller_settings,
        evaluation=EvaluationSpec(seeds=resolved_seeds),
    )


def _git_commit() -> str:
    """Return the current Git commit hash, or 'unavailable'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unavailable"


def _write_artifacts(
    output_dir: str,
    config: ExperimentConfig,
    seeds: tuple[int, ...],
) -> str:
    """Write config.yaml and manifest.json to the result directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Write resolved config (all fields, including nulls)
    resolved = {
        "name": config.name,
        "controller": config.controller,
        "action_space": config.action_space,
        "state": config.state,
        "reward": config.reward,
        "network": config.network,
        "optimizer": config.optimizer,
        "training": config.training,
        "scenario": config.scenario,
        "controller_settings": config.controller_settings,
        "evaluation": {"seeds": list(seeds)},
    }
    config_path = os.path.join(output_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(resolved, f, default_flow_style=False, sort_keys=False)

    # Write manifest with provenance
    manifest = {
        "schema_version": "1.0",
        "resolved": {
            "controller": config.controller,
            "scenario": config.scenario,
            "seeds": list(seeds),
            "controller_settings": config.controller_settings,
        },
        "ml_fields": {
            "action_space": config.action_space,
            "state": config.state,
            "reward": config.reward,
            "network": config.network,
            "optimizer": config.optimizer,
            "training": config.training,
        },
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "git_commit": _git_commit(),
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)

    return output_dir


def _resolve_output_dir(output_dir: str) -> str:
    """Resolve the output directory path relative to the project root."""
    if not os.path.isabs(output_dir):
        return os.path.join(_ROOT, output_dir)
    return output_dir


# ---------------------------------------------------------------------------
# Config-driven runner
# ---------------------------------------------------------------------------


class ExperimentRunner:
    """Run one experiment from a resolved config."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self, output_dir: Optional[str] = None) -> str:
        """Execute the experiment and write all artifacts.

        Returns the result directory path.
        """
        config = self.config

        # Preflight: validate controller is installed
        available = load_available_controllers()
        if config.controller not in available:
            raise SystemExit(f"controller {config.controller!r} is not installed. Available controllers: {available}")

        # Resolve output directory
        if output_dir is None:
            name = config.name or config.scenario
            stamp = int(time.time())
            output_dir = os.path.join(_ROOT, "results", f"{name}-{stamp}")
        else:
            output_dir = _resolve_output_dir(output_dir)

        result_dir = output_dir

        # Resolve scenario for the runner
        from scenarios import get_scenario

        scenario = get_scenario(config.scenario)
        if config.evaluation.seeds:
            scenario = scenario.with_seeds(list(config.evaluation.seeds))

        os.makedirs(result_dir, exist_ok=True)

        # Write provenance artifacts
        _write_artifacts(result_dir, config, config.evaluation.seeds)

        # TensorBoard writer for scalar metrics
        from tensorboard.summary import Writer

        tb_dir = os.path.join(result_dir, "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        writer = Writer(tb_dir)
        tracker = ExperimentTracker(writer)

        # Log experiment metadata as text file (Writer API doesn't support text)
        metadata_lines = [f"controller: {config.controller}"]
        metadata_lines.append(f"scenario: {config.scenario}")
        metadata_lines.append(f"seeds: {config.evaluation.seeds}")
        if config.controller_settings:
            metadata_lines.append(f"controller_settings: {config.controller_settings}")
        metadata_path = os.path.join(result_dir, "metadata.txt")
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines))

        # Run evaluation
        records = run_many(
            [config.controller],
            scenario,
            controller_settings=config.controller_settings,
        )

        # Validate every record
        for record in records:
            validate_result(record.to_dict())

        # Write JSONL
        jsonl_path = os.path.join(result_dir, "runs.jsonl")
        write_jsonl(records, jsonl_path)

        # Aggregate with scores
        rows = aggregate_with_scores(records)

        # Write CSV
        import pandas as pd

        summary_frame = pd.DataFrame(rows)
        summary_frame.to_csv(os.path.join(result_dir, "summary.csv"), index=False)

        # Write Markdown summary
        from evaluation import findings_table

        header = (
            f"# Evaluation: {config.scenario}\n\n"
            f"- duration: {scenario.duration_seconds:g}s\n"
            f"- seeds: {list(scenario.seeds)}\n"
            f"- score weights: {{}}\n\n"
        )
        summary_md = header + findings_table(rows) + "\n"
        with open(os.path.join(result_dir, "summary.md"), "w", encoding="utf-8") as f:
            f.write(summary_md)

        # Generate charts
        generate_charts(
            [record.to_dict() for record in records],
            rows,
            result_dir,
            weights=None,
        )

        # Log TensorBoard scalar events: per-seed and aggregate means
        from evaluation import AGGREGATED_METRICS

        for record in records:
            metrics = record.to_dict()["metrics"]
            for metric in AGGREGATED_METRICS:
                value = metrics[metric]
                if isinstance(value, dict):
                    tracker.log_scalar(f"evaluation/{metric}", value["mean"], step=record.seed)
                else:
                    tracker.log_scalar(f"evaluation/{metric}", float(value), step=record.seed)

        # Log aggregate means (step=0)
        for row in rows:
            for metric in AGGREGATED_METRICS:
                value = row[metric]
                if isinstance(value, dict):
                    tracker.log_scalar(f"evaluation/{metric}/mean", value["mean"], step=0)

        tracker.close()

        print(f"Experiment results: {os.path.relpath(result_dir, _ROOT)}")
        return result_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.experiment",
        description="Run a single experiment from a YAML config file.",
    )
    parser.add_argument("--config", required=True, help="path to experiment YAML file")
    parser.add_argument(
        "--output",
        default=None,
        help="output directory (default: results/<name>-<timestamp>)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Read the original YAML content for the exact snapshot
    with open(args.config, encoding="utf-8") as f:
        original_yaml = f.read()

    config = resolve_experiment(args.config)
    runner = ExperimentRunner(config)
    result_dir = runner.run(args.output)

    # Overwrite with the exact original YAML (CLI path preserves input)
    config_path = os.path.join(result_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(original_yaml)

    return 0


if __name__ == "__main__":
    sys.exit(main())
