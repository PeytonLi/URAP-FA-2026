from __future__ import annotations

import argparse
import json
from pathlib import Path

from .iat import load_model_specs
from .openrouter import OpenRouterClient
from .secrets import get_openrouter_api_key
from .visbias import (
    VisBiasRunner,
    load_run_config,
    load_tasks,
    prepare_visbias_inputs,
)
from .visbias_analysis import build_visbias_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, run, or analyze the VisBias pilot.")
    parser.add_argument("action", choices=("prepare", "dry-run", "run", "analyze"))
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path("data/private/visbias/source"),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/frozen/visbias"),
    )
    parser.add_argument(
        "--normalized-dir",
        type=Path,
        default=Path("data/private/visbias/normalized"),
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/models_aug31.json"),
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=Path("config/visbias_pilot_aug31.json"),
    )
    parser.add_argument("--run-id", default="visbias_race_eight_model_pilot_20260831_v1")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys from the selected model config, or 'all'",
    )
    parser.add_argument(
        "--task-kinds",
        default="all",
        help="Comma-separated task kinds (description,comparison), or 'all'",
    )
    parser.add_argument(
        "--task-limit-per-kind",
        type=int,
        help="Deterministically retain only the first N tasks of each selected kind",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/visbias"),
    )
    parser.add_argument("--images-per-cell", type=int, default=3)
    parser.add_argument("--normalized-max-dimension", type=int, default=512)
    parser.add_argument("--normalized-jpeg-quality", type=int, default=90)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Finish generation without paired analysis; intended only for compatibility probes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "prepare":
        result = prepare_visbias_inputs(
            args.source_repo,
            args.input_dir,
            expected_revision="6a129c5d6e0d2278325811196ac33a1bc522408d",
            images_per_cell=args.images_per_cell,
            normalized_root=args.normalized_dir,
            normalized_max_dimension=args.normalized_max_dimension,
            normalized_jpeg_quality=args.normalized_jpeg_quality,
        )
        print(json.dumps(result, indent=2))
        return 0

    config = load_run_config(args.run_config)
    _, models = load_model_specs(args.model_config)
    if args.models != "all":
        requested_models = {value.strip() for value in args.models.split(",") if value.strip()}
        known_models = {model.key for model in models}
        unknown_models = requested_models - known_models
        if unknown_models:
            raise SystemExit(f"Unknown model keys: {', '.join(sorted(unknown_models))}")
        models = [model for model in models if model.key in requested_models]
    tasks = load_tasks(args.input_dir)
    if args.task_kinds != "all":
        requested_kinds = {
            value.strip() for value in args.task_kinds.split(",") if value.strip()
        }
        unknown_kinds = requested_kinds - {"description", "comparison"}
        if unknown_kinds:
            raise SystemExit(f"Unknown task kinds: {', '.join(sorted(unknown_kinds))}")
        tasks = [task for task in tasks if task["task_kind"] in requested_kinds]
    if args.task_limit_per_kind is not None:
        if args.task_limit_per_kind < 1:
            raise SystemExit("--task-limit-per-kind must be at least 1")
        retained = []
        counts = {"description": 0, "comparison": 0}
        for task in tasks:
            kind = str(task["task_kind"])
            if counts[kind] < args.task_limit_per_kind:
                retained.append(task)
                counts[kind] += 1
        tasks = retained
    planned = len(models) * len(tasks)
    if args.action == "dry-run":
        print(
            json.dumps(
                {
                    "models": [model.__dict__ for model in models],
                    "description_tasks_per_model": sum(
                        task["task_kind"] == "description" for task in tasks
                    ),
                    "comparison_tasks_per_model": sum(
                        task["task_kind"] == "comparison" for task in tasks
                    ),
                    "planned_calls": planned,
                    "config": config.__dict__,
                },
                indent=2,
            )
        )
        return 0

    run_dir = args.output_root / args.run_id
    if args.action == "analyze":
        result = build_visbias_analysis(
            run_dir,
            args.input_dir / "race_stereotype_lexicons.json",
            bootstrap_samples=config.bootstrap_samples,
            seed=config.base_seed,
        )
        print(json.dumps(result, indent=2))
        return 0

    client = OpenRouterClient(
        get_openrouter_api_key(),
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    runner = VisBiasRunner(client, args.run_id, config, models, tasks, args.output_root)
    manifest = runner.run()
    if args.skip_analysis:
        print(json.dumps({"manifest": manifest, "analysis": "skipped"}, indent=2))
        return 0
    analysis = build_visbias_analysis(
        runner.output_dir,
        args.input_dir / "race_stereotype_lexicons.json",
        bootstrap_samples=config.bootstrap_samples,
        seed=config.base_seed,
    )
    print(json.dumps({"manifest": manifest, "analysis": analysis}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
