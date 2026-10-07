from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .bold import load_race_tasks, load_tasks, source_manifest, write_generation_analysis
from .iat import PROJECT_ROOT, load_model_specs
from .openrouter import OpenRouterClient
from .secrets import get_openrouter_api_key
from .text_runner import TextBenchmarkRunner, TextRunConfig, freeze_selected_prompts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BOLD completions through OpenRouter")
    parser.add_argument("--run-id")
    parser.add_argument("--sample-count", type=int, default=25)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--models", default="all")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--race-focus", action="store_true")
    parser.add_argument("--samples-per-category", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--base-seed", type=int, default=17082026)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--minimum-request-interval", type=float, default=6.5)
    parser.add_argument("--max-run-cost", type=float, default=2.0)
    parser.add_argument("--minimum-balance", type=float, default=5.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retry-empty-responses", action="store_true")
    parser.add_argument("--accept-existing-api-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, models = load_model_specs(args.model_config)
    model_universe = models
    if args.models != "all":
        requested = {item.strip() for item in args.models.split(",") if item.strip()}
        unknown = requested - {model.key for model in models}
        if unknown:
            raise SystemExit(f"Unknown models: {', '.join(sorted(unknown))}")
        models = [model for model in models if model.key in requested]
    if args.race_focus and args.full:
        raise SystemExit("--race-focus and --full cannot be combined")
    tasks = (
        load_race_tasks(args.samples_per_category, args.base_seed)
        if args.race_focus
        else load_tasks(None if args.full else args.sample_count, args.base_seed)
    )
    run_id = args.run_id or f"bold_smoke_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = PROJECT_ROOT / "results" / "bold" / run_id
    selected_rows = [
        {"task_key": task.key, "prompt": task.prompt, **task.metadata} for task in tasks
    ]
    config = TextRunConfig(
        benchmark_version="bold-inspect-evals-2-A",
        repeats=1,
        temperature=args.temperature,
        top_p=args.top_p,
        base_seed=args.base_seed,
        max_workers=args.max_workers,
        minimum_request_interval_seconds=args.minimum_request_interval,
        max_run_cost_usd=args.max_run_cost,
        minimum_account_balance_usd=args.minimum_balance,
        provider={"data_collection": "deny", "zdr": True, "allow_fallbacks": False},
        retry_empty_responses=args.retry_empty_responses,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "models": [model.key for model in models],
                    "tasks": len(tasks),
                    "planned_calls": len(models) * len(tasks),
                    "domains": sorted({task.metadata["domain"] for task in tasks}),
                },
                indent=2,
            )
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_selected_prompts(output_dir / "selected_prompts.csv", selected_rows)
    client = OpenRouterClient(
        get_openrouter_api_key(),
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    runner = TextBenchmarkRunner(
        client,
        run_id,
        config,
        models,
        tasks,
        output_dir,
        source_manifest=source_manifest(),
        model_universe=model_universe,
        retry_existing_api_errors=not args.accept_existing_api_errors,
    )
    records, manifest = runner.run()
    analysis_models = [type(models[0])(**item) for item in manifest["models"]]
    analysis = write_generation_analysis(
        records,
        output_dir,
        [(model.key, model.display_name) for model in analysis_models],
    )
    manifest["analysis"] = analysis
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "run_id": run_id,
                "records": manifest["record_count"],
                "valid_records": manifest["valid_record_count"],
                "reported_cost_usd": manifest["reported_response_cost_usd"],
                "remaining_credit_usd": manifest["credits_after"]["remaining"],
                "analysis": analysis,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
