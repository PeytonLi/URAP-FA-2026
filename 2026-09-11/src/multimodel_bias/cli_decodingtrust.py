from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .decodingtrust import (
    RACE_DEMOGRAPHIC_GROUPS,
    analyze,
    load_race_tasks,
    load_tasks,
    source_manifest,
)
from .iat import PROJECT_ROOT, load_model_specs
from .openrouter import OpenRouterClient
from .secrets import get_openrouter_api_key
from .text_runner import TextBenchmarkRunner, TextRunConfig, freeze_selected_prompts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DecodingTrust stereotype bias via OpenRouter")
    parser.add_argument("--run-id")
    parser.add_argument("--samples-per-regime", type=int, default=16)
    parser.add_argument("--full", action="store_true", help="Use all 1,152 prompts per regime")
    parser.add_argument("--models", default="all")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--race-focus", action="store_true")
    parser.add_argument("--topics-per-group", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
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
        load_race_tasks(args.topics_per_group, args.base_seed)
        if args.race_focus
        else load_tasks(None if args.full else args.samples_per_regime, args.base_seed)
    )
    run_id = args.run_id or f"decodingtrust_smoke_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir = PROJECT_ROOT / "results" / "decodingtrust" / run_id
    config = TextRunConfig(
        benchmark_version="decodingtrust-stereotype-commit-161ae83",
        repeats=args.repeats,
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
                    "planned_calls": len(models) * len(tasks) * args.repeats,
                    "regimes": sorted({task.metadata["regime"] for task in tasks}),
                    "demographic_groups": sorted(
                        {task.metadata["demographic_group"] for task in tasks}
                    ),
                },
                indent=2,
            )
        )
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_selected_prompts(
        output_dir / "selected_prompts.csv",
        [{"task_key": task.key, "prompt": task.prompt, **task.metadata} for task in tasks],
    )
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
    analysis = analyze(
        records,
        output_dir,
        [(model.key, model.display_name) for model in analysis_models],
        focus_groups=RACE_DEMOGRAPHIC_GROUPS if args.race_focus else None,
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
