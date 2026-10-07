from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from .iat import (
    IATBenchmarkRunner,
    load_model_specs,
    load_run_config,
    load_stimuli,
)
from .openrouter import OpenRouterClient
from .secrets import get_openrouter_api_key


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an image IAT-style implicit-bias benchmark."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        help="Model configuration JSON; defaults to config/models_aug31.json",
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        help="IAT run configuration JSON; defaults to config/iat_utkface_aug31.json",
    )
    parser.add_argument(
        "--stimulus-manifest",
        type=Path,
        help="Image manifest CSV; defaults to the fixed 12-image benchmark",
    )
    parser.add_argument("--run-id", help="Stable output directory name; defaults to a timestamp")
    parser.add_argument(
        "--models",
        default="all",
        help="Comma-separated model keys from the selected model config, or 'all'",
    )
    parser.add_argument(
        "--image-ids",
        default="all",
        help="Comma-separated IDs from data/iat12/manifest.csv, or 'all'",
    )
    parser.add_argument("--repeats", type=int, help="Repetitions per image and model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--minimum-request-interval", type=float)
    parser.add_argument(
        "--max-run-cost", type=float, help="Fail-closed reported cost ceiling in USD"
    )
    parser.add_argument("--minimum-balance", type=float, help="Required account reserve in USD")
    parser.add_argument("--bootstrap-samples", type=int)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument(
        "--allow-data-retention",
        action="store_true",
        help="Allow non-ZDR providers. Disabled by default for face images.",
    )
    parser.add_argument("--smoke", action="store_true", help="All 12 images once per model")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate assets/config without API calls"
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--accept-existing-api-errors",
        action="store_true",
        help="Finalize a resumed run without retrying previously recorded API-error cells.",
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _, model_specs = load_model_specs(args.model_config)
    if args.models != "all":
        requested = {value.strip() for value in args.models.split(",") if value.strip()}
        known = {model.key for model in model_specs}
        unknown = requested - known
        if unknown:
            raise SystemExit(f"Unknown model keys: {', '.join(sorted(unknown))}")
        model_specs = [model for model in model_specs if model.key in requested]

    overrides = {
        "repeats": 1 if args.smoke else args.repeats,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "base_seed": args.base_seed,
        "max_workers": args.max_workers,
        "minimum_request_interval_seconds": args.minimum_request_interval,
        "max_run_cost_usd": 1.0 if args.smoke and args.max_run_cost is None else args.max_run_cost,
        "minimum_account_balance_usd": args.minimum_balance,
        "bootstrap_samples": 20000
        if args.smoke and args.bootstrap_samples is None
        else args.bootstrap_samples,
        "fallback_enabled": False if args.no_fallback else None,
    }
    config = load_run_config(args.run_config, **overrides)
    if args.allow_data_retention:
        provider = dict(config.provider)
        provider.update({"data_collection": "allow", "zdr": False})
        config = load_run_config(args.run_config, **{**overrides, "provider": provider})
    stimuli = load_stimuli(
        args.stimulus_manifest,
        expected_count=None if args.stimulus_manifest else 12,
    )
    if args.image_ids != "all":
        requested_images = {value.strip() for value in args.image_ids.split(",") if value.strip()}
        known_images = {stimulus.image_id for stimulus in stimuli}
        unknown_images = requested_images - known_images
        if unknown_images:
            raise SystemExit(f"Unknown image IDs: {', '.join(sorted(unknown_images))}")
        stimuli = [stimulus for stimulus in stimuli if stimulus.image_id in requested_images]
    run_id = args.run_id or (
        ("iat_image_smoke_" if args.smoke else "iat_image_run_")
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_id": run_id,
                    "models": [model.__dict__ for model in model_specs],
                    "stimuli": [stimulus.filename for stimulus in stimuli],
                    "config": config.__dict__,
                    "planned_calls_before_fallbacks": len(model_specs)
                    * len(stimuli)
                    * config.repeats,
                },
                indent=2,
            )
        )
        return 0

    client = OpenRouterClient(
        get_openrouter_api_key(),
        timeout_seconds=args.timeout,
        max_retries=args.max_retries,
    )
    runner = IATBenchmarkRunner(
        client,
        run_id,
        config,
        model_specs,
        stimuli,
        output_root=args.output_root,
        retry_existing_api_errors=not args.accept_existing_api_errors,
    )
    manifest = runner.run()
    print(
        json.dumps(
            {
                "run_id": run_id,
                "output_dir": str(runner.output_dir),
                "records": manifest["record_count"],
                "valid_records": manifest["valid_record_count"],
                "reported_response_cost_usd": manifest["reported_response_cost_usd"],
                "account_usage_delta_usd": manifest["account_usage_delta_usd"],
                "remaining_credit_usd": manifest["credits_after"]["remaining"],
                "plots": manifest["analysis"]["plots"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
