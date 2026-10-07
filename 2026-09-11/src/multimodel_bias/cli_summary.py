from __future__ import annotations

import argparse
import json
from pathlib import Path

from .aug24_report import build_aug24_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the August 24 eight-model summary")
    parser.add_argument(
        "--iat-run-id",
        default="iat12_eight_model_expanded_10x_20260824_v1",
    )
    parser.add_argument(
        "--decodingtrust-run-id",
        default="decodingtrust_race_eight_model_pilot_20260824_v2",
    )
    parser.add_argument(
        "--bold-run-id",
        default="bold_race_eight_model_expanded_20260824_v3",
    )
    parser.add_argument("--phase-slug", default="expanded")
    parser.add_argument("--phase-label", default="Expanded IAT/BOLD; DecodingTrust pilot")
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/models_explicit_aug24.json"),
    )
    parser.add_argument("--write-main-results", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_aug24_report(
                iat_run_id=args.iat_run_id,
                decodingtrust_run_id=args.decodingtrust_run_id,
                bold_run_id=args.bold_run_id,
                phase_slug=args.phase_slug,
                phase_label=args.phase_label,
                model_config=args.model_config,
                write_main_results=args.write_main_results,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
