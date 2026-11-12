from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .analysis import write_iat_analysis
from .iat import ModelSpec
from .visbias import load_run_config
from .visbias_analysis import build_visbias_analysis


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _models_from_manifest(manifest: dict[str, Any]) -> list[ModelSpec]:
    models = [ModelSpec(**item) for item in manifest.get("models", [])]
    if not models:
        raise ValueError("Run manifest contains no model contracts")
    return models


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild August 31 UTKFace and VisBias analyses without model calls."
    )
    parser.add_argument(
        "--iat-run-dir",
        type=Path,
        default=Path("results/utkface_iat/utkface_iat_eight_model_20260831_v1"),
    )
    parser.add_argument(
        "--visbias-run-dir",
        type=Path,
        default=Path("results/visbias/visbias_race_eight_model_pilot_20260831_v1"),
    )
    parser.add_argument(
        "--visbias-run-config",
        type=Path,
        default=Path("config/visbias_pilot_aug31.json"),
    )
    parser.add_argument(
        "--visbias-lexicon",
        type=Path,
        default=Path("data/frozen/visbias/race_stereotype_lexicons.json"),
    )
    parser.add_argument("--skip-iat", action="store_true")
    parser.add_argument("--skip-visbias", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {"api_calls": 0}
    if not args.skip_iat:
        manifest = json.loads(
            (args.iat_run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        records = _read_csv(args.iat_run_dir / "responses.csv")
        models = _models_from_manifest(manifest)
        result["iat"] = write_iat_analysis(
            records,
            models,
            args.iat_run_dir,
            bootstrap_samples=int(manifest["config"]["bootstrap_samples"]),
            bootstrap_seed=int(manifest["config"]["base_seed"]) + 77,
        )

    if not args.skip_visbias:
        config = load_run_config(args.visbias_run_config)
        result["visbias"] = build_visbias_analysis(
            args.visbias_run_dir,
            args.visbias_lexicon,
            bootstrap_samples=config.bootstrap_samples,
            seed=config.base_seed,
        )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
