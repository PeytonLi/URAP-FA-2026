from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .analysis import write_iat_analysis
from .decodingtrust import analyze as analyze_decodingtrust
from .iat import PROJECT_ROOT, load_model_specs


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _require_complete(
    records: list[dict[str, Any]],
    models: list[Any],
    expected_per_model: int,
    label: str,
) -> None:
    counts = Counter(record.get("model_key") for record in records)
    expected = {model.key: expected_per_model for model in models}
    if counts != expected:
        raise ValueError(f"{label} model distribution is incomplete: {dict(counts)}")
    invalid = [record for record in records if record.get("status") != "completed"]
    if invalid:
        raise ValueError(f"{label} contains {len(invalid)} non-completed rows")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild IAT and DecodingTrust analysis from shared CSVs without API calls"
    )
    parser.add_argument("--iat-run-id", default="iat12_five_model_pilot_10x_v1")
    parser.add_argument(
        "--decodingtrust-run-id",
        default="decodingtrust_five_model_smoke_v1",
    )
    args = parser.parse_args(argv)

    _, models = load_model_specs()
    iat_dir = PROJECT_ROOT / "results" / "iat12" / args.iat_run_id
    iat_records = _read_csv(iat_dir / "responses.csv")
    iat_manifest = json.loads((iat_dir / "run_manifest.json").read_text(encoding="utf-8"))
    repeats = int(iat_manifest["config"]["repeats"])
    _require_complete(iat_records, models, repeats * 12, "IAT")
    iat_analysis = write_iat_analysis(
        iat_records,
        models,
        iat_dir,
        bootstrap_samples=int(iat_manifest["config"]["bootstrap_samples"]),
        bootstrap_seed=int(iat_manifest["config"]["base_seed"]) + 77,
    )

    dt_dir = PROJECT_ROOT / "results" / "decodingtrust" / args.decodingtrust_run_id
    dt_records = _read_csv(dt_dir / "responses.csv")
    dt_manifest = json.loads((dt_dir / "run_manifest.json").read_text(encoding="utf-8"))
    expected_dt_per_model = int(dt_manifest["task_count"]) * int(dt_manifest["config"]["repeats"])
    _require_complete(dt_records, models, expected_dt_per_model, "DecodingTrust")
    dt_analysis = analyze_decodingtrust(
        dt_records,
        dt_dir,
        [(model.key, model.display_name) for model in models],
    )

    print(
        json.dumps(
            {
                "iat_run_id": args.iat_run_id,
                "iat_analysis": iat_analysis,
                "decodingtrust_run_id": args.decodingtrust_run_id,
                "decodingtrust_analysis": dt_analysis,
                "api_calls": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
