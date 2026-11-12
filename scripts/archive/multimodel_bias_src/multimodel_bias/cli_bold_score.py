from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from .bold import score_records
from .iat import PROJECT_ROOT, load_model_specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score an existing BOLD generation run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-config", type=Path)
    args = parser.parse_args(argv)
    output_dir = PROJECT_ROOT / "results" / "bold" / args.run_id
    responses_path = output_dir / "responses.csv"
    if not responses_path.exists():
        raise SystemExit(f"Missing generation results: {responses_path}")
    with responses_path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    _, models = load_model_specs(args.model_config)
    present = {row["model_key"] for row in records}
    models = [model for model in models if model.key in present]
    domains = {row.get("domain") for row in records}
    analysis = score_records(
        records,
        output_dir,
        [(model.key, model.display_name) for model in models],
        batch_size=args.batch_size,
        focus_domain="race" if domains == {"race"} else None,
    )
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis"] = analysis
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"run_id": args.run_id, "analysis": analysis}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
