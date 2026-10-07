#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from august10_research.dataset_gate import assess_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    with args.dataset.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = assess_dataset(rows, config["dataset"])
    result["input_dataset"] = str(args.dataset)
    result["config"] = str(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "failed_checks": result["issues"]}))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
