#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august10_research.claude_usage import summarize_usage  # noqa: E402


def portable_path(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), ROOT_DIR)).as_posix()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Track Claude judgment cost and reserve.")
    parser.add_argument("--judgments-jsonl", action="append", type=Path, default=[])
    parser.add_argument(
        "--audit-root",
        action="append",
        type=Path,
        default=[],
        help="Discover completed paid Claude audit checkpoints recursively.",
    )
    parser.add_argument("--pricing", type=Path, default=ROOT_DIR / "config/claude_pricing.json")
    parser.add_argument("--starting-balance-usd", type=Decimal)
    parser.add_argument("--reserve-fraction", type=Decimal, default=Decimal("0.10"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pricing = json.loads(args.pricing.read_text(encoding="utf-8"))
    judgment_paths = list(args.judgments_jsonl)
    for root in args.audit_root:
        for manifest_path in sorted(root.rglob("claude_pair_audit_manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "complete" and not manifest.get("dry_run", False):
                judgment_paths.append(manifest_path.parent / "claude_judgments.jsonl")
    judgment_paths = list(dict.fromkeys(path.resolve() for path in judgment_paths))
    records = [record for path in judgment_paths for record in read_jsonl(path)]
    result = summarize_usage(
        records,
        pricing["rates_per_million_tokens"],
        starting_balance_usd=args.starting_balance_usd,
        reserve_fraction=args.reserve_fraction,
    )
    result.update(
        {
            "protocol_version": "august10-claude-usage-tracker-v1",
            "pricing_source": pricing["pricing_source"],
            "pricing_verified_date": pricing["pricing_verified_date"],
            "judgment_files": [portable_path(path) for path in judgment_paths],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if result["pause_judging"] is True else 0


if __name__ == "__main__":
    raise SystemExit(main())
