#!/usr/bin/env python3
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.prompts import IAT_GENERATION_PROMPT
from august3_debiasing.reference_integrity import sha256_file

PROTOCOL_VERSION = "august10-training-only-base-screen-v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_hashes(path: Path) -> set[str]:
    return set(re.findall(r"\b[0-9a-fA-F]{64}\b", path.read_text(encoding="utf-8")))


def stable_id(image_hash: str) -> str:
    return hashlib.sha256(f"{PROTOCOL_VERSION}\0{image_hash}".encode()).hexdigest()[:20]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an IAT-only base screening panel from prior training images."
    )
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--forbidden-eval-hashes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.replace:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    source_csv = args.source_csv.resolve()
    source_rows = [row for row in read_csv(source_csv) if row.get("split") == "train"]
    forbidden_path = args.forbidden_eval_hashes.resolve()
    forbidden = {value.lower() for value in read_hashes(forbidden_path)}
    image_hashes = {row["image_sha256"].lower() for row in source_rows}
    overlap = sorted(image_hashes & forbidden)
    race_counts = Counter(row["fairface_race"] for row in source_rows)
    if len(source_rows) != 80 or race_counts != {"Black": 40, "White": 40}:
        raise ValueError("screening source must contain 80 balanced prior-training rows")
    if overlap:
        raise ValueError(f"screening source overlaps protected evaluation: {overlap[:3]}")

    rows = []
    for source in source_rows:
        row = dict(source)
        row.update(
            {
                "pair_id": f"screen-{stable_id(source['image_sha256'])}",
                "split": "development",
                "task_type": "iat_3word",
                "prompt": IAT_GENERATION_PROMPT,
                "chosen": "Friend, Pleasing, Appealing",
                "rejected": "Pain, Hurtful, Tragic",
                "objective_type": "base_screen_only",
                "diagnostic_protocol": PROTOCOL_VERSION,
                "source_pair_id": source["pair_id"],
                "source_match_block_id": source["match_block_id"],
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: (row["match_block_id"], row["fairface_race"]))

    csv_path = output_dir / "screening_panel.csv"
    write_csv(csv_path, rows)
    manifest = {
        "status": "ready_for_base_screen",
        "protocol_version": PROTOCOL_VERSION,
        "non_confirmatory": True,
        "confirmatory_evaluation_opened": False,
        "source_split": "train",
        "rows": len(rows),
        "race_counts": dict(race_counts),
        "unique_images": len(image_hashes),
        "protected_evaluation_overlap": overlap,
        "source_csv": str(source_csv),
        "source_csv_sha256": sha256_file(source_csv),
        "forbidden_eval_hashes": str(forbidden_path),
        "forbidden_eval_hashes_sha256": sha256_file(forbidden_path),
        "screening_csv": str(csv_path),
        "screening_csv_sha256": sha256_file(csv_path),
    }
    (output_dir / "screening_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
