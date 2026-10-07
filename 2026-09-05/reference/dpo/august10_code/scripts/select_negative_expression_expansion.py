#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august10_research.negative_expression_selection import (  # noqa: E402
    NegativeSelectionPolicy,
    select_negative_expression_pairs,
)

POLICY_VERSION = "august3-pairwise-expression-v2-contract"
PROTOCOL_VERSION = "august10-negative-expression-expansion-v1"


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select and release reviewed negative-expression FairFace blocks."
    )
    parser.add_argument("--annotated-csv", required=True, type=Path)
    parser.add_argument("--exclude-metadata-csv", action="append", type=Path, default=[])
    parser.add_argument("--source-repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-blocks", type=int, default=9)
    parser.add_argument("--review-decisions-csv", type=Path)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.source_repo_root = args.source_repo_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotated = read_csv(args.annotated_csv)
    excluded = {
        row["image_sha256"]
        for path in args.exclude_metadata_csv
        for row in read_csv(path)
    }
    policy = NegativeSelectionPolicy(target_blocks=args.target_blocks)
    matches = select_negative_expression_pairs(annotated, excluded, policy)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    candidate_rows = []
    for match in matches:
        for source in (match.black, match.white):
            source_path = (args.source_repo_root / source["image_path"]).resolve()
            if not source_path.is_file() or sha256_file(source_path) != source["image_sha256"]:
                raise ValueError(f"source image is missing or changed: {source_path}")
            destination = image_dir / f"{source['image_id']}.png"
            if not destination.exists():
                shutil.copy2(source_path, destination)
            if sha256_file(destination) != source["image_sha256"]:
                raise ValueError(f"copied image hash differs: {destination}")
            row = dict(source)
            row.update(
                {
                    "image_path": str(destination.relative_to(ROOT_DIR)),
                    "sampling_match_id": match.sampling_match_id,
                    "expression_pair_equivalent": "true",
                    "expression_pair_min_confidence": str(match.minimum_confidence),
                    "expression_pair_max_cue_difference": str(
                        match.expression_intensity_difference
                    ),
                    "expression_pair_audit_version": POLICY_VERSION,
                    "selection_protocol": PROTOCOL_VERSION,
                    "match_age_bucket_difference": str(match.age_bucket_difference),
                    "match_expression_intensity_difference": str(
                        match.expression_intensity_difference
                    ),
                }
            )
            candidate_rows.append(row)
    candidate_path = args.output_dir / "candidate_negative_expression_pairs.csv"
    write_csv(candidate_path, candidate_rows)

    manifest: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "awaiting_visual_review",
        "annotated_csv": str(args.annotated_csv.resolve()),
        "annotated_csv_sha256": sha256_file(args.annotated_csv),
        "excluded_metadata": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.exclude_metadata_csv
        ],
        "candidate_metadata_csv": str(candidate_path.resolve()),
        "candidate_metadata_csv_sha256": sha256_file(candidate_path),
        "candidate_pairs": len(matches),
        "selection_policy": {
            "expression_class": "negative",
            "same_gender": True,
            "same_fine_expression": True,
            "minimum_confidence": policy.minimum_confidence,
            "max_age_bucket_difference": policy.max_age_bucket_difference,
            "max_expression_intensity_difference": (
                policy.max_expression_intensity_difference
            ),
            "excluded_preexisting_images": len(excluded),
            "seed": policy.seed,
        },
    }
    if args.review_decisions_csv is not None:
        decisions = {
            row["sampling_match_id"]: row for row in read_csv(args.review_decisions_csv)
        }
        expected = {match.sampling_match_id for match in matches}
        if set(decisions) != expected:
            raise ValueError("review decisions must cover every selected pair exactly once")
        rejected = [key for key, row in decisions.items() if row["decision"] != "accept"]
        if rejected:
            raise ValueError(f"visual review rejected selected pairs: {rejected}")
        released_path = args.output_dir / "expression_verified_matched_metadata.csv"
        write_csv(released_path, candidate_rows)
        manifest.update(
            {
                "status": "complete",
                "released_metadata_is_partial": False,
                "policy_version": POLICY_VERSION,
                "race_or_gender_metadata_in_prompt": False,
                "minimum_confidence": min(
                    float(row["expression_confidence"]) for row in candidate_rows
                ),
                "selected_pairs": len(matches),
                "accepted_pairs": len(matches),
                "target_accepted_pairs": len(matches),
                "accepted_metadata_csv": str(released_path.resolve()),
                "accepted_metadata_csv_sha256": sha256_file(released_path),
                "visual_review_decisions_csv": str(
                    args.review_decisions_csv.resolve()
                ),
                "visual_review_decisions_csv_sha256": sha256_file(
                    args.review_decisions_csv
                ),
                "visual_review_provenance": "codex_multimodal_ai_only_review",
            }
        )
    manifest_path = args.output_dir / "expression_pair_audit_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
