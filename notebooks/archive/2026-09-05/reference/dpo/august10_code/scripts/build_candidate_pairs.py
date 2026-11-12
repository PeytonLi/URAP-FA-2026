#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.candidate_pairs import CandidatePolicy, build_candidate_pairs
from august3_debiasing.expression_pair_artifact import verify_expression_pair_artifact
from august3_debiasing.prompts import PROMPT_BY_TASK, prompt_sha256

CANDIDATE_BUILD_PROTOCOL_VERSION = "august3-candidate-build-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble blinded mixed-task candidate pairs from frozen response pools."
    )
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--expression-pair-manifest", required=True, type=Path)
    parser.add_argument("--responses-csv", required=True, type=Path, action="append")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-profile-pairs-per-image", type=int, default=2)
    parser.add_argument("--max-iat-contrasts-per-match", type=int, default=3)
    parser.add_argument("--max-controlled-iat-matches", type=int)
    parser.add_argument(
        "--controlled-iat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add demographic-metadata-agnostic matched IAT templates so sparse "
            "sampled-tuple overlap cannot block quotas."
        ),
    )
    parser.add_argument("--seed", default="august3-candidates-v1")
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    pair_manifest = verify_expression_pair_artifact(
        args.metadata_csv, args.expression_pair_manifest
    )
    metadata = read_csv(args.metadata_csv)
    responses = [row for path in args.responses_csv for row in read_csv(path)]
    policy = CandidatePolicy(
        max_profile_pairs_per_image=args.max_profile_pairs_per_image,
        max_iat_contrasts_per_match=args.max_iat_contrasts_per_match,
        include_controlled_iat=args.controlled_iat,
        max_controlled_iat_matches=args.max_controlled_iat_matches,
        seed=args.seed,
    )
    result = build_candidate_pairs(metadata, responses, policy=policy)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "mixed_candidate_pairs.csv"
    write_csv(output_csv, result.rows)
    manifest = {
        **result.report,
        "status": "complete",
        "protocol_version": CANDIDATE_BUILD_PROTOCOL_VERSION,
        "metadata_csv": str(args.metadata_csv.resolve()),
        "metadata_csv_sha256": sha256_file(args.metadata_csv),
        "expression_pair_manifest": str(args.expression_pair_manifest.resolve()),
        "expression_pair_manifest_sha256": sha256_file(args.expression_pair_manifest),
        "expression_pair_policy_version": pair_manifest["policy_version"],
        "response_csvs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in args.responses_csv
        ],
        "output_csv": str(output_csv.resolve()),
        "output_csv_sha256": sha256_file(output_csv),
        "prompt_sha256": {task: prompt_sha256(task) for task in PROMPT_BY_TASK},
        "policy": {key: getattr(policy, key) for key in policy.__dataclass_fields__},
    }
    (args.output_dir / "candidate_build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
