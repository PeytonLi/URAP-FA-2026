#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august10_research.dataset_gate import assess_dataset  # noqa: E402
from august10_research.judgment_planner import (  # noqa: E402
    PlannerPolicy,
    ReadyBlock,
    build_ready_blocks,
    select_development_block_ids,
    select_existing_blocks,
)
from august10_research.profile_rematcher import (  # noqa: E402
    ProfileMatch,
    ProfileRematchPolicy,
    rematch_profiles,
)

ADDED_FIELDS = (
    "source_sampling_match_id",
    "source_match_block_id",
    "judge_audit_source",
    "matching_provenance",
    "match_age_bucket_difference",
    "match_expression_intensity_difference",
    "match_min_expression_confidence",
    "block_direct_harm_reduction",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
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


def portable_path(path: Path) -> str:
    return Path(os.path.relpath(path.resolve(), ROOT_DIR)).as_posix()


def materialize_selected_images(
    rows: list[dict[str, Any]],
    source_roots: list[Path],
    destination_dir: Path,
) -> int:
    roots = [root.resolve() for root in source_roots]
    destination_dir.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, Path] = {}
    for row in rows:
        expected = row["image_sha256"].strip().lower()
        if expected in materialized:
            row["image_path"] = portable_path(materialized[expected])
            continue
        raw = Path(row["image_path"])
        candidates = [raw.resolve()] if raw.is_absolute() else [root / raw for root in roots]
        source = next(
            (
                candidate.resolve()
                for candidate in candidates
                if candidate.is_file() and sha256_file(candidate) == expected
            ),
            None,
        )
        if source is None:
            raise ValueError(
                f"no source root contains the hash-verified image for {row['pair_id']}"
            )
        destination = destination_dir / f"{expected}.png"
        if destination.exists():
            if sha256_file(destination) != expected:
                raise ValueError(f"materialized image hash mismatch: {destination}")
        else:
            shutil.copy2(source, destination)
        materialized[expected] = destination
        row["image_path"] = portable_path(destination)
    return len(materialized)


def materialize_iat(block: ReadyBlock) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{block.black['pair_id']}|{block.white['pair_id']}".encode()
    ).hexdigest()[:16]
    block_id = f"aug10-iat-original-{digest}"
    rows = []
    for source in block.rows:
        row = dict(source)
        row.update(
            {
                "source_sampling_match_id": row.get("sampling_match_id", ""),
                "source_match_block_id": row.get("match_block_id", ""),
                "sampling_match_id": block.sampling_match_id,
                "match_block_id": block_id,
                "matching_provenance": "august10-original-expression-pair-v1",
                "match_age_bucket_difference": "0",
                "match_expression_intensity_difference": row.get(
                    "expression_pair_max_cue_difference", ""
                ),
                "match_min_expression_confidence": row.get(
                    "expression_pair_min_confidence", ""
                ),
                "block_direct_harm_reduction": "",
            }
        )
        rows.append(row)
    return {"block_id": block_id, "rows": rows}


def materialize_profile(match: ProfileMatch) -> dict[str, Any]:
    rows = []
    for source in (match.black, match.white):
        row = dict(source)
        row.update(
            {
                "source_sampling_match_id": row.get("sampling_match_id", ""),
                "source_match_block_id": row.get("match_block_id", ""),
                "sampling_match_id": match.match_id,
                "match_block_id": match.match_id,
                "matching_provenance": "august10-covariate-profile-rematch-v1",
                "match_age_bucket_difference": str(match.age_bucket_difference),
                "match_expression_intensity_difference": str(
                    match.expression_intensity_difference
                ),
                "match_min_expression_confidence": str(
                    match.minimum_expression_confidence
                ),
                "block_direct_harm_reduction": str(
                    match.direct_harm_reduction
                ).lower(),
                "expression_pair_equivalent": "true",
                "expression_pair_min_confidence": str(
                    match.minimum_expression_confidence
                ),
                "expression_pair_max_cue_difference": str(
                    match.expression_intensity_difference
                ),
                "expression_pair_audit_version": (
                    "august10-covariate-profile-rematch-v1"
                ),
            }
        )
        rows.append(row)
    return {"block_id": match.match_id, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the audited 100-row directional pilot and run its gate."
    )
    parser.add_argument("--eligible-csv", action="append", type=Path, default=[])
    parser.add_argument("--audit-root", action="append", type=Path, default=[])
    parser.add_argument("--gate-config", type=Path, default=ROOT_DIR / "config/pilot_gate.json")
    parser.add_argument("--source-asset-root", action="append", type=Path, default=[])
    parser.add_argument("--materialized-image-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default="august10-frozen-pilot-v1")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    sources: list[tuple[Path, str, dict[str, Any] | None]] = [
        (path.resolve(), f"explicit-{index:02d}", None)
        for index, path in enumerate(args.eligible_csv, 1)
    ]
    for root in args.audit_root:
        for manifest_path in sorted(root.rglob("claude_pair_audit_manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete" or manifest.get("dry_run", False):
                continue
            sources.append(
                (
                    manifest_path.parent / "claude_judged_pairs_eligible.csv",
                    manifest_path.parent.name,
                    manifest,
                )
            )
    unique_sources = []
    seen_paths = set()
    for path, label, manifest in sources:
        resolved = path.resolve()
        if resolved not in seen_paths:
            unique_sources.append((resolved, label, manifest))
            seen_paths.add(resolved)

    merged_by_pair: dict[str, dict[str, Any]] = {}
    source_records = []
    for path, label, manifest in unique_sources:
        rows = read_csv(path)
        digest = sha256_file(path)
        if manifest and digest != manifest.get("eligible_csv_sha256"):
            raise ValueError(f"eligible CSV hash does not match audit manifest: {path}")
        for source in rows:
            row = dict(source)
            row["judge_audit_source"] = label
            pair_id = row.get("pair_id", "").strip()
            if pair_id in merged_by_pair:
                comparable = {
                    key: value
                    for key, value in row.items()
                    if key != "judge_audit_source"
                }
                prior = {
                    key: value
                    for key, value in merged_by_pair[pair_id].items()
                    if key != "judge_audit_source"
                }
                if comparable != prior:
                    raise ValueError(f"conflicting duplicate eligible pair: {pair_id}")
                continue
            merged_by_pair[pair_id] = row
        source_records.append(
            {"path": portable_path(path), "label": label, "sha256": digest, "rows": len(rows)}
        )
    eligible = list(merged_by_pair.values())
    if not eligible:
        raise ValueError("no eligible Claude judgments were found")

    policy = json.loads(args.gate_config.read_text(encoding="utf-8"))
    dataset_policy = policy["dataset"]

    ready = build_ready_blocks(eligible)
    profile_row_counts = Counter(
        row["sampling_match_id"]
        for row in eligible
        if row.get("task_type") == "grounded_profile"
    )
    selected_iat, _ = select_existing_blocks(
        ready,
        PlannerPolicy(
            target_iat_blocks=15,
            target_profile_blocks=35,
            min_negative_iat_blocks=dataset_policy["min_negative_iat_blocks"],
            preserve_ready_profile_matches_for_iat=True,
        ),
        profile_row_counts=profile_row_counts,
    )
    if len(selected_iat) != 15:
        raise ValueError(f"expected 15 IAT blocks, found {len(selected_iat)}")
    reserved_images = {
        row["image_sha256"] for block in selected_iat for row in block.rows
    }
    profile_matches = rematch_profiles(
        eligible,
        reserved_images,
        ProfileRematchPolicy(
            target_blocks=35,
            min_negative_expression_blocks=dataset_policy["min_negative_profile_blocks"],
            seed=args.seed + "-profiles",
        ),
    )

    iat_blocks = [materialize_iat(block) for block in selected_iat]
    profile_blocks = [materialize_profile(match) for match in profile_matches]
    development_ids = select_development_block_ids(
        iat_blocks,
        profile_blocks,
        dataset_policy,
        args.seed,
    )
    frozen_rows = []
    for block in iat_blocks + profile_blocks:
        split = "development" if block["block_id"] in development_ids else "train"
        for row in block["rows"]:
            row["split"] = split
            frozen_rows.append(row)
    source_asset_roots = [ROOT_DIR, *args.source_asset_root]
    materialized_images = materialize_selected_images(
        frozen_rows,
        source_asset_roots,
        args.materialized_image_dir.resolve(),
    )

    output_fields = list(eligible[0])
    for row in eligible:
        for field in row:
            if field not in output_fields:
                output_fields.append(field)
    for field in ADDED_FIELDS:
        if field not in output_fields:
            output_fields.append(field)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "all_claude_eligible_pairs.csv"
    frozen_path = args.output_dir / "mixed_dpo_pairs_frozen.csv"
    write_csv(merged_path, eligible, output_fields)
    write_csv(frozen_path, frozen_rows, output_fields)

    gate = assess_dataset(frozen_rows, dataset_policy)
    gate_path = args.output_dir / "dataset_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profile_summary = [
        {
            "match_block_id": match.match_id,
            "black_pair_id": match.black["pair_id"],
            "white_pair_id": match.white["pair_id"],
            "objective_type": match.black["objective_type"],
            "fairface_gender": match.black["fairface_gender"],
            "expression_class": match.black["expression_class"],
            "expression_fine": match.black["expression_fine"],
            "age_bucket_difference": match.age_bucket_difference,
            "expression_intensity_difference": match.expression_intensity_difference,
            "minimum_expression_confidence": match.minimum_expression_confidence,
            "direct_harm_reduction": match.direct_harm_reduction,
        }
        for match in profile_matches
    ]
    profile_summary_path = args.output_dir / "selected_profile_matches.csv"
    write_csv(profile_summary_path, profile_summary, list(profile_summary[0]))

    manifest = {
        "protocol_version": "august10-frozen-directional-pilot-v1",
        "status": "ready_to_train" if gate["pass"] else "do_not_train",
        "seed": args.seed,
        "eligible_sources": source_records,
        "eligible_pair_count": len(eligible),
        "selected_rows": len(frozen_rows),
        "selected_blocks": len(iat_blocks) + len(profile_blocks),
        "selected_iat_blocks": len(iat_blocks),
        "selected_profile_blocks": len(profile_blocks),
        "materialized_images": materialized_images,
        "materialized_image_dir": portable_path(args.materialized_image_dir),
        "source_asset_roots": [str(path.resolve()) for path in source_asset_roots],
        "direct_profile_blocks": sum(
            match.direct_harm_reduction for match in profile_matches
        ),
        "negative_expression_blocks": sum(
            block["rows"][0].get("expression_class") == "negative"
            for block in iat_blocks + profile_blocks
        ),
        "gate_decision": gate["decision"],
        "outputs": {
            path.name: {"path": portable_path(path), "sha256": sha256_file(path)}
            for path in (merged_path, frozen_path, gate_path, profile_summary_path)
        },
        "profile_rematch_policy": {
            "same_objective": True,
            "same_gender": True,
            "same_expression_class": True,
            "same_expression_fine": True,
            "same_response_provenance": True,
            "max_age_bucket_difference": 1,
            "max_expression_intensity_difference": 1,
            "min_expression_confidence": 0.70,
            "min_negative_expression_blocks": dataset_policy[
                "min_negative_profile_blocks"
            ],
            "min_negative_iat_blocks": dataset_policy["min_negative_iat_blocks"],
            "profile_compatibility": {
                "max_word_count_ratio": 1.25,
                "max_grounded_detail_difference": 1,
                "max_quality_score_difference": 0.75,
            },
        },
        "frozen_counts": {
            "split": dict(Counter(row["split"] for row in frozen_rows)),
            "task": dict(Counter(row["task_type"] for row in frozen_rows)),
            "race": dict(Counter(row["fairface_race"] for row in frozen_rows)),
        },
    }
    manifest_path = args.output_dir / "freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if gate["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
