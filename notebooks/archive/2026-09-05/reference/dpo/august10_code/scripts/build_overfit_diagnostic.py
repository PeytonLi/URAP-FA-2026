#!/usr/bin/env python3
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.pair_policy import parse_iat_response
from august3_debiasing.prompts import IAT_GENERATION_PROMPT
from august3_debiasing.reference_integrity import sha256_file
from august10_research.overfit_diagnostic import (
    INFORMATIVE_PROTOCOL_VERSION,
    PROTOCOL_VERSION,
    assess_overfit_dataset,
)

SELECTED_BLOCK_IDS = (
    "aug10-iat-original-242f7da6516c54d6",
    "aug10-iat-original-3b9dc0589d2f9ee7",
    "aug10-iat-original-60a6e54d0e811b8c",
    "aug10-iat-original-b6ae20d7a666b877",
    "aug10-iat-original-e0351a126c12991f",
    "aug10-profile-rematch-342ec89407f2b71e",
)

TRAIN_TEMPLATES = (
    ("Friend, Pleasing, Appealing", "Pain, Hurtful, Tragic"),
    ("Glad, Cheerful, Delight", "Sadness, Horrible, Hate"),
    ("Joyful, Lovely, Excellent", "Angry, Ugly, Failure"),
    ("Friendship, Happy, Pleasure", "Grief, Nasty, Disaster"),
)
DEVELOPMENT_TEMPLATE = ("Cheer, Enjoy, Terrific", "Poison, Gross, Sick")
DEFAULT_MODEL_REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_hashes(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"\b[0-9a-fA-F]{64}\b", text.lower()))


def stable_id(*parts: str) -> str:
    payload = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def diagnostic_row(
    source: dict[str, str],
    *,
    image_path: str,
    split: str,
    block_id: str,
    variant: str,
    repeat: int,
    chosen: str,
    rejected: str,
) -> dict[str, Any]:
    row: dict[str, Any] = dict(source)
    row.update(
        {
            "pair_id": f"diag-{stable_id(source['image_sha256'], split, variant, str(repeat))}",
            "match_block_id": block_id,
            "image_path": image_path,
            "split": split,
            "task_type": "iat_3word",
            "prompt": IAT_GENERATION_PROMPT,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_response_provenance": "deterministic_diagnostic_template",
            "rejected_response_provenance": "deterministic_diagnostic_template",
            "judge_model": "not_used_controlled_diagnostic",
            "judge_image_consensus": "not_applicable",
            "judge_text_consensus": "not_applicable",
            "judge_order_stable": "not_applicable",
            "objective_type": "negative_count_reduction",
            "controlled_template_id": variant,
            "candidate_construction": "race_invariant_controlled_overfit_canary",
            "chosen_identity_leakage_count": 0,
            "rejected_identity_leakage_count": 0,
            "chosen_person_negative_count": 0,
            "rejected_person_negative_count": 3,
            "chosen_unwarranted_negative_count": 0,
            "rejected_unwarranted_negative_count": 3,
            "block_direct_harm_reduction": 3,
            "diagnostic_protocol": PROTOCOL_VERSION,
            "diagnostic_non_confirmatory": True,
            "diagnostic_image_seen_in_training": True,
            "diagnostic_repeat": repeat,
            "source_match_block_id": source["match_block_id"],
            "source_pair_id": source["pair_id"],
        }
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a protected, non-confirmatory DPO/LoRA overfit diagnostic."
    )
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--forbidden-eval-hashes", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-repeats", type=int, default=4)
    parser.add_argument("--screening-responses-csv", type=Path)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.train_repeats != 4:
        raise ValueError("v1 diagnostic requires exactly four training repeats")
    root = args.repo_root.resolve()
    source_csv = args.source_csv.resolve()
    forbidden_path = args.forbidden_eval_hashes.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.replace:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True)

    all_source_rows = [row for row in read_csv(source_csv) if row.get("split") == "train"]
    screening_metrics: dict[str, dict[str, Any]] = {}
    screening_responses_path = (
        args.screening_responses_csv.resolve()
        if args.screening_responses_csv is not None
        else None
    )
    selected_block_ids = SELECTED_BLOCK_IDS
    protocol_version = PROTOCOL_VERSION
    if screening_responses_path is not None:
        protocol_version = INFORMATIVE_PROTOCOL_VERSION
        for response in read_csv(screening_responses_path):
            image_hash = response["image_sha256"]
            parsed = parse_iat_response(response["response"])
            metric = screening_metrics.setdefault(
                image_hash,
                {"negative_terms": 0, "selected_terms": 0, "responses": 0},
            )
            metric["negative_terms"] += int(parsed["negative_count"])
            metric["selected_terms"] += 3 if parsed["valid"] else 0
            metric["responses"] += 1
        blocks: dict[str, list[dict[str, str]]] = {}
        for row in all_source_rows:
            blocks.setdefault(row["match_block_id"], []).append(row)
        scored_blocks = []
        for block_id, block_rows in blocks.items():
            if len(block_rows) != 2 or {row["fairface_race"] for row in block_rows} != {
                "Black",
                "White",
            }:
                continue
            scores = {
                row["fairface_race"]: int(
                    screening_metrics.get(row["image_sha256"], {}).get("negative_terms", 0)
                )
                for row in block_rows
            }
            scored_blocks.append((block_id, scores["Black"], scores["White"]))
        top_blocks = sorted(
            scored_blocks,
            key=lambda item: (item[1] + item[2], min(item[1], item[2]), item[0]),
            reverse=True,
        )[:20]
        candidates = []
        for combination in itertools.combinations(top_blocks, 6):
            black = sum(item[1] for item in combination)
            white = sum(item[2] for item in combination)
            candidates.append((min(black, white), black + white, black, white, combination))
        best = max(candidates, default=None)
        if best is None or best[0] < 6:
            raise ValueError("training-only screening did not find six informative matched blocks")
        selected_block_ids = tuple(item[0] for item in best[4])
    source_rows = [
        row for row in all_source_rows if row.get("match_block_id") in selected_block_ids
    ]
    selected_counts = {
        block_id: [row for row in source_rows if row["match_block_id"] == block_id]
        for block_id in selected_block_ids
    }
    if any(
        len(rows) != 2 or {row["fairface_race"] for row in rows} != {"Black", "White"}
        for rows in selected_counts.values()
    ):
        raise ValueError("selected source blocks must each contain one Black and one White row")
    if protocol_version == PROTOCOL_VERSION and {
        row["expression_class"] for row in source_rows
    } != {"neutral"}:
        raise ValueError("selected source images are not all neutral")

    copied_paths: dict[str, str] = {}
    for row in source_rows:
        source_path = (root / row["image_path"]).resolve()
        if not source_path.is_relative_to(root) or not source_path.is_file():
            raise FileNotFoundError(source_path)
        if sha256_file(source_path) != row["image_sha256"]:
            raise ValueError(f"source image hash mismatch: {source_path}")
        target = image_dir / f"{row['image_sha256']}{source_path.suffix.lower()}"
        shutil.copy2(source_path, target)
        copied_paths[row["image_sha256"]] = str(target.relative_to(root))

    rows: list[dict[str, Any]] = []
    screening_pair_ids = {}
    if screening_responses_path is not None:
        for response in read_csv(screening_responses_path):
            screening_pair_ids.setdefault(response["image_sha256"], response["image"])
    for source_block_index, source_block_id in enumerate(selected_block_ids, 1):
        block_rows = sorted(
            selected_counts[source_block_id], key=lambda row: row["fairface_race"]
        )
        for template_index, (chosen, rejected) in enumerate(TRAIN_TEMPLATES, 1):
            for repeat in range(1, args.train_repeats + 1):
                block_id = (
                    f"diag-train-b{source_block_index:02d}-"
                    f"t{template_index:02d}-r{repeat:02d}"
                )
                for source in block_rows:
                    row = diagnostic_row(
                            source,
                            image_path=copied_paths[source["image_sha256"]],
                            split="train",
                            block_id=block_id,
                            variant=f"train-template-{template_index:02d}",
                            repeat=repeat,
                            chosen=chosen,
                            rejected=rejected,
                        )
                    row["diagnostic_protocol"] = protocol_version
                    if screening_metrics:
                        metric = screening_metrics[source["image_sha256"]]
                        row.update(
                            {
                                "diagnostic_base_negative_terms": metric["negative_terms"],
                                "diagnostic_base_selected_terms": metric["selected_terms"],
                                "diagnostic_base_responses": metric["responses"],
                            }
                        )
                    rows.append(row)
        development_block = f"diag-development-b{source_block_index:02d}"
        for source in block_rows:
            row = diagnostic_row(
                    source,
                    image_path=copied_paths[source["image_sha256"]],
                    split="development",
                    block_id=development_block,
                    variant="development-template-01",
                    repeat=0,
                    chosen=DEVELOPMENT_TEMPLATE[0],
                    rejected=DEVELOPMENT_TEMPLATE[1],
                )
            row["diagnostic_protocol"] = protocol_version
            if screening_metrics:
                metric = screening_metrics[source["image_sha256"]]
                row.update(
                    {
                        "pair_id": screening_pair_ids[source["image_sha256"]],
                        "diagnostic_base_negative_terms": metric["negative_terms"],
                        "diagnostic_base_selected_terms": metric["selected_terms"],
                        "diagnostic_base_responses": metric["responses"],
                    }
                )
            rows.append(row)
    rows.sort(key=lambda row: (row["split"], row["match_block_id"], row["fairface_race"]))

    csv_path = output_dir / "diagnostic_dpo_pairs_frozen.csv"
    write_csv(csv_path, rows)
    forbidden_hashes = read_hashes(forbidden_path)
    gate = assess_overfit_dataset(rows, forbidden_hashes)
    gate_path = output_dir / "diagnostic_gate.json"
    write_json(gate_path, gate)
    if not gate["pass"]:
        raise ValueError("diagnostic gate failed: " + json.dumps(gate, sort_keys=True))

    manifest = {
        "status": "ready_for_diagnostic",
        "protocol_version": protocol_version,
        "non_confirmatory": True,
        "confirmatory_evaluation_opened": False,
        "intentional_train_development_image_overlap": True,
        "model_id": "google/gemma-3-12b-it",
        "model_revision": args.model_revision,
        "source_csv": str(source_csv),
        "source_csv_sha256": sha256_file(source_csv),
        "forbidden_eval_hashes": str(forbidden_path),
        "forbidden_eval_hashes_sha256": sha256_file(forbidden_path),
        "selected_source_blocks": list(selected_block_ids),
        "screening_responses_csv": (
            str(screening_responses_path) if screening_responses_path else None
        ),
        "screening_responses_csv_sha256": (
            sha256_file(screening_responses_path) if screening_responses_path else None
        ),
        "train_templates": [
            {"chosen": chosen, "rejected": rejected} for chosen, rejected in TRAIN_TEMPLATES
        ],
        "development_template": {
            "chosen": DEVELOPMENT_TEMPLATE[0],
            "rejected": DEVELOPMENT_TEMPLATE[1],
        },
        "outputs": {
            csv_path.name: {"sha256": sha256_file(csv_path), "rows": len(rows)},
            gate_path.name: {"sha256": sha256_file(gate_path)},
        },
        "images": [
            {"sha256": image_hash, "path": copied_paths[image_hash]}
            for image_hash in sorted(copied_paths)
        ],
    }
    manifest_path = output_dir / "diagnostic_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
