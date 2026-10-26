"""Aggregate current-protocol label support without publishing images or evidence."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import candidates_v3 as candidates
import common
import export_v3
import policy_v3


def audit(rows, records):
    records = [r for r in candidates.active_policy_records(records)
               if r.get("teacher_profile", "openai") == candidates.PROFILE]
    supported = candidates.approved_candidates(rows, records)
    teachers = {}
    for tag, model in candidates.TEACHERS.items():
        attempts = [r for r in records if r["tag"] == tag]
        valid = [r for r in attempts if r["status"] == "valid"]
        teachers[tag] = {
            "model": model, "calls": len(attempts),
            "statuses": dict(Counter(r["status"] for r in attempts)),
            "invalid_reasons": dict(Counter(
                r.get("error", "unknown") for r in attempts if r["status"] == "invalid")),
            "valid_empty_candidate_calls": sum(not r["value"]["candidates"] for r in valid),
            "valid_blocked_calls": sum(bool(r["value"]["review_flags"]) for r in valid),
            "valid_calls_with_notes": sum(bool(r["value"]["review_notes"]) for r in valid),
            "unblocked_confident_word_mentions": dict(Counter(
                c["word"] for r in valid if not r["value"]["review_flags"]
                for c in r["value"]["candidates"] if c["confidence"] >= 0.8)),
        }
    splits = {}
    for split in common.LABEL_SPLITS:
        members = [r for r in rows if r["split"] == split]
        counts = {}
        for race in ("Black", "White"):
            counts[race] = dict(Counter(
                w for r in members if r["source_race"] == race
                for w in supported.get(r["image_id"], {}).get("words", [])))
        union = sorted(set(counts["Black"]) | set(counts["White"]))
        splits[split] = {
            "images_with_supported_words": sum(r["image_id"] in supported for r in members),
            "images_with_at_least_three_supported_words": sum(
                len(supported.get(r["image_id"], {}).get("words", [])) >= 3 for r in members),
            "union_words": union, "union_count": len(union),
            "word_image_counts_by_race": counts,
        }
    variants = {}
    by_id = {r["image_id"]: r for r in rows}
    for n in (2, 3, 4):
        selected, dropped = export_v3.select_variants(rows, supported, n)
        dev_blocks = {by_id[i]["match_set_id"] for i in selected
                      if by_id[i]["split"] == "development"}
        retained = Counter(by_id[i]["split"] for i in selected)
        variants[str(n)] = {
            "retained_images_by_split": dict(retained),
            "development_blocks": len(dev_blocks),
            "drop_reasons": dict(Counter(r["reason"] for r in dropped)),
            "has_minimum_complete_blocks": bool(retained["train"] and len(dev_blocks) >= 3),
        }
    return {
        "policy_id": policy_v3.POLICY_ID, "policy_sha256": policy_v3.POLICY_SHA,
        "schema_sha256": policy_v3.SCHEMA_SHA, "teacher_profile": candidates.PROFILE,
        "confidence_threshold": 0.8, "scope": "All stored current-protocol calls",
        "teacher_statistics_are_call_counts_not_unique_image_counts": True,
        "teachers": teachers, "support_by_split": splits, "variants": variants,
        "test_used_for_labeling": False, "targets_are_ground_truth": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("openai", "bitdeer"), default="openai")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    candidates.configure(args.provider)
    result = audit(candidates.target_rows(), candidates.load_records())
    result["raw_responses_sha256"] = common.sha(candidates.STORE / "responses.jsonl")
    result["parent_manifest_sha256"] = common.sha(common.MANIFEST)
    common.write_json(args.out, result)
