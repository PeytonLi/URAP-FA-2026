"""Export approved gated targets in the curated_iat_targets.json schema and
audit them. The output binds records to the manifest's declared image_sha256
(upstream PNG hash); pixel-verified Linux-rebuilt files carry the same decoded
content and are checked via pixel_sha256 separately."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import common
import policy

CAP_SHUFFLE_SEED = 20261020

TARGET_VERSION = "iat-gated-v1"


def approval_sha() -> str:
    """SHA-256 fingerprint of the recorded two-pass gating authorization in
    2026-10-03/config/prime_budget.json (mirrors curated.ai_approval_authorization)."""
    import hashlib
    permission = json.loads((common.REPO / "2026-10-03/config/prime_budget.json").read_text())
    auth = permission.get("ai_target_approval_gated_two_pass")
    if not auth or auth.get("authorized") is not True:
        raise SystemExit("recorded gated two-pass AI-approval authorization missing")
    return hashlib.sha256(json.dumps(auth, sort_keys=True).encode()).hexdigest()


def load_resolved() -> list[dict]:
    out = []
    path = common.PRIVATE / "resolved.jsonl"
    if path.exists():
        for line in path.open():
            if line.strip():
                out.append(json.loads(line))
    # merge adjudication decisions (they carry the final chosen/rejected)
    adj = common.PRIVATE / "adjudication_decisions.jsonl"
    if adj.exists():
        for line in adj.open():
            if line.strip():
                rec = json.loads(line)
                if rec.get("decision") == "approved":
                    out.append(rec)
    return out


def export(name: str = "curated_iat_targets",
           cap_share: float = 0.08) -> dict:
    manifest = {r["image_id"]: r for r in common.load_manifest()}
    resolved = load_resolved()
    auth_sha = approval_sha()
    errors = []
    records = []
    for item in resolved:
        row = manifest.get(item["image_id"])
        if row is None:
            errors.append((item["image_id"], "not_in_manifest"))
            continue
        if row["split"] not in common.LABEL_SPLITS:
            errors.append((item["image_id"], "forbidden_split"))
            continue
        chosen = ", ".join(sorted(policy.word_set(item["chosen"])))
        rejected = ", ".join(sorted(policy.word_set(item["rejected"])))
        if not policy.exact_target(chosen) or not policy.exact_target(rejected):
            errors.append((item["image_id"], "bad_triple"))
            continue
        if policy.word_set(chosen) == policy.word_set(rejected):
            errors.append((item["image_id"], "chosen_equals_rejected_set"))
            continue
        rec = {
            "image_id": row["image_id"],
            "image_sha256": row["image_sha256"],
            "image_pixel_sha256": row.get("pixel_sha256", ""),
            "split": row["split"],
            "prompt": policy.IAT_PROMPT,
            "chosen": chosen,
            "rejected": rejected,
            "review_status": "approved",
            "reviewer": "devin-ai-gated-two-pass",
            "reviewer_kind": "ai",
            "draft_source": "two_pass_gated_teachers",
            "approval_authorization_sha256": auth_sha,
            "target_provenance": (
                f"{TARGET_VERSION}; passA={item.get('teacher_a','')}; "
                f"passB={item.get('teacher_b','')}; gate=identity+conf>={0.8}"),
            "dataset": row["dataset"],
            "match_set_id": row.get("match_set_id", ""),
        }
        records.append(rec)

    # Variety cap: no single chosen triple may cover more than cap_share of
    # the EXPORTED cohort (post-demotion). Within each triple, records are
    # kept in a seeded-random order so demotions are unbiased across
    # datasets/races — deterministic, not image_id-sorted. Capped-out records
    # are reported, not exported — a dominant generic triple is the October
    # collapse mode.
    by_triple = defaultdict(list)
    for r in records:
        by_triple[r["chosen"]].append(r)
    rng = random.Random(CAP_SHUFFLE_SEED)
    for group in by_triple.values():
        rng.shuffle(group)
    cap = max(3, int(cap_share * len(records)))
    while cap > 1:
        kept_count = sum(min(len(group), cap) for group in by_triple.values())
        if kept_count and cap / kept_count <= cap_share:
            break
        cap -= 1
    kept, demoted = [], []
    for triple, group in by_triple.items():
        kept.extend(group[:cap])
        demoted.extend(group[cap:])
    records = kept
    audit = {
        "records": len(records),
        "demoted_by_variety_cap": [r["image_id"] for r in demoted],
        "cap_share": cap_share,
        "cap_semantics": "max share of any chosen triple in the exported cohort",
        "cap_per_triple": cap,
        "errors": errors,
        "by_split": dict(Counter(r["split"] for r in records)),
        "by_dataset": dict(Counter(r["dataset"] for r in records)),
        "chosen_words": dict(Counter(
            w for r in records for w in policy.word_set(r["chosen"]))),
        "chosen_triples": dict(Counter(r["chosen"] for r in records)),
        "distinct_chosen_triples": len({r["chosen"] for r in records}),
        "distinct_rejected_triples": len({r["rejected"] for r in records}),
        "distinct_chosen_words": len({
            w for r in records for w in policy.word_set(r["chosen"])}),
        "top_triple_share": (round(
            max(Counter(r["chosen"] for r in records).values())
            / len(records), 4) if records else 0),
    }
    # demographic resolution coverage
    by_race = defaultdict(lambda: [0, 0])
    allrows = {r["image_id"]: r for r in common.labelable_rows()}
    for iid, r in allrows.items():
        by_race[r["source_race"]][1] += 1
    for r in records:
        by_race[manifest[r["image_id"]]["source_race"]][0] += 1
    audit["coverage_by_race"] = {
        k: {"resolved": v[0], "total": v[1],
            "rate": round(v[0] / v[1], 4)} for k, v in sorted(by_race.items())}
    by_gender = defaultdict(lambda: [0, 0])
    for iid, r in allrows.items():
        by_gender[r["source_gender"]][1] += 1
    for r in records:
        by_gender[manifest[r["image_id"]]["source_gender"]][0] += 1
    audit["coverage_by_gender"] = {
        k: {"resolved": v[0], "total": v[1],
            "rate": round(v[0] / v[1], 4)} for k, v in sorted(by_gender.items())}

    out_path = common.WORKDIR / "data" / "private" / f"{name}.json"
    if records and not errors:
        common.write_json(out_path, records)
        emit_gated_cohort(records, manifest)
    audit_path = common.PRIVATE / f"{name}_audit.json"
    common.write_json(audit_path, audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "chosen_words"},
                     indent=2))
    if errors:
        print(f"NOT EXPORTED: {len(errors)} errors")
    return audit


def emit_gated_cohort(records: list[dict], manifest: dict) -> None:
    """Declared gated cohort: a filtered manifest containing only rows with
    approved targets, plus a draft active_dataset.json pointing at it. The
    frozen September-29 manifest is never modified. Activation copies the
    draft config to 2026-10-03/config/active_dataset.json."""
    import csv as _csv
    import hashlib
    source = common.MANIFEST
    resolved_ids = {r["image_id"] for r in records}
    fieldnames = None
    kept = []
    with source.open(newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for row in reader:
            if row["image_id"] in resolved_ids:
                kept.append(row)
    gated_csv = common.WORKDIR / "data" / "selected_images_gated.csv"
    gated_csv.parent.mkdir(parents=True, exist_ok=True)
    with gated_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    counts = Counter(r["split"] for r in kept)
    datasets = Counter(r["dataset"] for r in kept)
    config = {
        "dataset_id": "image-curation-2026-09-29-v1",
        "cohort_note": "confidence-gated two-pass subset of the frozen pool",
        "parent_manifest_path": "2026-09-29/data/selected_images.csv",
        "manifest_path": "2026-10-20/data/selected_images_gated.csv",
        "manifest_sha256": hashlib.sha256(gated_csv.read_bytes()).hexdigest(),
        "asset_root": "2026-09-29/private/dataset",
        "targets_path": "2026-10-20/data/private/curated_iat_targets.json",
        "dataset_counts": dict(datasets),
        "split_counts": dict(counts),
        "preserve_published_splits": True,
        "target_splits": ["train", "development"],
        "test_preference_generation_allowed": False,
        "profile_targets_used": 0,
        "training_inputs": {"sft": ["image", "prompt", "chosen"],
                            "orpo": ["image", "prompt", "chosen", "rejected"]},
        "primary_benchmark": "2026-09-11/data/iat12/manifest.csv",
        "benchmark_images_training_allowed": False,
        "additional_test_pool": "Reserved; do not activate a new benchmark without defining its protocol",
        "superseded_pairs": "2026-10-03/data/pairs.csv",
        "target_policy": "Two-pass gated AI approval under recorded authorization; unresolved images carry no targets",
    }
    common.write_json(
        common.WORKDIR / "config" / "active_dataset_gated_draft.json", config)
    print(f"gated cohort: {len(kept)} manifest rows -> {gated_csv}")
    print("draft config -> 2026-10-20/config/active_dataset_gated_draft.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="curated_iat_targets")
    ap.add_argument("--cap-share", type=float, default=0.08)
    args = ap.parse_args()
    export(args.name, args.cap_share)
