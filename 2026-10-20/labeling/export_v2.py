"""Export v2 gated targets: same schema as export.py plus two new rules.

1. Pair balance (the debiasing anchor): match_set_id groups faces matched on
   expression/age/gender across race. All exported members of a match set must
   share the same dominant chosen valence; conflicting sets are dropped whole
   and audited — a pair labeled opposite directions reteaches race->valence.

2. Coverage selection (Peyton's variety objective): when a chosen triple is
   over the per-triple cap, keep the members whose words are RAREST in the
   cohort (seeded tiebreak) instead of an arbitrary prefix — the export keeps
   maximally distinct vocabulary. Audit reports distinct chosen words,
   coverage vs the 50 allowed words, chosen-word entropy and the valence-mix
   histogram.

Reads resolved_v2.jsonl + adjudication_decisions_v2.jsonl, writes
curated_iat_targets_v2.json + selected_images_gated_v2.csv +
active_dataset_gated_v2_draft.json + curated_iat_targets_v2_audit.json.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import common
import export as export_v1
import policy
import policy_v2

CAP_SHUFFLE_SEED = 20261021

TARGET_VERSION = "iat-gated-v2-mixedvalence"


def load_resolved_v2() -> list[dict]:
    out = []
    path = common.PRIVATE / "resolved_v2.jsonl"
    if path.exists():
        for line in path.open():
            if line.strip():
                out.append(json.loads(line))
    adj = common.PRIVATE / "adjudication_decisions_v2.jsonl"
    if adj.exists():
        for line in adj.open():
            if line.strip():
                rec = json.loads(line)
                if rec.get("decision") == "approved":
                    out.append(rec)
    return out


def word_freq(records) -> Counter:
    return Counter(w for r in records
                   for w in policy.word_set(r["chosen"]))


def pair_balance(records, manifest):
    """Drop whole match sets whose members disagree on dominant valence."""
    sets = defaultdict(list)
    for r in records:
        sets[r.get("match_set_id") or r["image_id"]].append(r)
    kept, violations = [], []
    for sid, members in sets.items():
        signs = {policy_v2.valence_sign(policy.word_set(m["chosen"]))
                 for m in members}
        if len(signs) > 1:
            violations.append({
                "match_set_id": sid,
                "members": [{"image_id": m["image_id"],
                             "chosen": m["chosen"],
                             "race": manifest[m["image_id"]]["source_race"]}
                            for m in members]})
            continue
        kept.extend(members)
    return kept, violations


def export(name: str = "curated_iat_targets_v2",
           cap_share: float = 0.08) -> dict:
    manifest = {r["image_id"]: r for r in common.load_manifest()}
    resolved = load_resolved_v2()
    auth_sha = export_v1.approval_sha()
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
            "draft_source": "two_pass_gated_teachers_v2",
            "approval_authorization_sha256": auth_sha,
            "target_provenance": (
                f"{TARGET_VERSION}; passA={item.get('teacher_a','')}; "
                f"passB={item.get('teacher_b','')}; gate=valence+conf>={0.8}"),
            "dataset": row["dataset"],
            "match_set_id": row.get("match_set_id", ""),
        }
        records.append(rec)

    # pair-balance: drop match sets whose members disagree on dominant valence
    records, pair_violations = pair_balance(records, manifest)

    # Variety cap + coverage selection: within each over-cap triple group,
    # keep the members whose chosen words are rarest cohort-wide.
    freqs = word_freq(records)
    rng = random.Random(CAP_SHUFFLE_SEED)
    by_triple = defaultdict(list)
    for r in records:
        by_triple[r["chosen"]].append(r)
    for group in by_triple.values():
        rng.shuffle(group)
        group.sort(key=lambda r: -sum(1.0 / freqs[w]
                                      for w in policy.word_set(r["chosen"])))
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

    mix = Counter()
    for r in records:
        pos = len(policy.word_set(r["chosen"]) & policy_v2.POSITIVE_IAT_SET)
        mix[f"{pos}pos{3 - pos}neg"] += 1
    allowed = policy_v2.WORD_SET - policy.FORBIDDEN_CHOSEN
    used = {w for r in records for w in policy.word_set(r["chosen"])}
    counts = Counter(w for r in records for w in policy.word_set(r["chosen"]))
    total = sum(counts.values()) or 1
    entropy = -sum(c / total * __import__("math").log2(c / total)
                   for c in counts.values()) / __import__("math").log2(max(2, len(counts)))

    audit = {
        "policy": policy_v2.POLICY_ID,
        "records": len(records),
        "pair_balance_violations": pair_violations,
        "dropped_match_sets": len(pair_violations),
        "demoted_by_variety_cap": [r["image_id"] for r in demoted],
        "cap_share": cap_share,
        "cap_per_triple": cap,
        "chosen_valence_mix": dict(mix),
        "chosen_words": dict(counts),
        "chosen_triples": dict(Counter(r["chosen"] for r in records)),
        "distinct_chosen_triples": len({r["chosen"] for r in records}),
        "distinct_rejected_triples": len({r["rejected"] for r in records}),
        "distinct_chosen_words": len(used),
        "allowed_words": len(allowed),
        "word_coverage_rate": round(len(used) / len(allowed), 4) if allowed else 0,
        "chosen_word_normalized_entropy": round(entropy, 4),
        "top_triple_share": (round(
            max(Counter(r["chosen"] for r in records).values())
            / len(records), 4) if records else 0),
        "errors": errors,
        "by_split": dict(Counter(r["split"] for r in records)),
        "by_dataset": dict(Counter(r["dataset"] for r in records)),
    }
    by_race = defaultdict(lambda: [0, 0])
    allrows = {r["image_id"]: r for r in common.labelable_rows()}
    for iid, r in allrows.items():
        by_race[r["source_race"]][1] += 1
    for r in records:
        by_race[manifest[r["image_id"]]["source_race"]][0] += 1
    audit["coverage_by_race"] = {
        k: {"resolved": v[0], "total": v[1],
            "rate": round(v[0] / v[1], 4)} for k, v in sorted(by_race.items())}

    out_path = common.WORKDIR / "data" / "private" / f"{name}.json"
    if records and not errors:
        common.write_json(out_path, records)
        emit_gated_cohort_v2(records, manifest, name)
    audit_path = common.PRIVATE / f"{name}_audit.json"
    common.write_json(audit_path, audit)
    print(json.dumps({k: v for k, v in audit.items()
                      if k not in ("chosen_words", "chosen_triples",
                                   "pair_balance_violations")}, indent=2))
    print("pair_balance_violations:", len(pair_violations))
    if errors:
        print(f"NOT EXPORTED: {len(errors)} errors")
    return audit


def emit_gated_cohort_v2(records: list[dict], manifest: dict,
                         name: str) -> None:
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
    gated_csv = common.WORKDIR / "data" / f"selected_images_gated_v2.csv"
    with gated_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    counts = Counter(r["split"] for r in kept)
    datasets = Counter(r["dataset"] for r in kept)
    config = {
        "dataset_id": "image-curation-2026-09-29-v2",
        "cohort_note": "confidence-gated two-pass mixed-valence subset",
        "parent_manifest_path": "2026-09-29/data/selected_images.csv",
        "manifest_path": "2026-10-20/data/selected_images_gated_v2.csv",
        "manifest_sha256": hashlib.sha256(gated_csv.read_bytes()).hexdigest(),
        "asset_root": "2026-09-29/private/dataset",
        "targets_path": f"2026-10-20/data/private/{name}.json",
        "dataset_counts": dict(datasets),
        "split_counts": dict(counts),
        "preserve_published_splits": True,
        "target_splits": ["train", "development"],
        "test_preference_generation_allowed": False,
        "profile_targets_used": 0,
        "training_inputs": {"sft": ["image", "prompt", "chosen"],
                            "orpo": ["image", "prompt", "chosen", "rejected"],
                            "kto": ["image", "prompt", "chosen", "rejected"]},
        "primary_benchmark": "2026-09-11/data/iat12/manifest.csv",
        "benchmark_images_training_allowed": False,
        "additional_test_pool": "Reserved; do not activate a new benchmark without defining its protocol",
        "superseded_pairs": "2026-10-03/data/pairs.csv",
        "target_policy": ("Two-pass gated AI approval v2: dominant-valence "
                          "agreement, mixed-valence chosen for ambiguous faces, "
                          "match-set pair balance, coverage-maximizing export"),
    }
    common.write_json(
        common.WORKDIR / "config" / "active_dataset_gated_v2_draft.json", config)
    print(f"gated cohort: {len(kept)} manifest rows -> {gated_csv}")
    print("draft config -> 2026-10-20/config/active_dataset_gated_v2_draft.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="curated_iat_targets_v2")
    ap.add_argument("--cap-share", type=float, default=0.08)
    args = ap.parse_args()
    export(args.name, args.cap_share)
