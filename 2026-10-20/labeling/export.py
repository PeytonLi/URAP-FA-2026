"""Export approved gated targets in the curated_iat_targets.json schema and
audit them. The output binds records to the manifest's declared image_sha256
(upstream PNG hash); pixel-verified Linux-rebuilt files carry the same decoded
content and are checked via pixel_sha256 separately."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import common
import policy

TARGET_VERSION = "iat-gated-v1"


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


def export(name: str = "curated_iat_targets") -> dict:
    manifest = {r["image_id"]: r for r in common.load_manifest()}
    resolved = load_resolved()
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
        chosen, rejected = item["chosen"], item["rejected"]
        if not policy.exact_target(chosen) or not policy.exact_target(rejected):
            errors.append((item["image_id"], "bad_triple"))
            continue
        if set(policy.word_set(chosen)) & set(policy.word_set(rejected)):
            # chosen and rejected must be different words, not just reordered
            pass  # shared words allowed only if sets differ — enforce below
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
            "target_provenance": (
                f"{TARGET_VERSION}; passA={item.get('teacher_a','')}; "
                f"passB={item.get('teacher_b','')}; gate=identity+conf>={0.8}"),
            "dataset": row["dataset"],
            "match_set_id": row.get("match_set_id", ""),
        }
        records.append(rec)

    audit = {
        "records": len(records),
        "errors": errors,
        "by_split": dict(Counter(r["split"] for r in records)),
        "by_dataset": dict(Counter(r["dataset"] for r in records)),
        "chosen_words": dict(Counter(
            w for r in records for w in policy.word_set(r["chosen"]))),
        "distinct_chosen_triples": len({r["chosen"] for r in records}),
        "distinct_rejected_triples": len({r["rejected"] for r in records}),
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
    audit_path = common.PRIVATE / f"{name}_audit.json"
    common.write_json(audit_path, audit)
    print(json.dumps({k: v for k, v in audit.items() if k != "chosen_words"},
                     indent=2))
    if errors:
        print(f"NOT EXPORTED: {len(errors)} errors")
    return audit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="curated_iat_targets")
    args = ap.parse_args()
    export(args.name)
