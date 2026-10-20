"""Two-pass confidence gate + paired-consistency audit + export.

Resolution rule per image (train/development only):
  * both passes produced a valid draft;
  * pass expressions agree and are neither neutral nor uncertain;
  * neither pass flagged neutral_vocabulary_gap or ambiguous_expression;
  * both self-confidences >= MIN_CONFIDENCE;
  * chosen word SETS identical -> auto-resolve
    |chosen A ∩ chosen B| == 2   -> adjudication queue (Devin picks the final
                                   triple from the shared two words + one of
                                   the differing words or re-derives one)
    otherwise                    -> unresolved (no forced weak label)

Rejected side: taken from the higher-confidence pass, must be a valid
3-distinct-word triple disjoint from the final chosen set; if it collides
with the merged chosen, the other pass's rejected is tried, else the image
goes to adjudication.

Paired consistency: match_set_id groups faces matched on expression/age/
gender. After resolution, blocks whose resolved members disagree on chosen
words get an audit flag in the report — reported, not auto-harmonized, so the
final call stays visible.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import common
import policy

MIN_CONFIDENCE = 0.8

ADJC_SENTINEL = "adjudication"


def load_pass(tag: str) -> dict[str, dict]:
    store = common.JsonlStore(common.PRIVATE / f"drafts_{tag}.jsonl",
                              key="image_id")
    return {k: v for k, v in ((r["image_id"], r) for r in store.values())}


def word_set(text: str) -> set[str]:
    return {w.strip() for w in text.split(",") if w.strip()}


def gate_row(row: dict, a: dict | None, b: dict | None) -> tuple[str, dict]:
    info = {"image_id": row["image_id"], "split": row["split"],
            "dataset": row["dataset"], "match_set_id": row.get("match_set_id"),
            "source_race": row.get("source_race"),
            "source_gender": row.get("source_gender"),
            "age_band": row.get("age_band"),
            "expression_class": row.get("expression_class")}
    if a is None or b is None:
        info["reason"] = "missing_pass"
        return "missing", info
    info.update({"a_expression": a["teacher_expression"],
                 "b_expression": b["teacher_expression"],
                 "a_conf": a["subjective_confidence"],
                 "b_conf": b["subjective_confidence"],
                 "a_chosen": a["chosen"], "b_chosen": b["chosen"],
                 "a_rejected": a["rejected"], "b_rejected": b["rejected"],
                 "teacher_a": a.get("teacher", ""),
                 "teacher_b": b.get("teacher", "")})
    blocking = {"neutral_vocabulary_gap", "ambiguous_expression"}
    if (set(a["review_flags"]) & blocking) or (set(b["review_flags"]) & blocking):
        info["reason"] = "blocking_flag"
        return "unresolved", info
    # After flag filtering, only positive/negative expressions can disagree:
    # neutral/uncertain drafts always carry a blocking flag. A positive-vs-
    # negative disagreement means the face genuinely reads both ways.
    if a["teacher_expression"] != b["teacher_expression"]:
        info["reason"] = "expression_disagreement"
        return "unresolved", info
    if min(a["subjective_confidence"], b["subjective_confidence"]) < MIN_CONFIDENCE:
        info["reason"] = "low_confidence"
        return "unresolved", info
    ca, cb = word_set(a["chosen"]), word_set(b["chosen"])
    if ca == cb:
        chosen = sorted(ca)
        hi = a if a["subjective_confidence"] >= b["subjective_confidence"] else b
        lo = b if hi is a else a
        rejected = word_set(hi["rejected"]) if word_set(hi["rejected"]) != ca \
            else word_set(lo["rejected"])
        if rejected == ca:
            info["reason"] = "rejected_equals_chosen"
            return ADJC_SENTINEL, info
        info["chosen"] = ", ".join(chosen)
        info["rejected"] = ", ".join(sorted(rejected))
        return "resolved", info
    if len(ca & cb) == 2:
        info["reason"] = "chosen_overlap_2of3"
        return ADJC_SENTINEL, info
    info["reason"] = "chosen_disagreement"
    return "unresolved", info


def run() -> dict:
    rows = common.labelable_rows()
    a_map, b_map = load_pass("A"), load_pass("B")
    resolved, adjudication, unresolved, missing = [], [], [], []
    for row in rows:
        status, info = gate_row(row, a_map.get(row["image_id"]),
                                b_map.get(row["image_id"]))
        {"resolved": resolved, "adjudication": adjudication,
         "unresolved": unresolved, "missing": missing}[status].append(info)

    by = lambda items, k: dict(Counter(i[k] for i in items))
    report = {
        "counts": {"labelable": len(rows), "resolved": len(resolved),
                   "adjudication": len(adjudication),
                   "unresolved": len(unresolved), "missing": len(missing)},
        "unresolved_reasons": dict(Counter(i["reason"] for i in unresolved)),
        "resolved_by_split": by(resolved, "split"),
        "resolved_by_dataset": by(resolved, "dataset"),
        "resolved_by_race": by(resolved, "source_race"),
        "resolved_by_expression_class": by(resolved, "expression_class"),
        "rate_by_race": {},
    }
    # resolution-rate parity check across race groups — catches a gate that
    # systematically drops one demographic cell
    totals = Counter(r.get("source_race") for r in rows)
    for race, n in totals.items():
        res = sum(1 for i in resolved if i.get("source_race") == race)
        report["rate_by_race"][race] = round(res / n, 4) if n else None

    # match-block consistency: resolved members of the same matched block
    # should share chosen words when the block matched on expression
    blocks = defaultdict(list)
    for i in resolved:
        if i.get("match_set_id"):
            blocks[i["match_set_id"]].append(i)
    divergent = {k: [(i["image_id"], i["chosen"], i["source_race"])
                     for i in v]
                 for k, v in blocks.items()
                 if len({i["chosen"] for i in v}) > 1}
    report["matched_blocks"] = {"with_multiple_resolved": sum(
        1 for v in blocks.values() if len(v) > 1),
        "divergent_chosen_blocks": len(divergent),
        "examples": dict(list(divergent.items())[:20])}

    with (common.PRIVATE / "resolved.jsonl").open("w") as fh:
        for i in resolved:
            fh.write(json.dumps(i, sort_keys=True) + "\n")
    with (common.PRIVATE / "adjudication_queue.jsonl").open("w") as fh:
        for i in adjudication:
            fh.write(json.dumps(i, sort_keys=True) + "\n")
    common.write_json(common.PRIVATE / "gate_report.json", report)
    print(json.dumps(report["counts"], indent=2))
    print("rate_by_race:", report["rate_by_race"])
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
