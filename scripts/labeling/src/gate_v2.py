"""Two-pass confidence gate, v2 rules: dominant-valence agreement instead of
expression-class equality, and graded/mixed chosen triples allowed.

Resolution rule per image (train/development only):
  * both v2 passes produced a valid draft;
  * chosen triples agree on dominant valence (a pure-pole-vs-pure-pole
    disagreement or opposite-dominant mixes count as disagreement);
  * expression classes do not directly contradict (positive vs negative);
  * both self-confidences >= MIN_CONFIDENCE;
  * chosen word SETS identical -> auto-resolve
    |chosen A ∩ chosen B| == 2   -> adjudication queue
    otherwise                    -> unresolved
  * a PURE-pole chosen is only allowed when both passes rate the face the
    same clear pole (positive+positive or negative+negative). A pure-pole
    chosen on a face either pass called neutral/uncertain is sent to
    adjudication as pure_pole_on_ambiguous instead of auto-resolving.

Flag handling changed from v1: neutral_vocabulary_gap / ambiguous_expression
no longer block — under a valence-only vocabulary, a confident mixed triple
is the correct answer for those faces, and v2 policy explicitly asks for it.
Flags are kept on records as metadata; the hard checks are dominant-valence
agreement + confidence.

Paired consistency audit is unchanged: match_set_id groups are reported when
resolved members disagree on chosen (audit only). Export adds the hard rule
that partners share dominant valence.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import common
import gate  # v1 helpers (word_set, JsonlStore loading)
import policy_v2

MIN_CONFIDENCE = 0.8
ADJC_SENTINEL = "adjudication"

CLEAR_POLES = {"positive", "negative"}


def load_pass(tag: str) -> dict[str, dict]:
    store = common.JsonlStore(common.PRIVATE / f"drafts_v2_{tag}.jsonl",
                              key="image_id")
    return {k: v for k, v in ((r["image_id"], r) for r in store.values())}


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
    # Direct contradiction: one pass says clearly positive, other clearly negative.
    if ({a["teacher_expression"], b["teacher_expression"]}
            == CLEAR_POLES):
        info["reason"] = "expression_disagreement"
        return "unresolved", info
    ca, cb = gate.word_set(a["chosen"]), gate.word_set(b["chosen"])
    # Dominant-valence disagreement: the teachers split on which pole the face
    # supports. These are exactly the ambiguous faces whose honest label is a
    # mixed triple — route them to disagreement adjudication rather than
    # dropping them, or every pure-pole trio survives and nothing mixed exists.
    sa, sb = policy_v2.valence_sign(ca), policy_v2.valence_sign(cb)
    if sa != sb:
        info["reason"] = "valence_disagreement"
        return "disagreement", info
    if min(a["subjective_confidence"], b["subjective_confidence"]) < MIN_CONFIDENCE:
        info["reason"] = "low_confidence"
        return "unresolved", info
    both_clear = (a["teacher_expression"] in CLEAR_POLES
                  and b["teacher_expression"] == a["teacher_expression"])
    if ca == cb:
        if policy_v2.is_pure_pole(ca) and not both_clear:
            info["reason"] = "pure_pole_on_ambiguous"
            return ADJC_SENTINEL, info
        chosen = sorted(ca)
        hi = a if a["subjective_confidence"] >= b["subjective_confidence"] else b
        lo = b if hi is a else a
        rejected = gate.word_set(hi["rejected"]) \
            if gate.word_set(hi["rejected"]) != ca else gate.word_set(lo["rejected"])
        if rejected == ca:
            info["reason"] = "rejected_equals_chosen"
            return ADJC_SENTINEL, info
        info["chosen"] = ", ".join(chosen)
        info["rejected"] = ", ".join(sorted(rejected))
        return "resolved", info
    if len(ca & cb) == 2:
        info["reason"] = "chosen_overlap_2of3"
        return ADJC_SENTINEL, info
    # Same dominant valence but disjoint or 1-word-overlap triples: honest
    # disagreement over word choice — adjudicate, do not drop.
    info["reason"] = "chosen_disagreement"
    return "disagreement", info


def run() -> dict:
    rows = common.labelable_rows()
    a_map, b_map = load_pass("A"), load_pass("B")
    resolved, adjudication, disagreement, unresolved, missing = [], [], [], [], []
    for row in rows:
        status, info = gate_row(row, a_map.get(row["image_id"]),
                                b_map.get(row["image_id"]))
        {"resolved": resolved, "adjudication": adjudication,
         "disagreement": disagreement,
         "unresolved": unresolved, "missing": missing}[status].append(info)

    by = lambda items, k: dict(Counter(i[k] for i in items))
    mix = Counter()
    for i in resolved:
        ws = gate.word_set(i["chosen"])
        pos = len(ws & policy_v2.POSITIVE_IAT_SET)
        mix[f"{pos}pos{3 - pos}neg"] += 1
    report = {
        "policy": policy_v2.POLICY_ID,
        "counts": {"labelable": len(rows), "resolved": len(resolved),
                   "adjudication": len(adjudication),
                   "disagreement_adjudication": len(disagreement),
                   "unresolved": len(unresolved), "missing": len(missing)},
        "unresolved_reasons": dict(Counter(i["reason"] for i in unresolved)),
        "adjudication_reasons": dict(Counter(i["reason"] for i in adjudication)),
        "disagreement_reasons": dict(Counter(i["reason"] for i in disagreement)),
        "resolved_by_split": by(resolved, "split"),
        "resolved_by_dataset": by(resolved, "dataset"),
        "resolved_by_race": by(resolved, "source_race"),
        "resolved_by_expression_class": by(resolved, "expression_class"),
        "resolved_chosen_valence_mix": dict(mix),
        "rate_by_race": {},
    }
    totals = Counter(r.get("source_race") for r in rows)
    for race, n in totals.items():
        res = sum(1 for i in resolved if i.get("source_race") == race)
        report["rate_by_race"][race] = round(res / n, 4) if n else None

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

    with (common.PRIVATE / "resolved_v2.jsonl").open("w") as fh:
        for i in resolved:
            fh.write(json.dumps(i, sort_keys=True) + "\n")
    with (common.PRIVATE / "adjudication_queue_v2.jsonl").open("w") as fh:
        for i in adjudication:
            fh.write(json.dumps(i, sort_keys=True) + "\n")
    with (common.PRIVATE / "adjudication_queue_v2_disagreement.jsonl").open("w") as fh:
        for i in disagreement:
            fh.write(json.dumps(i, sort_keys=True) + "\n")
    common.write_json(common.PRIVATE / "gate_report_v2.json", report)
    print(json.dumps(report["counts"], indent=2))
    print("valence mix:", report["resolved_chosen_valence_mix"])
    print("rate_by_race:", report["rate_by_race"])
    return report


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    run()
