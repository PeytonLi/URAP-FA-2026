"""V2 fan-out: shared-VM agents view the adjudication_queue_v2.jsonl items and
write decision JSONL, merged+validated into adjudication_decisions_v2.jsonl.

Queue item kinds (see gate_v2):
  * chosen_overlap_2of3   — teachers shared exactly 2 words; keep the shared 2
                            and pick the better differing word (v1 behavior).
  * pure_pole_on_ambiguous — teachers agreed on an identical pure-valence
                            triple on a face at least one called neutral/
                            uncertain; decide whether the pure-pole is honest
                            or a valence-MIXED triple (swap ONE word for an
                            opposite-pole word) better fits the visible
                            expression.
  * rejected_equals_chosen — rare; treat like chosen_overlap_2of3.
"""
import asyncio
import hashlib
import json
from pathlib import Path

REPO = Path("/home/ubuntu/repos/URAP-FA-2026")
QUEUE = REPO / "2026-10-20/data/private/target_review/adjudication_queue_v2.jsonl"
OUT_DIR = REPO / "2026-10-20/data/private/target_review"
MERGED = OUT_DIR / "adjudication_decisions_v2.jsonl"

IAT_WORDS = ("Love, Cheer, Friend, Pleasure, Adore, Cheerful, Friendship, "
             "Joyful, Smiling, Cherish, Excellent, Glad, Joyous, Spectacular, "
             "Appealing, Delight, Excitement, Laughing, Attractive, Fabulous, "
             "Glorious, Pleasing, Beautiful, Fantastic, Happy, Lovely, "
             "Terrific, Celebrate, Enjoy, Magnificent, Abuse, Grief, Poison, "
             "Sadness, Pain, Despise, Failure, Nasty, Angry, Detest, Horrible, "
             "Negative, Ugly, Dirty, Gross, Evil, Rotten, Annoy, Disaster, "
             "Horrific, Scorn, Awful, Disgust, Hate, Humiliate, Selfish, "
             "Tragic, Bothersome, Hatred, Hurtful, Sick")
FORBIDDEN_CHOSEN = {"Sick", "Poison", "Evil", "Selfish", "Abuse", "Failure",
                    "Dirty", "Rotten", "Hatred", "Hate", "Humiliate"}
WORD_SET = {w.strip() for w in IAT_WORDS.split(",")}

N_AGENTS = 8


def manifest_paths():
    import csv
    out = {}
    with (REPO / "2026-09-29/data/selected_images.csv").open() as fh:
        for row in csv.DictReader(fh):
            out[row["image_id"]] = row["image_path"]
    return out


SCHEMA = {"type": "object", "properties": {
    "decisions": {"type": "integer"},
    "skipped": {"type": "integer"},
    "notes": {"type": "string"}},
    "required": ["decisions", "skipped"]}


def agent_prompt(idx, items, paths):
    out_file = OUT_DIR / f"adjudication_decisions_v2_w{idx}.jsonl"
    lines = []
    for it in items:
        lines.append(json.dumps({
            "image_id": it["image_id"],
            "image_file": str(REPO / "2026-09-29/private/dataset" /
                              paths[it["image_id"]]),
            "reason": it.get("reason"),
            "split": it.get("split"), "dataset": it.get("dataset"),
            "match_set_id": it.get("match_set_id"),
            "source_race": it.get("source_race"),
            "source_gender": it.get("source_gender"),
            "expression_class": it.get("expression_class"),
            "teacher_a": it.get("teacher_a"),
            "teacher_b": it.get("teacher_b"),
            "a_chosen": it["a_chosen"], "b_chosen": it["b_chosen"],
            "a_rejected": it["a_rejected"], "b_rejected": it["b_rejected"],
            "a_expression": it["a_expression"],
            "b_expression": it["b_expression"],
            "a_conf": it["a_conf"], "b_conf": it["b_conf"],
        }, sort_keys=True))
    return f"""You are adjudicating labeling items for a face-expression
word-choice study (v2). Two independent teachers each picked 3 words from a
fixed 61-word vocabulary for the same face photo. Your job: VIEW each image
and pick the final chosen triple.

Vocabulary (use ONLY these words, exact casing):
{IAT_WORDS}

Each item has a "reason" field telling you why it needs a decision:

* "chosen_overlap_2of3": the teachers shared exactly 2 words and disagreed on
  the third. Final chosen = the 2 shared words + whichever differing word
  better fits the visible expression. Rarely, if neither fits, pick a
  different vocabulary word — say why in "reason".

* "pure_pole_on_ambiguous": BOTH teachers agreed on the same 3 words, but all
  three sit on one valence pole (all-positive or all-negative) while at least
  one teacher rated the face neutral/ambiguous. Look at the face:
  - if the visible expression clearly supports the agreed triple, keep it as
    chosen verbatim (a clear pole is honest on a clear face);
  - if the face is mild, ambiguous or neutral-looking, a valence-MIXED triple
    is the honest encoding: keep the 2 best-fitting agreed words and swap the
    weakest one for ONE word from the opposite pole (e.g. an ambiguous stern
    face agreed as "Angry, Sadness, Pain" could become
    "Sadness, Pain, Pleasing" only if a softening is genuinely visible —
    do not mix just to mix).

* "rejected_equals_chosen": handle like chosen_overlap_2of3 — keep the agreed
  chosen words if honest; pick any disjoint valid rejected triple.

Rules for every item:
1. Read the image file with your read tool and judge the face's visible
   expression only (mouth corners, lips, brow, eyes). Do NOT infer race,
   identity, health or personality.
2. Chosen = 3 DISTINCT vocabulary words, sharing at least 2 words with one of
   the teachers' chosen triples, containing NONE of: {sorted(FORBIDDEN_CHOSEN)}.
3. Rejected = one pass's rejected triple that shares NO word with your chosen
   (prefer the higher-confidence pass's). If both collide, write any valid
   incongruent 3-distinct-word vocabulary triple that disagrees with the
   visible expression.
4. Append ONE JSON line per image to {out_file} (create parent dirs if
   needed), fields exactly:
   {{"image_id","split","dataset","match_set_id","source_race","source_gender",
   "expression_class","a_chosen","b_chosen","a_rejected","b_rejected",
   "teacher_a","teacher_b","chosen","rejected","decision":"approved",
   "adjudicator":"devin-workflow-v2-w{idx}","reason":"<under 25 words>"}}
   Copy split/dataset/match_set_id/source_race/source_gender/expression_class/
   a_*/b_*/teacher_* fields from the item JSON verbatim. chosen/rejected are
   comma-space joined strings like "Happy, Smiling, Cheerful".
5. If the image file cannot be read, skip the item (do not guess) and count
   it in "skipped".

Items ({len(items)}):
""" + "\n".join(lines)


def valid_decision(item, rec):
    if rec.get("decision") != "approved":
        return False, "not_approved"
    chosen = {w.strip() for w in str(rec.get("chosen", "")).split(",")}
    rejected = {w.strip() for w in str(rec.get("rejected", "")).split(",")}
    if len(chosen) != 3 or len(rejected) != 3:
        return False, "not_3_distinct"
    if not chosen <= WORD_SET or not rejected <= WORD_SET:
        return False, "off_vocabulary"
    if chosen & FORBIDDEN_CHOSEN:
        return False, "forbidden_chosen"
    if chosen == rejected or not chosen.isdisjoint(rejected):
        return False, "chosen_rejected_overlap"
    ca = {w.strip() for w in item["a_chosen"].split(",")}
    cb = {w.strip() for w in item["b_chosen"].split(",")}
    if len(ca & chosen) < 2 and len(cb & chosen) < 2:
        return False, "drifted_from_both_passes"
    return True, ""


async def main():
    await register_workflow({
        "name": "iat-adjudication-v2-fanout",
        "description": "Shared-VM agents view v2-queued face images and resolve "
                       "2/3 overlaps + pure-pole-on-ambiguous items into targets",
        "product": "URAP-FA-2026 gated labeling v2",
        "soft_time_limit_minutes": 30,
        "subscribe": "none",
        "phases": [{"title": "adjudicate",
                    "detail": "view queued images, write v2 decision JSONL",
                    "count": N_AGENTS}],
    })
    items = [json.loads(line) for line in QUEUE.open() if line.strip()]
    log(f"adjudication v2 queue: {len(items)} items across {N_AGENTS} agents")
    chunks = [items[i::N_AGENTS] for i in range(N_AGENTS)]
    paths = manifest_paths()
    results = await parallel([
        (lambda i=i, c=c: agent(
            agent_prompt(i, c, paths),
            phase="adjudicate",
            schema=SCHEMA,
            label=f"adjudicate-v2-w{i}",
            vm_mode="shared",   # the images exist only on this VM
            soft_time_limit_minutes=30))
        for i, c in enumerate(chunks)])
    merged, rejected_decisions = [], []
    seen = set()
    item_map = {it["image_id"]: it for it in items}
    for i in range(N_AGENTS):
        path = OUT_DIR / f"adjudication_decisions_v2_w{i}.jsonl"
        if not path.exists():
            log(f"w{i}: no decisions file")
            continue
        for line in path.open():
            if not line.strip():
                continue
            rec = json.loads(line)
            iid = rec.get("image_id")
            if iid in seen or iid not in item_map:
                rejected_decisions.append((iid, "dup_or_unknown"))
                continue
            ok, why = valid_decision(item_map[iid], rec)
            if not ok:
                rejected_decisions.append((iid, why))
                continue
            seen.add(iid)
            merged.append(rec)
    with MERGED.open("w") as fh:
        for rec in merged:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    log(f"merged {len(merged)} approved decisions; "
        f"{len(rejected_decisions)} rejected: {rejected_decisions[:10]}")
    log(f"agent summaries: {[r for r in results]}")
    uncovered = len(items) - len(merged)
    log(f"queue coverage: {len(merged)}/{len(items)} "
        f"({uncovered} left unresolved)")


asyncio.run(main())
