"""V2b fan-out: shared-VM agents adjudicate the disagreement queue — the items
where the two teacher passes split on dominant valence (valence_disagreement)
or on word choice within one valence (chosen_disagreement). These are the
ambiguous faces; honest labels here are frequently MIXED-valence triples.

Merged+validated into adjudication_decisions_v2b.jsonl.
"""
import asyncio
import hashlib
import json
from pathlib import Path

REPO = Path("/home/ubuntu/repos/URAP-FA-2026")
QUEUE = REPO / "2026-10-20/data/private/target_review/adjudication_queue_v2_disagreement.jsonl"
OUT_DIR = REPO / "2026-10-20/data/private/target_review"
MERGED = OUT_DIR / "adjudication_decisions_v2b.jsonl"

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

N_AGENTS = 6


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
    out_file = OUT_DIR / f"adjudication_decisions_v2b_w{idx}.jsonl"
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
    return f"""You are adjudicating the HARD cases of a face-expression
word-choice study. Two independent teachers each picked 3 words from a fixed
61-word vocabulary for the same face photo and DISAGREED. Your job: VIEW each
image and pick the final chosen triple.

Vocabulary (use ONLY these words, exact casing):
{IAT_WORDS}

Each item has a "reason" field:

* "valence_disagreement": one teacher's chosen triple is positive-dominant and
  the other's is negative-dominant. This happens on MILD, AMBIGUOUS or NEUTRAL
  faces — faces that honestly sit between poles. This vocabulary has NO
  genuinely neutral words, so the honest encoding for a between-poles face is
  a MIXED triple:
  - First decide which pole the visible expression better supports (look at
    mouth corners, lips, brow, eyes).
  - Keep up to 2 words from that better-supported teacher's triple.
  - Fill the third slot with ONE word from the OTHER teacher's triple (the
    opposite pole), producing e.g. 2-negative+1-positive or
    2-positive+1-negative.
  - ONLY if the expression genuinely reads as a clear pole do you keep a
    pure-valence triple — most of these items are here BECAUSE the face was
    ambiguous, so mixed is the expected majority outcome.

* "chosen_disagreement": both teachers picked the same dominant valence but
  shared at most one word. View the face and build the honest triple:
  prefer words that appeared in either teacher's chosen set; pick the 3 best-
  fitting from the union. Pure-valence is expected here since the valence
  agreed — but if the face is mild/ambiguous you may still mix one opposite-
  pole word.

Rules for every item:
1. Judge the face's VISIBLE expression only. Do NOT infer race, identity,
   health or personality.
2. Chosen = 3 DISTINCT vocabulary words sharing at least 2 words with the
   union of the two teachers' chosen sets, containing NONE of:
   {sorted(FORBIDDEN_CHOSEN)}.
3. Rejected = one pass's rejected triple sharing NO word with your chosen
   (prefer the higher-confidence pass's). If both collide, write any valid
   incongruent 3-distinct-word vocabulary triple that disagrees with the
   visible expression.
4. Append ONE JSON line per image to {out_file} (create parent dirs if
   needed), fields exactly:
   {{"image_id","split","dataset","match_set_id","source_race","source_gender",
   "expression_class","a_chosen","b_chosen","a_rejected","b_rejected",
   "teacher_a","teacher_b","chosen","rejected","decision":"approved",
   "adjudicator":"devin-workflow-v2b-w{idx}","reason":"<under 25 words>"}}
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
    # Disagreement items may legitimately need words from BOTH passes; require
    # >=2 overlap with the UNION of both chosen sets (not a single pass).
    if len((ca | cb) & chosen) < 2:
        return False, "drifted_from_both_passes"
    return True, ""


async def main():
    await register_workflow({
        "name": "iat-adjudication-v2b-disagreement",
        "description": "Shared-VM agents adjudicate valence/chosen-disagreement "
                       "items into (often mixed-valence) targets",
        "product": "URAP-FA-2026 gated labeling v2",
        "soft_time_limit_minutes": 30,
        "subscribe": "none",
        "phases": [{"title": "adjudicate",
                    "detail": "view queued images, write v2b decision JSONL",
                    "count": N_AGENTS}],
    })
    items = [json.loads(line) for line in QUEUE.open() if line.strip()]
    log(f"disagreement queue: {len(items)} items across {N_AGENTS} agents")
    chunks = [items[i::N_AGENTS] for i in range(N_AGENTS)]
    paths = manifest_paths()
    results = await parallel([
        (lambda i=i, c=c: agent(
            agent_prompt(i, c, paths),
            phase="adjudicate",
            schema=SCHEMA,
            label=f"adjudicate-v2b-w{i}",
            vm_mode="shared",   # the images exist only on this VM
            soft_time_limit_minutes=30))
        for i, c in enumerate(chunks)])
    merged, rejected_decisions = [], []
    seen = set()
    item_map = {it["image_id"]: it for it in items}
    for i in range(N_AGENTS):
        path = OUT_DIR / f"adjudication_decisions_v2b_w{i}.jsonl"
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
