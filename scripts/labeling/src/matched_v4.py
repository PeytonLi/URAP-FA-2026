"""Race-matched answer sets: every face in a match block gets the same triples.

The targets are a fairness-and-variety design, not claims about any face.
Each block leans with its automated expression class (metadata, not ground
truth), every chosen triple mixes valence, and word use is spread evenly over
the 50 eligible words within each split. The rejected answer of each variant
is a single-valence triple: the collapse mode seen in earlier rounds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict

import candidates_v3
import common
import policy
import policy_v3
from multimodel_bias.constants import NEGATIVE_IAT_SET, POSITIVE_IAT_SET

DATASET_ID = "image-curation-2026-09-29-v4"
DRAFT_SOURCE = "matched_block_answers_v4"
AUTHORIZATION_KEY = "ai_target_approval_matched_v4"
# Positive words per variant. Every triple mixes valence; the lean sets the mix.
SCHEDULES = {"positive": (2, 2, 2, 1), "negative": (1, 1, 1, 2),
             "neutral": (1, 2, 1, 2), "uncertain": (2, 1, 2, 1)}
# The rejected answer uses the block's leaning pole only.
REJECTED_POLE = {"positive": "positive", "negative": "negative",
                 "neutral": "negative", "uncertain": "positive"}
POSITIVE = sorted(set(policy_v3.ELIGIBLE) & POSITIVE_IAT_SET)
NEGATIVE = sorted(set(policy_v3.ELIGIBLE) & NEGATIVE_IAT_SET)
REJECT_POOLS = {"positive": sorted(POSITIVE_IAT_SET), "negative": sorted(NEGATIVE_IAT_SET)}


def key(*parts):
    return candidates_v3.stable_key(*parts)


def blocks(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["split"], row["match_set_id"]].append(row)
    for (split, sid), members in grouped.items():
        races = Counter(r["source_race"] for r in members)
        if (split not in common.LABEL_SPLITS or set(races) != {"Black", "White"}
                or races["Black"] != races["White"]):
            raise ValueError(f"Race-unbalanced or unknown-split block: {sid}")
        if len({r["expression_class"] for r in members}) != 1:
            raise ValueError(f"Block expression class differs: {sid}")
        if members[0]["expression_class"] not in SCHEDULES:
            raise ValueError(f"Unknown expression class: {sid}")
    return dict(sorted(grouped.items(), key=lambda item: key("v4-block", item[0])))


def least_used(pool, counts, count, exclude, *salt):
    options = [w for w in pool if w not in exclude]
    options.sort(key=lambda w: (counts[w], key(*salt, w)))
    return options[:count]


def design(rows, variants=4):
    """Return {(split, sid): [(chosen_triple, rejected_triple), ...]}."""
    if variants not in (2, 3, 4):
        raise ValueError("two_to_four_variants_required")
    chosen_counts: dict[str, Counter[str]] = defaultdict(Counter)
    rejected_counts: dict[str, Counter[str]] = defaultdict(Counter)
    triple_counts: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    plan = {}
    for (split, sid), members in blocks(rows).items():
        lean = members[0]["expression_class"]
        used: set[tuple[str, ...]] = set()
        answers = []
        for step, positives in enumerate(SCHEDULES[lean][:variants]):
            counts = chosen_counts[split]
            for attempt in range(len(POSITIVE) * len(NEGATIVE)):
                skip = set().union(*used) if attempt else set()
                pos = least_used(POSITIVE, counts, positives, skip if attempt % 2 else set(),
                                 split, sid, step, attempt, "p")
                neg = least_used(NEGATIVE, counts, 3 - positives, skip if attempt % 2 == 0 else set(),
                                 split, sid, step, attempt, "n")
                triple = tuple(sorted(pos + neg))
                if len(triple) == 3 and triple not in used:
                    break
            else:
                raise ValueError(f"No distinct triple for block {sid}")
            pole = REJECTED_POLE[lean]
            rejected = tuple(sorted(least_used(
                REJECT_POOLS[pole], rejected_counts[split], 3, set(triple),
                split, sid, step, "r")))
            used.add(triple)
            answers.append((triple, rejected))
            counts.update(triple)
            rejected_counts[split].update(rejected)
            triple_counts[split][triple] += 1
        plan[split, sid] = answers
    return plan


def entropy(counts, support=61):
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(c / total * math.log2(c / total) for c in counts.values() if c) / math.log2(support)


def audit(rows, targets):
    by_id = {r["image_id"]: r for r in rows}
    result = {}
    for split in common.LABEL_SPLITS:
        words: dict[str, Counter[str]] = defaultdict(Counter)
        triples: Counter[tuple[str, ...]] = Counter()
        mixes: dict[str, Counter[str]] = defaultdict(Counter)
        rejected: Counter[str] = Counter()
        for target in targets:
            if target["split"] != split:
                continue
            race = by_id[target["image_id"]]["source_race"]
            for variant in target["preference_variants"]:
                chosen = policy.word_set(variant["chosen"])
                words[race].update(chosen)
                triples[tuple(sorted(chosen))] += 1
                mixes[race][str(len(chosen & POSITIVE_IAT_SET))] += 1
                rejected.update(policy.word_set(variant["rejected"]))
        union = set(words["Black"]) | set(words["White"])
        totals = {race: sum(words[race].values()) for race in ("Black", "White")}
        tv = 0.5 * sum(abs(words["Black"][w] / totals["Black"] - words["White"][w] / totals["White"])
                       for w in policy.IAT_WORDS) if all(totals.values()) else None
        aggregate = words["Black"] + words["White"]
        result[split] = {
            "images": sum(t["split"] == split for t in targets),
            "examples": sum(triples.values()),
            "distinct_chosen_words": len(union),
            "uncovered_eligible_words": sorted(set(policy_v3.ELIGIBLE) - union),
            "chosen_word_counts": dict(sorted(aggregate.items())),
            "min_max_word_count": [min(aggregate.values()), max(aggregate.values())],
            "word_entropy_normalized_to_61": entropy(aggregate),
            "effective_vocabulary_size": 61 ** entropy(aggregate),
            "black_white_word_tv": tv,
            "distinct_unordered_triples": len(triples),
            "dominant_triple_share": max(triples.values()) / sum(triples.values()),
            "positive_words_per_answer_by_race": {r: dict(sorted(c.items())) for r, c in mixes.items()},
            "single_valence_chosen_answers": sum(c for t, c in triples.items()
                                                 if len(set(t) & POSITIVE_IAT_SET) in (0, 3)),
            "rejected_distinct_words": len(rejected),
        }
    return result


def export(variants=4):
    rows = candidates_v3.target_rows()
    forbidden = {r["sha256"] for r in common.read_csv(common.REPO / "2026-09-11/data/iat12/manifest.csv")}
    if any(r["image_sha256"] in forbidden for r in rows):
        raise ValueError("benchmark_overlap")
    permission = json.loads(
        (common.REPO / "configs" / "prompt_sweep" / "prime_budget.json").read_text()
    )
    authorization = permission.get(AUTHORIZATION_KEY, {})
    if authorization.get("authorized") is not True:
        raise ValueError("Matched-answer targets not authorized")
    auth = hashlib.sha256(json.dumps(authorization, sort_keys=True).encode()).hexdigest()
    plan = design(rows, variants)
    targets = []
    for row in sorted(rows, key=lambda r: r["image_id"]):
        lean = row["expression_class"]
        prefs = []
        for step, (chosen, rejected) in enumerate(plan[row["split"], row["match_set_id"]]):
            # Seeded word order per image: same word sets, no canonical string.
            ordered = list(chosen)
            random.Random(key(row["image_id"], step, "chosen")).shuffle(ordered)
            negative = list(rejected)
            random.Random(key(row["image_id"], step, "rejected")).shuffle(negative)
            prefs.append({"chosen": ", ".join(ordered), "rejected": ", ".join(negative)})
        targets.append({
            "image_id": row["image_id"], "image_sha256": row["image_sha256"],
            "image_pixel_sha256": row["pixel_sha256"], "split": row["split"],
            "dataset": row["dataset"], "match_set_id": row["match_set_id"],
            "prompt": policy.IAT_PROMPT, **prefs[0], "preference_variants": prefs,
            "review_status": "approved", "reviewer": "devin-matched-block-design-v4",
            "reviewer_kind": "ai", "draft_source": DRAFT_SOURCE,
            "approval_authorization_sha256": auth, "block_lean": lean,
            "target_provenance": (f"{DRAFT_SOURCE}; identical answer sets across the race-matched "
                                  "block; lean from automated expression_class; not a claim about the face"),
        })
    summary = audit(rows, targets)
    common.write_json(common.WORKDIR / "data/private/curated_iat_targets_v4.json", targets)
    manifest_path = common.WORKDIR / "data/selected_images_gated_v4.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "dataset_id": DATASET_ID,
        "parent_manifest_path": "2026-09-29/data/selected_images.csv",
        "parent_manifest_sha256": common.sha(common.MANIFEST),
        "target_races": ["Black", "White"],
        "manifest_path": str(manifest_path.relative_to(common.REPO)),
        "manifest_sha256": common.sha(manifest_path),
        "asset_root": "2026-09-29/private/dataset",
        "targets_path": "datasets/private_targets/curated_iat_targets_v4.json",
        "dataset_counts": dict(Counter(r["dataset"] for r in rows)),
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "example_split_counts": {s: a["examples"] for s, a in summary.items()},
        "variants_per_image": variants, "identical_answers_within_block": True,
        "preserve_published_splits": True, "target_splits": list(common.LABEL_SPLITS),
        "test_preference_generation_allowed": False, "profile_targets_used": 0,
        "training_methods": ["orpo"],
        "training_inputs": {"orpo": ["image", "prompt", "chosen", "rejected"]},
        "primary_benchmark": "2026-09-11/data/iat12/manifest.csv",
        "benchmark_images_training_allowed": False,
        "superseded_pairs": "scripts/train_eval/_data/pairs.csv",
        "target_policy": (f"Matched answers v4: {variants} mixed-valence triples per block, identical "
                          "for all Black and White members, even 50-word coverage per split; "
                          "single-valence rejected answers"),
        "benchmark_scope": "iat_only",
    }
    common.write_json(common.WORKDIR / "config/active_dataset_gated_v4_draft.json", config)
    report = {"dataset_id": DATASET_ID, "variants_per_image": variants,
              "schedules": {k: list(v[:variants]) for k, v in SCHEDULES.items()},
              "rejected_pole": REJECTED_POLE, "splits": summary,
              "targets_sha256": common.sha(common.WORKDIR / "data/private/curated_iat_targets_v4.json"),
              "parent_manifest_sha256": config["parent_manifest_sha256"],
              "test_used_for_selection": False, "targets_are_claims_about_faces": False}
    common.write_json(
        common.REPO / "results" / "v4" / "iat" / "matched_v4" / "audit.json", report
    )
    return config, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, choices=(2, 3, 4), default=4)
    args = parser.parse_args()
    _, report = export(args.variants)
    print(json.dumps({s: {k: v for k, v in a.items() if k != "chosen_word_counts"}
                      for s, a in report["splits"].items()}, indent=2))
