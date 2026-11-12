"""Deterministic, split-local coverage selection from independently supported words.

All members of a retained match set get four distinct unordered triples with
the same positive-word-count schedule. No image is added or dropped alone.
Vocabulary equality is measured separately: valence parity does not prove it.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import Counter, defaultdict

import candidates_v3
import common
import policy
import policy_v3
from multimodel_bias.constants import NEGATIVE_IAT_SET, POSITIVE_IAT_SET


def entropy(counts, denominator=61):
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum(c / total * math.log2(c / total) for c in counts.values() if c) / math.log2(denominator)


def distribution_audit(records, manifest):
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    triples: Counter[tuple[str, ...]] = Counter()
    mixes: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        race = manifest[record["image_id"]]["source_race"]
        for variant in record["preference_variants"]:
            words = policy.word_set(variant["chosen"])
            counts[race].update(words)
            triples[tuple(sorted(words))] += 1
            mixes[race][str(len(words & POSITIVE_IAT_SET))] += 1
    totals = {race: sum(c.values()) for race, c in counts.items()}
    union = set().union(*(set(c) for c in counts.values())) if counts else set()
    tv: float | None = None
    if totals.get("Black") and totals.get("White"):
        tv = 0.5 * sum(abs(counts["Black"][w] / totals["Black"]
                          - counts["White"][w] / totals["White"]) for w in policy.IAT_WORDS)
    aggregate: Counter[str] = sum(counts.values(), Counter())
    return {
        "images": len(records), "examples": sum(triples.values()),
        "union_words": sorted(union), "distinct_words": len(union),
        "target_50_met": len(union) >= 50,
        "uncovered_eligible_words": sorted(set(policy_v3.ELIGIBLE) - union),
        "ineligible_preferred_words": sorted(policy.FORBIDDEN_CHOSEN),
        "by_race_word_counts": {r: dict(c) for r, c in counts.items()},
        "by_race_distinct_words": {r: len(c) for r, c in counts.items()},
        "word_entropy_normalized_to_61": entropy(aggregate),
        "word_tv": tv, "positive_word_count_histogram_by_race": dict(mixes),
        "distinct_unordered_triples": len(triples),
        "dominant_unordered_triple_share": max(triples.values(), default=0) / (sum(triples.values()) or 1),
    }


def triple_pool(words, positive_count):
    pos = sorted(set(words) & POSITIVE_IAT_SET)
    neg = sorted(set(words) & NEGATIVE_IAT_SET)
    # At most C(50,3)=19600, bounded independently of cohort size.
    return [tuple(sorted(p + n))
            for p in itertools.combinations(pos, positive_count)
            for n in itertools.combinations(neg, 3 - positive_count)]


def select_variants(rows, approved, variants=4):
    if variants not in (2, 3, 4):
        raise ValueError("two_to_four_variants_required")
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["split"], row["match_set_id"]].append(row)
    selected, dropped = {}, []
    # Counts never cross splits: development coverage cannot steer training.
    frequencies: dict[str | tuple[str, str], Counter[str]] = defaultdict(Counter)
    triple_frequencies: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for (split, sid), members in sorted(blocks.items(), key=lambda item: candidates_v3.stable_key(item[0])):
        races = Counter(r["source_race"] for r in members)
        if (split not in common.LABEL_SPLITS or set(races) != {"Black", "White"}
                or races["Black"] != races["White"]
                or any(r["image_id"] not in approved for r in members)):
            dropped.append({"match_set_id": sid, "split": split,
                            "reason": "incomplete_supported_matched_block"})
            continue
        pools = {r["image_id"]: {k: triple_pool(approved[r["image_id"]]["words"], k)
                                 for k in range(4)} for r in members}
        used: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        block_variants = defaultdict(list)
        block_counts: dict[str, Counter[str]] = defaultdict(Counter)
        # Commit coverage frequencies only if the complete block survives.
        for step in range(variants):
            options = []
            for k in range(4):
                choices, score = {}, 0.0
                local_counts: Counter[str] = Counter()
                for row in sorted(members, key=lambda r: candidates_v3.stable_key(sid, step, r["image_id"])):
                    iid, race = row["image_id"], row["source_race"]
                    available = [t for t in pools[iid][k] if t not in used[iid]]
                    if not available:
                        break
                    freq = frequencies[split] + block_counts["all"]
                    other = "White" if race == "Black" else "Black"
                    race_freq = frequencies[split, race] + block_counts[race]
                    other_freq = frequencies[split, other] + block_counts[other]

                    def quality(triple):
                        novelty = sum(10 if freq[w] == 0 else 1 / (1 + freq[w]) for w in triple)
                        balance = sum((other_freq[w] - race_freq[w]) / (1 + freq[w]) for w in triple)
                        return (novelty + balance + sum(1 / (1 + local_counts[w]) for w in triple)
                                - 2 * triple_frequencies[split][triple])

                    best = max(available, key=lambda t: (quality(t), t))
                    choices[iid] = best
                    score += quality(best)
                    local_counts.update(best)
                if len(choices) == len(members):
                    options.append((score, k, choices))
            if not options:
                break
            _, _, choices = max(options, key=lambda x: (x[0], -x[1]))
            for row in members:
                iid, race = row["image_id"], row["source_race"]
                triple = choices[iid]
                block_variants[iid].append(triple)
                used[iid].add(triple)
                block_counts["all"].update(triple)
                block_counts[race].update(triple)
        if any(len(block_variants[r["image_id"]]) != variants for r in members):
            dropped.append({"match_set_id": sid, "split": split,
                            "reason": "insufficient_distinct_valence_matched_triples"})
            continue
        for row in members:
            iid, race = row["image_id"], row["source_race"]
            selected[iid] = block_variants[iid]
            for triple in selected[iid]:
                frequencies[split].update(triple)
                frequencies[split, race].update(triple)
                triple_frequencies[split][triple] += 1
    return selected, dropped


def export(variants=4):
    suffix = "" if candidates_v3.PROFILE == "openai" else "_" + candidates_v3.PROFILE
    rows = candidates_v3.target_rows()
    records = candidates_v3.load_records()
    approved = candidates_v3.approved_candidates(rows, records)
    selected, dropped = select_variants(rows, approved, variants)
    manifest = {r["image_id"]: r for r in rows}
    permission = json.loads((common.REPO / "2026-10-03/config/prime_budget.json").read_text())
    authorization = permission.get("ai_target_approval_candidates_v3", {})
    if authorization.get("authorized") is not True:
        raise ValueError("Candidate-set approval not authorized")
    import hashlib
    auth = hashlib.sha256(json.dumps(authorization, sort_keys=True).encode()).hexdigest()
    targets = []
    for iid, triples in sorted(selected.items()):
        row, proposal = manifest[iid], approved[iid]
        prefs = []
        for i, triple in enumerate(triples):
            # Seeded presentation order avoids a canonical alphabetical answer.
            ordered = list(triple)
            random.Random(candidates_v3.stable_key(iid, i)).shuffle(ordered)
            prefs.append({"chosen": ", ".join(ordered),
                          "rejected": ", ".join(proposal["rejected"])})
        targets.append({
            "image_id": iid, "image_sha256": row["image_sha256"],
            "image_pixel_sha256": row["pixel_sha256"], "split": row["split"],
            "dataset": row["dataset"], "match_set_id": row["match_set_id"],
            "prompt": policy.IAT_PROMPT, **prefs[0], "preference_variants": prefs,
            "review_status": "approved", "reviewer": "devin-ai-word-intersection",
            "reviewer_kind": "ai", "draft_source": "two_pass_candidates_v3",
            "approval_authorization_sha256": auth,
            "target_provenance": f"{policy_v3.POLICY_ID}; independent word agreement >=0.8; not ground truth",
            "teacher_profile": candidates_v3.PROFILE,
            "approved_candidates": proposal["provenance"],
            "rejected_reason": proposal["rejected_reason"],
            "rejected_request_id": proposal["rejected_request_id"],
        })
    split_audits = {split: distribution_audit(
        [t for t in targets if t["split"] == split], manifest) for split in common.LABEL_SPLITS}
    audit = {"policy": policy_v3.POLICY_ID, "candidate_policy_sha256": policy_v3.POLICY_SHA,
             "teacher_profile": candidates_v3.PROFILE, "teachers": dict(candidates_v3.TEACHERS),
             "raw_responses_sha256": common.sha(candidates_v3.STORE / "responses.jsonl"),
             "variants_per_image": variants, "dropped_blocks": dropped,
             "eligible_by_stratum": dict(Counter(
                 "|".join(r[k] for k in ("split", "dataset", "source_race",
                                        "source_gender", "expression_class")) for r in rows)),
             "retained_by_stratum": dict(Counter(
                 "|".join(r[k] for k in ("split", "dataset", "source_race",
                                        "source_gender", "expression_class"))
                 for r in rows if r["image_id"] in selected)),
             "splits": split_audits,
             "test_used_for_selection": False, "targets_are_ground_truth": False}
    common.write_json(candidates_v3.STORE / f"export_audit{suffix}.json", audit)
    if not all(split_audits[s]["images"] for s in common.LABEL_SPLITS):
        raise ValueError("Both training and development supported images required")
    common.write_json(common.WORKDIR / f"data/private/curated_iat_targets_v3{suffix}.json", targets)
    selected_rows = [r for r in rows if r["image_id"] in selected]
    manifest_path = common.WORKDIR / f"data/selected_images_gated_v3{suffix}.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(selected_rows)
    config = {
        "dataset_id": "image-curation-2026-09-29-v3",
        "parent_manifest_path": "2026-09-29/data/selected_images.csv",
        "parent_manifest_sha256": common.sha(common.MANIFEST),
        "target_races": ["Black", "White"],
        "manifest_path": str(manifest_path.relative_to(common.REPO)),
        "manifest_sha256": common.sha(manifest_path),
        "asset_root": "2026-09-29/private/dataset",
        "targets_path": f"2026-10-20/data/private/curated_iat_targets_v3{suffix}.json",
        "teacher_profile": candidates_v3.PROFILE,
        "dataset_counts": dict(Counter(r["dataset"] for r in selected_rows)),
        "split_counts": dict(Counter(r["split"] for r in selected_rows)),
        "example_split_counts": {s: a["examples"] for s, a in split_audits.items()},
        "variants_per_image": variants,
        "preserve_published_splits": True, "target_splits": list(common.LABEL_SPLITS),
        "test_preference_generation_allowed": False, "profile_targets_used": 0,
        "training_methods": ["orpo"],
        "training_inputs": {"orpo": ["image", "prompt", "chosen", "rejected"]},
        "primary_benchmark": "2026-09-11/data/iat12/manifest.csv",
        "benchmark_images_training_allowed": False,
        "superseded_pairs": "2026-10-03/data/pairs.csv",
        "target_policy": f"Two independent word supports >=0.8; {variants} non-permutation triples; complete valence-matched race blocks",
        "benchmark_scope": "iat_only",
    }
    common.write_json(common.WORKDIR / f"config/active_dataset_gated_v3{suffix}_draft.json", config)
    print(json.dumps(audit, indent=2))
    return config, audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--provider", choices=("openai", "bitdeer"), default="openai")
    args = parser.parse_args()
    candidates_v3.configure(args.provider)
    export(args.variants)
