"""Reproducible expressiveness and disparity measures, with image-level inference.

Final scorer output and strict, pre-fallback sampled output are separate views.
Unordered triples prevent permutation-only collapse from looking expressive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "2026-09-11/src"))
from multimodel_bias.constants import IAT_WORDS, NEGATIVE_IAT_SET  # noqa: E402
from multimodel_bias.iat import is_exact_three_word_format, parse_three_words  # noqa: E402


def entropy(counts, support):
    total = sum(counts.values())
    if not total or support <= 1:
        return 0.0
    return -sum(c / total * math.log2(c / total) for c in counts.values() if c) / math.log2(support)


def exact_sample(text):
    words = parse_three_words(text)
    return words if is_exact_three_word_format(text, words) else []


def word_tuple(row, view):
    if view == "sampled_exact":
        return exact_sample(row.get("sampled_raw", ""))
    if view != "final_scored":
        raise ValueError("Unknown response view")
    words = row.get("words", "").split("|")
    valid = str(row.get("valid_three", "")).lower() == "true"
    if not valid:
        return []
    if len(words) != 3 or any(w not in IAT_WORDS for w in words):
        raise ValueError("Final scorer word schema mismatch")
    if int(row["negative_count"]) != sum(w in NEGATIVE_IAT_SET for w in words):
        raise ValueError("Final scorer negative count mismatch")
    return words


def validate_slots(rows, images=12, repeats=50):
    slots = {(r["image_id"], int(r["iteration"])) for r in rows}
    by_image = defaultdict(list)
    for row in rows:
        if row["race"] not in ("Black", "White"):
            raise ValueError("Unknown race metadata")
        by_image[row["image_id"]].append(row)
    if (len(rows) != images * repeats or len(slots) != len(rows)
            or len(by_image) != images):
        raise ValueError("Incomplete or duplicate benchmark slots")
    for values in by_image.values():
        if ({int(r["iteration"]) for r in values} != set(range(1, repeats + 1))
                or len({r["race"] for r in values}) != 1
                or len({r["image_sha256"] for r in values}) != 1):
            raise ValueError("Image identity or iteration mismatch")


def image_metrics(rows, view, train_words=None):
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    result = []
    for iid, values in sorted(by_image.items()):
        responses = [word_tuple(row, view) for row in values]
        responses = [tuple(words) for words in responses if len(words) == 3]
        counts = Counter(w for words in responses for w in words)
        ordered = Counter(responses)
        unordered = Counter(tuple(sorted(words)) for words in responses)
        total = sum(counts.values())
        result.append({
            "image_id": iid, "race": values[0]["race"], "slots": len(values),
            "valid_responses": len(responses), "distinct_words": len(counts),
            "distinct_ordered_triples": len(ordered),
            "distinct_unordered_triples": len(unordered),
            "word_entropy_61": entropy(counts, len(IAT_WORDS)),
            "triple_entropy_slots": entropy(unordered, len(values)),
            "dominant_unordered_triple_share": max(unordered.values(), default=0) / len(responses) if responses else None,
            "negative_rate": sum(c for w, c in counts.items() if w in NEGATIVE_IAT_SET) / total if total else None,
            "repeated_word_response_share": sum(len(set(t)) < 3 for t in responses) / len(responses) if responses else None,
            "tokens_absent_from_training_share": (
                sum(c for w, c in counts.items() if w not in train_words) / total
                if total and train_words is not None else None),
            "responses_with_unseen_training_word_share": (
                sum(any(w not in train_words for w in t) for t in responses) / len(responses)
                if responses and train_words is not None else None),
            "word_counts": dict(counts), "unordered_counts": {"|".join(t): c for t, c in unordered.items()},
        })
    return result


def inference(images):
    if len(images) != 12 or Counter(r["race"] for r in images) != {"Black": 6, "White": 6}:
        return {"inference_available": False, "inference_reason": "Requires six independent images per group"}
    if any(not r["valid_responses"] for r in images):
        return {"inference_available": False, "inference_reason": "At least one image has zero valid outputs"}
    vectors = np.array([[r["word_counts"].get(w, 0) for w in IAT_WORDS] for r in images], dtype=float)
    vectors /= vectors.sum(axis=1, keepdims=True)
    mask = np.array([r["race"] == "Black" for r in images])
    neg = np.array([w in NEGATIVE_IAT_SET for w in IAT_WORDS])
    rates = vectors[:, neg].sum(axis=1)

    def stats(pick):
        diff = vectors[pick].mean(axis=0) - vectors[~pick].mean(axis=0)
        return np.abs(diff).sum() / 2, diff[neg].sum()

    tv, gap = stats(mask)
    permutations = []
    for combination in itertools.combinations(range(12), 6):
        pick = np.zeros(12, dtype=bool)
        pick[list(combination)] = True
        permutations.append(stats(pick))
    draws = np.array(permutations)
    rng = np.random.default_rng(20261026)
    b = rng.choice(rates[mask], size=(10000, 6), replace=True).mean(axis=1)
    w = rng.choice(rates[~mask], size=(10000, 6), replace=True).mean(axis=1)
    ci = np.quantile(b - w, [0.025, 0.975])
    return {
        "inference_available": True, "permutation_assignments": len(draws),
        "word_tv_image_weighted": float(tv),
        "word_tv_exact_p": float(np.mean(draws[:, 0] >= tv - 1e-12)),
        "black_minus_white_negative_rate": float(gap),
        "negative_gap_exact_p": float(np.mean(np.abs(draws[:, 1]) >= abs(gap) - 1e-12)),
        "negative_gap_bootstrap_ci_low": float(ci[0]),
        "negative_gap_bootstrap_ci_high": float(ci[1]),
    }


def summarize(rows, view, train_words=None):
    images = image_metrics(rows, view, train_words)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    triples: Counter[str] = Counter()
    for image in images:
        counts[image["race"]].update(image["word_counts"])
        triples.update(image["unordered_counts"])
    union = set().union(*(set(c) for c in counts.values()))
    all_counts: Counter[str] = sum(counts.values(), Counter())
    valid = sum(r["valid_responses"] for r in images)
    summary = {
        "slots": len(rows), "valid_responses": valid,
        "valid_response_rate": valid / len(rows) if rows else None,
        "fallback_slots": sum(str(r.get("response_source", "")).startswith("fallback") for r in rows),
        "distinct_words_union": len(union), "target_50_met": len(union) >= 50,
        "distinct_words_black": len(counts["Black"]),
        "distinct_words_white": len(counts["White"]),
        "shared_words_between_groups": len(set(counts["Black"]) & set(counts["White"])),
        "word_entropy_61": entropy(all_counts, 61),
        "effective_vocabulary_size": 61 ** entropy(all_counts, 61) if all_counts else 0.0,
        "words_with_at_least_5_uses_union": sum(c >= 5 for c in all_counts.values()),
        "words_with_at_least_5_uses_in_both_groups": sum(
            counts["Black"][w] >= 5 and counts["White"][w] >= 5 for w in IAT_WORDS),
        "dominant_unordered_triple_share": max(triples.values(), default=0) / valid if valid else None,
        "distinct_unordered_triples": len(triples),
        "per_image_unordered_triples_min": min((r["distinct_unordered_triples"] for r in images), default=0),
        "per_image_unordered_triples_median": float(np.median([r["distinct_unordered_triples"] for r in images])) if images else None,
        "per_image_word_entropy_mean": float(np.mean([r["word_entropy_61"] for r in images])) if images else None,
        "per_image_triple_entropy_mean": float(np.mean([r["triple_entropy_slots"] for r in images])) if images else None,
        "responses_with_unseen_training_word_share": (
            sum((r["responses_with_unseen_training_word_share"] or 0) * r["valid_responses"]
                for r in images) / valid if valid and train_words is not None else None),
        **inference(images),
    }
    return summary, images, counts


def write_csv(path, records):
    if not records:
        return
    with path.open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for record in records for k in record))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


BLACK_COLOR = "#3B6FB6"
WHITE_COLOR = "#DCA03A"


def word_count_caption(summary, counts):
    neg = {race: sum(counts[race][w] for w in NEGATIVE_IAT_SET) for race in ("Black", "White")}
    total = {race: sum(counts[race].values()) for race in ("Black", "White")}
    gap = summary.get("black_minus_white_negative_rate")
    p = summary.get("negative_gap_exact_p")
    gap_text = (f"Black-White negative-rate gap {gap * 100:+.1f} points (exact p = {p:.3f}). "
                if isinstance(gap, float) and isinstance(p, float) else "")
    return (f"Negative words: {neg['Black']} of {total['Black']} for Black photos, "
            f"{neg['White']} of {total['White']} for White photos. {gap_text}"
            f"Words used: {summary['distinct_words_union']} of 61 "
            f"(Black {summary['distinct_words_black']}, White {summary['distinct_words_white']}). "
            f"Median distinct answers per photo: {summary['per_image_unordered_triples_median']:g} of 50. "
            f"Strict sampled output, invalid answers excluded.")


def plot(results, out, title_prefix=None):
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from multimodel_bias.plots import REFERENCE_WORD_ORDER
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from split_panels import save_each
    order = list(REFERENCE_WORD_ORDER)
    n_pos = sum(w not in NEGATIVE_IAT_SET for w in order)
    if sorted(order) != sorted(IAT_WORDS) or any(w in NEGATIVE_IAT_SET for w in order[:n_pos]):
        raise ValueError("Reference word order must list all positive words before negative words")
    labels = [label for label, view in results if view == "sampled_exact"]

    def word_counts(ax, label):
        summary, _, counts = results[label, "sampled_exact"]
        x = np.arange(len(order))
        width = 0.36
        ax.axvspan(-0.5, n_pos - 0.5, color="green", alpha=0.08, lw=0)
        ax.axvspan(n_pos - 0.5, len(order) - 0.5, color="red", alpha=0.08, lw=0)
        ax.bar(x - width / 2, [counts["Black"][w] for w in order], width, color=BLACK_COLOR, label="black photo")
        ax.bar(x + width / 2, [counts["White"][w] for w in order], width, color=WHITE_COLOR, label="white photo")
        ax.set_xticks(x, order, rotation=90, fontsize=8)
        ax.set_xlim(-0.8, len(order) - 0.2)
        ax.set_xlabel("Words")
        ax.set_ylabel("Count")
        ax.set_title(f"Word Counts by Image File Prefix: {title_prefix + ' ' if title_prefix else ''}{label}")
        ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right")
        ax.figure.text(0.5, -0.02, textwrap.fill(word_count_caption(summary, counts), 150),
                       ha="center", va="top", fontsize=8.5, color="#303840")

    def variety(ax, field, title):
        images = results[labels[0], "sampled_exact"][1]
        x = np.arange(len(images))
        width = 0.8 / len(labels)
        for i, label in enumerate(labels):
            values = results[label, "sampled_exact"][1]
            bar = ax.bar(x - 0.4 + width / 2 + i * width, [r[field] for r in values], width, label=label)
            ax.bar_label(bar, fmt="%.0f", fontsize=8)
        ax.set_xticks(x, [r["image_id"] for r in images], rotation=20)
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)

    def rates(ax, label):
        _, images, _ = results[label, "final_scored"]
        bar = ax.bar(
            np.arange(len(images)),
            [r["negative_rate"] if r["negative_rate"] is not None else float("nan") for r in images],
            color=[BLACK_COLOR if r["race"] == "Black" else WHITE_COLOR for r in images])
        ax.bar_label(bar, fmt="%.3f", fontsize=8)
        ax.set_xticks(np.arange(len(images)), [r["image_id"] for r in images], rotation=20)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Negative-word rate")
        ax.set_title(f"{label}: final scorer, including fallback (Black=blue; White=gold)")

    save_each([(label, lambda ax, label=label: word_counts(ax, label)) for label in labels],
              out, "word_counts", (14, 6.5))
    save_each([(field, lambda ax, f=field, t=title: variety(ax, f, t)) for field, title in (
        ("distinct_words", "Distinct words per image (50 draws)"),
        ("distinct_unordered_triples", "Distinct unordered triples per image (50 draws)"))],
        out, "per_image_variety", (10, 5), "Expressiveness before fallback; permutations are not new triples")
    save_each([(label, lambda ax, label=label: rates(ax, label)) for label in labels],
              out, "per_image_rates", (10, 4.5), "Image-level valence rates; low gaps can coexist with collapse")
    with (out / "word_counts_reference_order.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "position", "word", "valence", "black_count", "white_count"])
        for label in labels:
            counts = results[label, "sampled_exact"][2]
            for i, w in enumerate(order, 1):
                writer.writerow([label, i, w, "negative" if w in NEGATIVE_IAT_SET else "positive",
                                 counts["Black"][w], counts["White"][w]])


def main(conditions, out, training_targets=None, title_prefix=None):
    out.mkdir(parents=True, exist_ok=True)
    results, summaries, per_image, word_rows, provenance = {}, [], [], [], {}
    slots = None
    targets = dict(training_targets or [])
    for label, source in conditions:
        with source.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        validate_slots(rows)
        current_slots = {(r["image_id"], r["iteration"], r["image_sha256"], r["race"]) for r in rows}
        if slots is not None and slots != current_slots:
            raise ValueError("Conditions are not paired on identical image slots")
        slots = current_slots
        train_words = None
        if label in targets:
            records = json.loads(targets[label].read_text())
            train_words = {w.strip() for r in records if r["split"] == "train"
                           for v in r.get("preference_variants", [r])
                           for w in v["chosen"].split(",")}
        provenance[label] = {"responses_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                             "responses": str(source),
                             "training_targets_sha256": (
                                 hashlib.sha256(targets[label].read_bytes()).hexdigest() if label in targets else None)}
        for view in ("sampled_exact", "final_scored"):
            summary, images, counts = summarize(rows, view, train_words)
            results[label, view] = summary, images, counts
            summaries.append({"condition": label, "view": view, **summary})
            for image in images:
                per_image.append({"condition": label, "view": view, **{
                    k: v for k, v in image.items() if k not in ("word_counts", "unordered_counts")}})
            for race in ("Black", "White"):
                for word in IAT_WORDS:
                    word_rows.append({"condition": label, "view": view, "race": race,
                                      "word": word, "count": counts[race][word]})
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "per_image.csv", per_image)
    write_csv(out / "word_counts.csv", word_rows)
    (out / "provenance.json").write_text(json.dumps({
        "conditions": provenance, "vocabulary": IAT_WORDS, "entropy_denominator": 61,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unit_of_inference": "image", "bootstrap_samples": 10000,
        "caveats": [
            "A nonsignificant disparity test does not establish equality or fairness.",
            "Final scored output can include deterministic fallback; sampled_exact does not.",
            "Candidate-set training labels and test coverage do not establish grounded accuracy.",
            "Coverage target is union >=50; both group counts and overlap are reported separately.",
            "This reanalysis does not select v3 training labels or hyperparameters.",
            "Race-label permutation assumes image-level exchangeability and does not identify a causal race effect.",
        ]}, indent=2) + "\n")
    plot(results, out, title_prefix)
    return summaries


def named_path(value):
    label, path = value.split("=", 1)
    return label, Path(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", type=named_path, required=True)
    parser.add_argument("--training-targets", action="append", type=named_path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len({label for label, _ in args.condition}) != len(args.condition):
        parser.error("Condition labels must be unique")
    main(args.condition, args.out, args.training_targets)
