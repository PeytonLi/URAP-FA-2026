#!/usr/bin/env python3
"""Rebuild audited row-level tables and cluster-aware exploratory statistics offline."""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

from common import ROOT, read_csv, sha, write_csv, write_json

sys.path.insert(0, str(ROOT / "src"))
from multimodel_bias.analysis import _as_bool, _image_level_rows, _model_metrics
from multimodel_bias.constants import NEGATIVE_IAT_SET
from multimodel_bias.decodingtrust import classify_response
from multimodel_bias.iat import is_exact_three_word_format, parse_three_words
from multimodel_bias.uncertainty import (
    benjamini_hochberg,
    stratified_exact_binary_label_permutation,
)
from multimodel_bias.visbias_analysis import score_descriptions

OUT = ROOT / "results/analysis"
N = 100_000
SEED = 9052026
ORDER = (
    "gemma_local",
    "qwen35_9b",
    "kimi_k3",
    "gpt56_luna",
    "gemini36_flash",
    "mimo_v25",
    "minimax_m3",
    "step37_flash",
    "grok46",
)
NAMES = dict(
    zip(
        ORDER,
        (
            "Gemma 3 12B (local pinned)",
            "Qwen 3.5 9B",
            "Kimi K3",
            "GPT-5.6 Luna",
            "Gemini 3.6 Flash",
            "MiMo-V2.5",
            "MiniMax M3",
            "Step 3.7 Flash",
            "Grok 4.6",
        ),
        strict=True,
    )
)
A24 = ROOT / "reference/august24/results"
A31 = ROOT / "reference/august31/results"
DPO = ROOT / "reference/dpo/august10"


def load(path):
    return read_csv(path) if path.exists() else []


def subset(rows, key):
    return [r for r in rows if r["model_key"] == key]


def mean_ci(values, seed=SEED):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.concatenate(
        [
            values[rng.integers(len(values), size=(min(2000, N - i), len(values)))].mean(1)
            for i in range(0, N, 2000)
        ]
    )
    return tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def sign_test(values, seed=SEED):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return math.nan
    observed = abs(values.mean())
    if len(values) <= 16:
        signs = np.asarray(list(itertools.product((-1, 1), repeat=len(values))))
        return float(np.mean(np.abs((signs * values).mean(1)) >= observed - 1e-12))
    rng, count = np.random.default_rng(seed), 0
    for i in range(0, N, 2000):
        signs = rng.choice((-1, 1), size=(min(2000, N - i), len(values)))
        count += int(np.sum(np.abs((signs * values).mean(1)) >= observed - 1e-12))
    return (count + 1) / (N + 1)


def adjust(rows, family):
    for row, value in zip(rows, benjamini_hochberg([r["raw_p_value"] for r in rows]), strict=True):
        row.update(
            bh_adjusted_q_value=value,
            correction_family=family,
            family_tests=len(rows),
            exploratory=True,
        )
    return rows


def normalize_iat(rows, model=None):
    normalized = []
    for original in rows:
        row = dict(original)
        if model:
            row["model_key"] = model
        row["race"] = row.get("race") or row.get("fairface_race")
        row["gender"] = row.get("gender") or row.get("fairface_gender")
        row["image_id"] = row.get("image_id") or row.get("image", "").split("_")[0]
        row["image_filename"] = row.get("image_filename") or row.get("image") or row.get("filename")
        words = parse_three_words(row.get("response", ""))
        row.update(
            words="|".join(words),
            valid_three=len(words) == 3,
            word_count=len(words),
            negative_count=sum(w in NEGATIVE_IAT_SET for w in words),
        )
        row["model_display_name"] = NAMES[row["model_key"]]
        if "exact_format" not in row:
            raw = (
                row.get("fallback_raw")
                if row.get("response_source", "").startswith("fallback")
                else row.get("sampled_raw")
            )
            row["exact_format"] = is_exact_three_word_format(raw, words) if raw else ""
        normalized.append(row)
    return normalized


def iat_analysis(rows, dataset):
    metrics, significance, clusters, word_rows, intersections = [], [], [], [], []
    pair_lookup = {
        r["image_id"]: r["match_pair_id"]
        for r in load(ROOT / "data/frozen/utkface/utkface_iat_evaluation_manifest.csv")
    }
    for index, key in enumerate(ORDER):
        selected = subset(rows, key)
        if not selected:
            continue
        metric = _model_metrics(selected, N, SEED + index)
        images = _image_level_rows(selected)
        clusters.extend(dict(r, model_key=key, dataset=dataset) for r in images)
        effect = metric["black_minus_white_negative_rate_gap"]
        low, high = metric["gap_ci_low"], metric["gap_ci_high"]
        if dataset == "utkface":
            pairs = defaultdict(dict)
            for row in images:
                pairs[pair_lookup[row["image_id"]]][row["race"]] = row["negative_rate"]
            differences = [
                p["Black"] - p["White"] for p in pairs.values() if set(p) == {"Black", "White"}
            ]
            effect = float(np.mean(differences))
            low, high = mean_ci(differences, SEED + index)
            p = sign_test(differences, SEED + index)
            test, unit, count = (
                "matched-pair two-sided sign-flip",
                "complete age/expression/quality-matched image pairs",
                len(differences),
            )
        else:
            _, p, _ = stratified_exact_binary_label_permutation(
                images, "negative_rate", "race", "gender", group_a="Black", group_b="White"
            )
            test, unit, count = (
                "exact race-label permutation within recorded gender",
                "distinct images",
                len(images),
            )
        significance.append(
            dict(
                benchmark=f"IAT61 {dataset}",
                dataset=dataset,
                metric="negative_term_race_gap",
                model_key=key,
                model_display_name=NAMES[key],
                effect=effect,
                ci_low=low,
                ci_high=high,
                raw_p_value=p,
                sample_count=count,
                sample_unit=unit,
                test=test,
                interval_method="95% percentile bootstrap; matched pairs for UTK, race/gender image clusters for neutral12",
            )
        )
        rates = metric.pop("race")
        for group, values in metric.pop("intersectional").items():
            intersections.append(dict(dataset=dataset, model_key=key, group=group, **values))
        metrics.append(
            dict(
                dataset=dataset,
                model_key=key,
                model_display_name=NAMES[key],
                **metric,
                black_rate=rates["Black"]["negative_rate"],
                white_rate=rates["White"]["negative_rate"],
                black_ci_low=rates["Black"]["ci_low"],
                black_ci_high=rates["Black"]["ci_high"],
                white_ci_low=rates["White"]["ci_low"],
                white_ci_high=rates["White"]["ci_high"],
            )
        )
        for race in ("Black", "White"):
            counts = Counter(
                word
                for row in selected
                if row["race"] == race and _as_bool(row["valid_three"])
                for word in row["words"].split("|")
            )
            word_rows.extend(
                dict(
                    dataset=dataset,
                    model_key=key,
                    race=race,
                    word=word,
                    count=count,
                    negative=word in NEGATIVE_IAT_SET,
                )
                for word, count in counts.items()
            )
    adjust(significance, f"IAT61 {dataset}: all available models")
    for name, data in (
        ("responses", rows),
        ("summary", metrics),
        ("significance", significance),
        ("image_scores", clusters),
        ("word_counts", word_rows),
        ("intersectional", intersections),
    ):
        write_csv(OUT / f"iat_{dataset}_{name}.csv", data)
    return significance


def group_range(rows, field, group, block=None, seed=SEED):
    """Exchange group labels within topics for DT; independent prompts for BOLD."""
    groups = sorted({r[group] for r in rows})
    arrays = [np.asarray([float(r[field]) for r in rows if r[group] == g]) for g in groups]
    observed = max(a.mean() for a in arrays) - min(a.mean() for a in arrays)
    rng = np.random.default_rng(seed)
    if block:
        blocks = sorted({r[block] for r in rows})
        matrix = np.array(
            [
                [
                    next(float(r[field]) for r in rows if r[block] == b and r[group] == g)
                    for g in groups
                ]
                for b in blocks
            ]
        )
        draws = matrix[rng.integers(len(blocks), size=(N, len(blocks)))].mean(1)
        permutations = np.concatenate(
            [
                rng.permuted(
                    np.broadcast_to(matrix, (min(2000, N - i), *matrix.shape)), axis=2
                ).mean(1)
                for i in range(0, N, 2000)
            ]
        )
    else:
        draws = np.column_stack([a[rng.integers(len(a), size=(N, len(a)))].mean(1) for a in arrays])
        values = np.concatenate(arrays)
        boundaries = np.cumsum([0] + [len(a) for a in arrays])
        batches = []
        for i in range(0, N, 2000):
            shuffled = rng.permuted(
                np.broadcast_to(values, (min(2000, N - i), len(values))), axis=1
            )
            batches.append(
                np.column_stack(
                    [
                        shuffled[:, boundaries[j] : boundaries[j + 1]].mean(1)
                        for j in range(len(groups))
                    ]
                )
            )
        permutations = np.concatenate(batches)
    ranges = draws.max(1) - draws.min(1)
    null = permutations.max(1) - permutations.min(1)
    low, high = np.percentile(ranges, [2.5, 97.5])
    p = (int(np.sum(null >= observed - 1e-12)) + 1) / (N + 1)
    return float(observed), float(low), float(high), p


def subject_cluster_range(rows, field, seed=SEED):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["subject_name"])].append(float(row[field]))
    categories = sorted({category for category, _ in grouped})
    arrays = [
        np.array(
            [
                [sum(values), len(values)]
                for (category, _), values in sorted(grouped.items())
                if category == g
            ],
            dtype=float,
        )
        for g in categories
    ]
    rng = np.random.default_rng(seed)
    observed_means = [a[:, 0].sum() / a[:, 1].sum() for a in arrays]
    observed = max(observed_means) - min(observed_means)
    means = []
    for a in arrays:
        sums = a[rng.integers(len(a), size=(N, len(a)))].sum(1)
        means.append(sums[:, 0] / sums[:, 1])
    bootstrap = np.column_stack(means)
    pooled = np.concatenate(arrays)
    boundaries = np.cumsum([0] + [len(a) for a in arrays])
    exceed = 0
    for i in range(0, N, 2000):
        # One permutation index moves each subject's sum/count together.
        indexes = rng.permuted(
            np.broadcast_to(np.arange(len(pooled)), (min(2000, N - i), len(pooled))), axis=1
        )
        shuffled = pooled[indexes]
        group_means = []
        for j in range(len(categories)):
            sums = shuffled[:, boundaries[j] : boundaries[j + 1], :].sum(1)
            group_means.append(sums[:, 0] / sums[:, 1])
        null = np.column_stack(group_means)
        exceed += int(np.sum(null.max(1) - null.min(1) >= observed - 1e-12))
    low, high = np.percentile(bootstrap.max(1) - bootstrap.min(1), [2.5, 97.5])
    return float(observed), float(low), float(high), (exceed + 1) / (N + 1)


def subject_cluster_ci(rows, field, seed=SEED):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["subject_name"]].append(float(r[field]))
    a = np.array([[sum(v), len(v)] for v in grouped.values()], dtype=float)
    rng = np.random.default_rng(seed)
    sums = a[rng.integers(len(a), size=(N, len(a)))].sum(1)
    return tuple(float(x) for x in np.percentile(sums[:, 0] / sums[:, 1], [2.5, 97.5]))


def text_analysis(local):
    dt = load(A24 / "decodingtrust/responses.csv") + [
        r for r in local if r["task_kind"] == "decodingtrust"
    ]
    bold = load(A24 / "bold/scored_responses.csv") + load(
        ROOT / "results/gemma_bold/scored_responses.csv"
    )
    tests, summaries = [], []
    for r in dt:
        r["truncated"] = _as_bool(r.get("truncated")) or r.get("finish_reason") == "length"
        r["classification"] = (
            classify_response(r.get("response", ""))
            if r.get("status") not in {"runtime_error", "api_error"}
            else "API_ERROR"
        )
        r["agreement"] = int(r["classification"] == "AGREE")
    quality = []
    for benchmark, table in (("DecodingTrust", dt), ("BOLD", bold)):
        for key in ORDER:
            selected = subset(table, key)
            if selected:
                quality.append(
                    dict(
                        benchmark=benchmark,
                        model_key=key,
                        rows=len(selected),
                        truncated_rows=sum(
                            _as_bool(r.get("truncated")) or r.get("finish_reason") == "length"
                            for r in selected
                        ),
                        invalid_rows=sum(r.get("status") != "completed" for r in selected),
                        full_response_interpretation="unreliable:36/48 local Gemma DT outputs reached cap"
                        if benchmark == "DecodingTrust" and key == "gemma_local"
                        else "inspect truncation and validity counts",
                    )
                )
    write_csv(OUT / "generation_quality_by_task.csv", quality)
    write_csv(OUT / "decodingtrust_responses.csv", dt)
    write_csv(OUT / "bold_scored_responses.csv", bold)
    for benchmark, rows, group, fields in (
        ("DecodingTrust", dt, "demographic_group", ["agreement"]),
        ("BOLD", bold, "category", ["toxic_any", "sentiment_class", "regard_class"]),
    ):
        for index, key in enumerate(ORDER):
            for regime in (
                ("benign", "untargeted", "targeted")
                if benchmark == "DecodingTrust"
                else ("continuation",)
            ):
                selected = [
                    r
                    for r in subset(rows, key)
                    if (
                        r.get("regime") == regime
                        if benchmark == "DecodingTrust"
                        else r.get("toxic_any", "") != ""
                    )
                ]
                if not selected:
                    continue
                for field in fields:
                    usable = [
                        r
                        for r in selected
                        if r.get(field, "") != "" and r.get("classification") != "API_ERROR"
                    ]
                    effect, low, high, p = (
                        group_range(
                            usable,
                            field,
                            group,
                            block="topic" if benchmark == "DecodingTrust" else None,
                            seed=SEED + index,
                        )
                        if benchmark == "DecodingTrust"
                        else subject_cluster_range(usable, field, SEED + index)
                    )
                    tests.append(
                        dict(
                            benchmark=benchmark,
                            dataset="text_only",
                            model_key=key,
                            model_display_name=NAMES[key],
                            metric=field,
                            regime=regime,
                            effect=effect,
                            ci_low=low,
                            ci_high=high,
                            raw_p_value=p,
                            sample_count=len(usable)
                            if benchmark == "DecodingTrust"
                            else len({r["subject_name"] for r in usable}),
                            sample_unit="4 topic blocks, 4 race prompts each"
                            if benchmark == "DecodingTrust"
                            else "86 subject clusters; 100 prompts (25/category)",
                            test="max-minus-min race/category mean; within-topic permutation"
                            if benchmark == "DecodingTrust"
                            else "max-minus-min category mean; whole-subject cluster label permutation",
                            interval_method="95% topic-block bootstrap"
                            if benchmark == "DecodingTrust"
                            else "95% category-stratified subject-cluster bootstrap",
                        )
                    )
                    for category in sorted({r[group] for r in usable}):
                        items = [r for r in usable if r[group] == category]
                        values = [float(r[field]) for r in items]
                        ci = (
                            mean_ci(values, SEED + index)
                            if benchmark == "DecodingTrust"
                            else subject_cluster_ci(items, field, SEED + index)
                        )
                        summaries.append(
                            dict(
                                benchmark=benchmark,
                                model_key=key,
                                model_display_name=NAMES[key],
                                regime=regime,
                                metric=field,
                                category=category,
                                value=float(np.mean(values)),
                                ci_low=ci[0],
                                ci_high=ci[1],
                                responses=len(items),
                                unclassified_rate=sum(
                                    r.get("classification") == "NO_RESPONSE" for r in items
                                )
                                / len(items),
                                invalid_rate=sum(r.get("status") != "completed" for r in items)
                                / len(items),
                            )
                        )
        family = [r for r in tests if r["benchmark"] == benchmark]
        adjust(
            family,
            f"{benchmark}: all models x {'3 regimes' if benchmark == 'DecodingTrust' else '3 metrics'}",
        )
    write_csv(OUT / "text_category_summary.csv", summaries)
    write_csv(OUT / "text_significance.csv", tests)
    return tests


def visbias_analysis(rows, dataset):
    lexicon = json.loads((ROOT / "data/frozen/visbias/race_stereotype_lexicons.json").read_text())
    descriptions = score_descriptions(rows, lexicon)
    tests, comparisons, pair_scores = [], [], []
    for index, key in enumerate(ORDER):
        for metric in (
            "vader_compound",
            "iat_negative_rate",
            "stereotype_hits_per_100_words",
            "identity_mention",
        ):
            pairs = defaultdict(dict)
            for row in subset(descriptions, key):
                if row["status"] == "completed":
                    pairs[row["matched_pair_id"]][row["race"]] = float(row[metric])
            differences = [
                p["black"] - p["white"] for p in pairs.values() if set(p) == {"black", "white"}
            ]
            if not differences:
                continue
            low, high = mean_ci(differences, SEED + index)
            tests.append(
                dict(
                    benchmark=f"VisBias {dataset}",
                    dataset=dataset,
                    model_key=key,
                    model_display_name=NAMES[key],
                    metric=metric,
                    effect=float(np.mean(differences)),
                    ci_low=low,
                    ci_high=high,
                    raw_p_value=sign_test(differences, SEED + index),
                    sample_count=len(differences),
                    sample_unit="matched image pairs",
                    test="two-sided pair sign-flip",
                    interval_method="95% matched-pair bootstrap",
                )
            )
        selected = [r for r in subset(rows, key) if r["task_kind"] == "comparison"]
        if not selected:
            continue
        pairs = defaultdict(list)
        for r in selected:
            pairs[r["matched_pair_id"]].append(r)
        directions, unsupported = [], []
        for pair_id, items in sorted(pairs.items()):
            complete = len(items) == 4 and len({(r["attribute"], r["order"]) for r in items}) == 4
            valid = complete and all(r["status"] == "completed" for r in items)
            direction = np.mean(
                [
                    1
                    if r["selected_race"] == "black"
                    else -1
                    if r["selected_race"] == "white"
                    else 0
                    for r in items
                ]
            )
            choices = np.mean([r["parsed_answer"] in {"A", "B"} for r in items])
            pair_scores.append(
                dict(
                    dataset=dataset,
                    model_key=key,
                    matched_pair_id=pair_id,
                    presentations=len(items),
                    complete_and_valid=valid,
                    directional_score=float(direction),
                    unsupported_rate=float(choices),
                )
            )
            if valid:
                directions.append(direction)
                unsupported.append(choices)
        lo, hi = mean_ci(directions, SEED + index)
        tests.append(
            dict(
                benchmark=f"VisBias {dataset}",
                dataset=dataset,
                model_key=key,
                model_display_name=NAMES[key],
                metric="unsupported_selection_gap",
                effect=float(np.mean(directions)) if directions else math.nan,
                ci_low=lo,
                ci_high=hi,
                raw_p_value=sign_test(directions, SEED + index),
                sample_count=len(directions),
                sample_unit="face-pair clusters (2 attributes x 2 orders)",
                test="two-sided pair sign-flip",
                interval_method="95% face-pair bootstrap",
            )
        )
        ulo, uhi = mean_ci(unsupported, SEED + index)
        comparisons.append(
            dict(
                dataset=dataset,
                model_key=key,
                model_display_name=NAMES[key],
                responses=len(selected),
                complete_pairs=len(directions),
                unsupported_rate=sum(
                    r["status"] == "completed" and r["parsed_answer"] in {"A", "B"}
                    for r in selected
                )
                / len(selected),
                cannot_determine_rate=sum(
                    r["status"] == "completed" and r["parsed_answer"] == "CANNOT_DETERMINE"
                    for r in selected
                )
                / len(selected),
                invalid_rate=sum(r["status"] != "completed" for r in selected) / len(selected),
                black_selection_rate=sum(
                    r["status"] == "completed" and r["selected_race"] == "black" for r in selected
                )
                / len(selected),
                white_selection_rate=sum(
                    r["status"] == "completed" and r["selected_race"] == "white" for r in selected
                )
                / len(selected),
                complete_pair_unsupported_rate=float(np.mean(unsupported))
                if unsupported
                else math.nan,
                unsupported_ci_low=ulo,
                unsupported_ci_high=uhi,
            )
        )
    adjust(tests, f"VisBias {dataset}: all models x 5 metrics")
    for label, table in (
        ("responses", rows),
        ("scored_descriptions", descriptions),
        ("significance", tests),
        ("comparison_summary", comparisons),
        ("comparison_pair_scores", pair_scores),
    ):
        write_csv(OUT / f"visbias_{dataset}_{label}.csv", table)
    return tests


def continuity(local):
    old = normalize_iat(
        load(
            ROOT / "reference/august26/reference/local_pinned_gemma_iat/base_gemma3_neutral_iat.csv"
        ),
        "gemma_local",
    )
    new = normalize_iat(
        [r for r in local if r["task_kind"] == "iat" and r["dataset"] == "neutral12"]
    )
    lookup = {(r["image_id"], int(r["iteration"])): r for r in old}
    comparisons = []
    for r in new:
        prior = lookup.get((r["image_id"], int(r["iteration"])))
        comparisons.append(
            dict(
                image_id=r["image_id"],
                iteration=r["iteration"],
                old_words=prior["words"] if prior else "",
                new_words=r["words"],
                exact_match=prior is not None and prior["words"] == r["words"],
            )
        )
    write_csv(OUT / "gemma_baseline_continuity.csv", comparisons)
    write_json(
        OUT / "gemma_baseline_continuity.json",
        dict(
            archived_rows=len(old),
            new_rows=len(new),
            matched_response_tuples=sum(r["exact_match"] for r in comparisons),
            baseline_replaced=False,
            archive_sha256=sha(
                ROOT
                / "reference/august26/reference/local_pinned_gemma_iat/base_gemma3_neutral_iat.csv"
            ),
            reason="New inference is a continuity audit, never a replacement for the frozen pre-DPO baseline.",
        ),
    )
    return old


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    local = load(ROOT / "results/gemma_local/responses.csv")
    if len(local) != 956:
        raise ValueError(
            f"Expected 956 local tasks; found {len(local)}. Do not silently report a partial run."
        )
    manifest = json.loads((ROOT / "results/gemma_local/manifest.json").read_text())
    if not manifest.get("base_hashes_unchanged") or manifest.get("adapter_loaded"):
        raise ValueError("Pinned base integrity failed")
    neutral = normalize_iat(load(A24 / "iat12/responses.csv")) + continuity(local)
    utk = normalize_iat(
        [
            r
            for r in load(A31 / "utkface_iat/utkface_iat_eight_model_20260831_v1/responses.csv")
            if r["model_key"] != "gpt56_luna"
        ]
    )
    recovered = load(
        ROOT / "results/utkface_recovered/gpt_utkface_recovered_20260905/responses.csv"
    )
    utk += normalize_iat(
        recovered + [r for r in local if r["task_kind"] == "iat" and r["dataset"] == "utkface400"]
    )
    tests = iat_analysis(neutral, "neutral12") + iat_analysis(utk, "utkface") + text_analysis(local)
    hosted = load(ROOT / "results/hosted_visbias12/responses.csv")
    tests += visbias_analysis(
        hosted + [r for r in local if r["dataset"] == "neutral12" and r["task_kind"] != "iat"],
        "neutral12",
    )
    original = load(A31 / "visbias/visbias_race_eight_model_pilot_20260831_v1/responses.csv")
    tests += visbias_analysis(
        [r for r in original if r["model_key"] != "gpt56_luna"]
        + [r for r in local if r["dataset"] == "visbias_source84"],
        "original",
    )
    write_csv(OUT / "all_significance_tests.csv", tests)
    print(
        json.dumps(
            {
                "tests": len(tests),
                "bh_significant": sum(r["bh_adjusted_q_value"] < 0.05 for r in tests),
                "output": str(OUT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
