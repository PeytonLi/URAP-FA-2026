from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import IAT_WORDS, NEGATIVE_IAT_SET
from .plots import write_iat_plots
from .uncertainty import benjamini_hochberg, stratified_binary_label_permutation

if TYPE_CHECKING:
    from .iat import ModelSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, percentile))


def _bootstrap_by_image(
    records: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, dict[str, float | None]]:
    clusters: dict[tuple[str, str, str], tuple[int, int]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not _as_bool(record.get("valid_three")):
            continue
        grouped[(record["race"], record["gender"], record["image_id"])].append(record)
    for key, values in grouped.items():
        negative = sum(int(value.get("negative_count") or 0) for value in values)
        total = sum(int(value.get("word_count") or 0) for value in values)
        clusters[key] = (negative, total)

    rng = np.random.default_rng(seed)
    black_negative = np.zeros(samples, dtype=np.float64)
    black_total = np.zeros(samples, dtype=np.float64)
    white_negative = np.zeros(samples, dtype=np.float64)
    white_total = np.zeros(samples, dtype=np.float64)
    keys_by_stratum: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in clusters:
        keys_by_stratum[(key[0], key[1])].append(key)

    chunk_size = 5_000
    for (race, _gender), keys in keys_by_stratum.items():
        values = np.asarray([clusters[key] for key in keys], dtype=np.float64)
        target_negative = black_negative if race == "Black" else white_negative
        target_total = black_total if race == "Black" else white_total
        for start in range(0, samples, chunk_size):
            stop = min(samples, start + chunk_size)
            indexes = rng.integers(0, len(keys), size=(stop - start, len(keys)))
            selected = values[indexes]
            target_negative[start:stop] += selected[:, :, 0].sum(axis=1)
            target_total[start:stop] += selected[:, :, 1].sum(axis=1)
    black_rates = np.divide(
        black_negative,
        black_total,
        out=np.full(samples, np.nan, dtype=np.float64),
        where=black_total > 0,
    )
    white_rates = np.divide(
        white_negative,
        white_total,
        out=np.full(samples, np.nan, dtype=np.float64),
        where=white_total > 0,
    )
    gaps = black_rates - white_rates

    return {
        "Black": {
            "ci_low": _percentile(black_rates, 2.5),
            "ci_high": _percentile(black_rates, 97.5),
        },
        "White": {
            "ci_low": _percentile(white_rates, 2.5),
            "ci_high": _percentile(white_rates, 97.5),
        },
        "gap": {
            "ci_low": _percentile(gaps, 2.5),
            "ci_high": _percentile(gaps, 97.5),
            "probability_gap_gt_zero": float(np.mean(gaps[np.isfinite(gaps)] > 0))
            if np.isfinite(gaps).any()
            else None,
        },
    }


def _image_level_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    image_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if _as_bool(record.get("valid_three")):
            image_groups[(record["image_id"], record["race"], record["gender"])].append(
                record
            )
    rows = []
    for (image_id, race, gender), image_records in image_groups.items():
        negative = sum(int(record.get("negative_count") or 0) for record in image_records)
        terms = sum(int(record.get("word_count") or 0) for record in image_records)
        if terms:
            rows.append(
                {
                    "image_id": image_id,
                    "race": race,
                    "gender": gender,
                    "negative_rate": negative / terms,
                }
            )
    return rows


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if _as_bool(record.get("valid_three"))]
    negative = sum(int(record.get("negative_count") or 0) for record in valid)
    total = sum(int(record.get("word_count") or 0) for record in valid)
    exact = sum(_as_bool(record.get("exact_format")) for record in records)
    return {
        "responses": len(records),
        "valid_responses": len(valid),
        "validity_rate": _rate(len(valid), len(records)),
        "distinct_images": len({record.get("image_id") for record in records}),
        "images_with_valid_outputs": len(
            {record.get("image_id") for record in valid}
        ),
        "exact_format_rate": _rate(exact, len(records)),
        "negative_terms": negative,
        "total_terms": total,
        "negative_rate": _rate(negative, total),
        "mean_negative_terms_per_valid_response": _rate(negative, len(valid)),
    }


def _model_metrics(
    records: list[dict[str, Any]], bootstrap_samples: int, bootstrap_seed: int
) -> dict[str, Any]:
    race = {
        race_name: _group_metrics([record for record in records if record.get("race") == race_name])
        for race_name in ("Black", "White")
    }
    intersectional = {
        f"{race_name}_{gender}": _group_metrics(
            [
                record
                for record in records
                if record.get("race") == race_name and record.get("gender") == gender
            ]
        )
        for race_name in ("Black", "White")
        for gender in ("Female", "Male")
    }
    valid = [record for record in records if _as_bool(record.get("valid_three"))]
    word_counts = Counter()
    tuples = Counter()
    for record in valid:
        words = [word for word in str(record.get("words") or "").split("|") if word]
        word_counts.update(words)
        tuples[tuple(words)] += 1
    total_words = sum(word_counts.values())
    entropy = 0.0
    if total_words:
        probabilities = np.array(list(word_counts.values()), dtype=np.float64) / total_words
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
    normalized_entropy = entropy / math.log2(len(IAT_WORDS)) if entropy else 0.0
    gap = race["Black"]["negative_rate"] - race["White"]["negative_rate"]
    bootstrap = _bootstrap_by_image(records, bootstrap_samples, bootstrap_seed)
    race["Black"].update(bootstrap["Black"])
    race["White"].update(bootstrap["White"])
    return {
        "responses": len(records),
        "valid_responses": len(valid),
        "validity_rate": _rate(len(valid), len(records)),
        "distinct_images": len({record.get("image_id") for record in records}),
        "images_with_valid_outputs": len(
            {record.get("image_id") for record in valid}
        ),
        "total_cost_usd": sum(float(record.get("cost_usd") or 0.0) for record in records),
        "race": race,
        "intersectional": intersectional,
        "black_minus_white_negative_rate_gap": _finite_or_none(gap),
        "gap_ci_low": bootstrap["gap"]["ci_low"],
        "gap_ci_high": bootstrap["gap"]["ci_high"],
        "probability_gap_gt_zero": bootstrap["gap"]["probability_gap_gt_zero"],
        "unique_response_tuples": len(tuples),
        "response_tuple_diversity_rate": _rate(len(tuples), len(valid)),
        "distinct_iat_words": len(word_counts),
        "word_entropy_bits": entropy,
        "normalized_word_entropy": normalized_entropy,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_iat_analysis(
    records: list[dict[str, Any]],
    model_specs: list[ModelSpec],
    output_dir: Path,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "bootstrap": {
            "unit": "image cluster stratified by race and gender",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
        },
        "models": {},
    }
    summary_rows = []
    intersectional_rows = []
    word_rows = []
    significance_rows = []
    for model_index, model in enumerate(model_specs):
        model_records = [record for record in records if record.get("model_key") == model.key]
        model_metrics = _model_metrics(
            model_records,
            bootstrap_samples,
            bootstrap_seed + model_index,
        )
        metrics["models"][model.key] = model_metrics
        image_rows = _image_level_rows(model_records)
        observed, p_value, permutations, permutation_method = (
            stratified_binary_label_permutation(
            image_rows,
            "negative_rate",
            "race",
            "gender",
            group_a="Black",
            group_b="White",
            samples=max(20_000, bootstrap_samples),
            seed=bootstrap_seed + 10_000 + model_index,
            )
        )
        significance_rows.append(
            {
                "model_key": model.key,
                "model_display_name": model.display_name,
                "images_with_valid_outputs": len(image_rows),
                "observed_black_minus_white_gap": observed,
                "two_sided_permutation_p_value": p_value,
                "permutation_method": permutation_method,
                "stratified_label_permutations": permutations,
            }
        )
        black = model_metrics["race"]["Black"]
        white = model_metrics["race"]["White"]
        summary_rows.append(
            {
                "model_key": model.key,
                "model_display_name": model.display_name,
                "model_id": model.model_id,
                "responses": model_metrics["responses"],
                "valid_responses": model_metrics["valid_responses"],
                "validity_rate": model_metrics["validity_rate"],
                "distinct_images": model_metrics["distinct_images"],
                "images_with_valid_outputs": model_metrics["images_with_valid_outputs"],
                "black_negative_rate": black["negative_rate"],
                "black_ci_low": black["ci_low"],
                "black_ci_high": black["ci_high"],
                "white_negative_rate": white["negative_rate"],
                "white_ci_low": white["ci_low"],
                "white_ci_high": white["ci_high"],
                "black_minus_white_gap": model_metrics["black_minus_white_negative_rate_gap"],
                "gap_ci_low": model_metrics["gap_ci_low"],
                "gap_ci_high": model_metrics["gap_ci_high"],
                "probability_gap_gt_zero": model_metrics["probability_gap_gt_zero"],
                "unique_response_tuples": model_metrics["unique_response_tuples"],
                "response_tuple_diversity_rate": model_metrics["response_tuple_diversity_rate"],
                "distinct_iat_words": model_metrics["distinct_iat_words"],
                "normalized_word_entropy": model_metrics["normalized_word_entropy"],
                "total_cost_usd": model_metrics["total_cost_usd"],
            }
        )
        for group_name, group_metrics in model_metrics["intersectional"].items():
            race, gender = group_name.split("_", 1)
            intersectional_rows.append(
                {
                    "model_key": model.key,
                    "model_display_name": model.display_name,
                    "race": race,
                    "gender": gender,
                    **group_metrics,
                }
            )
        for race in ("Black", "White"):
            counts = Counter()
            for record in model_records:
                if record.get("race") != race or not _as_bool(record.get("valid_three")):
                    continue
                counts.update(word for word in str(record.get("words") or "").split("|") if word)
            for word in IAT_WORDS:
                word_rows.append(
                    {
                        "model_key": model.key,
                        "model_display_name": model.display_name,
                        "race": race,
                        "word": word,
                        "valence": "negative" if word in NEGATIVE_IAT_SET else "positive",
                        "count": counts[word],
                    }
                )

    adjusted = benjamini_hochberg(
        [float(row["two_sided_permutation_p_value"]) for row in significance_rows]
    )
    for row, q_value in zip(significance_rows, adjusted, strict=True):
        row["bh_adjusted_p_value_across_models"] = q_value
        metrics["models"][row["model_key"]]["exploratory_race_permutation"] = row
    metrics["exploratory_significance"] = {
        "test": "two-sided Black/White image-label permutation stratified by gender",
        "multiplicity": (
            f"Benjamini-Hochberg correction across {len(significance_rows)} models"
        ),
        "unit": "mean negative-term rate per distinct image",
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "model_summary.csv", summary_rows)
    _write_csv(output_dir / "intersectional_metrics.csv", intersectional_rows)
    _write_csv(output_dir / "word_counts.csv", word_rows)
    _write_csv(output_dir / "exploratory_significance.csv", significance_rows)
    plot_paths = write_iat_plots(metrics, word_rows, model_specs, output_dir / "plots")
    return {
        "metrics_json": _portable_path(output_dir / "metrics.json"),
        "model_summary_csv": _portable_path(output_dir / "model_summary.csv"),
        "intersectional_metrics_csv": _portable_path(output_dir / "intersectional_metrics.csv"),
        "word_counts_csv": _portable_path(output_dir / "word_counts.csv"),
        "exploratory_significance_csv": _portable_path(
            output_dir / "exploratory_significance.csv"
        ),
        "plots": [_portable_path(path) for path in plot_paths],
    }
