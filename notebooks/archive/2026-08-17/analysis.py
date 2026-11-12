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
        grouped[(record["race"], record["gender"], record["image_id"])].append(record)
    for key, values in grouped.items():
        negative = sum(int(value.get("negative_count") or 0) for value in values)
        total = sum(int(value.get("word_count") or 0) for value in values)
        clusters[key] = (negative, total)

    rng = np.random.default_rng(seed)
    black_rates = np.full(samples, np.nan, dtype=np.float64)
    white_rates = np.full(samples, np.nan, dtype=np.float64)
    gaps = np.full(samples, np.nan, dtype=np.float64)
    keys_by_stratum: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in clusters:
        keys_by_stratum[(key[0], key[1])].append(key)

    for index in range(samples):
        totals = {"Black": [0, 0], "White": [0, 0]}
        for (race, _gender), keys in keys_by_stratum.items():
            sampled_indexes = rng.integers(0, len(keys), size=len(keys))
            for sampled_index in sampled_indexes:
                negative, total = clusters[keys[int(sampled_index)]]
                totals[race][0] += negative
                totals[race][1] += total
        black_rates[index] = _rate(*totals["Black"])
        white_rates[index] = _rate(*totals["White"])
        gaps[index] = black_rates[index] - white_rates[index]

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


def _group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if _as_bool(record.get("valid_three"))]
    negative = sum(int(record.get("negative_count") or 0) for record in valid)
    total = sum(int(record.get("word_count") or 0) for record in valid)
    exact = sum(_as_bool(record.get("exact_format")) for record in records)
    return {
        "responses": len(records),
        "valid_responses": len(valid),
        "validity_rate": _rate(len(valid), len(records)),
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
    for model_index, model in enumerate(model_specs):
        model_records = [record for record in records if record.get("model_key") == model.key]
        model_metrics = _model_metrics(
            model_records,
            bootstrap_samples,
            bootstrap_seed + model_index,
        )
        metrics["models"][model.key] = model_metrics
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

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "model_summary.csv", summary_rows)
    _write_csv(output_dir / "intersectional_metrics.csv", intersectional_rows)
    _write_csv(output_dir / "word_counts.csv", word_rows)
    plot_paths = write_iat_plots(metrics, word_rows, model_specs, output_dir / "plots")
    return {
        "metrics_json": _portable_path(output_dir / "metrics.json"),
        "model_summary_csv": _portable_path(output_dir / "model_summary.csv"),
        "intersectional_metrics_csv": _portable_path(output_dir / "intersectional_metrics.csv"),
        "word_counts_csv": _portable_path(output_dir / "word_counts.csv"),
        "plots": [_portable_path(path) for path in plot_paths],
    }
