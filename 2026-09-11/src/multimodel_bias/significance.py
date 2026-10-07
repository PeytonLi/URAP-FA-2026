from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ORDER = (
    "qwen35_9b",
    "kimi_k3",
    "gpt56_luna",
    "gemini36_flash",
    "mimo_v25",
    "minimax_m3",
    "step37_flash",
    "grok46",
)
REGIME_LABELS = {
    "benign": "Benign",
    "untargeted": "Untargeted",
    "targeted": "Race-targeted",
}
METRIC_LABELS = {
    "toxicity_rate": "Toxicity flag rate",
    "mean_sentiment": "Mean sentiment class",
    "mean_regard": "Mean regard class",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, str], *fields: str) -> float:
    for field in fields:
        value = row.get(field, "")
        if value not in {"", None}:
            return float(value)
    return math.nan


def _bootstrap_group_range(
    rows: list[dict[str, str]],
    group_field: str,
    value: Callable[[dict[str, str]], float],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    grouped: dict[str, np.ndarray] = {}
    raw: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        item = value(row)
        if math.isfinite(item):
            raw[str(row[group_field])].append(item)
    for group, values in raw.items():
        grouped[group] = np.asarray(values, dtype=np.float64)
    if len(grouped) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = []
    for group in sorted(grouped):
        values = grouped[group]
        indexes = rng.integers(0, len(values), size=(samples, len(values)))
        means.append(values[indexes].mean(axis=1))
    matrix = np.vstack(means)
    ranges = matrix.max(axis=0) - matrix.min(axis=0)
    low, high = np.percentile(ranges, [2.5, 97.5])
    return float(low), float(high)


def build_iat_significance(reference_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    summaries = {
        row["model_key"]: row for row in _read_csv(reference_dir / "model_summary.csv")
    }
    significance = _read_csv(reference_dir / "exploratory_significance.csv")
    rows: list[dict[str, Any]] = []
    for row in significance:
        summary = summaries[row["model_key"]]
        rows.append(
            {
                "benchmark": "IAT-style 12-image",
                "metric": "Black minus White negative-term rate",
                "model_key": row["model_key"],
                "model_display_name": row["model_display_name"],
                "effect": _number(summary, "black_minus_white_gap"),
                "ci_low": _number(summary, "gap_ci_low"),
                "ci_high": _number(summary, "gap_ci_high"),
                "raw_p_value": _number(
                    row,
                    "two_sided_exact_permutation_p_value",
                    "two_sided_permutation_p_value",
                ),
                "bh_adjusted_q_value": _number(
                    row, "bh_adjusted_p_value_across_models"
                ),
                "sample_count": int(float(row["images_with_valid_outputs"])),
                "sample_unit": "distinct image clusters",
                "test": "two-sided race-label permutation stratified by recorded gender",
            }
        )
    _write_csv(output_dir / "iat_significance_with_ci.csv", rows)
    return rows


def build_decodingtrust_significance(
    reference_dir: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    responses = _read_csv(reference_dir / "responses.csv")
    significance = _read_csv(reference_dir / "demographic_significance.csv")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(significance):
        selected = [
            response
            for response in responses
            if response["model_key"] == row["model_key"]
            and response["regime"] == row["regime"]
            and response["classification"] not in {"API_ERROR"}
        ]
        low, high = _bootstrap_group_range(
            selected,
            "demographic_group",
            lambda response: 1.0 if response["classification"] == "AGREE" else 0.0,
            samples=bootstrap_samples,
            seed=seed + index,
        )
        rows.append(
            {
                "benchmark": "DecodingTrust race",
                "metric": f"{REGIME_LABELS[row['regime']]} max-minus-min agreement",
                "regime": row["regime"],
                "model_key": row["model_key"],
                "model_display_name": row["model_display_name"],
                "effect": _number(row, "observed_max_minus_min_agreement"),
                "ci_low": low,
                "ci_high": high,
                "raw_p_value": _number(row, "permutation_p_value"),
                "bh_adjusted_q_value": _number(row, "bh_adjusted_p_value"),
                "sample_count": int(float(row["responses"])),
                "sample_unit": "prompt responses across four race groups",
                "test": "exploratory max-minus-min race-group label permutation",
            }
        )
    _write_csv(output_dir / "decodingtrust_significance_with_ci.csv", rows)
    return rows


def build_bold_significance(
    reference_dir: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    responses = _read_csv(reference_dir / "scored_responses.csv")
    significance = _read_csv(reference_dir / "category_significance.csv")
    value_functions: dict[str, Callable[[dict[str, str]], float]] = {
        "toxicity_rate": lambda row: float(row["toxic_any"]),
        "mean_sentiment": lambda row: float(row["sentiment_class"]),
        "mean_regard": lambda row: float(row["regard_class"]),
    }
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(significance):
        selected = [
            response
            for response in responses
            if response["model_key"] == row["model_key"]
            and response["status"] == "completed"
        ]
        low, high = _bootstrap_group_range(
            selected,
            "category",
            value_functions[row["metric"]],
            samples=bootstrap_samples,
            seed=seed + index,
        )
        rows.append(
            {
                "benchmark": "BOLD race",
                "metric": METRIC_LABELS[row["metric"]],
                "metric_key": row["metric"],
                "model_key": row["model_key"],
                "model_display_name": row["model_display_name"],
                "effect": _number(row, "observed_max_minus_min"),
                "ci_low": low,
                "ci_high": high,
                "raw_p_value": _number(row, "permutation_p_value"),
                "bh_adjusted_q_value": _number(row, "bh_adjusted_p_value"),
                "sample_count": int(float(row["responses"])),
                "sample_unit": "continuations across four race categories",
                "test": "exploratory max-minus-min race-category label permutation",
            }
        )
    _write_csv(output_dir / "bold_significance_with_ci.csv", rows)
    return rows


def _ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rank = {key: index for index, key in enumerate(MODEL_ORDER)}
    return sorted(rows, key=lambda row: rank[row["model_key"]])


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_iat(rows: list[dict[str, Any]], output: Path) -> None:
    rows = _ordered(rows)
    y = np.arange(len(rows))
    effects = np.asarray([100 * row["effect"] for row in rows])
    low = np.asarray([100 * row["ci_low"] for row in rows])
    high = np.asarray([100 * row["ci_high"] for row in rows])
    fig, ax = plt.subplots(figsize=(12, 7.3))
    colors = ["#b2182b" if row["bh_adjusted_q_value"] < 0.05 else "#2166ac" for row in rows]
    for index, color in enumerate(colors):
        ax.errorbar(
            effects[index],
            y[index],
            xerr=[[effects[index] - low[index]], [high[index] - effects[index]]],
            fmt="none",
            ecolor=color,
            capsize=4,
            linewidth=2,
        )
    ax.scatter(effects, y, c=colors, s=65, zorder=3)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks(y, [row["model_display_name"] for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Black minus White negative-term rate (percentage points)")
    ax.set_title("IAT-Style Race Gap: Effect Size, 95% CI, and Significance")
    ax.grid(axis="x", alpha=0.22)
    for index, row in enumerate(rows):
        ax.text(
            high[index] + 1.0,
            index,
            f"p={row['raw_p_value']:.3f}; q={row['bh_adjusted_q_value']:.3f}; "
            f"n={row['sample_count']}",
            va="center",
            fontsize=8.5,
        )
    fig.text(
        0.07,
        0.01,
        "Positive values mean the model selected a larger share of negative IAT words for "
        "Black-face images. Points are observed gaps; bars are image-cluster bootstrap intervals. "
        "No model met "
        "the BH-adjusted q < 0.05 threshold in this 12-image study.",
        fontsize=9.5,
        color="#333333",
        wrap=True,
    )
    fig.subplots_adjust(left=0.20, right=0.77, bottom=0.16, top=0.90)
    _save_figure(fig, output)


def plot_decodingtrust(rows: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.7), sharey=True)
    for axis, regime in zip(axes, ("benign", "untargeted", "targeted"), strict=True):
        selected = _ordered([row for row in rows if row["regime"] == regime])
        y = np.arange(len(selected))
        effects = np.asarray([100 * row["effect"] for row in selected])
        low = np.asarray([100 * row["ci_low"] for row in selected])
        high = np.asarray([100 * row["ci_high"] for row in selected])
        axis.errorbar(
            effects,
            y,
            xerr=np.vstack([effects - low, high - effects]),
            fmt="o",
            color="#2166ac",
            ecolor="#6b8fb3",
            capsize=3,
        )
        axis.set_title(REGIME_LABELS[regime])
        axis.set_xlabel("Largest race-group agreement gap (pp)")
        axis.set_xlim(-2, max(20.0, float(high.max()) + 18.0))
        axis.grid(axis="x", alpha=0.22)
        axis.set_yticks(y, [row["model_display_name"] for row in selected])
        axis.invert_yaxis()
        for index, row in enumerate(selected):
            axis.text(
                high[index] + 1.5,
                index,
                f"p={row['raw_p_value']:.2f}\nq={row['bh_adjusted_q_value']:.2f}",
                va="center",
                fontsize=7.5,
            )
    fig.suptitle("DecodingTrust Race-Group Differences by Prompt Regime", fontsize=15)
    fig.text(
        0.06,
        0.01,
        "The x-axis is the difference between the highest and lowest stereotype-agreement rate "
        "among four race groups; 0 means equal observed rates. Intervals use response bootstrap "
        "resampling. All exploratory race-group tests had BH-adjusted q = 1.00.",
        fontsize=9.5,
        color="#333333",
        wrap=True,
    )
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.16, top=0.86, wspace=0.22)
    _save_figure(fig, output)


def plot_bold(rows: list[dict[str, Any]], output: Path) -> None:
    metric_order = ("toxicity_rate", "mean_sentiment", "mean_regard")
    axis_labels = {
        "toxicity_rate": "Largest category toxicity gap (pp)",
        "mean_sentiment": "Largest category mean-sentiment gap",
        "mean_regard": "Largest category mean-regard gap",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.7), sharey=True)
    for axis, metric in zip(axes, metric_order, strict=True):
        selected = _ordered([row for row in rows if row["metric_key"] == metric])
        y = np.arange(len(selected))
        scale = 100.0 if metric == "toxicity_rate" else 1.0
        effects = np.asarray([scale * row["effect"] for row in selected])
        low = np.asarray([scale * row["ci_low"] for row in selected])
        high = np.asarray([scale * row["ci_high"] for row in selected])
        axis.errorbar(
            effects,
            y,
            xerr=np.vstack([effects - low, high - effects]),
            fmt="o",
            color="#1b7837",
            ecolor="#75a784",
            capsize=3,
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel(axis_labels[metric])
        right_margin = 2.0 if metric == "toxicity_rate" else 0.15
        axis.set_xlim(
            -0.01 if metric != "toxicity_rate" else -0.2,
            float(high.max()) + right_margin,
        )
        axis.grid(axis="x", alpha=0.22)
        axis.set_yticks(y, [row["model_display_name"] for row in selected])
        axis.invert_yaxis()
        for index, row in enumerate(selected):
            axis.text(
                high[index] + (0.025 if metric != "toxicity_rate" else 0.3),
                index,
                f"p={row['raw_p_value']:.3f}\nq={row['bh_adjusted_q_value']:.3f}",
                va="center",
                fontsize=7.5,
            )
    fig.suptitle("BOLD Race-Category Differences: Effect Size and Significance", fontsize=15)
    fig.text(
        0.055,
        0.008,
        "Each x-axis shows the largest minus smallest score among four race categories. "
        "Toxicity is "
        "the percentage flagged when any Detoxify 'unbiased' score exceeded 0.3; sentiment and "
        "regard use classes from -1 (negative) to +1 (positive). No BH-adjusted q was below 0.05.",
        fontsize=9.5,
        color="#333333",
        wrap=True,
    )
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.17, top=0.86, wspace=0.25)
    _save_figure(fig, output)


def build_all_significance_outputs(
    reference_root: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int = 20_000,
    seed: int = 31082026,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    iat = build_iat_significance(reference_root / "iat12", output_dir)
    decodingtrust = build_decodingtrust_significance(
        reference_root / "decodingtrust",
        output_dir,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    bold = build_bold_significance(
        reference_root / "bold",
        output_dir,
        bootstrap_samples=bootstrap_samples,
        seed=seed + 1000,
    )
    plots = output_dir / "plots"
    plot_iat(iat, plots / "iat_significance_forest.png")
    plot_decodingtrust(decodingtrust, plots / "decodingtrust_significance_forest.png")
    plot_bold(bold, plots / "bold_significance_forest.png")
    combined = iat + decodingtrust + bold
    _write_csv(output_dir / "all_significance_tests.csv", combined)
    return {
        "iat_rows": len(iat),
        "decodingtrust_rows": len(decodingtrust),
        "bold_rows": len(bold),
        "corrected_significant": sum(row["bh_adjusted_q_value"] < 0.05 for row in combined),
        "plots": [
            str(plots / "iat_significance_forest.png"),
            str(plots / "decodingtrust_significance_forest.png"),
            str(plots / "bold_significance_forest.png"),
        ],
    }
