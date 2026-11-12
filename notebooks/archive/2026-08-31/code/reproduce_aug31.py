#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "aug31-matplotlib"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
MODEL_RANK = {key: index for index, key in enumerate(MODEL_ORDER)}
BLACK_COLOR = "#0072B2"
WHITE_COLOR = "#E69F00"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in {"", None} else math.nan


def truthy(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def ordered(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: MODEL_RANK[row["model_key"]])


def require_close(observed: float, expected: float, label: str) -> None:
    if not math.isclose(observed, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"{label}: observed {observed}, expected {expected}")


def add_finding(fig: plt.Figure, finding: str, *, width: int = 145) -> None:
    fig.text(
        0.5,
        0.015,
        textwrap.fill(finding, width=width),
        ha="center",
        fontsize=9.5,
        color="#303840",
    )


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def error_extent(value: float, low: float, high: float) -> tuple[float, float]:
    return max(0.0, value - low), max(0.0, high - value)


def validate_raw_data(root: Path) -> dict[str, Any]:
    raw = root / "data/raw"
    frozen = root / "data/frozen"
    derived = root / "data/derived"

    utk_rows = read_csv(raw / "utkface_responses.csv")
    utk_summary = read_csv(derived / "utkface_model_summary.csv")
    if len(utk_rows) != 2_800:
        raise ValueError(f"UTKFace rows: observed {len(utk_rows)}, expected 2800")
    for summary in utk_summary:
        model_rows = [row for row in utk_rows if row["model_key"] == summary["model_key"]]
        valid = [row for row in model_rows if truthy(row["valid_three"])]
        if len(model_rows) != int(summary["responses"]):
            raise ValueError(f"UTKFace response-count drift for {summary['model_key']}")
        if len(valid) != int(summary["valid_responses"]):
            raise ValueError(f"UTKFace validity-count drift for {summary['model_key']}")
        for race, field in (
            ("Black", "black_negative_rate"),
            ("White", "white_negative_rate"),
        ):
            selected = [row for row in valid if row["race"] == race]
            observed = sum(int(row["negative_count"]) for row in selected) / sum(
                int(row["word_count"]) for row in selected
            )
            require_close(observed, float(summary[field]), f"{summary['model_key']} {race}")

    vis_rows = read_csv(raw / "visbias_responses.csv")
    comparison_summary = read_csv(derived / "visbias_comparison_summary.csv")
    if len(vis_rows) != 1_764:
        raise ValueError(f"VisBias rows: observed {len(vis_rows)}, expected 1764")
    for summary in comparison_summary:
        rows = [
            row
            for row in vis_rows
            if row["model_key"] == summary["model_key"] and row["task_kind"] == "comparison"
        ]
        total = len(rows)
        completed = sum(row["status"] == "completed" for row in rows)
        unsupported = sum(row["selected_race"] in {"black", "white"} for row in rows)
        cannot = sum(row["parsed_answer"] == "CANNOT_DETERMINE" for row in rows)
        unclassified = total - unsupported - cannot
        if total != int(summary["responses"]) or completed != int(summary["valid_responses"]):
            raise ValueError(f"VisBias response-count drift for {summary['model_key']}")
        require_close(
            unsupported / total,
            float(summary["unsupported_choice_rate"]),
            f"{summary['model_key']} unsupported rate",
        )
        require_close(
            cannot / total,
            float(summary["cannot_determine_rate"]),
            f"{summary['model_key']} cannot-determine rate",
        )
        require_close(
            unclassified / total,
            float(summary["unclassified_rate"]),
            f"{summary['model_key']} unclassified rate",
        )

    evaluation = read_csv(frozen / "utkface_evaluation_manifest.csv")
    training = read_csv(frozen / "utkface_finetuning_manifest.csv")
    if len(evaluation) != 400 or len(training) != 400:
        raise ValueError("UTKFace evaluation and future-training pools must each contain 400 rows")
    overlap = {row["sha256"] for row in evaluation} & {row["sha256"] for row in training}
    if overlap:
        raise ValueError(f"UTKFace evaluation/training hash overlap: {len(overlap)}")
    for pool_name, rows in (("evaluation", evaluation), ("training", training)):
        cells = Counter((row["race"], row["gender"]) for row in rows)
        if set(cells.values()) != {100} or len(cells) != 4:
            raise ValueError(f"UTKFace {pool_name} race/gender balance drift: {dict(cells)}")

    dpo_rows = read_csv(frozen / "utkface_future_dpo_tasks.csv")
    if len(dpo_rows) != 800:
        raise ValueError(f"Future DPO tasks: observed {len(dpo_rows)}, expected 800")
    if any(truthy(row["dpo_training_ready"]) for row in dpo_rows):
        raise ValueError("Prompt-only future DPO rows must not be marked training-ready")
    if any(row["chosen"] or row["rejected"] for row in dpo_rows):
        raise ValueError("Prompt-only future DPO rows must have blank chosen/rejected fields")

    description_tasks = read_csv(frozen / "visbias_description_tasks.csv")
    comparison_tasks = read_csv(frozen / "visbias_comparison_tasks.csv")
    if len(description_tasks) != 84 or len(comparison_tasks) != 168:
        raise ValueError("VisBias frozen task counts must be 84 descriptions and 168 comparisons")

    archived_inputs = {
        "iat": len(read_csv(raw / "archived_iat12_responses.csv")),
        "decodingtrust": len(read_csv(raw / "archived_decodingtrust_responses.csv")),
        "bold": len(read_csv(raw / "archived_bold_scored_responses.csv")),
    }
    if any(count == 0 for count in archived_inputs.values()):
        raise ValueError("An archived significance input is empty")

    return {
        "utkface_rows": len(utk_rows),
        "utkface_valid_rows": sum(truthy(row["valid_three"]) for row in utk_rows),
        "visbias_rows": len(vis_rows),
        "visbias_valid_rows": sum(row["status"] == "completed" for row in vis_rows),
        "utkface_evaluation_rows": len(evaluation),
        "utkface_future_training_rows": len(training),
        "utkface_cross_pool_hash_overlap": len(overlap),
        "future_dpo_tasks": len(dpo_rows),
        "visbias_description_tasks": len(description_tasks),
        "visbias_comparison_tasks": len(comparison_tasks),
        "archived_input_rows": archived_inputs,
    }


def plot_archived_iat(rows: list[dict[str, str]], output: Path) -> Path:
    selected = ordered([row for row in rows if row["benchmark"] == "IAT-style 12-image"])
    y = np.arange(len(selected))
    effect = np.asarray([100 * number(row, "effect") for row in selected])
    low = np.asarray([100 * number(row, "ci_low") for row in selected])
    high = np.asarray([100 * number(row, "ci_high") for row in selected])
    fig, ax = plt.subplots(figsize=(12, 7.3))
    ax.errorbar(effect, y, xerr=np.vstack([effect - low, high - effect]), fmt="o", capsize=4)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_yticks(y, [row["model_display_name"] for row in selected])
    ax.invert_yaxis()
    ax.set_xlabel("Black minus White negative-term rate (percentage points)")
    ax.set_title("IAT-Style Race Gap: Effect Size, 95% CI, and Significance")
    ax.grid(axis="x", alpha=0.22)
    for index, row in enumerate(selected):
        ax.text(
            high[index] + 1,
            index,
            f"p={number(row, 'raw_p_value'):.3f}; q={number(row, 'bh_adjusted_q_value'):.3f}; "
            f"n={row['sample_count']}",
            va="center",
            fontsize=8.5,
        )
    add_finding(
        fig,
        "Positive values mean a higher negative-word rate for Black-face images. No model "
        "met the BH-adjusted q < 0.05 threshold in this archived 12-image study.",
    )
    fig.subplots_adjust(left=0.20, right=0.77, bottom=0.16, top=0.90)
    return save(fig, output)


def plot_archived_decodingtrust(rows: list[dict[str, str]], output: Path) -> Path:
    labels = {"benign": "Benign", "untargeted": "Untargeted", "targeted": "Race-targeted"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.7), sharey=True)
    for axis, regime in zip(axes, labels, strict=True):
        selected = ordered(
            [row for row in rows if row["benchmark"] == "DecodingTrust race" and row["regime"] == regime]
        )
        y = np.arange(len(selected))
        effect = np.asarray([100 * number(row, "effect") for row in selected])
        low = np.asarray([100 * number(row, "ci_low") for row in selected])
        high = np.asarray([100 * number(row, "ci_high") for row in selected])
        axis.errorbar(effect, y, xerr=np.vstack([effect - low, high - effect]), fmt="o", capsize=3)
        axis.set_title(labels[regime])
        axis.set_xlabel("Largest race-group agreement gap (pp)")
        axis.set_yticks(y, [row["model_display_name"] for row in selected])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.22)
        axis.set_xlim(-2, max(20.0, float(high.max()) + 18.0))
        for index, row in enumerate(selected):
            axis.text(
                high[index] + 1.5,
                index,
                f"p={number(row, 'raw_p_value'):.2f}\nq={number(row, 'bh_adjusted_q_value'):.2f}",
                va="center",
                fontsize=7.5,
            )
    fig.suptitle("DecodingTrust Race-Group Differences by Prompt Regime", fontsize=15)
    add_finding(
        fig,
        "The x-axis is the largest minus smallest stereotype-agreement rate among four race "
        "groups; zero means equal observed rates. All exploratory tests had BH-adjusted q=1.00.",
    )
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.16, top=0.86, wspace=0.22)
    return save(fig, output)


def plot_archived_bold(rows: list[dict[str, str]], output: Path) -> Path:
    metric_order = ("toxicity_rate", "mean_sentiment", "mean_regard")
    titles = {
        "toxicity_rate": "Toxicity flag rate",
        "mean_sentiment": "Mean sentiment class",
        "mean_regard": "Mean regard class",
    }
    axis_labels = {
        "toxicity_rate": "Largest category toxicity gap (pp)",
        "mean_sentiment": "Largest category mean-sentiment gap",
        "mean_regard": "Largest category mean-regard gap",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.7), sharey=True)
    for axis, metric in zip(axes, metric_order, strict=True):
        selected = ordered(
            [row for row in rows if row["benchmark"] == "BOLD race" and row["metric_key"] == metric]
        )
        scale = 100.0 if metric == "toxicity_rate" else 1.0
        y = np.arange(len(selected))
        effect = np.asarray([scale * number(row, "effect") for row in selected])
        low = np.asarray([scale * number(row, "ci_low") for row in selected])
        high = np.asarray([scale * number(row, "ci_high") for row in selected])
        axis.errorbar(effect, y, xerr=np.vstack([effect - low, high - effect]), fmt="o", capsize=3)
        axis.set_title(titles[metric])
        axis.set_xlabel(axis_labels[metric])
        axis.set_yticks(y, [row["model_display_name"] for row in selected])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.22)
        right_margin = 2.0 if metric == "toxicity_rate" else 0.15
        axis.set_xlim(-0.2 if metric == "toxicity_rate" else -0.01, float(high.max()) + right_margin)
        for index, row in enumerate(selected):
            offset = 0.3 if metric == "toxicity_rate" else 0.025
            axis.text(
                high[index] + offset,
                index,
                f"p={number(row, 'raw_p_value'):.3f}\nq={number(row, 'bh_adjusted_q_value'):.3f}",
                va="center",
                fontsize=7.5,
            )
    fig.suptitle("BOLD Race-Category Differences: Effect Size and Significance", fontsize=15)
    add_finding(
        fig,
        "Each x-axis is the largest minus smallest score among four race categories. Toxicity "
        "is the percentage where any Detoxify unbiased score exceeded 0.3; sentiment and regard "
        "range from -1 (negative) to +1 (positive). No BH-adjusted q was below 0.05.",
    )
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.17, top=0.86, wspace=0.25)
    return save(fig, output)


def plot_utkface_rates(rows: list[dict[str, str]], output: Path) -> Path:
    rows = ordered(rows)
    labels = [row["model_display_name"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    black = np.asarray([number(row, "black_negative_rate") for row in rows])
    white = np.asarray([number(row, "white_negative_rate") for row in rows])
    black_errors = np.asarray(
        [error_extent(black[i], number(row, "black_ci_low"), number(row, "black_ci_high")) for i, row in enumerate(rows)]
    ).T
    white_errors = np.asarray(
        [error_extent(white[i], number(row, "white_ci_low"), number(row, "white_ci_high")) for i, row in enumerate(rows)]
    ).T
    fig, ax = plt.subplots(figsize=(12, 6.2))
    black_bars = ax.bar(x - width / 2, black * 100, width, label="Black photo", color=BLACK_COLOR, yerr=black_errors * 100, capsize=4)
    white_bars = ax.bar(x + width / 2, white * 100, width, label="White photo", color=WHITE_COLOR, yerr=white_errors * 100, capsize=4)
    ax.bar_label(black_bars, labels=[f"{value * 100:.1f}%" for value in black], padding=3, fontsize=8)
    ax.bar_label(white_bars, labels=[f"{value * 100:.1f}%" for value in white], padding=3, fontsize=8)
    ax.set_title(f"{len(rows)}-Model UTKFace IAT-Style Negative-Term Comparison\n95% image-cluster bootstrap CI")
    ax.set_ylabel("Negative terms among valid IAT terms (%)")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    gaps = black - white
    largest = int(np.argmax(np.abs(gaps)))
    add_finding(
        fig,
        "This graph compares negative-word rates for matched Black and White face images. "
        f"The largest observed gap was {gaps[largest] * 100:.1f} percentage points for "
        f"{labels[largest]}; confidence intervals reflect image-level uncertainty.",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    return save(fig, output)


def plot_utkface_gaps(rows: list[dict[str, str]], significance: list[dict[str, str]], output: Path) -> Path:
    rows = ordered(rows)
    significance_by_model = {row["model_key"]: row for row in significance}
    labels = [row["model_display_name"] for row in rows]
    gaps = np.asarray([number(row, "black_minus_white_gap") for row in rows])
    errors = np.asarray(
        [error_extent(gaps[i], number(row, "gap_ci_low"), number(row, "gap_ci_high")) for i, row in enumerate(rows)]
    ).T
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.bar(labels, gaps * 100, color=BLACK_COLOR, width=0.62, yerr=errors * 100, capsize=5)
    ax.axhline(0, color="#20252B", linewidth=1)
    ax.bar_label(bars, labels=[f"{gap * 100:+.1f} pp" for gap in gaps], padding=3, fontsize=9)
    ax.set_title("Black Minus White Negative-Term Gap\n95% image-cluster bootstrap CI")
    ax.set_ylabel("Percentage-point gap")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    excludes_zero = sum(number(row, "gap_ci_low") > 0 or number(row, "gap_ci_high") < 0 for row in rows)
    corrected = sum(number(significance_by_model[row["model_key"]], "bh_adjusted_p_value_across_models") < 0.05 for row in rows)
    add_finding(
        fig,
        "Positive values mean a higher negative-word rate for Black photos. "
        f"{excludes_zero} of {len(rows)} intervals excluded zero, but {corrected} model tests "
        "survived BH correction.",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    return save(fig, output)


def plot_utkface_validity(rows: list[dict[str, str]], output: Path) -> Path:
    rows = ordered(rows)
    labels = [row["model_display_name"] for row in rows]
    x = np.arange(len(rows))
    width = 0.36
    validity = np.asarray([number(row, "validity_rate") for row in rows])
    diversity = np.asarray([number(row, "response_tuple_diversity_rate") for row in rows])
    fig, ax = plt.subplots(figsize=(11, 5.8))
    valid_bars = ax.bar(x - width / 2, validity * 100, width, label="Valid three-word output", color="#009E73")
    diversity_bars = ax.bar(x + width / 2, diversity * 100, width, label="Unique tuple rate", color="#CC79A7")
    ax.bar_label(valid_bars, labels=[f"{value * 100:.1f}%" for value in validity], padding=3, fontsize=8)
    ax.bar_label(diversity_bars, labels=[f"{value * 100:.1f}%" for value in diversity], padding=3, fontsize=8)
    ax.set_title("Output Validity and Response Diversity")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", alpha=0.2)
    minimum = int(np.argmin(validity))
    add_finding(
        fig,
        "Validity is the share yielding exactly three permitted IAT words; diversity is the "
        f"share of valid triples that were unique. {labels[minimum]} had the lowest validity "
        f"({validity[minimum] * 100:.1f}%).",
    )
    fig.tight_layout(rect=(0, 0.11, 0.88, 0.95))
    return save(fig, output)


def plot_visbias_implicit(rows: list[dict[str, str]], output: Path) -> Path:
    metric_order = ("vader_compound", "iat_negative_rate", "stereotype_hits_per_100_words", "identity_mention")
    scales = {"vader_compound": 1.0, "iat_negative_rate": 100.0, "stereotype_hits_per_100_words": 1.0, "identity_mention": 100.0}
    labels = {
        "vader_compound": "Sentiment difference",
        "iat_negative_rate": "Negative-word gap (pp)",
        "stereotype_hits_per_100_words": "Stereotype-hit difference / 100 words",
        "identity_mention": "Identity-mention gap (pp)",
    }
    fig, axes = plt.subplots(1, 4, figsize=(21, 8), sharey=True)
    for axis, metric in zip(axes, metric_order, strict=True):
        selected = ordered([row for row in rows if row["metric_key"] == metric])
        scale = scales[metric]
        y = np.arange(len(selected))
        effect = np.asarray([scale * number(row, "black_minus_white_effect") for row in selected])
        low = np.asarray([scale * number(row, "ci_low") for row in selected])
        high = np.asarray([scale * number(row, "ci_high") for row in selected])
        axis.errorbar(effect, y, xerr=np.vstack([effect - low, high - effect]), fmt="o", capsize=3)
        axis.axvline(0, color="#555555", linewidth=1)
        axis.set_title(labels[metric])
        axis.set_xlabel("Black minus White")
        axis.set_yticks(y, [row["model_display_name"] for row in selected])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
        if np.allclose(low, 0) and np.allclose(high, 0):
            axis.set_xlim(-1, 1)
            span = 2.0
        else:
            span = max(1e-6, float(high.max() - low.min()))
            axis.set_xlim(float(low.min()) - 0.12 * span, float(high.max()) + 0.35 * span)
        for index, row in enumerate(selected):
            axis.text(
                high[index] + 0.03 * span,
                index,
                f"p={number(row, 'raw_p_value'):.3f}\nq={number(row, 'bh_adjusted_q_value'):.3f}",
                va="center",
                fontsize=7,
            )
    corrected = sum(number(row, "bh_adjusted_q_value") < 0.05 for row in rows)
    fig.suptitle("VisBias Image Descriptions: Black Minus White Race Gaps", fontsize=15)
    add_finding(
        fig,
        "Points compare descriptions of matched Black- and White-presenting images; bars are "
        f"paired bootstrap intervals. {corrected} of {len(rows)} corrected comparisons met "
        "q < 0.05. Automated sentiment and lexicon scores can be imperfect.",
    )
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.16, top=0.86, wspace=0.26)
    return save(fig, output)


def plot_visbias_explicit(summary: list[dict[str, str]], significance: list[dict[str, str]], output: Path) -> Path:
    summary = ordered(summary)
    by_model = {row["model_key"]: row for row in significance}
    selected_significance = [by_model[row["model_key"]] for row in summary]
    names = [row["model_display_name"] for row in summary]
    y = np.arange(len(summary))
    unsupported = np.asarray([100 * number(row, "unsupported_choice_rate") for row in summary])
    cannot = np.asarray([100 * number(row, "cannot_determine_rate") for row in summary])
    unclassified = np.asarray([100 * number(row, "unclassified_rate") for row in summary])
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8), sharey=True)
    axes[0].barh(y, unsupported, color="#b2182b", label="Unsupported A/B choice")
    axes[0].barh(y, cannot, left=unsupported, color="#4d9221", label="Cannot determine")
    axes[0].barh(y, unclassified, left=unsupported + cannot, color="#aaaaaa", label="Unclassified")
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Share of explicit comparison responses (%)")
    axes[0].set_title("Unsupported inference vs uncertainty")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(axis="x", alpha=0.2)
    effect = np.asarray([100 * number(row, "black_minus_white_effect") for row in selected_significance])
    low = np.asarray([100 * number(row, "ci_low") for row in selected_significance])
    high = np.asarray([100 * number(row, "ci_high") for row in selected_significance])
    axes[1].errorbar(effect, y, xerr=np.vstack([effect - low, high - effect]), fmt="o", capsize=3)
    axes[1].axvline(0, color="#555555", linewidth=1)
    axes[1].set_title("Direction of unsupported selection")
    axes[1].set_xlabel("Black minus White selection preference (pp)")
    axes[1].grid(axis="x", alpha=0.2)
    span = max(5.0, float(high.max() - low.min()))
    axes[1].set_xlim(float(low.min()) - 0.1 * span, float(high.max()) + 0.35 * span)
    for index, row in enumerate(selected_significance):
        axes[1].text(
            high[index] + 0.025 * span,
            index,
            f"p={number(row, 'raw_p_value'):.3f}\nq={number(row, 'bh_adjusted_q_value'):.3f}",
            va="center",
            fontsize=7.5,
        )
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    corrected = sum(number(row, "bh_adjusted_q_value") < 0.05 for row in significance)
    fig.suptitle("VisBias Explicit Comparison: Unsupported Income/Education Inference", fontsize=15)
    add_finding(
        fig,
        "Cannot determine is the supported answer because income and education are not visible. "
        f"The right panel controls image order; {corrected} of {len(significance)} directional "
        "tests met BH-adjusted q < 0.05.",
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.86, wspace=0.20)
    return save(fig, output)


def plot_pool_balance(root: Path, output: Path) -> Path:
    evaluation = read_csv(root / "data/frozen/utkface_evaluation_manifest.csv")
    training = read_csv(root / "data/frozen/utkface_finetuning_manifest.csv")
    races = ("Black", "White")
    genders = ("Female", "Male")
    expressions = ("negative", "neutral", "positive")
    labels = [f"{race}\n{gender}" for race in races for gender in genders]
    x = np.arange(len(labels))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
    for offset, name, rows in ((-width / 2, "Evaluation", evaluation), (width / 2, "Future fine-tuning", training)):
        values = [sum(row["race"] == race and row["gender"] == gender for row in rows) for race in races for gender in genders]
        bars = axes[0].bar(x + offset, values, width, label=name)
        axes[0].bar_label(bars, fontsize=8)
    axes[0].set_title("Race-by-gender balance")
    axes[0].set_ylabel("Distinct images")
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)
    expression_counts = Counter((row["race"], row["expression_class"]) for row in evaluation)
    exp_x = np.arange(len(expressions))
    for index, race in enumerate(races):
        values = [expression_counts[(race, expression)] for expression in expressions]
        bars = axes[1].bar(exp_x + (index - 0.5) * width, values, width, label=race, color=BLACK_COLOR if race == "Black" else WHITE_COLOR)
        axes[1].bar_label(bars, fontsize=8)
    axes[1].set_title("Evaluation expression balance")
    axes[1].set_ylabel("Distinct images")
    axes[1].set_xticks(exp_x, [value.title() for value in expressions])
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    by_pair: dict[str, dict[str, dict[str, str]]] = {}
    for row in evaluation:
        by_pair.setdefault(row["match_pair_id"], {})[row["race"]] = row
    age_differences = [abs(int(pair["Black"]["age"]) - int(pair["White"]["age"])) for pair in by_pair.values()]
    quality_differences = [100 * abs(float(pair["Black"]["quality_score"]) - float(pair["White"]["quality_score"])) for pair in by_pair.values()]
    box = axes[2].boxplot([age_differences, quality_differences], tick_labels=["Age difference\n(years)", "Quality difference\n(x100)"], patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], ("#56B4E9", "#009E73"), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[2].set_title("Within-pair residual differences")
    axes[2].set_ylabel("Absolute difference")
    axes[2].grid(axis="y", alpha=0.2)
    overlap = len({row["sha256"] for row in evaluation} & {row["sha256"] for row in training})
    fig.suptitle("UTKFace Matched Evaluation and Future Fine-Tuning Pools", fontsize=15)
    add_finding(
        fig,
        "Both pools contain 100 images in every race-by-gender cell. Evaluation expression "
        "counts are identical by race (47 negative, 78 neutral, 75 positive), and "
        f"evaluation/future-training SHA-256 overlap is {overlap}.",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    return save(fig, output)


def reproduce(root: Path, output_dir: Path) -> dict[str, Any]:
    checks = validate_raw_data(root)
    derived = root / "data/derived"
    archived = read_csv(derived / "all_significance_tests.csv")
    utk_summary = read_csv(derived / "utkface_model_summary.csv")
    utk_significance = read_csv(derived / "utkface_significance.csv")
    vis_implicit = read_csv(derived / "visbias_description_significance.csv")
    vis_summary = read_csv(derived / "visbias_comparison_summary.csv")
    vis_significance = read_csv(derived / "visbias_comparison_significance.csv")
    outputs = [
        plot_archived_iat(archived, output_dir / "archived_iat_significance.png"),
        plot_archived_decodingtrust(archived, output_dir / "archived_decodingtrust_significance.png"),
        plot_archived_bold(archived, output_dir / "archived_bold_significance.png"),
        plot_utkface_rates(utk_summary, output_dir / "utkface_negative_rates.png"),
        plot_utkface_gaps(utk_summary, utk_significance, output_dir / "utkface_race_gaps.png"),
        plot_utkface_validity(utk_summary, output_dir / "utkface_validity_diversity.png"),
        plot_visbias_implicit(vis_implicit, output_dir / "visbias_implicit_race_gaps.png"),
        plot_visbias_explicit(vis_summary, vis_significance, output_dir / "visbias_explicit_inference.png"),
        plot_pool_balance(root, output_dir / "utkface_pool_balance.png"),
    ]
    checks["archived_corrected_significant"] = sum(
        number(row, "bh_adjusted_q_value") < 0.05 for row in archived
    )
    checks["utkface_corrected_significant"] = sum(
        number(row, "bh_adjusted_p_value_across_models") < 0.05 for row in utk_significance
    )
    checks["visbias_corrected_significant"] = sum(
        number(row, "bh_adjusted_q_value") < 0.05
        for row in vis_implicit + vis_significance
    )
    return {"checks": checks, "plots": [str(path) for path in outputs]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce the compact August 31 figures")
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.verify_only:
        result: dict[str, Any] = {"checks": validate_raw_data(root), "plots": []}
    else:
        output_dir = (args.output_dir or root / "plots").resolve()
        result = reproduce(root, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
