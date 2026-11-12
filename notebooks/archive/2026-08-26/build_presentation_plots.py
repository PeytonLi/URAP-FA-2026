#!/usr/bin/env python3
"""Regenerate the August 26 presentation figures from frozen summaries."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "urap_aug26_matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "urap_aug26_cache")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LOCAL_GEMMA_REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"
BLACK_COLOR = "#0072B2"
WHITE_COLOR = "#E69F00"
REGIMES = ("benign", "untargeted", "targeted")
REGIME_COLORS = ("#4C78A8", "#F2A541", "#C44E52")

SHORT_LABELS = {
    "gemma3_12b_local_pinned": "Gemma 3 12B IT\n(local pinned)",
    "gemma3_12b_hosted": "Gemma 3 12B IT\n(OpenRouter/DeepInfra)",
    "qwen35_9b": "Qwen 3.5 9B",
    "kimi_k3": "Kimi K3",
    "gpt56_luna": "GPT-5.6 Luna",
    "gemini36_flash": "Gemini 3.6 Flash",
    "mimo_v25": "MiMo-V2.5",
    "minimax_m3": "MiniMax M3",
    "step37_flash": "Step 3.7 Flash",
    "grok46": "Grok 4.6",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen input: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field!r} in row for {row.get('model_key')}") from exc


def error_extent(value: float, low: float, high: float) -> tuple[float, float]:
    return max(0.0, value - low), max(0.0, high - value)


def add_finding(fig: plt.Figure, finding: str) -> None:
    fig.text(
        0.5,
        0.022,
        textwrap.fill(finding, width=170),
        ha="center",
        va="bottom",
        fontsize=10,
        color="#303840",
        linespacing=1.35,
    )


def validate_model_family(rows: list[dict[str, str]], family: str) -> None:
    expected = 9 if family == "with_grok" else 8
    keys = [row["model_key"] for row in rows]
    if len(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError(
            f"{family} must contain {expected} unique models; found {len(keys)} rows"
        )
    has_grok = "grok46" in keys
    if has_grok != (family == "with_grok"):
        raise ValueError(f"Grok membership does not match family {family}")


def model_count_word(count: int) -> str:
    words = {8: "Eight", 9: "Nine"}
    try:
        return words[count]
    except KeyError as exc:
        raise ValueError(f"Unsupported presentation family size: {count}") from exc


def plot_iat(family: str, output_dir: Path) -> Path:
    rows = read_csv(ROOT / f"iat_{family}_summary.csv")
    significance = read_csv(ROOT / f"iat_{family}_significance.csv")
    validate_model_family(rows, family)

    labels = [SHORT_LABELS[row["model_key"]] for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.36
    black = np.array([as_float(row, "black_negative_rate") for row in rows])
    white = np.array([as_float(row, "white_negative_rate") for row in rows])
    black_error = np.array(
        [
            error_extent(
                value,
                as_float(row, "black_ci_low"),
                as_float(row, "black_ci_high"),
            )
            for value, row in zip(black, rows, strict=True)
        ]
    ).T
    white_error = np.array(
        [
            error_extent(
                value,
                as_float(row, "white_ci_low"),
                as_float(row, "white_ci_high"),
            )
            for value, row in zip(white, rows, strict=True)
        ]
    ).T

    fig, ax = plt.subplots(figsize=(16, 9))
    black_bars = ax.bar(
        x - width / 2,
        black * 100,
        width,
        color=BLACK_COLOR,
        label="Black face images",
        yerr=black_error * 100,
        capsize=4,
    )
    white_bars = ax.bar(
        x + width / 2,
        white * 100,
        width,
        color=WHITE_COLOR,
        label="White face images",
        yerr=white_error * 100,
        capsize=4,
    )
    ax.bar_label(
        black_bars,
        labels=[f"{value:.0%}" for value in black],
        padding=3,
        fontsize=8,
    )
    ax.bar_label(
        white_bars,
        labels=[f"{value:.0%}" for value in white],
        padding=3,
        fontsize=8,
    )

    grok_label = "Included" if family == "with_grok" else "Excluded"
    ax.set_title(
        f"Implicit Bias: IAT-Style Negative-Term Rates ({len(rows)} Models; "
        f"Grok {grok_label})\n"
        f"Gemma uses local pinned HF commit {LOCAL_GEMMA_REVISION[:12]}...; "
        "other rows use frozen OpenRouter routes",
        fontsize=16,
        pad=16,
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Negative terms among valid selected IAT words (%)", fontsize=12)
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)

    gaps = black - white
    largest = int(np.argmax(np.abs(gaps)))
    significant = sum(
        as_float(row, "bh_adjusted_p_value_across_models") < 0.05
        for row in significance
    )
    largest_label = labels[largest].replace("\n", " ")
    add_finding(
        fig,
        "The y-axis is the share of the three selected words that came from the negative "
        "IAT lexicon; the within-model Black-versus-White separation is the bias signal, "
        "not the overall bar height. "
        f"The largest observed gap was {gaps[largest] * 100:+.1f} percentage points for "
        f"{largest_label}; {significant} of {len(rows)} exploratory race tests remained "
        "significant at BH-adjusted q<0.05. Whiskers are 95% image-cluster bootstrap "
        "intervals, and only 12 distinct faces were tested.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.25)
    output = output_dir / f"01_iat_gemma_local_{family}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_iat_v2(family: str, output_dir: Path) -> Path:
    rows = read_csv(ROOT / f"iat_{family}_summary.csv")
    validate_model_family(rows, family)

    labels = [SHORT_LABELS[row["model_key"]] for row in rows]
    x = np.arange(len(rows), dtype=float)
    width = 0.36
    black = np.array([as_float(row, "black_negative_rate") for row in rows])
    white = np.array([as_float(row, "white_negative_rate") for row in rows])
    black_error = np.array(
        [
            error_extent(
                value,
                as_float(row, "black_ci_low"),
                as_float(row, "black_ci_high"),
            )
            for value, row in zip(black, rows, strict=True)
        ]
    ).T
    white_error = np.array(
        [
            error_extent(
                value,
                as_float(row, "white_ci_low"),
                as_float(row, "white_ci_high"),
            )
            for value, row in zip(white, rows, strict=True)
        ]
    ).T

    fig, ax = plt.subplots(figsize=(16, 9))
    black_bars = ax.bar(
        x - width / 2,
        black * 100,
        width,
        color=BLACK_COLOR,
        label="Black images",
        yerr=black_error * 100,
        capsize=4,
    )
    white_bars = ax.bar(
        x + width / 2,
        white * 100,
        width,
        color=WHITE_COLOR,
        label="White images",
        yerr=white_error * 100,
        capsize=4,
    )
    ax.bar_label(
        black_bars,
        labels=[f"{value:.0%}" for value in black],
        padding=3,
        fontsize=9,
    )
    ax.bar_label(
        white_bars,
        labels=[f"{value:.0%}" for value in white],
        padding=3,
        fontsize=9,
    )
    ax.set_title(
        f"{model_count_word(len(rows))}-Model IAT-Style Negative-Term Comparison",
        fontsize=19,
        pad=18,
    )
    ax.set_ylabel("Negative-term rate (%)", fontsize=13)
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.89, bottom=0.15)
    output = output_dir / f"01_iat_gemma_local_{family}_v2.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_decodingtrust(family: str, output_dir: Path) -> Path:
    rows = read_csv(ROOT / f"decodingtrust_{family}_summary.csv")
    significance = read_csv(ROOT / f"decodingtrust_{family}_significance.csv")
    model_keys = list(dict.fromkeys(row["model_key"] for row in rows))
    model_rows = [{"model_key": key} for key in model_keys]
    validate_model_family(model_rows, family)

    labels = [SHORT_LABELS[key] for key in model_keys]
    x = np.arange(len(model_keys), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(16, 9))
    regime_labels = {
        "benign": "Benign instruction",
        "untargeted": "Untargeted adversarial",
        "targeted": "Race-targeted adversarial",
    }
    all_values: list[tuple[float, str, str]] = []

    for regime_index, regime in enumerate(REGIMES):
        selected = [
            next(
                row
                for row in rows
                if row["model_key"] == model_key and row["regime"] == regime
            )
            for model_key in model_keys
        ]
        values = np.array([as_float(row, "agreement_rate") for row in selected])
        errors = np.array(
            [
                error_extent(
                    value,
                    as_float(row, "agreement_ci_low"),
                    as_float(row, "agreement_ci_high"),
                )
                for value, row in zip(values, selected, strict=True)
            ]
        ).T
        bars = ax.bar(
            x + (regime_index - 1) * width,
            values * 100,
            width,
            color=REGIME_COLORS[regime_index],
            label=regime_labels[regime],
            yerr=errors * 100,
            capsize=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1%}" for value in values],
            padding=3,
            fontsize=8,
        )
        all_values.extend(
            (value, labels[index].replace("\n", " "), regime)
            for index, value in enumerate(values)
        )

    peak_value = max(value for value, _, _ in all_values)
    peak_conditions = [
        f"{model} ({regime})"
        for value, model, regime in all_values
        if math.isclose(value, peak_value)
    ]
    grok_label = "Included" if family == "with_grok" else "Excluded"
    ax.set_title(
        "Explicit Bias: DecodingTrust Stereotype Agreement "
        f"({len(model_keys)} Models; Grok {grok_label})\n"
        "Gemma uses google/gemma-3-12b-it via OpenRouter / DeepInfra BF16; "
        "this benchmark is text-only",
        fontsize=16,
        pad=16,
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Responses classified as agreeing with the stereotype (%)", fontsize=12)
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)

    significant = sum(
        as_float(row, "bh_adjusted_p_value_within_figure_family") < 0.05
        for row in significance
    )
    add_finding(
        fig,
        "The y-axis is the percentage of responses that explicitly agreed with a supplied "
        "race stereotype; higher is worse. "
        f"The highest observed agreement was {peak_value:.1%} for "
        f"{' and '.join(peak_conditions)}; {significant} of {len(significance)} "
        "exploratory race-group tests remained significant at BH-adjusted q<0.05. "
        "Whiskers are 95% Wilson intervals; this is a 48-prompt-per-model pilot, and "
        "refusal or unclassified rates are reported in the CSVs.",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.25)
    output = output_dir / f"02_decodingtrust_gemma_hosted_{family}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_decodingtrust_v2(family: str, output_dir: Path) -> Path:
    rows = read_csv(ROOT / f"decodingtrust_{family}_summary.csv")
    model_keys = list(dict.fromkeys(row["model_key"] for row in rows))
    validate_model_family([{"model_key": key} for key in model_keys], family)

    labels = [SHORT_LABELS[key] for key in model_keys]
    x = np.arange(len(model_keys), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(16, 9))
    regime_labels = {
        "benign": "Benign",
        "untargeted": "Untargeted",
        "targeted": "Race-targeted",
    }

    for regime_index, regime in enumerate(REGIMES):
        selected = [
            next(
                row
                for row in rows
                if row["model_key"] == model_key and row["regime"] == regime
            )
            for model_key in model_keys
        ]
        values = np.array([as_float(row, "agreement_rate") for row in selected])
        errors = np.array(
            [
                error_extent(
                    value,
                    as_float(row, "agreement_ci_low"),
                    as_float(row, "agreement_ci_high"),
                )
                for value, row in zip(values, selected, strict=True)
            ]
        ).T
        bars = ax.bar(
            x + (regime_index - 1) * width,
            values * 100,
            width,
            color=REGIME_COLORS[regime_index],
            label=regime_labels[regime],
            yerr=errors * 100,
            capsize=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1%}" for value in values],
            padding=3,
            fontsize=9,
        )

    ax.set_title(
        f"{model_count_word(len(model_keys))}-Model DecodingTrust Comparison",
        fontsize=19,
        pad=18,
    )
    ax.set_ylabel("Stereotype agreement (%)", fontsize=13)
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.89, bottom=0.15)
    output = output_dir / f"02_decodingtrust_gemma_hosted_{family}_v2.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_iat_gap_v2(family: str, output_dir: Path) -> Path:
    rows = read_csv(ROOT / f"iat_{family}_summary.csv")
    validate_model_family(rows, family)

    labels = [SHORT_LABELS[row["model_key"]] for row in rows]
    x = np.arange(len(rows), dtype=float)
    gaps = np.array([as_float(row, "black_minus_white_gap") for row in rows])
    lows = np.array([as_float(row, "gap_ci_low") for row in rows])
    highs = np.array([as_float(row, "gap_ci_high") for row in rows])
    errors = np.array(
        [
            error_extent(value, low, high)
            for value, low, high in zip(gaps, lows, highs, strict=True)
        ]
    ).T
    colors = [BLACK_COLOR if value >= 0 else WHITE_COLOR for value in gaps]

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axhline(0, color="#303840", linewidth=1.2, zorder=1)
    bars = ax.bar(
        x,
        gaps * 100,
        width=0.58,
        color=colors,
        yerr=errors * 100,
        capsize=5,
        zorder=2,
    )
    ax.bar_label(
        bars,
        labels=[f"{value * 100:+.1f}" for value in gaps],
        padding=4,
        fontsize=10,
    )
    ax.set_title(
        f"{model_count_word(len(rows))}-Model IAT-12 Race-Gap Comparison",
        fontsize=19,
        pad=18,
    )
    ax.set_ylabel("Black - White negative-term gap (pp)", fontsize=13)
    ax.set_xticks(x, labels=labels)
    lower = min(-5.0, math.floor((float(lows.min()) * 100 - 3) / 5) * 5)
    upper = max(5.0, math.ceil((float(highs.max()) * 100 + 6) / 5) * 5)
    ax.set_ylim(lower, upper)
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.89, bottom=0.15)
    output = output_dir / f"03_iat_black_minus_white_gap_{family}_v2.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for regenerated PNGs (default: this folder)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_iat("without_grok", args.output_dir),
        plot_iat("with_grok", args.output_dir),
        plot_decodingtrust("without_grok", args.output_dir),
        plot_decodingtrust("with_grok", args.output_dir),
        plot_iat_v2("without_grok", args.output_dir),
        plot_iat_v2("with_grok", args.output_dir),
        plot_decodingtrust_v2("without_grok", args.output_dir),
        plot_decodingtrust_v2("with_grok", args.output_dir),
        plot_iat_gap_v2("without_grok", args.output_dir),
        plot_iat_gap_v2("with_grok", args.output_dir),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
