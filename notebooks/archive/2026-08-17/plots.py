from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .constants import NEGATIVE_IAT_WORDS, POSITIVE_IAT_WORDS  # noqa: E402

if TYPE_CHECKING:
    from .iat import ModelSpec


BLACK_COLOR = "#0072B2"
WHITE_COLOR = "#E69F00"
NEUTRAL_COLOR = "#5B6573"


def _save(fig: plt.Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _word_order(black: dict[str, int], white: dict[str, int]) -> list[str]:
    def nonzero(word: str) -> bool:
        return black.get(word, 0) + white.get(word, 0) > 0

    positive_used = sorted(
        [word for word in POSITIVE_IAT_WORDS if nonzero(word)],
        key=lambda word: (-white.get(word, 0), word),
    )
    positive_zero = [word for word in POSITIVE_IAT_WORDS if not nonzero(word)]
    negative_used = sorted(
        [word for word in NEGATIVE_IAT_WORDS if nonzero(word)],
        key=lambda word: (-black.get(word, 0), word),
    )
    negative_zero = [word for word in NEGATIVE_IAT_WORDS if not nonzero(word)]
    return positive_used + positive_zero + negative_used + negative_zero


def _model_word_maps(
    word_rows: list[dict[str, Any]], model_key: str
) -> tuple[dict[str, int], dict[str, int]]:
    black = {
        row["word"]: int(row["count"])
        for row in word_rows
        if row["model_key"] == model_key and row["race"] == "Black"
    }
    white = {
        row["word"]: int(row["count"])
        for row in word_rows
        if row["model_key"] == model_key and row["race"] == "White"
    }
    return black, white


def _plot_model_word_counts(
    model: ModelSpec, word_rows: list[dict[str, Any]], output_dir: Path
) -> str:
    black, white = _model_word_maps(word_rows, model.key)
    order = _word_order(black, white)
    x = np.arange(len(order), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(
        x - width / 2,
        [black[word] for word in order],
        width,
        label="Black photo",
        color=BLACK_COLOR,
    )
    ax.bar(
        x + width / 2,
        [white[word] for word in order],
        width,
        label="White photo",
        color=WHITE_COLOR,
    )
    ax.axvline(len(POSITIVE_IAT_WORDS) - 0.5, color="#B9BEC6", linewidth=1.2)
    ax.text(
        (len(POSITIVE_IAT_WORDS) - 1) / 2,
        1.01,
        "Positive terms",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=10,
        color=NEUTRAL_COLOR,
    )
    ax.text(
        len(POSITIVE_IAT_WORDS) + (len(NEGATIVE_IAT_WORDS) - 1) / 2,
        1.01,
        "Negative terms",
        transform=ax.get_xaxis_transform(),
        ha="center",
        fontsize=10,
        color=NEUTRAL_COLOR,
    )
    ax.set_title(f"{model.display_name}: IAT-Style Word Counts")
    ax.set_xlabel("Words")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=8)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    return _save(fig, output_dir / f"{model.key}_01_word_counts.png")


def _error_extent(rate: float, low: float | None, high: float | None) -> tuple[float, float]:
    if low is None or high is None or not math.isfinite(rate):
        return 0.0, 0.0
    return max(0.0, rate - low), max(0.0, high - rate)


def _plot_model_rates(model: ModelSpec, model_metrics: dict[str, Any], output_dir: Path) -> str:
    races = ("Black", "White")
    rates = [float(model_metrics["race"][race]["negative_rate"]) for race in races]
    plotted_rates = np.nan_to_num(np.array(rates, dtype=np.float64), nan=0.0)
    errors = [
        _error_extent(
            rates[index],
            model_metrics["race"][race].get("ci_low"),
            model_metrics["race"][race].get("ci_high"),
        )
        for index, race in enumerate(races)
    ]
    yerr = np.array(errors, dtype=np.float64).T
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    bars = ax.bar(
        races,
        plotted_rates * 100,
        color=[BLACK_COLOR, WHITE_COLOR],
        yerr=yerr * 100,
        capsize=5,
        width=0.62,
    )
    ax.bar_label(
        bars,
        labels=[f"{rate * 100:.1f}%" if math.isfinite(rate) else "N/A" for rate in rates],
        padding=4,
    )
    gap = model_metrics.get("black_minus_white_negative_rate_gap")
    gap_label = "unavailable" if gap is None else f"{gap * 100:+.1f} percentage points"
    ax.set_title(
        f"{model.display_name}: Negative IAT-Term Rate\nBlack minus White gap: {gap_label}"
    )
    ax.set_ylabel("Negative terms among valid IAT terms (%)")
    ax.set_ylim(0, max(10.0, float(plotted_rates.max(initial=0.0)) * 125))
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    return _save(fig, output_dir / f"{model.key}_02_negative_rates.png")


def _plot_comparison(
    metrics: dict[str, Any], model_specs: list[ModelSpec], output_dir: Path
) -> list[str]:
    labels = [model.display_name for model in model_specs]
    x = np.arange(len(model_specs), dtype=np.float64)
    width = 0.36
    black_raw = np.array(
        [metrics["models"][model.key]["race"]["Black"]["negative_rate"] for model in model_specs]
    )
    white_raw = np.array(
        [metrics["models"][model.key]["race"]["White"]["negative_rate"] for model in model_specs]
    )
    black = np.nan_to_num(black_raw, nan=0.0)
    white = np.nan_to_num(white_raw, nan=0.0)
    black_errors = np.array(
        [
            _error_extent(
                float(black_raw[index]),
                metrics["models"][model.key]["race"]["Black"].get("ci_low"),
                metrics["models"][model.key]["race"]["Black"].get("ci_high"),
            )
            for index, model in enumerate(model_specs)
        ],
        dtype=float,
    ).T
    white_errors = np.array(
        [
            _error_extent(
                float(white_raw[index]),
                metrics["models"][model.key]["race"]["White"].get("ci_low"),
                metrics["models"][model.key]["race"]["White"].get("ci_high"),
            )
            for index, model in enumerate(model_specs)
        ],
        dtype=float,
    ).T

    fig, ax = plt.subplots(figsize=(12, 6.2))
    black_bars = ax.bar(
        x - width / 2,
        black * 100,
        width,
        label="Black photo",
        color=BLACK_COLOR,
        yerr=black_errors * 100,
        capsize=4,
    )
    white_bars = ax.bar(
        x + width / 2,
        white * 100,
        width,
        label="White photo",
        color=WHITE_COLOR,
        yerr=white_errors * 100,
        capsize=4,
    )
    ax.bar_label(
        black_bars,
        labels=[f"{value * 100:.1f}%" if math.isfinite(value) else "N/A" for value in black_raw],
        padding=3,
        fontsize=8,
    )
    ax.bar_label(
        white_bars,
        labels=[f"{value * 100:.1f}%" if math.isfinite(value) else "N/A" for value in white_raw],
        padding=3,
        fontsize=8,
    )
    ax.set_title("Five-Model IAT-Style Negative-Term Comparison\n95% image-cluster bootstrap CI")
    ax.set_ylabel("Negative terms among valid IAT terms (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    negative_path = _save(fig, output_dir / "comparison_01_negative_rates.png")

    gaps_raw = np.array(
        [
            metrics["models"][model.key]["black_minus_white_negative_rate_gap"]
            for model in model_specs
        ],
        dtype=np.float64,
    )
    gaps = np.nan_to_num(gaps_raw, nan=0.0)
    gap_errors = np.array(
        [
            _error_extent(
                float(gaps_raw[index]),
                metrics["models"][model.key].get("gap_ci_low"),
                metrics["models"][model.key].get("gap_ci_high"),
            )
            for index, model in enumerate(model_specs)
        ],
        dtype=float,
    ).T
    colors = [BLACK_COLOR if gap >= 0 else WHITE_COLOR for gap in gaps]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.bar(
        labels,
        gaps * 100,
        color=colors,
        width=0.62,
        yerr=gap_errors * 100,
        capsize=5,
    )
    ax.axhline(0, color="#20252B", linewidth=1)
    ax.bar_label(
        bars,
        labels=[f"{gap * 100:+.1f} pp" if math.isfinite(gap) else "N/A" for gap in gaps_raw],
        padding=3,
        fontsize=9,
    )
    ax.set_title("Black Minus White Negative-Term Gap\n95% image-cluster bootstrap CI")
    ax.set_ylabel("Percentage-point gap")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    gap_path = _save(fig, output_dir / "comparison_02_race_gaps.png")

    validity_raw = np.array(
        [metrics["models"][model.key]["validity_rate"] for model in model_specs]
    )
    diversity_raw = np.array(
        [metrics["models"][model.key]["response_tuple_diversity_rate"] for model in model_specs]
    )
    validity = np.nan_to_num(validity_raw, nan=0.0)
    diversity = np.nan_to_num(diversity_raw, nan=0.0)
    fig, ax = plt.subplots(figsize=(11, 5.8))
    valid_bars = ax.bar(
        x - width / 2, validity * 100, width, label="Valid three-word output", color="#009E73"
    )
    diversity_bars = ax.bar(
        x + width / 2, diversity * 100, width, label="Unique tuple rate", color="#CC79A7"
    )
    ax.bar_label(
        valid_bars,
        labels=[f"{value * 100:.0f}%" if math.isfinite(value) else "N/A" for value in validity_raw],
        padding=3,
        fontsize=8,
    )
    ax.bar_label(
        diversity_bars,
        labels=[
            f"{value * 100:.0f}%" if math.isfinite(value) else "N/A" for value in diversity_raw
        ],
        padding=3,
        fontsize=8,
    )
    ax.set_title("Output Validity and Response Diversity")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    quality_path = _save(fig, output_dir / "comparison_03_validity_diversity.png")
    return [negative_path, gap_path, quality_path]


def write_iat_plots(
    metrics: dict[str, Any],
    word_rows: list[dict[str, Any]],
    model_specs: list[ModelSpec],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for model in model_specs:
        paths.append(_plot_model_word_counts(model, word_rows, output_dir))
        paths.append(_plot_model_rates(model, metrics["models"][model.key], output_dir))
    paths.extend(_plot_comparison(metrics, model_specs, output_dir))
    return paths
