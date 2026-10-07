from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .iat import load_model_specs  # noqa: E402
from .text_runner import write_csv  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing completed benchmark output: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value not in {"", None} else math.nan


def _extent(value: float, low: float, high: float) -> list[float]:
    return [max(0.0, value - low), max(0.0, high - value)]


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["model_key"]: row for row in rows}


def build_summary(
    iat_run_id: str,
    decodingtrust_run_id: str,
    bold_run_id: str,
) -> dict[str, Any]:
    output_dir = PROJECT_ROOT / "results" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    iat_dir = PROJECT_ROOT / "results" / "iat12" / iat_run_id
    dt_dir = PROJECT_ROOT / "results" / "decodingtrust" / decodingtrust_run_id
    bold_dir = PROJECT_ROOT / "results" / "bold" / bold_run_id

    iat = _index(_read_csv(iat_dir / "model_summary.csv"))
    dt = _index(_read_csv(dt_dir / "model_summary.csv"))
    dt_regimes = _read_csv(dt_dir / "regime_summary.csv")
    bold = _index(_read_csv(bold_dir / "model_summary.csv"))
    _, models = load_model_specs()

    rows = []
    for model in models:
        iat_row = iat[model.key]
        dt_row = dt[model.key]
        bold_row = bold[model.key]
        rows.append(
            {
                "model_key": model.key,
                "model_display_name": model.display_name,
                "weight_access": model.weight_access,
                "canonical_revision": model.api_revision_hint,
                "iat_responses": iat_row["responses"],
                "iat_black_negative_rate": iat_row["black_negative_rate"],
                "iat_white_negative_rate": iat_row["white_negative_rate"],
                "iat_black_minus_white_gap": iat_row["black_minus_white_gap"],
                "iat_gap_ci_low": iat_row["gap_ci_low"],
                "iat_gap_ci_high": iat_row["gap_ci_high"],
                "iat_probability_gap_gt_zero": iat_row["probability_gap_gt_zero"],
                "iat_tuple_diversity_rate": iat_row["response_tuple_diversity_rate"],
                "decodingtrust_valid_responses": dt_row["valid_responses"],
                "decodingtrust_overall_agreement_rate": dt_row["overall_agreement_rate"],
                "decodingtrust_benign_agreement_rate": dt_row["benign_agreement_rate"],
                "decodingtrust_untargeted_agreement_rate": dt_row["untargeted_agreement_rate"],
                "decodingtrust_targeted_agreement_rate": dt_row["targeted_agreement_rate"],
                "decodingtrust_overall_no_response_rate": dt_row["overall_no_response_rate"],
                "bold_scored_responses": bold_row["scored_responses"],
                "bold_toxicity_rate": bold_row["toxicity_rate"],
                "bold_toxicity_ci_low": bold_row["toxicity_rate_ci_low"],
                "bold_toxicity_ci_high": bold_row["toxicity_rate_ci_high"],
                "bold_mean_sentiment": bold_row["mean_sentiment"],
                "bold_mean_sentiment_ci_low": bold_row["mean_sentiment_ci_low"],
                "bold_mean_sentiment_ci_high": bold_row["mean_sentiment_ci_high"],
                "bold_mean_regard": bold_row["mean_regard"],
                "bold_mean_regard_ci_low": bold_row["mean_regard_ci_low"],
                "bold_mean_regard_ci_high": bold_row["mean_regard_ci_high"],
            }
        )

    csv_path = output_dir / "five_model_pilot_summary.csv"
    write_csv(csv_path, rows)
    labels = [model.display_name for model in models]
    x = np.arange(len(models), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

    iat_gaps = [_number(iat[model.key], "black_minus_white_gap") for model in models]
    iat_errors = np.array(
        [
            _extent(
                iat_gaps[index],
                _number(iat[model.key], "gap_ci_low"),
                _number(iat[model.key], "gap_ci_high"),
            )
            for index, model in enumerate(models)
        ]
    ).T
    axes[0, 0].bar(x, np.array(iat_gaps) * 100, yerr=iat_errors * 100, capsize=4)
    axes[0, 0].axhline(0, color="#20252B", linewidth=1)
    axes[0, 0].set_title("IAT-Style Black Minus White Gap\n95% image-cluster bootstrap CI")
    axes[0, 0].set_ylabel("Percentage points")

    width = 0.24
    for regime_index, (regime, color) in enumerate(
        (("benign", "#009E73"), ("untargeted", "#E69F00"), ("targeted", "#D55E00"))
    ):
        regime_rows = {row["model_key"]: row for row in dt_regimes if row["regime"] == regime}
        values = [_number(regime_rows[model.key], "agreement_rate") for model in models]
        errors = np.array(
            [
                _extent(
                    values[index],
                    _number(regime_rows[model.key], "agreement_ci_low"),
                    _number(regime_rows[model.key], "agreement_ci_high"),
                )
                for index, model in enumerate(models)
            ]
        ).T
        axes[0, 1].bar(
            x + (regime_index - 1) * width,
            np.array(values) * 100,
            width,
            yerr=errors * 100,
            capsize=3,
            label=regime.title(),
            color=color,
        )
    axes[0, 1].set_title("DecodingTrust Stereotype Agreement\n95% Wilson CI")
    axes[0, 1].set_ylabel("Responses (%)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    toxicity = [_number(bold[model.key], "toxicity_rate") for model in models]
    toxicity_errors = np.array(
        [
            _extent(
                toxicity[index],
                _number(bold[model.key], "toxicity_rate_ci_low"),
                _number(bold[model.key], "toxicity_rate_ci_high"),
            )
            for index, model in enumerate(models)
        ]
    ).T
    axes[1, 0].bar(x, np.array(toxicity) * 100, yerr=toxicity_errors * 100, capsize=4)
    axes[1, 0].set_title("BOLD Toxicity\n95% Wilson CI")
    axes[1, 0].set_ylabel("Responses flagged toxic (%)")

    for offset, (metric, label, color) in enumerate(
        (("mean_sentiment", "Sentiment", "#56B4E9"), ("mean_regard", "Regard", "#CC79A7"))
    ):
        values = [_number(bold[model.key], metric) for model in models]
        errors = np.array(
            [
                _extent(
                    values[index],
                    _number(bold[model.key], f"{metric}_ci_low"),
                    _number(bold[model.key], f"{metric}_ci_high"),
                )
                for index, model in enumerate(models)
            ]
        ).T
        axes[1, 1].bar(
            x + (offset - 0.5) * 0.32,
            values,
            0.32,
            yerr=errors,
            capsize=4,
            label=label,
            color=color,
        )
    axes[1, 1].axhline(0, color="#20252B", linewidth=1)
    axes[1, 1].set_title("BOLD Sentiment And Regard\n95% domain-stratified bootstrap CI")
    axes[1, 1].set_ylabel("Mean class (-1 to +1)")
    axes[1, 1].legend(frameon=False)

    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=16, ha="right")
        ax.grid(axis="y", color="#E4E7EB", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Five-Model Bias Benchmark: August 17 Pilot", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = output_dir / "five_model_pilot_overview.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    markdown_path = output_dir / "five_model_pilot_summary.md"
    lines = [
        "# Five-Model Pilot Summary",
        "",
        "These are pilot estimates from 10 IAT responses per image, 48 DecodingTrust "
        "prompts, and 25 BOLD prompts per model. Do not combine the columns into one score.",
        "",
        "| Model | IAT gap (pp, 95% CI) | DT agreement B/U/T | BOLD toxicity | "
        "BOLD sentiment | BOLD regard |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        iat_row = iat[model.key]
        dt_row = dt[model.key]
        bold_row = bold[model.key]
        lines.append(
            f"| {model.display_name} | "
            f"{_number(iat_row, 'black_minus_white_gap') * 100:+.1f} "
            f"[{_number(iat_row, 'gap_ci_low') * 100:+.1f}, "
            f"{_number(iat_row, 'gap_ci_high') * 100:+.1f}] | "
            f"{_number(dt_row, 'benign_agreement_rate') * 100:.1f}% / "
            f"{_number(dt_row, 'untargeted_agreement_rate') * 100:.1f}% / "
            f"{_number(dt_row, 'targeted_agreement_rate') * 100:.1f}% | "
            f"{_number(bold_row, 'toxicity_rate') * 100:.1f}% | "
            f"{_number(bold_row, 'mean_sentiment'):+.2f} | "
            f"{_number(bold_row, 'mean_regard'):+.2f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_path = output_dir / "summary_manifest.json"
    manifest = {
        "iat_run_id": iat_run_id,
        "decodingtrust_run_id": decodingtrust_run_id,
        "bold_run_id": bold_run_id,
        "outputs": [
            str(csv_path.relative_to(PROJECT_ROOT)),
            str(markdown_path.relative_to(PROJECT_ROOT)),
            str(plot_path.relative_to(PROJECT_ROOT)),
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
