from __future__ import annotations

import csv
import json
import math
import os
import textwrap
from collections import defaultdict
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
        raise FileNotFoundError(f"Missing benchmark artifact: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["model_key"]: row for row in rows}


def _number(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _finite(value: float, default: float = 0.0) -> float:
    return value if math.isfinite(value) else default


def _error(value: float, low: float, high: float) -> tuple[float, float]:
    if not all(math.isfinite(item) for item in (value, low, high)):
        return 0.0, 0.0
    return max(0.0, value - low), max(0.0, high - value)


def _percent(value: float, digits: int = 1) -> str:
    return "N/A" if not math.isfinite(value) else f"{value * 100:.{digits}f}%"


def _signed_pp(value: float, digits: int = 1) -> str:
    return "N/A" if not math.isfinite(value) else f"{value * 100:+.{digits}f} pp"


def _decimal(value: float, digits: int = 2) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:+.{digits}f}"


def _observed_providers(path: Path) -> dict[str, str]:
    providers: dict[str, set[str]] = defaultdict(set)
    for row in _read_csv(path):
        provider = row.get("provider", "").strip()
        if provider:
            providers[row["model_key"]].add(provider)
    return {key: "; ".join(sorted(values)) for key, values in providers.items()}


def _minimum_q(rows: list[dict[str, str]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _number(row, "bh_adjusted_p_value")
        if math.isfinite(value):
            grouped[row["model_key"]].append(value)
    return {key: min(values) for key, values in grouped.items()}


def _add_finding(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.5,
        0.012,
        textwrap.fill(text, width=185),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#303840",
    )


def _build_overview(rows: list[dict[str, Any]], output_path: Path, phase_label: str) -> None:
    labels = [row["model_display_name"] for row in rows]
    x = np.arange(len(rows), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(17, 12.5))

    iat_values = np.array([row["iat_gap"] for row in rows], dtype=float)
    iat_errors = np.array(
        [_error(row["iat_gap"], row["iat_gap_ci_low"], row["iat_gap_ci_high"]) for row in rows]
    ).T
    axes[0, 0].bar(
        x,
        np.nan_to_num(iat_values) * 100,
        yerr=iat_errors * 100,
        capsize=4,
        color=["#0072B2" if value >= 0 else "#E69F00" for value in iat_values],
    )
    axes[0, 0].axhline(0, color="#20252B", linewidth=1)
    axes[0, 0].set_title("Implicit IAT-Style Black Minus White Gap\n95% image-cluster CI")
    axes[0, 0].set_ylabel("Negative-term gap (percentage points)")

    width = 0.25
    for offset, (field, label, color) in enumerate(
        (
            ("dt_benign", "Benign", "#009E73"),
            ("dt_untargeted", "Untargeted", "#E69F00"),
            ("dt_targeted", "Targeted", "#D55E00"),
        )
    ):
        values = np.array([row[field] for row in rows], dtype=float)
        axes[0, 1].bar(
            x + (offset - 1) * width,
            np.nan_to_num(values) * 100,
            width,
            label=label,
            color=color,
        )
    axes[0, 1].set_title("Explicit DecodingTrust Stereotype Agreement")
    axes[0, 1].set_ylabel("Valid responses classified AGREE (%)")
    axes[0, 1].legend(frameon=False, fontsize=8)

    toxicity = np.array([row["bold_toxicity"] for row in rows], dtype=float)
    toxicity_errors = np.array(
        [
            _error(
                row["bold_toxicity"],
                row["bold_toxicity_ci_low"],
                row["bold_toxicity_ci_high"],
            )
            for row in rows
        ]
    ).T
    axes[1, 0].bar(
        x,
        np.nan_to_num(toxicity) * 100,
        yerr=toxicity_errors * 100,
        capsize=4,
        color="#CC6677",
    )
    axes[1, 0].set_title("BOLD Toxicity\n95% Wilson CI")
    axes[1, 0].set_ylabel("Responses flagged (%)")

    for offset, (field, label, color) in enumerate(
        (
            ("bold_sentiment", "Sentiment", "#56B4E9"),
            ("bold_regard", "Regard", "#CC79A7"),
        )
    ):
        values = np.array([row[field] for row in rows], dtype=float)
        axes[1, 1].bar(
            x + (offset - 0.5) * 0.34,
            np.nan_to_num(values),
            0.34,
            label=label,
            color=color,
        )
    axes[1, 1].axhline(0, color="#20252B", linewidth=1)
    axes[1, 1].set_ylim(-1, 1)
    axes[1, 1].set_title("BOLD Sentiment And Regard")
    axes[1, 1].set_ylabel("Mean class (-1 negative, 0 neutral, +1 positive)")
    axes[1, 1].legend(frameon=False)

    for axis in axes.flat:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=18, ha="right")
        axis.grid(axis="y", color="#E4E7EB", linewidth=0.8)
        axis.set_axisbelow(True)

    largest_iat = max(rows, key=lambda row: abs(_finite(row["iat_gap"])))
    highest_dt = max(rows, key=lambda row: _finite(row["dt_targeted"]))
    flagged = sum(round(_finite(row["bold_toxicity"]) * row["bold_scored"]) for row in rows)
    _add_finding(
        fig,
        "The panels measure different constructs and should not be combined into one score. "
        f"The largest observed implicit gap was {_signed_pp(largest_iat['iat_gap'])} for "
        f"{largest_iat['model_display_name']}, while {highest_dt['model_display_name']} had "
        f"the highest targeted stereotype-agreement rate ({_percent(highest_dt['dt_targeted'])}). "
        f"BOLD flagged {flagged} response(s) across all models; confidence intervals and "
        "corrected exploratory tests are required before interpreting category differences.",
    )
    fig.suptitle(f"Eight-Model Race-Bias Benchmark: {phase_label}", fontsize=18)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_aug24_report(
    *,
    iat_run_id: str,
    decodingtrust_run_id: str,
    bold_run_id: str,
    phase_slug: str,
    phase_label: str,
    model_config: Path,
    write_main_results: bool = False,
) -> dict[str, Any]:
    output_dir = PROJECT_ROOT / "results" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    iat_dir = PROJECT_ROOT / "results" / "iat12" / iat_run_id
    dt_dir = PROJECT_ROOT / "results" / "decodingtrust" / decodingtrust_run_id
    bold_dir = PROJECT_ROOT / "results" / "bold" / bold_run_id

    iat = _index(_read_csv(iat_dir / "model_summary.csv"))
    iat_q = _index(_read_csv(iat_dir / "exploratory_significance.csv"))
    dt = _index(_read_csv(dt_dir / "model_summary.csv"))
    dt_significance_rows = _read_csv(dt_dir / "demographic_significance.csv")
    dt_q = _minimum_q(dt_significance_rows)
    bold = _index(_read_csv(bold_dir / "model_summary.csv"))
    bold_generation = _index(_read_csv(bold_dir / "generation_summary.csv"))
    bold_significance_rows = _read_csv(bold_dir / "category_significance.csv")
    bold_q = _minimum_q(bold_significance_rows)
    bold_category_rows = _read_csv(bold_dir / "category_summary.csv")
    iat_providers = _observed_providers(iat_dir / "responses.csv")
    dt_providers = _observed_providers(dt_dir / "responses.csv")
    bold_providers = _observed_providers(bold_dir / "responses.csv")
    _, models = load_model_specs(model_config)

    rows: list[dict[str, Any]] = []
    for model in models:
        iat_row = iat[model.key]
        dt_row = dt[model.key]
        bold_row = bold[model.key]
        bold_generation_row = bold_generation[model.key]
        dt_responses = int(float(dt_row["responses"]))
        dt_valid = int(float(dt_row["valid_responses"]))
        row = {
            "model_key": model.key,
            "model_display_name": model.display_name,
            "canonical_revision": model.api_revision_hint,
            "expected_provider_endpoint": model.provider_endpoint,
            "observed_iat_provider": iat_providers.get(model.key, ""),
            "observed_decodingtrust_provider": dt_providers.get(model.key, ""),
            "observed_bold_provider": bold_providers.get(model.key, ""),
            "iat_responses": int(float(iat_row["responses"])),
            "iat_valid_responses": int(float(iat_row["valid_responses"])),
            "iat_validity": _number(iat_row, "validity_rate"),
            "iat_black_rate": _number(iat_row, "black_negative_rate"),
            "iat_white_rate": _number(iat_row, "white_negative_rate"),
            "iat_gap": _number(iat_row, "black_minus_white_gap"),
            "iat_gap_ci_low": _number(iat_row, "gap_ci_low"),
            "iat_gap_ci_high": _number(iat_row, "gap_ci_high"),
            "iat_q": _number(iat_q[model.key], "bh_adjusted_p_value_across_models"),
            "iat_diversity": _number(iat_row, "response_tuple_diversity_rate"),
            "dt_responses": dt_responses,
            "dt_valid_responses": dt_valid,
            "dt_validity": dt_valid / dt_responses if dt_responses else math.nan,
            "dt_benign": _number(dt_row, "benign_agreement_rate"),
            "dt_untargeted": _number(dt_row, "untargeted_agreement_rate"),
            "dt_targeted": _number(dt_row, "targeted_agreement_rate"),
            "dt_overall": _number(dt_row, "overall_agreement_rate"),
            "dt_overall_ci_low": _number(dt_row, "overall_agreement_ci_low"),
            "dt_overall_ci_high": _number(dt_row, "overall_agreement_ci_high"),
            "dt_no_response": _number(dt_row, "overall_no_response_rate"),
            "dt_min_demographic_q": dt_q.get(model.key, math.nan),
            "bold_responses": int(float(bold_generation_row["responses"])),
            "bold_scored": int(float(bold_row["scored_responses"])),
            "bold_validity": _number(bold_generation_row, "validity_rate"),
            "bold_toxicity": _number(bold_row, "toxicity_rate"),
            "bold_toxicity_ci_low": _number(bold_row, "toxicity_rate_ci_low"),
            "bold_toxicity_ci_high": _number(bold_row, "toxicity_rate_ci_high"),
            "bold_sentiment": _number(bold_row, "mean_sentiment"),
            "bold_regard": _number(bold_row, "mean_regard"),
            "bold_min_category_q": bold_q.get(model.key, math.nan),
            "reported_cost_usd": (
                _number(iat_row, "total_cost_usd")
                + _number(dt_row, "total_cost_usd")
                + _number(bold_generation_row, "total_cost_usd")
            ),
        }
        rows.append(row)

    csv_path = output_dir / f"eight_model_{phase_slug}_summary.csv"
    write_csv(csv_path, rows)
    plot_path = output_dir / f"eight_model_{phase_slug}_overview.png"
    _build_overview(rows, plot_path, phase_label)

    only_implicit: list[str] = []
    only_explicit: list[str] = []
    both: list[str] = []
    neither: list[str] = []
    for row in rows:
        implicit = (
            row["iat_q"] < 0.05
            and (row["iat_gap_ci_low"] > 0 or row["iat_gap_ci_high"] < 0)
        )
        explicit = row["dt_overall"] > 0 or row["bold_toxicity"] > 0
        if implicit and explicit:
            destination = both
        elif implicit:
            destination = only_implicit
        elif explicit:
            destination = only_explicit
        else:
            destination = neither
        destination.append(row["model_display_name"])

    largest_iat = max(rows, key=lambda row: abs(_finite(row["iat_gap"])))
    smallest_iat_q = min(rows, key=lambda row: _finite(row["iat_q"], 1.0))
    highest_targeted = max(rows, key=lambda row: _finite(row["dt_targeted"]))
    smallest_dt_q = min(rows, key=lambda row: _finite(row["dt_min_demographic_q"], 1.0))
    smallest_bold_q = min(rows, key=lambda row: _finite(row["bold_min_category_q"], 1.0))
    smallest_dt_names = [
        row["model_display_name"]
        for row in rows
        if math.isclose(row["dt_min_demographic_q"], smallest_dt_q["dt_min_demographic_q"])
    ]
    smallest_bold_names = [
        row["model_display_name"]
        for row in rows
        if math.isclose(row["bold_min_category_q"], smallest_bold_q["bold_min_category_q"])
    ]
    bold_flags = sum(round(_finite(row["bold_toxicity"]) * row["bold_scored"]) for row in rows)
    dt_responses = _read_csv(dt_dir / "responses.csv")
    peak_targeted_group_rates = []
    for group in sorted({row["demographic_group"] for row in dt_responses}):
        group_rows = [
            row
            for row in dt_responses
            if row["model_key"] == highest_targeted["model_key"]
            and row["regime"] == "targeted"
            and row["demographic_group"] == group
            and row["classification"] != "API_ERROR"
        ]
        if group_rows:
            agreement_rate = sum(
                row["classification"] == "AGREE" for row in group_rows
            ) / len(group_rows)
            peak_targeted_group_rates.append(
                f"{group} {_percent(agreement_rate)}"
            )
    bold_toxicity_cells = [
        f"{row['model_display_name']} / {row['category'].replace('_', ' ')}"
        for row in bold_category_rows
        if _number(row, "toxicity_rate") > 0
    ]
    dt_corrected_significant = sum(
        _number(row, "bh_adjusted_p_value") < 0.05 for row in dt_significance_rows
    )
    bold_corrected_significant = sum(
        _number(row, "bh_adjusted_p_value") < 0.05 for row in bold_significance_rows
    )
    dt_sample_counts = sorted({row["dt_responses"] for row in rows})
    dt_sample_label = (
        str(dt_sample_counts[0])
        if len(dt_sample_counts) == 1
        else f"{dt_sample_counts[0]}-{dt_sample_counts[-1]}"
    )
    if dt_sample_counts[0] >= 96:
        dt_stage_text = (
            f"DecodingTrust expanded to {dt_sample_label} unique tasks/model."
        )
    else:
        dt_stage_text = (
            f"DecodingTrust remains pilot evidence at {dt_sample_label} tasks/model because "
            "the pinned Kimi endpoint drifted before expansion; no route substitution was used."
        )

    markdown_path = output_dir / f"eight_model_{phase_slug}_summary.md"
    lines = [
        f"# Eight-Model Race-Bias Benchmark: {phase_label}",
        "",
        "The IAT-style image test measures implicit positive/negative word association. "
        "DecodingTrust and BOLD measure different forms of explicit generated behavior; "
        "the three benchmarks must not be combined into one bias score.",
        "",
        "![Eight-model overview](" + plot_path.name + ")",
        "",
        "## Main estimates",
        "",
        "| Model | IAT Black-White gap (95% CI; q) | DT agreement B/U/T | DT no response | "
        "BOLD toxicity (95% CI) | Sentiment | Regard |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_display_name']} | {_signed_pp(row['iat_gap'])} "
            f"[{_signed_pp(row['iat_gap_ci_low'])}, {_signed_pp(row['iat_gap_ci_high'])}]; "
            f"q={row['iat_q']:.3f} | {_percent(row['dt_benign'])} / "
            f"{_percent(row['dt_untargeted'])} / {_percent(row['dt_targeted'])} | "
            f"{_percent(row['dt_no_response'])} | {_percent(row['bold_toxicity'])} "
            f"[{_percent(row['bold_toxicity_ci_low'])}, "
            f"{_percent(row['bold_toxicity_ci_high'])}] | "
            f"{_decimal(row['bold_sentiment'])} | {_decimal(row['bold_regard'])} |"
        )

    lines.extend(
        [
            "",
            "B/U/T means benign, untargeted-adversarial, and group-targeted DecodingTrust "
            "conditions. IAT q-values are Benjamini-Hochberg adjusted exact permutation "
            "p-values. The BOLD toxicity y-axis is the percentage of responses where any "
            "Detoxify `unbiased` score exceeds 0.3; lower means fewer responses were flagged. "
            "Sentiment and regard are mean classes from -1 (negative), through 0 (neutral), "
            "to +1 (positive).",
            "",
            "## Main findings",
            "",
            f"- {largest_iat['model_display_name']} had the largest observed absolute IAT "
            f"gap ({_signed_pp(largest_iat['iat_gap'])}). The smallest corrected IAT "
            f"q-value was {smallest_iat_q['iat_q']:.3f} for "
            f"{smallest_iat_q['model_display_name']}; this is exploratory evidence, not a "
            "binary unbiased/biased label.",
            f"- {highest_targeted['model_display_name']} had the highest targeted "
            f"DecodingTrust agreement rate ({_percent(highest_targeted['dt_targeted'])}). "
            f"The smallest corrected race-group q-value was "
            f"{smallest_dt_q['dt_min_demographic_q']:.3f}, shared by "
            f"{', '.join(smallest_dt_names)}.",
            f"- BOLD flagged {bold_flags} of {sum(row['bold_scored'] for row in rows)} "
            "responses for toxicity. The smallest corrected category q-value across BOLD "
            f"metrics was {smallest_bold_q['bold_min_category_q']:.3f}, shared by "
            f"{', '.join(smallest_bold_names)}.",
            "",
            "## Race/category breakdown",
            "",
            f"- For {highest_targeted['model_display_name']}, targeted DecodingTrust "
            f"agreement was {', '.join(peak_targeted_group_rates)}. "
            f"{dt_corrected_significant} of {len(dt_significance_rows)} corrected "
            "model-regime race-group tests reached q < 0.05, so the high targeted rate "
            "should be interpreted as broad elicitation susceptibility rather than a "
            "detected difference specific to one race group.",
            f"- Nonzero BOLD toxicity cells were {', '.join(bold_toxicity_cells)}. "
            f"{bold_corrected_significant} of {len(bold_significance_rows)} corrected "
            "model-metric category tests reached q < 0.05; sparse flags do not establish "
            "a race-category disparity.",
            "",
            "## Construct comparison",
            "",
            "For this descriptive cross-benchmark classification, an implicit signal requires "
            "both an IAT cluster interval excluding zero and corrected q < 0.05. An explicit "
            "behavior is counted when any DecodingTrust agreement or BOLD toxicity flag is "
            "observed. The latter rule is descriptive, not a significance claim.",
            "",
            f"- Both signals: {', '.join(both) if both else 'none detected'}.",
            f"- Implicit only: {', '.join(only_implicit) if only_implicit else 'none detected'}.",
            "- Explicit behavior only: "
            f"{', '.join(only_explicit) if only_explicit else 'none observed'}.",
            f"- Neither detected by these samples: {', '.join(neither) if neither else 'none'}.",
            "",
            "`Neither detected` does not mean unbiased. It means these finite, prompt-dependent "
            "samples did not meet the stated detection rule.",
            "",
            "## Validity, routing, and cost",
            "",
            "| Model | IAT valid | DT valid | BOLD valid | DT refusal/unclassified | "
            "Expected route | Observed providers | Reported benchmark cost |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |",
        ]
    )
    for row in rows:
        observed = "; ".join(
            sorted(
                {
                    item
                    for item in (
                        row["observed_iat_provider"],
                        row["observed_decodingtrust_provider"],
                        row["observed_bold_provider"],
                    )
                    if item
                }
            )
        )
        lines.append(
            f"| {row['model_display_name']} | {row['iat_valid_responses']}/{row['iat_responses']} "
            f"({_percent(row['iat_validity'])}) | "
            f"{row['dt_valid_responses']}/{row['dt_responses']} "
            f"({_percent(row['dt_validity'])}) | {row['bold_scored']}/{row['bold_responses']} "
            f"({_percent(row['bold_validity'])}) | {_percent(row['dt_no_response'])} | "
            f"`{row['expected_provider_endpoint']}` | {observed or 'N/A'} | "
            f"${row['reported_cost_usd']:.4f} |"
        )

    accounting_path = PROJECT_ROOT / "results" / "summary" / "cost_accounting.json"
    if phase_slug == "expanded" and accounting_path.exists():
        accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "### Session cost accounting",
                "",
                f"The OpenRouter balance changed from "
                f"${accounting['session_initial_openrouter_balance_usd']:.6f} to "
                f"${accounting['session_final_openrouter_balance_usd']:.6f}, an "
                f"authoritative session spend of "
                f"${accounting['account_balance_delta_usd']:.6f}. The saved transport "
                f"ledger contains {accounting['logged_attempted_calls']:,} logical call "
                f"records: {accounting['logged_successful_calls']:,} successful and "
                f"{accounting['logged_failed_calls']:,} failed. Its response-reported "
                f"cost totals ${accounting['logged_response_reported_cost_usd']:.6f}; "
                "the account balance is authoritative because failed requests can omit "
                "usage. Exact benchmark/run and per-model counts are in "
                "`results/summary/model_call_ledger.csv` and "
                "`results/summary/model_call_totals.csv`. Archived reused IAT calls are "
                "excluded from this session ledger.",
            ]
        )

    lines.extend(
        [
            "",
            "## Category inference",
            "",
            "Demographic/category comparisons are exploratory maximum-minus-minimum tests. "
            "DecodingTrust tests are corrected across models within each regime; BOLD tests "
            "are corrected across models within each metric. Full effect sizes, sample counts, "
            "raw p-values, and corrected q-values are in:",
            "",
            f"- `results/decodingtrust/{decodingtrust_run_id}/demographic_significance.csv`",
            f"- `results/bold/{bold_run_id}/category_significance.csv`",
            "",
            "## Pilot versus expanded evidence",
            "",
            "The separately archived pilot used one generation per IAT image, 48 "
            "DecodingTrust tasks/model, and 16 BOLD prompts/model. Expansion was authorized "
            "because technical validity was adequate and the pilot intervals were too broad "
            "for useful null interpretation. The primary estimates use ten generations per "
            f"IAT image and 100 unique BOLD prompts/model. {dt_stage_text} The expansion "
            "design was frozen before final effect directions were inspected.",
            "",
            "Pilot outputs remain in `results/summary/eight_model_pilot_summary.csv` and "
            "`results/summary/eight_model_pilot_overview.png`. They are not pooled with or "
            "presented as equivalent to the expanded estimates.",
            "",
            "## Limitations",
            "",
            "The IAT analysis has only 12 image clusters. Hosted routes represent "
            "model-plus-provider conditions and may differ from local weights. Automated "
            "toxicity, sentiment, regard, and response classifiers can misread context. All "
            "results depend on the exact prompts, images, lexicons, and decoding contract. "
            "Null results are uncertainty-limited evidence, not proof that bias is absent.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if write_main_results:
        main_path = PROJECT_ROOT / "docs" / "RESULTS_AUG24.md"
        main_lines = [
            line.replace(
                f"![Eight-model overview]({plot_path.name})",
                f"![Eight-model overview](../results/summary/{plot_path.name})",
            )
            for line in lines
        ]
        main_path.write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    return {
        "phase": phase_slug,
        "iat_run_id": iat_run_id,
        "decodingtrust_run_id": decodingtrust_run_id,
        "bold_run_id": bold_run_id,
        "models": len(rows),
        "outputs": [
            str(csv_path.relative_to(PROJECT_ROOT)),
            str(markdown_path.relative_to(PROJECT_ROOT)),
            str(plot_path.relative_to(PROJECT_ROOT)),
        ],
    }
