#!/usr/bin/env python3
# ruff: noqa: E402, E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.constants import NEGATIVE_IAT_SET
from august10_research.neutral_retrospective import (
    BASE_NEUTRAL_CSV_SHA256,
    EVALUATION_SPECS,
    NEUTRAL_METADATA_SHA256,
    metric_row,
    validate_evaluation,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected}, observed {actual}")


def copy_evaluation(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in (
        "adapter_neutral_responses.csv",
        "adapter_neutral_metrics.json",
        "neutral_paired_bootstrap.json",
        "neutral_micro_outcome.json",
        "neutral_micro_evaluation_manifest.json",
    ):
        shutil.copy2(source / name, target / name)


def markdown_row(row: dict[str, Any]) -> str:
    return (
        f"| {row['label']} | {row['black_negative_rate']:.2%} | "
        f"{row['white_negative_rate']:.2%} | {row['absolute_race_gap']:.2%} | "
        f"{row['response_entropy']:.3f} | {row['unique_response_rate']:.2%} |"
    )


def build_plots(
    output_dir: Path,
    metric_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    intersection_rows: list[dict[str, Any]],
    word_rows: list[dict[str, Any]],
    best_key: str,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold"})
    colors = {"Black": "#D55E00", "White": "#0072B2"}
    condition_colors = ["#666666", "#CC79A7", "#E69F00", "#C44E52", "#56B4E9", "#009E73"]
    labels = ["Base"] + [spec.short_label for spec in EVALUATION_SPECS]
    x = np.arange(len(metric_rows))
    width = 0.36
    artifacts: list[str] = []

    fig, axis = plt.subplots(figsize=(12, 6), dpi=180)
    black = [row["black_negative_rate"] for row in metric_rows]
    white = [row["white_negative_rate"] for row in metric_rows]
    black_bars = axis.bar(x - width / 2, black, width, label="Black", color=colors["Black"])
    white_bars = axis.bar(x + width / 2, white, width, label="White", color=colors["White"])
    axis.bar_label(black_bars, labels=[f"{value:.1%}" for value in black], padding=3, fontsize=8)
    axis.bar_label(white_bars, labels=[f"{value:.1%}" for value in white], padding=3, fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylim(0, max(black + white) * 1.22)
    axis.set_ylabel("Negative selected-term rate")
    axis.set_title("12-image neutral IAT word comparison")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "01_neutral12_negative_rates.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=180)
    gaps = [row["absolute_race_gap"] for row in metric_rows]
    bars = axes[0].bar(x, gaps, color=condition_colors)
    axes[0].bar_label(bars, labels=[f"{value:.1%}" for value in gaps], padding=3, fontsize=8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, max(gaps) * 1.22)
    axes[0].set_ylabel("Absolute Black-White gap")
    axes[0].set_title("Absolute race gap")

    bx = np.arange(len(bootstrap_rows))
    means = [row["black_mean_rate_change"] for row in bootstrap_rows]
    lows = [row["black_ci_low"] for row in bootstrap_rows]
    highs = [row["black_ci_high"] for row in bootstrap_rows]
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].errorbar(
        bx,
        means,
        yerr=[
            [mean - low for mean, low in zip(means, lows, strict=False)],
            [high - mean for mean, high in zip(means, highs, strict=False)],
        ],
        fmt="o",
        capsize=5,
        color="#333333",
        ecolor="#666666",
    )
    axes[1].set_xticks(bx, labels[1:])
    axes[1].set_ylabel("Adapter minus base rate")
    axes[1].set_title("Black image-level bootstrap, 95% CI")
    fig.tight_layout()
    path = output_dir / "02_gap_and_bootstrap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path.name)

    adapters = metric_rows[1:]
    x_adapter = np.arange(len(adapters))
    fig, axis = plt.subplots(figsize=(10, 5.5), dpi=180)
    entropy = [row["entropy_retention"] for row in adapters]
    unique = [row["unique_response_retention"] for row in adapters]
    entropy_bars = axis.bar(
        x_adapter - width / 2, entropy, width, label="Entropy retention", color="#009E73"
    )
    unique_bars = axis.bar(
        x_adapter + width / 2, unique, width, label="Unique-response retention", color="#E69F00"
    )
    axis.bar_label(
        entropy_bars, labels=[f"{value:.1%}" for value in entropy], padding=3, fontsize=8
    )
    axis.bar_label(unique_bars, labels=[f"{value:.1%}" for value in unique], padding=3, fontsize=8)
    axis.axhline(
        0.95, color="#C44E52", linestyle="--", linewidth=1.3, label="95% preservation gate"
    )
    axis.set_xticks(x_adapter, labels[1:])
    axis.set_ylim(0, max(1.08, max(entropy + unique) * 1.12))
    axis.set_ylabel("Fraction of base metric")
    axis.set_title("Response-distribution preservation")
    axis.legend(loc="lower left")
    fig.tight_layout()
    path = output_dir / "03_diversity_preservation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path.name)

    best_words = [row for row in word_rows if row["model_key"] in ("base", best_key)]
    totals: dict[str, int] = {}
    for row in best_words:
        totals[row["word"]] = totals.get(row["word"], 0) + int(row["count"])
    words = [
        name
        for name, count in sorted(totals.items(), key=lambda item: item[1], reverse=True)
        if count
    ][:10][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=180, sharex=True, sharey=True)
    for axis, race in zip(axes, ("Black", "White"), strict=False):
        y = np.arange(len(words))
        base_counts = [
            next(
                (
                    int(row["count"])
                    for row in best_words
                    if row["model_key"] == "base" and row["race"] == race and row["word"] == word
                ),
                0,
            )
            for word in words
        ]
        adapter_counts = [
            next(
                (
                    int(row["count"])
                    for row in best_words
                    if row["model_key"] == best_key and row["race"] == race and row["word"] == word
                ),
                0,
            )
            for word in words
        ]
        axis.barh(y - width / 2, base_counts, width, label="Base", color="#666666")
        axis.barh(y + width / 2, adapter_counts, width, label="Mean 2e-5", color="#009E73")
        axis.set_yticks(y, [word.title() for word in words])
        axis.set_xlabel("Selected count")
        axis.set_title(race)
        axis.legend()
    fig.suptitle("Negative-word counts: base vs directional adapter", fontweight="bold")
    fig.tight_layout()
    path = output_dir / "04_directional_adapter_word_counts.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path.name)

    selected_intersections = [
        row for row in intersection_rows if row["model_key"] in ("base", best_key)
    ]
    groups = ("black_female", "black_male", "white_female", "white_male")
    group_labels = ("Black female", "Black male", "White female", "White male")
    fig, axis = plt.subplots(figsize=(10, 5.5), dpi=180)
    gx = np.arange(len(groups))
    base_values = [
        next(
            row["negative_rate"]
            for row in selected_intersections
            if row["model_key"] == "base" and row["group"] == group
        )
        for group in groups
    ]
    adapter_values = [
        next(
            row["negative_rate"]
            for row in selected_intersections
            if row["model_key"] == best_key and row["group"] == group
        )
        for group in groups
    ]
    base_bars = axis.bar(gx - width / 2, base_values, width, label="Base", color="#666666")
    adapter_bars = axis.bar(
        gx + width / 2, adapter_values, width, label="Mean 2e-5", color="#009E73"
    )
    axis.bar_label(
        base_bars, labels=[f"{value:.1%}" for value in base_values], padding=3, fontsize=8
    )
    axis.bar_label(
        adapter_bars, labels=[f"{value:.1%}" for value in adapter_values], padding=3, fontsize=8
    )
    axis.set_xticks(gx, group_labels)
    axis.set_ylim(0, max(base_values + adapter_values) * 1.18)
    axis.set_ylabel("Negative selected-term rate")
    axis.set_title("Intersectional rates on the 12-image panel")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "05_intersectional_rates.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path.name)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    black_bars = axes[0, 0].bar(x - width / 2, black, width, color=colors["Black"], label="Black")
    white_bars = axes[0, 0].bar(x + width / 2, white, width, color=colors["White"], label="White")
    axes[0, 0].bar_label(
        black_bars, labels=[f"{value:.1%}" for value in black], fontsize=7, padding=2
    )
    axes[0, 0].bar_label(
        white_bars, labels=[f"{value:.1%}" for value in white], fontsize=7, padding=2
    )
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0, max(black + white) * 1.24)
    axes[0, 0].set_title("Negative-term rates")
    axes[0, 0].legend()
    gap_bars = axes[0, 1].bar(x, gaps, color=condition_colors)
    axes[0, 1].bar_label(gap_bars, labels=[f"{value:.1%}" for value in gaps], fontsize=7, padding=2)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylim(0, max(gaps) * 1.24)
    axes[0, 1].set_title("Absolute race gap")
    axes[1, 0].bar(x_adapter - width / 2, entropy, width, color="#009E73", label="Entropy")
    axes[1, 0].bar(x_adapter + width / 2, unique, width, color="#E69F00", label="Unique")
    axes[1, 0].axhline(0.95, color="#C44E52", linestyle="--", linewidth=1.2)
    axes[1, 0].set_xticks(x_adapter, labels[1:])
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].set_title("Diversity retention")
    axes[1, 0].legend()
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        "Decision\n\nDirectional movement: mean-token 2e-5\n"
        "Research preservation gate: fail\n\n"
        "Retrospective 12-image analysis\n"
        "12 images x 10 repeats\n"
        "Frozen base responses reused\n"
        "100,000-sample image bootstrap\n\n"
        "No adapter is eligible for deployment\n"
        "or confirmatory fairness claims.",
        va="top",
        fontsize=12,
    )
    fig.suptitle(
        "First August 10 LoRAs on the neutral IAT benchmark", fontsize=16, fontweight="bold"
    )
    fig.tight_layout()
    path = output_dir / "00_retrospective_dashboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.insert(0, path.name)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile the first August 10 LoRAs on the frozen 12-image neutral IAT panel."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--base-neutral-csv", required=True, type=Path)
    parser.add_argument("--neutral-metadata-csv", required=True, type=Path)
    parser.add_argument("--frozen-input-manifest", required=True, type=Path)
    parser.add_argument("--reference-checksums", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_hash(args.base_neutral_csv, BASE_NEUTRAL_CSV_SHA256, "base neutral CSV")
    require_hash(args.neutral_metadata_csv, NEUTRAL_METADATA_SHA256, "neutral metadata")

    records: list[dict[str, Any]] = []
    canonical_base: dict[str, Any] | None = None
    for spec in EVALUATION_SPECS:
        source = args.run_dir / spec.key
        manifest = read_json(source / "neutral_micro_evaluation_manifest.json")
        response_rows = read_csv(source / "adapter_neutral_responses.csv")
        validate_evaluation(spec, manifest, response_rows)
        response_hash = sha256_file(source / "adapter_neutral_responses.csv")
        if response_hash != manifest["adapter_response_csv_sha256"]:
            raise ValueError(f"{spec.key} response CSV hash mismatch")
        base = read_json(source / "base_neutral_metrics.json")
        adapter = read_json(source / "adapter_neutral_metrics.json")
        bootstrap = read_json(source / "neutral_paired_bootstrap.json")
        if canonical_base is None:
            canonical_base = base
        elif base != canonical_base:
            raise ValueError(f"{spec.key} base metrics differ from the other evaluations")
        records.append(
            {
                "spec": spec,
                "source": source,
                "manifest": manifest,
                "adapter": adapter,
                "bootstrap": bootstrap,
            }
        )
    if canonical_base is None:
        raise ValueError("no evaluations were loaded")

    if args.output_dir.exists():
        if not args.replace:
            raise FileExistsError(f"output directory exists: {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    base_row = metric_row(
        key="base",
        label="Pinned base",
        objective="base",
        learning_rate=None,
        adapter_sha256="",
        metrics=canonical_base,
        base=canonical_base,
    )
    metric_rows = [base_row]
    bootstrap_rows: list[dict[str, Any]] = []
    intersection_rows: list[dict[str, Any]] = []
    word_rows: list[dict[str, Any]] = []

    for group, values in canonical_base["intersectional_metrics"].items():
        intersection_rows.append(
            {"model_key": "base", "label": "Pinned base", "group": group, **values}
        )
    for race, counts in canonical_base["word_counts"].items():
        for word, count in sorted(counts.items()):
            if word in NEGATIVE_IAT_SET:
                word_rows.append(
                    {
                        "model_key": "base",
                        "label": "Pinned base",
                        "race": race,
                        "word": word,
                        "count": count,
                    }
                )

    for record in records:
        spec = record["spec"]
        row = metric_row(
            key=spec.key,
            label=spec.label,
            objective=spec.objective,
            learning_rate=spec.learning_rate,
            adapter_sha256=spec.adapter_sha256,
            metrics=record["adapter"],
            base=canonical_base,
        )
        metric_rows.append(row)
        bootstrap = record["bootstrap"]
        black = bootstrap["groups"]["Black"]
        white = bootstrap["groups"]["White"]
        gap = bootstrap["race_gap_change"]
        bootstrap_rows.append(
            {
                "model_key": spec.key,
                "label": spec.label,
                "samples": bootstrap["samples"],
                "image_count_per_race": black["images"],
                "black_mean_rate_change": black["mean_rate_change"],
                "black_ci_low": black["ci_95"][0],
                "black_ci_high": black["ci_95"][1],
                "black_probability_decreased": black["probability_rate_decreased"],
                "white_mean_rate_change": white["mean_rate_change"],
                "white_ci_low": white["ci_95"][0],
                "white_ci_high": white["ci_95"][1],
                "white_probability_decreased": white["probability_rate_decreased"],
                "gap_mean_change": gap["mean"],
                "gap_ci_low": gap["ci_95"][0],
                "gap_ci_high": gap["ci_95"][1],
                "gap_probability_decreased": gap["probability_absolute_direction_decreased"],
            }
        )
        for group, values in record["adapter"]["intersectional_metrics"].items():
            intersection_rows.append(
                {"model_key": spec.key, "label": spec.label, "group": group, **values}
            )
        for race, counts in record["adapter"]["word_counts"].items():
            for word, count in sorted(counts.items()):
                if word in NEGATIVE_IAT_SET:
                    word_rows.append(
                        {
                            "model_key": spec.key,
                            "label": spec.label,
                            "race": race,
                            "word": word,
                            "count": count,
                        }
                    )
        copy_evaluation(record["source"], args.output_dir / "evaluations" / spec.key)

    directional = [row for row in metric_rows[1:] if row["directional_rate_pass"]]
    best = min(metric_rows[1:], key=lambda row: row["absolute_race_gap"])
    research_candidates = [row for row in metric_rows[1:] if row["research_gate_pass"]]
    decision = (
        "retrospective_directional_improvement_no_preservation_pass"
        if directional and not research_candidates
        else "no_directional_improvement"
    )

    write_csv(args.output_dir / "comparison_metrics.csv", metric_rows)
    write_csv(args.output_dir / "bootstrap_metrics.csv", bootstrap_rows)
    write_csv(args.output_dir / "intersectional_metrics.csv", intersection_rows)
    write_csv(args.output_dir / "negative_word_counts.csv", word_rows)

    reference_dir = args.output_dir / "reference"
    reference_dir.mkdir()
    for source in (
        args.base_neutral_csv,
        args.neutral_metadata_csv,
        args.frozen_input_manifest,
        args.reference_checksums,
    ):
        shutil.copy2(source, reference_dir / source.name)
    write_json(reference_dir / "base_neutral_metrics.json", canonical_base)

    inventory = [
        {
            "model_key": record["spec"].key,
            "label": record["spec"].label,
            "adapter_sha256": record["spec"].adapter_sha256,
            "adapter_config_sha256": record["manifest"]["adapter_hashes"]["adapter_config.json"],
            "training_manifest_sha256": record["manifest"]["training_manifest_sha256"],
            "response_csv_sha256": record["manifest"]["adapter_response_csv_sha256"],
        }
        for record in records
    ]
    write_json(args.output_dir / "adapter_inventory.json", inventory)

    plots = build_plots(
        args.output_dir,
        metric_rows,
        bootstrap_rows,
        intersection_rows,
        word_rows,
        best["model_key"],
    )
    best_bootstrap = next(row for row in bootstrap_rows if row["model_key"] == best["model_key"])
    summary = {
        "decision": decision,
        "confirmatory_claim_allowed": False,
        "multiple_comparisons": len(EVALUATION_SPECS),
        "base_neutral_csv_sha256": BASE_NEUTRAL_CSV_SHA256,
        "neutral_metadata_sha256": NEUTRAL_METADATA_SHA256,
        "images": 12,
        "images_per_race": 6,
        "repeats_per_image": 10,
        "base_inference_rerun": False,
        "best_directional_model": best,
        "best_directional_bootstrap": best_bootstrap,
        "directional_model_keys": [row["model_key"] for row in directional],
        "research_candidate_keys": [row["model_key"] for row in research_candidates],
        "preservation_rule": {
            "status": "conservative_retrospective_reporting_rule_not_preregistered",
            "minimum_entropy_retention": 0.95,
            "minimum_unique_response_retention": 0.95,
            "maximum_validity_drop": 0.01,
            "identity_leakage_must_not_increase": True,
        },
        "plots": plots,
    }
    write_json(args.output_dir / "research_summary.json", summary)

    results = [
        "# First August 10 LoRAs on the 12-Image Neutral IAT Benchmark",
        "",
        "**Decision: RETROSPECTIVE DIRECTIONAL MOVEMENT; NO RESEARCH OR DEPLOYMENT CANDIDATE.**",
        "",
        "The frozen pinned-base response CSV was reused; base inference was not rerun. Five final adapters from the first August 10 pilot were each evaluated on the same 12 neutral images with 10 repetitions, eager attention, seed `7192026`, temperature/top-p `1.0`, and the pinned Gemma revision.",
        "",
        "| Condition | Black negative | White negative | Absolute gap | Entropy | Unique rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[markdown_row(row) for row in metric_rows],
        "",
        "## Directional finding",
        "",
        f"`{best['label']}` was the only adapter that lowered both the Black negative rate and the absolute race gap. Black fell from {base_row['black_negative_rate']:.2%} to {best['black_negative_rate']:.2%}, White from {base_row['white_negative_rate']:.2%} to {best['white_negative_rate']:.2%}, and the absolute gap from {base_row['absolute_race_gap']:.2%} to {best['absolute_race_gap']:.2%}.",
        "",
        f"The paired image-level bootstrap estimated a Black rate change of {100 * best_bootstrap['black_mean_rate_change']:+.2f} percentage points with 95% interval [{100 * best_bootstrap['black_ci_low']:+.2f}, {100 * best_bootstrap['black_ci_high']:+.2f}] points and probability of a decrease {best_bootstrap['black_probability_decreased']:.2%}. The absolute-gap change was {100 * best_bootstrap['gap_mean_change']:+.2f} percentage points with interval [{100 * best_bootstrap['gap_ci_low']:+.2f}, {100 * best_bootstrap['gap_ci_high']:+.2f}] points.",
        "",
        "At the word level for Black images, `Pain` fell from 21 to 10, `Sadness` from 20 to 9, `Grief` from 13 to 6, and `Hurtful` from 7 to 4. One new `Poison` selection appeared, however, so the vocabulary change was not uniformly beneficial. White negative selections fell from 6 to 0.",
        "",
        "## Preservation finding",
        "",
        f"Validity remained {best['valid_rate']:.2%} and identity leakage remained {best['identity_leakage_rate']:.2%}. Entropy retention was {best['entropy_retention']:.1%}, but unique-response retention was only {best['unique_response_retention']:.1%} ({base_row['unique_responses']} to {best['unique_responses']} unique responses). It therefore fails the conservative 95% preservation rule applied in this retrospective report.",
        "",
        "The legacy micro evaluator marks this adapter as a guardrail pass because its historical diversity check required only 75% entropy retention and did not require 95% unique-response retention. That legacy pass is not equivalent to preserving the base response distribution.",
        "",
        "The other four adapters increased Black negativity and widened the gap. This establishes strong hyperparameter sensitivity and explains why the unchanged FairFace development rates were insufficient to characterize neutral-benchmark behavior.",
        "",
        "## Limits",
        "",
        "- This is a retrospective analysis after five adapters were available, not a confirmatory experiment.",
        "- The panel has only six images per race. Repeated generations estimate behavior on those images but do not increase visual population size.",
        "- The benchmark was previously observed, so it must not be used to tune and then claim final confirmation.",
        "- The first August 10 dataset still had zero negative-expression IAT blocks in training.",
        "- No adapter should be deployed or described as successfully debiased from this result alone.",
        "",
        "## Next step",
        "",
        "Use the mean-token `2e-5` result as evidence that the endpoint can move, not as a selected final model. Construct expression-conditioned pairs, choose checkpoints on a separate FairFace development panel with a 95% diversity gate, repeat across seeds, and reserve different matched images for confirmation.",
        "",
    ]
    (args.output_dir / "RESULTS.md").write_text("\n".join(results), encoding="utf-8")

    ledger_lines = []
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_SHA256SUMS":
            ledger_lines.append(f"{sha256_file(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "ARTIFACT_SHA256SUMS").write_text(
        "\n".join(ledger_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
