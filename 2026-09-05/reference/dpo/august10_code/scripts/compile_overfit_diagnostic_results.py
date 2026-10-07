#!/usr/bin/env python3
# ruff: noqa: E402

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


def iat_values(metrics: dict[str, Any]) -> dict[str, float]:
    race = metrics["race_metrics"]
    negative = sum(int(race[name]["negative_terms"]) for name in ("Black", "White"))
    selected = sum(int(race[name]["selected_terms"]) for name in ("Black", "White"))
    return {
        "black_negative_rate": float(race["Black"]["negative_rate"]),
        "white_negative_rate": float(race["White"]["negative_rate"]),
        "overall_negative_rate": negative / selected if selected else 0.0,
        "absolute_race_gap": float(metrics["negative_rate_gap_abs"]),
        "valid_rate": float(metrics["valid_rate"]),
        "identity_leakage_rate": float(metrics["identity_leakage_rate"]),
        "response_entropy": float(metrics["response_entropy"]),
        "unique_response_rate": float(metrics["unique_response_rate"]),
    }


def expression_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for condition in ("base", "adapter"):
        output[condition] = {}
        for expression in ("negative", "neutral", "positive"):
            selected = [
                row
                for row in rows
                if row["condition"] == condition
                and row["expression_class"] == expression
            ]
            words = [
                word.strip().lower()
                for row in selected
                for word in row["response"].split(",")
            ]
            negative = sum(word in NEGATIVE_IAT_SET for word in words)
            output[condition][expression] = {
                "responses": len(selected),
                "negative_terms": negative,
                "selected_terms": len(words),
                "negative_rate": negative / len(words) if words else 0.0,
            }
    return output


def copy_artifacts(source: Path, target: Path, names: tuple[str, ...]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = source / name
        if path.is_file():
            shutil.copy2(path, target / name)


def markdown_metric_row(label: str, values: dict[str, float]) -> str:
    cells = (
        label,
        f"{values['black_negative_rate']:.2%}",
        f"{values['white_negative_rate']:.2%}",
        f"{values['absolute_race_gap']:.2%}",
        f"{values['valid_rate']:.2%}",
        f"{values['response_entropy']:.3f}",
        f"{values['unique_response_rate']:.2%}",
    )
    return "| " + " | ".join(cells) + " |"


def plot_results(
    output_dir: Path,
    checkpoint_rows: list[dict[str, Any]],
    expressions: dict[str, dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"Base": "#5B6770", "Update 8": "#0B6E75", "Update 16": "#B55239"}
    artifacts = []
    labels = ["Base", "Update 8", "Update 16"]
    races = ("Black", "White")
    x = np.arange(len(races))
    width = 0.24
    fig, axis = plt.subplots(figsize=(8, 4.8), dpi=180)
    for index, label in enumerate(labels):
        row = next(item for item in checkpoint_rows if item["condition"] == label)
        values = [row[f"{race.lower()}_negative_rate"] for race in races]
        bars = axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=colors[label],
        )
        axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3)
    axis.set_xticks(x, races)
    axis.set_ylim(0, 0.20)
    axis.set_ylabel("Negative selected-term rate")
    axis.set_title("Exact-image IAT canary")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "01_canary_negative_rates.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path.name)

    expression_labels = ("negative", "neutral", "positive")
    x = np.arange(len(expression_labels))
    fig, axis = plt.subplots(figsize=(8, 4.8), dpi=180)
    base_expression_values = [
        expressions["base"][name]["negative_rate"] for name in expression_labels
    ]
    base_bars = axis.bar(
        x - width / 2,
        base_expression_values,
        width,
        label="Base",
        color=colors["Base"],
    )
    adapter_expression_values = [
        expressions["adapter"][name]["negative_rate"] for name in expression_labels
    ]
    adapter_bars = axis.bar(
        x + width / 2,
        adapter_expression_values,
        width,
        label="Update 8",
        color=colors["Update 8"],
    )
    axis.bar_label(
        base_bars,
        labels=[f"{value:.1%}" for value in base_expression_values],
        padding=3,
    )
    axis.bar_label(
        adapter_bars,
        labels=[f"{value:.1%}" for value in adapter_expression_values],
        padding=3,
    )
    axis.set_xticks(x, [name.title() for name in expression_labels])
    axis.set_ylim(0, 0.32)
    axis.set_ylabel("Negative selected-term rate")
    axis.set_title("Expression sensitivity check")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "02_expression_sensitivity.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path.name)

    base = checkpoint_rows[0]
    retention_metrics = (
        ("Entropy", "response_entropy"),
        ("Unique responses", "unique_response_rate"),
        ("Valid format", "valid_rate"),
    )
    fig, axis = plt.subplots(figsize=(8, 4.8), dpi=180)
    x = np.arange(len(retention_metrics))
    for index, label in enumerate(("Update 8", "Update 16")):
        row = next(item for item in checkpoint_rows if item["condition"] == label)
        values = [
            row[key] / base[key] if base[key] else 1.0
            for _, key in retention_metrics
        ]
        axis.bar(
            x + (index - 0.5) * 0.34,
            values,
            0.34,
            label=label,
            color=colors[label],
        )
    axis.axhline(0.80, color="black", linestyle="--", linewidth=0.9, label="80% floor")
    axis.set_xticks(x, [label for label, _ in retention_metrics])
    axis.set_ylim(0, 1.15)
    axis.set_ylabel("Retention relative to base")
    axis.set_title("Generation-quality preservation")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "03_checkpoint_preservation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(path.name)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    for index, label in enumerate(labels):
        row = next(item for item in checkpoint_rows if item["condition"] == label)
        values = [row["black_negative_rate"], row["white_negative_rate"]]
        bars = axes[0].bar(
            x=np.arange(2) + (index - 1) * width,
            height=values,
            width=width,
            label=label,
            color=colors[label],
        )
        axes[0].bar_label(
            bars, labels=[f"{value:.1%}" for value in values], padding=2, fontsize=7
        )
    axes[0].set_xticks(np.arange(2), races)
    axes[0].set_ylim(0, 0.20)
    axes[0].set_title("Negative terms by race")
    axes[0].set_ylabel("Rate")
    axes[0].legend(fontsize=8)
    base_bars = axes[1].bar(
        np.arange(3) - width / 2,
        base_expression_values,
        width,
        label="Base",
        color=colors["Base"],
    )
    adapter_bars = axes[1].bar(
        np.arange(3) + width / 2,
        adapter_expression_values,
        width,
        label="Update 8",
        color=colors["Update 8"],
    )
    axes[1].bar_label(
        base_bars,
        labels=[f"{value:.1%}" for value in base_expression_values],
        padding=2,
        fontsize=7,
    )
    axes[1].bar_label(
        adapter_bars,
        labels=[f"{value:.1%}" for value in adapter_expression_values],
        padding=2,
        fontsize=7,
    )
    axes[1].set_xticks(np.arange(3), [name.title() for name in expression_labels])
    axes[1].set_ylim(0, 0.32)
    axes[1].set_title("Expression sensitivity")
    axes[1].legend(fontsize=8)
    for index, label in enumerate(("Update 8", "Update 16")):
        row = next(item for item in checkpoint_rows if item["condition"] == label)
        axes[2].bar(
            np.arange(2) + (index - 0.5) * 0.34,
            [
                row["response_entropy"] / base["response_entropy"],
                row["unique_response_rate"] / base["unique_response_rate"],
            ],
            0.34,
            label=label,
            color=colors[label],
        )
    axes[2].axhline(0.80, color="black", linestyle="--", linewidth=0.9)
    axes[2].set_xticks(np.arange(2), ["Entropy", "Unique"])
    axes[2].set_ylim(0, 1.10)
    axes[2].set_title("Quality retention")
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "Gemma 3 DPO/LoRA mechanistic diagnostic: controllable, not deployable",
        fontsize=14,
    )
    fig.tight_layout()
    path = output_dir / "00_diagnostic_dashboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.insert(0, path.name)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the August 10 overfit diagnostic.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--screening-data-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    screening_data_dir = args.screening_data_dir.resolve()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.replace:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    base_dir = run_dir / "base_generation"
    step8_dir = run_dir / "evaluation_step0008"
    step16_dir = run_dir / "evaluation_step0016"
    training_dir = run_dir / "training_mean_lr5e5_seed81020261"
    screen_dir = run_dir.parent / "diagnostic_overfit_v1" / "base_training_pool_screen"

    base_summary = read_json(base_dir / "development_generation_summary.json")
    step8_summary = read_json(step8_dir / "development_generation_summary.json")
    step16_summary = read_json(step16_dir / "development_generation_summary.json")
    base_values = iat_values(base_summary["base_iat"])
    step8_values = iat_values(step8_summary["adapter_iat"])
    step16_values = iat_values(step16_summary["adapter_iat"])
    checkpoint_rows = [
        {"condition": "Base", **base_values},
        {"condition": "Update 8", **step8_values},
        {"condition": "Update 16", **step16_values},
    ]
    write_csv(output_dir / "checkpoint_metrics.csv", checkpoint_rows)

    responses = read_csv(step8_dir / "development_generation_responses.csv")
    expressions = expression_metrics(responses)
    write_json(output_dir / "expression_metrics.json", expressions)

    mechanistic_checks = {
        "black_negative_rate_reduced_at_least_10pp": (
            base_values["black_negative_rate"] - step8_values["black_negative_rate"] >= 0.10
        ),
        "white_negative_rate_reduced_at_least_10pp": (
            base_values["white_negative_rate"] - step8_values["white_negative_rate"] >= 0.10
        ),
        "absolute_gap_not_increased": (
            step8_values["absolute_race_gap"] <= base_values["absolute_race_gap"] + 1e-12
        ),
        "validity_preserved": step8_values["valid_rate"] >= base_values["valid_rate"],
        "identity_leakage_not_increased": (
            step8_values["identity_leakage_rate"]
            <= base_values["identity_leakage_rate"]
        ),
        "entropy_retained_75pct": (
            step8_values["response_entropy"] >= 0.75 * base_values["response_entropy"]
        ),
        "unique_response_rate_retained_80pct": (
            step8_values["unique_response_rate"]
            >= 0.80 * base_values["unique_response_rate"]
        ),
    }
    expression_checks = {
        "neutral_negative_rate_reduced": (
            expressions["adapter"]["neutral"]["negative_rate"]
            < expressions["base"]["neutral"]["negative_rate"]
        ),
        "positive_negative_rate_not_increased": (
            expressions["adapter"]["positive"]["negative_rate"]
            <= expressions["base"]["positive"]["negative_rate"]
        ),
        "negative_expression_signal_retained_50pct": (
            expressions["adapter"]["negative"]["negative_rate"]
            >= 0.50 * expressions["base"]["negative"]["negative_rate"]
        ),
    }
    summary = {
        "protocol_version": "august10-overfit-diagnostic-results-v1",
        "decision": "mechanistic_pass_not_research_candidate",
        "non_confirmatory": True,
        "confirmatory_evaluation_opened": False,
        "base_model": "google/gemma-3-12b-it",
        "base_revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        "recommended_diagnostic_checkpoint": "update_8",
        "promoted_for_research_training": False,
        "mechanistic_pass": all(mechanistic_checks.values()),
        "mechanistic_checks": mechanistic_checks,
        "expression_sensitivity_pass": all(expression_checks.values()),
        "expression_sensitivity_checks": expression_checks,
        "checkpoint_metrics": checkpoint_rows,
        "expression_metrics": expressions,
        "interpretation": (
            "DPO/LoRA can causally lower the generated negative-term metric. The blunt "
            "all-positive target also removes negative-expression sensitivity, so this "
            "adapter is a pipeline diagnostic and must not be reported as a debiased model."
        ),
    }
    write_json(output_dir / "diagnostic_summary.json", summary)
    plots = plot_results(output_dir, checkpoint_rows, expressions)

    copy_artifacts(
        data_dir,
        output_dir / "dataset",
        (
            "diagnostic_dpo_pairs_frozen.csv",
            "diagnostic_gate.json",
            "diagnostic_manifest.json",
        ),
    )
    copy_artifacts(
        screening_data_dir,
        output_dir / "screening",
        ("screening_panel.csv", "screening_manifest.json"),
    )
    common_eval = (
        "development_generation_responses.csv",
        "development_generation_summary.json",
        "development_generation_gate.json",
        "development_generation_manifest.json",
    )
    copy_artifacts(base_dir, output_dir / "base", common_eval)
    copy_artifacts(step8_dir, output_dir / "update_0008", common_eval)
    copy_artifacts(step16_dir, output_dir / "update_0016", common_eval)
    copy_artifacts(
        screen_dir,
        output_dir / "screening",
        (
            "development_generation_responses.csv",
            "development_generation_summary.json",
            "development_generation_manifest.json",
        ),
    )
    copy_artifacts(
        training_dir,
        output_dir / "training",
        (
            "training_manifest.json",
            "training_loss_history.csv",
            "development_preference_metrics.csv",
            "development_preference_summary.json",
            "development_preference_gate.json",
            "01_training_loss_by_race.png",
            "02_development_preference_by_group.png",
            "03_margin_and_gradient_diagnostics.png",
        ),
    )

    manifest = read_json(training_dir / "training_manifest.json")
    write_json(
        output_dir / "adapter_inventory.json",
        {
            "diagnostic_only": True,
            "recommended_checkpoint": "step_0008",
            "checkpoint_artifacts": manifest["checkpoint_artifacts"],
            "final_adapter": manifest["final_adapter"],
            "final_adapter_hashes": manifest["final_adapter_hashes"],
        },
    )

    metric_table = "\n".join(
        (
            markdown_metric_row("Base", base_values),
            markdown_metric_row("Update 8", step8_values),
            markdown_metric_row("Update 16", step16_values),
        )
    )
    results_text = f"""# August 10 DPO/LoRA Mechanistic Diagnostic

**Decision: MECHANISTIC PASS; NOT A RESEARCH OR DEPLOYMENT CANDIDATE.**

The immutable base is `google/gemma-3-12b-it` revision
`96b6f1eccf38110c56df3a15bffe176da04bfd80`. This run trained a fresh rank-8
LoRA only. No prior adapter or base-weight update was used. No Claude or ChatGPT
API calls were used.

## Why this diagnostic was needed

The first 12-image neutral panel produced zero negative terms for either race,
so it could not measure a reduction. A separate screen of 80 prior training
images found 15 negative terms for Black images and 15 for White images. Six
matched blocks containing that signal were frozen into this explicitly
non-confirmatory exact-image canary.

## Main result

| Condition | Black negative rate | White negative rate | Gap | Valid | Entropy | Unique |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{metric_table}

Update 8 proves that DPO plus LoRA can causally change the sampled word metric:
both race-group negative rates fell from 16.67% to 0%, the gap stayed at zero,
validity remained 100%, entropy retention was
{step8_values['response_entropy'] / base_values['response_entropy']:.1%}, and
unique-response retention was
{step8_values['unique_response_rate'] / base_values['unique_response_rate']:.1%}.

Update 16 added no metric benefit and collapsed the unique-response rate to
{step16_values['unique_response_rate']:.1%}. Later checkpoints were not opened
for generation after this early-stop failure.

## Accuracy warning

Base negative-term rates were
{expressions['base']['negative']['negative_rate']:.1%} on negative-expression
images, {expressions['base']['neutral']['negative_rate']:.1%} on neutral images,
and {expressions['base']['positive']['negative_rate']:.1%} on positive images.
Update 8 drove every expression class to 0%. Therefore, the current all-positive
preference construction removes expression sensitivity as well as unjustified
negativity. It must not be described as a successfully debiased adapter.

## Next experiment

Use expression-conditioned preference pairs. Neutral images should prefer
low-negativity, image-grounded outputs. Negative-expression images should retain
calibrated terms such as sadness when visible while rejecting unrelated or
extreme terms such as evil, hate, ugly, or criminality. Evaluate on separate
images, preserve the exact fixed base responses, and promote only checkpoints
that reduce neutral unjustified negativity without erasing expression signal or
response diversity.

The 419-image protected evaluation manifest remained closed and disjoint.
"""
    (output_dir / "RESULTS.md").write_text(results_text, encoding="utf-8")

    checksum_path = output_dir / "ARTIFACT_SHA256SUMS"
    files = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path != checksum_path
    )
    checksum_path.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output_dir)}\n" for path in files),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "decision": summary["decision"],
                "mechanistic_pass": summary["mechanistic_pass"],
                "expression_sensitivity_pass": summary["expression_sensitivity_pass"],
                "plots": plots,
                "artifacts": len(files),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
