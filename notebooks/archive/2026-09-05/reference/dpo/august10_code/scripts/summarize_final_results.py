#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.result_summary import build_final_summary
from august3_debiasing.reference_integrity import sha256_file


CURATED_ARTIFACTS = {
    "training": (
        "training_manifest.json",
        "training_loss_history.csv",
        "development_preference_metrics.csv",
        "development_preference_summary.json",
        "development_preference_gate.json",
        "01_training_loss_by_race.png",
        "02_development_preference_by_group.png",
        "03_margin_and_gradient_diagnostics.png",
    ),
    "development_generation": (
        "development_generation_manifest.json",
        "development_generation_responses.csv",
        "development_generation_summary.json",
        "development_generation_gate.json",
        "01_development_iat_negative_rates.png",
        "02_development_profile_safeguards.png",
        "03_development_iat_bootstrap.png",
    ),
    "iat_evaluation": (
        "evaluation_manifest.json",
        "adapter_fairface_responses.csv",
        "adapter_neutral_responses.csv",
        "base_fairface_metrics.json",
        "adapter_fairface_metrics.json",
        "base_neutral_metrics.json",
        "adapter_neutral_metrics.json",
        "fairface_paired_bootstrap.json",
        "neutral_paired_bootstrap.json",
        "acceptance_criteria.json",
        "word_counts.csv",
        "base_vs_adapter_summary.csv",
        "01_base_vs_adapter_negative_rates.png",
        "02_negative_word_distribution.png",
        "03_expression_stratified_signed_gaps.png",
        "04_quality_safeguards.png",
        "05_paired_bootstrap_forest.png",
        "06_intersectional_negative_rates.png",
    ),
    "profile_evaluation": (
        "profile_evaluation_manifest.json",
        "profile_image_level_results.csv",
        "profile_quality_summary.json",
        "profile_paired_bootstrap.json",
        "profile_acceptance_criteria.json",
        "profile_preference_counts.csv",
        "01_profile_quality_by_race.png",
        "02_profile_harms_by_race.png",
        "03_profile_preference_outcomes.png",
        "04_profile_paired_bootstrap.png",
        "05_profile_mechanical_safeguards.png",
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    iat_rows = [
        row
        for row in metrics
        if row["family"] == "fairface_iat" and row["metric"] == "negative_rate"
    ]
    profile_quality = next(
        row
        for row in metrics
        if row["family"] == "grounded_profile"
        and row["group"] == "overall"
        and row["metric"] == "quality_score"
    )
    lines = [
        "# August 3 Final Results",
        "",
        f"Overall prespecified acceptance: **{'PASS' if summary['pass'] else 'FAIL'}**",
        "",
        "## Primary IAT endpoint",
        "",
        "| Race | Frozen base | August 3 LoRA | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in iat_rows:
        lines.append(
            f"| {row['group']} | {row['base']:.4%} | {row['adapter']:.4%} | "
            f"{row['adapter_minus_base']:+.4%} |"
        )
    lines.extend(
        [
            "",
            "## Profile safeguard",
            "",
            f"Mean Claude quality score changed from {profile_quality['base']:.3f} to "
            f"{profile_quality['adapter']:.3f} "
            f"({profile_quality['adapter_minus_base']:+.3f}).",
            "",
            "## Prespecified checks",
            "",
        ]
    )
    for name, passed in summary["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines.extend(
        [
            "",
            "This report is generated from hashed training, IAT-evaluation, and profile-evaluation "
            "manifests. See `final_research_summary.json` and `key_metrics.csv` for complete values.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_dashboard(output_dir: Path, summary: dict[str, Any], metrics: list[dict[str, Any]]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"base": "#5B6770", "adapter": "#0B6E75"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180)
    iat = [
        row
        for row in metrics
        if row["family"] == "fairface_iat" and row["metric"] == "negative_rate"
    ]
    x = np.arange(len(iat))
    width = 0.36
    axes[0, 0].bar(x - width / 2, [row["base"] for row in iat], width, label="Base", color=colors["base"])
    axes[0, 0].bar(x + width / 2, [row["adapter"] for row in iat], width, label="LoRA", color=colors["adapter"])
    axes[0, 0].set_xticks(x, [row["group"] for row in iat])
    axes[0, 0].set_ylabel("Negative selected-term rate")
    axes[0, 0].set_title("Frozen FairFace IAT endpoint")
    axes[0, 0].legend()

    profile_metrics = ("quality_score", "grounded_detail_count", "visible_expression_support")
    profile = [
        next(
            row
            for row in metrics
            if row["family"] == "grounded_profile"
            and row["group"] == "overall"
            and row["metric"] == name
        )
        for name in profile_metrics
    ]
    x = np.arange(len(profile))
    axes[0, 1].bar(x - width / 2, [row["base"] for row in profile], width, label="Base", color=colors["base"])
    axes[0, 1].bar(x + width / 2, [row["adapter"] for row in profile], width, label="LoRA", color=colors["adapter"])
    axes[0, 1].set_xticks(x, [name.replace("_", " ").title() for name in profile_metrics], rotation=20, ha="right")
    axes[0, 1].set_title("Grounded-profile quality")
    axes[0, 1].legend()

    harmful_names = ("unwarranted_negative_count", "unsupported_trait_count", "invented_biography_count", "identity_leakage_count")
    harms = [
        next(
            row
            for row in metrics
            if row["family"] == "grounded_profile"
            and row["group"] == "overall"
            and row["metric"] == name
        )
        for name in harmful_names
    ]
    x = np.arange(len(harms))
    axes[1, 0].bar(x - width / 2, [row["base"] for row in harms], width, label="Base", color=colors["base"])
    axes[1, 0].bar(x + width / 2, [row["adapter"] for row in harms], width, label="LoRA", color=colors["adapter"])
    axes[1, 0].set_xticks(x, [name.replace("_count", "").replace("_", " ").title() for name in harmful_names], rotation=20, ha="right")
    axes[1, 0].set_ylabel("Mean count per response")
    axes[1, 0].set_title("Unsupported-inference safeguards")
    axes[1, 0].legend()

    check_names = list(summary["checks"])
    check_values = [1 if summary["checks"][name] else 0 for name in check_names]
    y = np.arange(len(check_names))
    axes[1, 1].barh(
        y,
        check_values,
        color=["#2E7D32" if value else "#B23A48" for value in check_values],
    )
    axes[1, 1].set_yticks(y, [name.replace("_", " ") for name in check_names], fontsize=8)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_xticks([0, 1], ["Fail", "Pass"])
    axes[1, 1].set_title("Prespecified final checks")
    fig.suptitle("Gemma 3 FairFace Debiasing - August 3")
    fig.tight_layout()
    path = output_dir / "00_august3_results_dashboard.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def copy_curated_artifacts(
    output_dir: Path,
    source_dirs: dict[str, Path],
) -> list[dict[str, str]]:
    copied = []
    for family, filenames in CURATED_ARTIFACTS.items():
        source_dir = source_dirs[family].resolve()
        destination_dir = output_dir / family
        destination_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            source = source_dir / filename
            if not source.is_file():
                raise FileNotFoundError(f"required public result artifact is missing: {source}")
            destination = destination_dir / filename
            shutil.copy2(source, destination)
            source_hash = sha256_file(source)
            if sha256_file(destination) != source_hash:
                raise RuntimeError(f"copied artifact hash mismatch: {destination}")
            copied.append(
                {
                    "family": family,
                    "path": str(destination.relative_to(output_dir)),
                    "sha256": source_hash,
                }
            )
    return copied


def write_checksum_ledger(output_dir: Path) -> Path:
    ledger = output_dir / "ARTIFACT_SHA256SUMS"
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path != ledger
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(output_dir)}"
        for path in paths
    ]
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and combine final August 3 training, IAT, and profile results."
    )
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--development-generation-dir", required=True, type=Path)
    parser.add_argument("--iat-evaluation-dir", required=True, type=Path)
    parser.add_argument("--profile-evaluation-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training = read_json(args.training_dir / "training_manifest.json")
    development_generation_manifest = read_json(
        args.development_generation_dir / "development_generation_manifest.json"
    )
    development_generation_gate = read_json(
        args.development_generation_dir / "development_generation_gate.json"
    )
    iat_manifest = read_json(args.iat_evaluation_dir / "evaluation_manifest.json")
    profile_manifest = read_json(
        args.profile_evaluation_dir / "profile_evaluation_manifest.json"
    )
    summary, metrics = build_final_summary(
        training_manifest=training,
        development_generation_manifest=development_generation_manifest,
        development_generation_gate=development_generation_gate,
        iat_manifest=iat_manifest,
        profile_manifest=profile_manifest,
        base_iat=read_json(args.iat_evaluation_dir / "base_fairface_metrics.json"),
        adapter_iat=read_json(args.iat_evaluation_dir / "adapter_fairface_metrics.json"),
        base_neutral=read_json(args.iat_evaluation_dir / "base_neutral_metrics.json"),
        adapter_neutral=read_json(args.iat_evaluation_dir / "adapter_neutral_metrics.json"),
        profile_summary=read_json(args.profile_evaluation_dir / "profile_quality_summary.json"),
        iat_acceptance=read_json(args.iat_evaluation_dir / "acceptance_criteria.json"),
        profile_acceptance=read_json(
            args.profile_evaluation_dir / "profile_acceptance_criteria.json"
        ),
    )
    write_csv(args.output_dir / "key_metrics.csv", metrics)
    dashboard = plot_dashboard(args.output_dir, summary, metrics)
    summary["dashboard"] = dashboard
    write_json(args.output_dir / "final_research_summary.json", summary)
    write_markdown(args.output_dir / "FINAL_RESULTS.md", summary, metrics)
    copied = copy_curated_artifacts(
        args.output_dir,
        {
            "training": args.training_dir,
            "development_generation": args.development_generation_dir,
            "iat_evaluation": args.iat_evaluation_dir,
            "profile_evaluation": args.profile_evaluation_dir,
        },
    )
    write_json(
        args.output_dir / "research_bundle_manifest.json",
        {
            "status": "complete",
            "experiment_revision": "2026-08-03",
            "public_bundle": True,
            "excluded": [
                "face images",
                "LoRA weights",
                "provider checkpoint JSONL",
                "credentials",
            ],
            "artifacts": copied,
        },
    )
    ledger = write_checksum_ledger(args.output_dir)
    summary["artifact_checksum_ledger"] = str(ledger)
    write_json(args.output_dir / "final_research_summary.json", summary)
    write_checksum_ledger(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
