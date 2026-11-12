#!/usr/bin/env python3

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

from august3_debiasing.constants import NEGATIVE_IAT_SET  # noqa: E402, I001


TRAINING_RUNS = (
    {
        "key": "sum_lr2e-6",
        "label": "Sum, LR 2e-6",
        "directory": "training_v4_rank8_seed8102026",
    },
    {
        "key": "sum_lr5e-6",
        "label": "Sum, LR 5e-6",
        "directory": "training_v4b_lr5e6_rank8_seed8102026",
    },
    {
        "key": "sum_lr1e-5",
        "label": "Sum, LR 1e-5",
        "directory": "training_v4c_lr1e5_rank8_seed8102026",
    },
    {
        "key": "mean_lr5e-6",
        "label": "Mean, LR 5e-6",
        "directory": "training_v4d_meanlogp_lr5e6_rank8_seed8102026",
    },
    {
        "key": "mean_lr2e-5",
        "label": "Mean, LR 2e-5",
        "directory": "training_v4e_meanlogp_lr2e5_rank8_seed8102026",
    },
)

CHECKPOINTS = (
    ("sum_lr2e-6_s1", "Sum 2e-6, step 1", "sum_lr2e-6", "development_checkpoint_step0001_v4"),
    ("sum_lr2e-6_s5", "Sum 2e-6, step 5", "sum_lr2e-6", "development_checkpoint_step0005_v4"),
    ("sum_lr2e-6_s10", "Sum 2e-6, step 10", "sum_lr2e-6", "development_checkpoint_step0010_v4"),
    (
        "sum_lr5e-6_s5",
        "Sum 5e-6, step 5",
        "sum_lr5e-6",
        "development_checkpoint_v4b_lr5e6_step0005",
    ),
    (
        "sum_lr5e-6_s10",
        "Sum 5e-6, step 10",
        "sum_lr5e-6",
        "development_checkpoint_v4b_lr5e6_step0010",
    ),
    (
        "sum_lr1e-5_s10",
        "Sum 1e-5, step 10",
        "sum_lr1e-5",
        "development_checkpoint_v4c_lr1e5_step0010",
    ),
    (
        "mean_lr5e-6_s10",
        "Mean 5e-6, step 10",
        "mean_lr5e-6",
        "development_checkpoint_v4d_meanlogp_lr5e6_step0010",
    ),
    (
        "mean_lr2e-5_s10",
        "Mean 2e-5, step 10",
        "mean_lr2e-5",
        "development_checkpoint_v4e_meanlogp_lr2e5_step0010",
    ),
)

TRAINING_FILES = (
    "training_manifest.json",
    "training_loss_history.csv",
    "development_preference_metrics.csv",
    "development_preference_summary.json",
    "development_preference_gate.json",
    "01_training_loss_by_race.png",
    "02_development_preference_by_group.png",
    "03_margin_and_gradient_diagnostics.png",
)

CHECKPOINT_FILES = (
    "development_generation_manifest.json",
    "development_generation_responses.csv",
    "development_generation_summary.json",
    "development_generation_gate.json",
    "checkpoint_gate_v2.json",
    "01_development_iat_negative_rates.png",
    "02_development_profile_safeguards.png",
    "03_development_iat_bootstrap.png",
)

BASE_FILES = (
    "development_generation_manifest.json",
    "development_generation_responses.csv",
    "development_generation_summary.json",
    "development_generation_gate.json",
    "01_base_development_iat_negative_rates.png",
    "02_base_development_profile_safeguards.png",
)

DATASET_FILES = (
    "mixed_dpo_pairs_frozen.csv",
    "selected_profile_matches.csv",
    "all_claude_eligible_pairs.csv",
    "dataset_gate.json",
    "freeze_manifest.json",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overall_iat_rate(iat: dict[str, Any]) -> float:
    races = iat["race_metrics"]
    negative = sum(int(races[race]["negative_terms"]) for race in ("Black", "White"))
    selected = sum(int(races[race]["selected_terms"]) for race in ("Black", "White"))
    return negative / selected


def ratio(adapter: float, base: float) -> float:
    return adapter / base if base else float(adapter == 0)


def checkpoint_row(
    key: str,
    label: str,
    run_key: str,
    directory: Path,
) -> dict[str, Any]:
    summary = read_json(directory / "development_generation_summary.json")
    manifest = read_json(directory / "development_generation_manifest.json")
    gate = read_json(directory / "checkpoint_gate_v2.json")
    base = summary["base_iat"]
    adapter = summary["adapter_iat"]
    base_profiles = summary["base_profiles"]["overall"]
    adapter_profiles = summary["adapter_profiles"]["overall"]
    return {
        "checkpoint_key": key,
        "label": label,
        "training_run": run_key,
        "base_black_negative_rate": base["race_metrics"]["Black"]["negative_rate"],
        "adapter_black_negative_rate": adapter["race_metrics"]["Black"]["negative_rate"],
        "black_change": adapter["race_metrics"]["Black"]["negative_rate"]
        - base["race_metrics"]["Black"]["negative_rate"],
        "base_white_negative_rate": base["race_metrics"]["White"]["negative_rate"],
        "adapter_white_negative_rate": adapter["race_metrics"]["White"]["negative_rate"],
        "white_change": adapter["race_metrics"]["White"]["negative_rate"]
        - base["race_metrics"]["White"]["negative_rate"],
        "base_overall_negative_rate": overall_iat_rate(base),
        "adapter_overall_negative_rate": overall_iat_rate(adapter),
        "overall_change": overall_iat_rate(adapter) - overall_iat_rate(base),
        "base_absolute_race_gap": base["negative_rate_gap_abs"],
        "adapter_absolute_race_gap": adapter["negative_rate_gap_abs"],
        "absolute_gap_change": adapter["negative_rate_gap_abs"]
        - base["negative_rate_gap_abs"],
        "iat_valid_rate": adapter["valid_rate"],
        "iat_identity_leakage_rate": adapter["identity_leakage_rate"],
        "iat_entropy_retention": ratio(adapter["response_entropy"], base["response_entropy"]),
        "iat_unique_response_retention": ratio(
            adapter["unique_response_rate"], base["unique_response_rate"]
        ),
        "profile_distinct_bigram_retention": ratio(
            adapter_profiles["distinct_bigram_rate"], base_profiles["distinct_bigram_rate"]
        ),
        "profile_unique_response_retention": ratio(
            adapter_profiles["unique_response_rate"], base_profiles["unique_response_rate"]
        ),
        "gate_pass": gate["pass"],
        "gate_decision": gate["decision"],
        "adapter_sha256": manifest["adapter_hashes"]["adapter_model.safetensors"],
    }


def training_row(run: dict[str, str], directory: Path) -> dict[str, Any]:
    manifest = read_json(directory / "training_manifest.json")
    preference = manifest["development_preference_summary"]
    groups = preference["groups"]
    return {
        "training_run": run["key"],
        "label": run["label"],
        "learning_rate": manifest["hyperparameters"]["learning_rate"],
        "sequence_logp_reduction": manifest["hyperparameters"].get(
            "sequence_logp_reduction", "sum"
        ),
        "optimizer_updates": manifest["optimizer_updates"],
        "development_preference_accuracy": preference["preference_accuracy"],
        "development_mean_dpo_loss": preference["mean_dpo_loss"],
        "profile_black_margin": groups["grounded_profile/Black"]["mean_margin_delta"],
        "profile_white_margin": groups["grounded_profile/White"]["mean_margin_delta"],
        "iat_black_margin": groups["iat_3word/Black"]["mean_margin_delta"],
        "iat_white_margin": groups["iat_3word/White"]["mean_margin_delta"],
        "development_preference_gate_pass": manifest["development_preference_gate"]["pass"],
        "peak_reserved_gib": manifest["gpu_memory"]["peak_reserved_gib"],
        "wall_seconds": manifest["wall_seconds"],
        "final_adapter_sha256": manifest["final_adapter_hashes"][
            "adapter_model.safetensors"
        ],
    }


def aggregate_claude_usage(early: dict[str, Any], expansion: dict[str, Any]) -> dict[str, Any]:
    models = sorted(set(early["by_model"]) | set(expansion["by_model"]))
    by_model = {}
    for model in models:
        values = [source.get("by_model", {}).get(model, {}) for source in (early, expansion)]
        by_model[model] = {
            "calls": sum(int(value.get("calls", 0)) for value in values),
            "input_tokens": sum(int(value.get("input_tokens", 0)) for value in values),
            "output_tokens": sum(int(value.get("output_tokens", 0)) for value in values),
            "estimated_cost_usd": sum(
                float(value.get("estimated_cost_usd", 0.0)) for value in values
            ),
        }
    return {
        "protocol_version": "august10-public-claude-usage-v1",
        "by_model": by_model,
        "cumulative_august10": {
            "calls": early["calls"] + expansion["calls"],
            "input_tokens": early["input_tokens"] + expansion["input_tokens"],
            "output_tokens": early["output_tokens"] + expansion["output_tokens"],
            "estimated_cost_usd": early["estimated_cost_usd"]
            + expansion["estimated_cost_usd"],
        },
        "negative_expression_expansion_session": {
            "calls": expansion["calls"],
            "input_tokens": expansion["input_tokens"],
            "output_tokens": expansion["output_tokens"],
            "estimated_cost_usd": expansion["estimated_cost_usd"],
            "authorized_budget_usd": expansion.get("starting_balance_usd"),
            "estimated_remaining_usd": expansion.get("estimated_remaining_usd"),
        },
        "balance_caveat": expansion["balance_caveat"],
        "pricing_source": expansion["pricing_source"],
        "pricing_verified_date": expansion["pricing_verified_date"],
    }


def copy_file(
    source: Path,
    destination: Path,
    records: list[dict[str, str]],
    output_root: Path,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = sha256_file(source)
    if sha256_file(destination) != source_hash:
        raise RuntimeError(f"artifact copy hash mismatch: {destination}")
    records.append(
        {"path": str(destination.relative_to(output_root)), "sha256": source_hash}
    )


def plot_dashboard(
    output_dir: Path,
    checkpoints: list[dict[str, Any]],
    training: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"Black": "#0B6E75", "White": "#5B6770", "accent": "#B23A48"}
    labels = ["Base", *[row["label"] for row in checkpoints]]
    black = [checkpoints[0]["base_black_negative_rate"]] + [
        row["adapter_black_negative_rate"] for row in checkpoints
    ]
    white = [checkpoints[0]["base_white_negative_rate"]] + [
        row["adapter_white_negative_rate"] for row in checkpoints
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), dpi=180)
    axes[0, 0].bar(x - width / 2, black, width, label="Black", color=colors["Black"])
    axes[0, 0].bar(x + width / 2, white, width, label="White", color=colors["White"])
    axes[0, 0].set_xticks(x, labels, rotation=32, ha="right")
    axes[0, 0].set_ylabel("Negative selected-term rate")
    axes[0, 0].set_title("Development IAT rates")
    axes[0, 0].legend()

    changes = (
        [row["black_change"] * 100 for row in checkpoints],
        [row["overall_change"] * 100 for row in checkpoints],
        [row["absolute_gap_change"] * 100 for row in checkpoints],
    )
    change_labels = ["Black rate", "Overall rate", "Absolute gap"]
    x = np.arange(len(checkpoints))
    width = 0.24
    for offset, values, name, color in zip(
        (-width, 0, width),
        changes,
        change_labels,
        (colors["Black"], "#C99720", colors["accent"]),
        strict=True,
    ):
        axes[0, 1].bar(x + offset, values, width, label=name, color=color)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set_xticks(x, [row["label"] for row in checkpoints], rotation=32, ha="right")
    axes[0, 1].set_ylabel("Change from base (percentage points)")
    axes[0, 1].set_ylim(-0.25, 0.25)
    axes[0, 1].set_title("Primary metric movement")
    axes[0, 1].legend()
    if not any(abs(value) > 1e-12 for series in changes for value in series):
        axes[0, 1].text(
            0.5,
            0.64,
            "All evaluated changes = 0.00 pp",
            transform=axes[0, 1].transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "#5B6770", "pad": 5},
        )

    for metric, label, color in (
        ("iat_entropy_retention", "IAT entropy", "#0B6E75"),
        ("iat_unique_response_retention", "IAT unique responses", "#C99720"),
        ("profile_distinct_bigram_retention", "Profile bigrams", "#B23A48"),
    ):
        axes[1, 0].plot(
            x,
            [row[metric] for row in checkpoints],
            marker="o",
            label=label,
            color=color,
        )
    axes[1, 0].axhline(0.9, color="black", linewidth=0.8, linestyle="--")
    axes[1, 0].set_xticks(x, [row["label"] for row in checkpoints], rotation=32, ha="right")
    axes[1, 0].set_ylabel("Retention relative to base")
    axes[1, 0].set_title("Diversity safeguards")
    axes[1, 0].legend()

    x = np.arange(len(training))
    axes[1, 1].bar(
        x,
        [row["development_preference_accuracy"] for row in training],
        color=[
            "#0B6E75" if row["development_preference_gate_pass"] else "#B23A48"
            for row in training
        ],
    )
    axes[1, 1].axhline(0.55, color="black", linewidth=0.8, linestyle="--")
    axes[1, 1].set_xticks(x, [row["label"] for row in training], rotation=25, ha="right")
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_ylabel("Held-out preference accuracy")
    axes[1, 1].set_title("DPO fit does not imply generation improvement")

    fig.suptitle("Gemma 3 August 10 Development Pilot", fontsize=16)
    fig.text(
        0.5,
        0.005,
        "Development-only screen. No checkpoint was promoted; confirmatory panels remain closed.",
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(output_dir / "00_august10_pilot_dashboard.png", bbox_inches="tight")
    plt.close(fig)


def plot_dataset(output_dir: Path, gate: dict[str, Any], usage: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = gate["metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    categories = ("Black", "White")
    axes[0].bar(
        categories,
        [metrics["rows_by_race"][key] for key in categories],
        color=("#0B6E75", "#5B6770"),
    )
    axes[0].set_title("Rows by FairFace race")
    axes[0].set_ylabel("Rows")
    axes[0].set_ylim(0, 60)

    categories = ("train", "development")
    axes[1].bar(
        categories,
        [metrics["rows_by_split"][key] for key in categories],
        color=("#0B6E75", "#C99720"),
    )
    axes[1].set_title("Frozen split")
    axes[1].set_ylabel("Rows")

    models = list(usage["by_model"])
    x = np.arange(len(models))
    axes[2].bar(
        x,
        [usage["by_model"][model]["estimated_cost_usd"] for model in models],
        color=("#5B6770", "#B23A48"),
    )
    axes[2].set_xticks(x, [model.replace("claude-", "") for model in models])
    axes[2].set_ylabel("Estimated cost (USD)")
    axes[2].set_title("Cumulative Claude audit cost")
    total = usage["cumulative_august10"]["estimated_cost_usd"]
    axes[2].text(
        0.5,
        0.95,
        f"Total: ${total:.2f}",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
    )
    fig.suptitle("Dataset and Judge Audit")
    fig.tight_layout()
    fig.savefig(output_dir / "01_dataset_and_judge_audit.png", bbox_inches="tight")
    plt.close(fig)


def plot_negative_words(
    output_dir: Path,
    checkpoint_summaries: dict[str, dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    panels = (
        ("Base", next(iter(checkpoint_summaries.values()))["base_iat"]),
        ("Sum 2e-6, step 10", checkpoint_summaries["sum_lr2e-6_s10"]["adapter_iat"]),
        ("Sum 1e-5, step 10", checkpoint_summaries["sum_lr1e-5_s10"]["adapter_iat"]),
        ("Mean 2e-5, step 10", checkpoint_summaries["mean_lr2e-5_s10"]["adapter_iat"]),
    )
    words = sorted(
        {
            word
            for _, metrics in panels
            for race in ("Black", "White")
            for word, count in metrics["word_counts"][race].items()
            if word in NEGATIVE_IAT_SET and count
        }
    )
    x = np.arange(len(words))
    width = 0.38
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=180, sharey=True)
    for axis, (label, metrics) in zip(axes.flat, panels, strict=True):
        axis.bar(
            x - width / 2,
            [metrics["word_counts"]["Black"].get(word, 0) for word in words],
            width,
            label="Black",
            color="#0B6E75",
        )
        axis.bar(
            x + width / 2,
            [metrics["word_counts"]["White"].get(word, 0) for word in words],
            width,
            label="White",
            color="#5B6770",
        )
        axis.set_xticks(x, words, rotation=35, ha="right")
        axis.set_title(label)
        axis.set_ylabel("Selections")
        axis.legend()
    fig.suptitle("Development IAT Negative-Word Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "02_negative_word_distribution.png", bbox_inches="tight")
    plt.close(fig)


def write_results_markdown(
    path: Path,
    dataset_gate: dict[str, Any],
    future_gate: dict[str, Any],
    usage: dict[str, Any],
    checkpoints: list[dict[str, Any]],
) -> None:
    lines = [
        "# August 10 Negative-Expression Pilot Results",
        "",
        "**Decision: STOP. No checkpoint is eligible for seed replication or "
        "confirmatory evaluation.**",
        "",
        "The immutable base is `google/gemma-3-12b-it` revision "
        "`96b6f1eccf38110c56df3a15bffe176da04bfd80`. Every candidate started as a "
        "fresh rank-8 LoRA; no base weights or prior adapters were updated.",
        "",
        "## Frozen data",
        "",
        f"- Historical v1 dataset gate: **{'PASS' if dataset_gate['pass'] else 'FAIL'}**.",
        "- 100 rows, 50 Black and 50 White, 80 train and 20 development.",
        "- 70 grounded profiles and 30 controlled three-word IAT rows.",
        "- Seven negative-expression blocks: five development and two train.",
        "- Post-run v2 gate: **FAIL** because training contains zero negative-expression "
        "IAT blocks; "
        "the new minimum is three.",
        f"- v2 failed checks: `{', '.join(future_gate['issues'])}`.",
        "",
        "## Development results",
        "",
        "The negative-expression challenge base selected 12/45 negative terms for Black images "
        "(26.67%) and 15/45 for White images (33.33%), an absolute gap of 6.67 points. "
        "This challenge is not the neutral 12-face confirmatory benchmark.",
        "",
        "| Checkpoint | Black | White | Overall | Abs. gap | Leakage | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in checkpoints:
        lines.append(
            f"| {row['label']} | {row['adapter_black_negative_rate']:.2%} | "
            f"{row['adapter_white_negative_rate']:.2%} | "
            f"{row['adapter_overall_negative_rate']:.2%} | "
            f"{row['adapter_absolute_race_gap']:.2%} | "
            f"{row['iat_identity_leakage_rate']:.2%} | `{row['gate_decision']}` |"
        )
    session = usage["negative_expression_expansion_session"]
    cumulative = usage["cumulative_august10"]
    lines.extend(
        [
            "",
            "Every evaluated checkpoint left Black, White, overall, and absolute-gap rates "
            "unchanged. The summed `1e-5` checkpoint also introduced one identity-leakage "
            "response (3.33%). Profile validity, subject reference, uniqueness, and bigram "
            "diversity otherwise passed their safeguards.",
            "",
            "Mean-token DPO removed the extreme profile-versus-IAT margin imbalance, but it did "
            "not change the primary generation rates. This narrows the next bottleneck to training "
            "coverage and task weighting rather than another learning-rate sweep.",
            "",
            "## Claude audit",
            "",
            f"The negative-expression expansion used {session['calls']} calls and an estimated "
            f"${session['estimated_cost_usd']:.6f} of the authorized $10 session budget. "
            f"Cumulative August 10 auditing used {cumulative['calls']} calls and an estimated "
            f"${cumulative['estimated_cost_usd']:.6f}. Provider account balance remains "
            "an estimate.",
            "",
            "## Next iteration",
            "",
            "1. Add at least six negative-expression IAT blocks so at least three can be frozen "
            "into train and three into development without overlap.",
            "2. Keep mean-token scoring available, but prespecify a task-balanced DPO weight or "
            "matched accumulation schedule before training.",
            "3. Re-run the base development challenge only if its image panel changes; otherwise "
            "reuse the hashed base responses.",
            "4. Require strict primary movement before three-seed replication. Keep the 400-image "
            "FairFace and 12-face neutral confirmatory panels closed until replication passes.",
            "",
            "See `checkpoint_metrics.csv`, `training_run_metrics.csv`, and "
            "`research_summary.json` for machine-readable values.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def checksum_ledger(output_dir: Path) -> None:
    ledger = output_dir / "ARTIFACT_SHA256SUMS"
    files = sorted(path for path in output_dir.rglob("*") if path.is_file() and path != ledger)
    ledger.write_text(
        "".join(f"{sha256_file(path)}  {path.relative_to(output_dir)}\n" for path in files),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile publication-safe August 10 pilot results."
    )
    parser.add_argument("--outputs-root", type=Path, default=ROOT_DIR / "outputs" / "august10")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "august10_negative_expression_pilot_v4",
    )
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    outputs_root = args.outputs_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.replace:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = outputs_root / "frozen_pilot_v4_negative_expression"
    dataset_gate = read_json(dataset_dir / "dataset_gate.json")
    future_gate = read_json(outputs_root / "postrun_dataset_gate_v2.json")
    early_usage = read_json(outputs_root / "claude_usage_cost_only.json")
    expansion_usage = read_json(
        outputs_root / "negative_expression_expansion_v1" / "claude_session_usage_final_v2.json"
    )
    usage = aggregate_claude_usage(early_usage, expansion_usage)

    training_dirs = {
        run["key"]: outputs_root / run["directory"] for run in TRAINING_RUNS
    }
    training_rows = [training_row(run, training_dirs[run["key"]]) for run in TRAINING_RUNS]
    checkpoint_rows = []
    checkpoint_summaries = {}
    for key, label, run_key, directory_name in CHECKPOINTS:
        directory = outputs_root / directory_name
        checkpoint_rows.append(checkpoint_row(key, label, run_key, directory))
        checkpoint_summaries[key] = read_json(directory / "development_generation_summary.json")

    base_signatures = {
        (
            row["base_black_negative_rate"],
            row["base_white_negative_rate"],
            row["base_overall_negative_rate"],
            row["base_absolute_race_gap"],
        )
        for row in checkpoint_rows
    }
    if len(base_signatures) != 1:
        raise RuntimeError("checkpoint screens do not share one immutable base signature")

    write_csv(output_dir / "training_run_metrics.csv", training_rows)
    write_csv(output_dir / "checkpoint_metrics.csv", checkpoint_rows)
    write_json(output_dir / "claude_usage_summary.json", usage)
    write_json(output_dir / "postrun_dataset_gate_v2.json", future_gate)

    copied: list[dict[str, str]] = []
    for filename in DATASET_FILES:
        copy_file(
            dataset_dir / filename,
            output_dir / "dataset" / filename,
            copied,
            output_dir,
        )
    base_dir = outputs_root / "base_development_challenge_v4"
    for filename in BASE_FILES:
        copy_file(
            base_dir / filename,
            output_dir / "base_development" / filename,
            copied,
            output_dir,
        )
    for run in TRAINING_RUNS:
        source = training_dirs[run["key"]]
        for filename in TRAINING_FILES:
            copy_file(
                source / filename,
                output_dir / "training" / run["key"] / filename,
                copied,
                output_dir,
            )
    for key, _label, _run_key, directory_name in CHECKPOINTS:
        source = outputs_root / directory_name
        for filename in CHECKPOINT_FILES:
            copy_file(
                source / filename,
                output_dir / "checkpoints" / key / filename,
                copied,
                output_dir,
            )

    adapter_inventory = {
        row["training_run"]: {
            "sha256": row["final_adapter_sha256"],
            "local_ignored_path": str(
                training_dirs[row["training_run"]]
                / "checkpoints"
                / "final"
                / "adapter_model.safetensors"
            ),
            "tracked_in_git": False,
        }
        for row in training_rows
    }
    write_json(output_dir / "adapter_inventory.json", adapter_inventory)

    plot_dashboard(output_dir, checkpoint_rows, training_rows)
    plot_dataset(output_dir, dataset_gate, usage)
    plot_negative_words(output_dir, checkpoint_summaries)
    write_results_markdown(
        output_dir / "RESULTS.md", dataset_gate, future_gate, usage, checkpoint_rows
    )
    summary = {
        "protocol_version": "august10-negative-expression-pilot-results-v1",
        "decision": "stop_no_checkpoint_promoted",
        "confirmatory_evaluation_opened": False,
        "immutable_base": {
            "model_id": "google/gemma-3-12b-it",
            "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
        },
        "historical_dataset_gate_pass": dataset_gate["pass"],
        "postrun_dataset_gate_pass": future_gate["pass"],
        "postrun_dataset_gate_failed_checks": future_gate["issues"],
        "training_runs": len(training_rows),
        "evaluated_checkpoints": len(checkpoint_rows),
        "promoted_checkpoints": sum(bool(row["gate_pass"]) for row in checkpoint_rows),
        "base_signature": list(next(iter(base_signatures))),
        "claude_usage": usage,
        "copied_artifacts": copied,
    }
    write_json(output_dir / "research_summary.json", summary)
    checksum_ledger(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "decision": summary["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
