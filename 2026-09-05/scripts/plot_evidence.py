#!/usr/bin/env python3
"""Presentation figures from frozen/derived CSVs only; no model calls."""

import json
import textwrap

from common import ROOT, read_csv, write_csv
from analyze_evidence import NAMES, ORDER

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = ROOT / "results/analysis"
PLOTS = ROOT / "plots"
REGISTRY = []
COLORS = ("#3268A8", "#D48C35", "#38917A", "#B75778")
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
    }
)


def rows(name):
    return read_csv(DATA / name)


def ordered(table):
    return sorted(table, key=lambda r: ORDER.index(r["model_key"]))


def names(table):
    return [NAMES[r["model_key"]].replace(" (local pinned)", "\n(local pinned)") for r in table]


def save(fig, filename, title, finding, inputs):
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.035, 0.025, textwrap.fill(finding, 150), va="bottom", fontsize=10, linespacing=1.5)
    fig.tight_layout(rect=(0.015, 0.13, 0.985, 0.935))
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / filename, dpi=200, metadata={"Software": "research-evidence-september5"})
    fig.savefig(
        (PLOTS / filename).with_suffix(".pdf"),
        metadata={"Creator": "research-evidence-september5", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    REGISTRY.append(
        dict(
            plot_id=filename[:-4],
            title=title,
            finding=finding,
            plot_purpose=finding,
            plot_code_path=str(ROOT / "scripts/plot_evidence.py"),
            plot_input_data_paths="|".join(str(DATA / p) for p in inputs),
            plot_config_path=str(ROOT / "config/study.json"),
            plot_reproduction_command="python scripts/plot_evidence.py",
            plot_output_path=str(PLOTS / filename),
            plot_code_commit_sha="not committed; see SHA256SUMS",
        )
    )


def forest(ax, table, scale=100, label="Black minus White (percentage points)", annotate=True):
    table = ordered(table)
    y = np.arange(len(table))
    effects = np.array([float(r["effect"]) * scale for r in table])
    low = np.array([float(r["ci_low"]) * scale for r in table])
    high = np.array([float(r["ci_high"]) * scale for r in table])
    # A percentile range interval need not contain its plug-in estimate.
    ax.hlines(y, low, high, color=COLORS[0])
    ax.plot(low, y, "|", color=COLORS[0])
    ax.plot(high, y, "|", color=COLORS[0])
    ax.plot(effects, y, "o", color=COLORS[0])
    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.set_yticks(y, names(table), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(label)
    ax.grid(axis="x", alpha=0.2)
    if annotate:
        for i, r in enumerate(table):
            ax.text(
                1.02,
                i,
                f"p={float(r['raw_p_value']):.3f}  q={float(r['bh_adjusted_q_value']):.3f}",
                transform=ax.get_yaxis_transform(),
                va="center",
                fontsize=8,
            )
    return table


def iat(dataset, prefix):
    summary = ordered(rows(f"iat_{dataset}_summary.csv"))
    tests = ordered(rows(f"iat_{dataset}_significance.csv"))
    count = 400 if dataset == "utkface" else 12
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(summary))
    for offset, race, color in ((-0.19, "black", COLORS[0]), (0.19, "white", COLORS[1])):
        means = np.array([float(r[f"{race}_rate"]) * 100 for r in summary])
        low = np.array([float(r[f"{race}_ci_low"]) * 100 for r in summary])
        high = np.array([float(r[f"{race}_ci_high"]) * 100 for r in summary])
        bars = ax.bar(
            x + offset,
            means,
            0.36,
            label=race.title(),
            color=color,
            yerr=[np.maximum(0, means - low), np.maximum(0, high - means)],
            capsize=2,
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
    ax.set_xticks(x, names(summary), rotation=22, ha="right")
    ax.set_ylabel("Negative terms (%)")
    ax.set_ylim(
        0,
        min(
            110,
            max(max(float(r["black_ci_high"]), float(r["white_ci_high"])) for r in summary) * 100
            + 15,
        ),
    )
    ax.legend(frameon=False)
    gemma = next(r for r in summary if r["model_key"] == "gemma_local")
    finding = (
        f"Bars show the share of three selected words that belong to the negative IAT list; whiskers are image-cluster 95% intervals. "
        f"Gemma selects {100 * float(gemma['black_rate']):.1f}% negative words for Black images and {100 * float(gemma['white_rate']):.1f}% for White images. "
        + (
            "The archived Gemma baseline is retained; local and hosted serving paths differ."
            if count == 12
            else "UTKFace expressions are mixed and matching is imperfect; differences are not causal race effects."
        )
    )
    save(
        fig,
        f"{prefix}_iat_{dataset}_rates.png",
        f"IAT-style negative terms: {count} images",
        finding,
        [f"iat_{dataset}_summary.csv"],
    )
    fig, ax = plt.subplots(figsize=(14, 7))
    forest(ax, tests)
    fig.subplots_adjust(right=0.75)
    significant = sum(float(r["bh_adjusted_q_value"]) < 0.05 for r in tests)
    save(
        fig,
        f"{prefix}b_iat_{dataset}_significance.png",
        f"IAT-style race gap: {count} images",
        f"A positive value means more negative words for Black images. {significant} of {len(tests)} model tests have BH-adjusted q < 0.05; "
        f"these exploratory tests use {'200 matched image pairs, fewer if outputs are invalid' if count == 400 else 'only 12 image clusters, not 120 independent responses'}. "
        "Error bars are 95% intervals; q adjusts for the family of model tests.",
        [f"iat_{dataset}_significance.csv"],
    )


def diversity():
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for i, dataset in enumerate(("neutral12", "utkface")):
        table = ordered(rows(f"iat_{dataset}_summary.csv"))
        for j, (field, label) in enumerate(
            (
                ("validity_rate", "Valid three-word outputs (%)"),
                ("response_tuple_diversity_rate", "Unique triples / valid responses (%)"),
            )
        ):
            ax = axes[i, j]
            ax.bar(range(len(table)), [float(r[field]) * 100 for r in table], color=COLORS[j])
            ax.set_xticks(range(len(table)), names(table), rotation=30, ha="right", fontsize=8)
            ax.set_ylim(0, 105)
            ax.set_ylabel(label)
            ax.set_title("12 faces" if i == 0 else "400 UTKFace images")
    save(
        fig,
        "03_validity_diversity.png",
        "Output validity and response diversity",
        "Validity measures whether the parser recovered three vocabulary words, not whether a description is correct or fair. "
        "Diversity measures distinct ordered triples at the fixed sample size; low diversity can expose repetitive outputs, but neither metric establishes general model accuracy.",
        ["iat_neutral12_summary.csv", "iat_utkface_summary.csv"],
    )


def text_plots():
    categories = rows("text_category_summary.csv")
    tests = rows("text_significance.csv")
    dt = rows("decodingtrust_responses.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharey=True)
    for ax, regime in zip(axes, ("benign", "untargeted", "targeted"), strict=True):
        selected = [r for r in dt if r["regime"] == regime]
        keys = [k for k in ORDER if any(r["model_key"] == k for r in selected)]
        left = np.zeros(len(keys))
        for cls, color in zip(
            ("AGREE", "DISAGREE", "NO_RESPONSE"), (COLORS[3], COLORS[2], "#B5BBC1"), strict=True
        ):
            values = np.array(
                [
                    100
                    * sum(r["classification"] == cls for r in selected if r["model_key"] == k)
                    / len([r for r in selected if r["model_key"] == k])
                    for k in keys
                ]
            )
            ax.barh(
                np.arange(len(keys)),
                values,
                left=left,
                color=color,
                label=cls.replace("NO_RESPONSE", "Unclassified"),
            )
            left += values
        ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys], fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Responses (%)")
        ax.set_title(regime.title())
    axes[0].invert_yaxis()
    axes[-1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, -0.20), ncol=3)
    n = sum(r["classification"] == "AGREE" for r in dt)
    save(
        fig,
        "04_decodingtrust_agreement.png",
        "DecodingTrust: responses to racial stereotypes",
        f"Bars show agreement, disagreement and unclassified responses under three instruction regimes (16 prompts per model per regime). "
        f"The phrase classifier found {n} agreement responses across {len(dt)} outputs. Gemma hit the 150-token cap on 36/48 outputs, so its full-response bias estimate is unreliable. Unclassified is not automatically refusal.",
        ["decodingtrust_responses.csv"],
    )
    for benchmark, metrics, file, title in (
        (
            "DecodingTrust",
            ("benign", "untargeted", "targeted"),
            "05_decodingtrust_significance.png",
            "DecodingTrust: race-group differences",
        ),
        (
            "BOLD",
            ("toxic_any", "sentiment_class", "regard_class"),
            "07_bold_significance.png",
            "BOLD: race-category differences",
        ),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        for ax, metric in zip(axes, metrics, strict=True):
            selected = [
                r
                for r in tests
                if r["benchmark"] == benchmark
                and r["regime" if benchmark == "DecodingTrust" else "metric"] == metric
            ]
            forest(
                ax,
                selected,
                scale=100 if benchmark == "DecodingTrust" or metric == "toxic_any" else 1,
                label="Largest minus smallest group mean",
                annotate=False,
            )
            ax.set_title(metric.replace("_", " ").title())
            labels = [
                f"{NAMES[r['model_key']]}  q={float(r['bh_adjusted_q_value']):.3f}"
                for r in ordered(selected)
            ]
            ax.set_yticklabels(labels, fontsize=8)
        selected = [r for r in tests if r["benchmark"] == benchmark]
        save(
            fig,
            file,
            title,
            f"Points show the range between four race/category means, with 95% intervals; zero means no measured group difference. "
            f"{sum(float(r['bh_adjusted_q_value']) < 0.05 for r in selected)} of {len(selected)} exploratory tests have BH q < 0.05. "
            + (
                "Toxicity ranges are percentage points; sentiment/regard are class-score units. Range-bootstrap intervals do not invert the permutation test: an interval above zero is not proof of significance."
                if benchmark == "BOLD"
                else "Only four topic blocks are available per regime; Gemma has 36/48 truncated outputs. Its point is a capped-output diagnostic, not evidence of absent bias."
            ),
            ["text_significance.csv"],
        )
    fig, axes = plt.subplots(1, 3, figsize=(19, 8))
    labels = {
        "toxic_any": "Toxicity flags (%)",
        "sentiment_class": "Mean sentiment class",
        "regard_class": "Mean regard class",
    }
    for ax, metric in zip(axes, labels, strict=True):
        selected = [r for r in categories if r["benchmark"] == "BOLD" and r["metric"] == metric]
        groups = sorted({r["category"] for r in selected})
        keys = [k for k in ORDER if any(r["model_key"] == k for r in selected)]
        matrix = np.array(
            [
                [
                    float(
                        next(
                            r["value"]
                            for r in selected
                            if r["model_key"] == key and r["category"] == group
                        )
                    )
                    for group in groups
                ]
                for key in keys
            ]
        )
        if metric == "toxic_any":
            matrix *= 100
        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="YlOrRd" if metric == "toxic_any" else "RdYlGn",
            vmin=0 if metric == "toxic_any" else -1,
            vmax=max(4, float(matrix.max())) if metric == "toxic_any" else 1,
        )
        ax.set_xticks(
            range(len(groups)),
            [
                g.replace("_Americans", "").replace("_American", "").replace("_", " ")
                for g in groups
            ],
            rotation=35,
            ha="right",
        )
        ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys], fontsize=9)
        ax.set_title(labels[metric])
        for i in range(len(keys)):
            for j in range(len(groups)):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}" if metric != "toxic_any" else f"{matrix[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    save(
        fig,
        "06_bold_categories.png",
        "BOLD: toxicity, sentiment and regard by race category",
        "Each cell has 25 planned continuations, with 24-25 valid scored outputs. Toxicity is the percentage with any Detoxify 'unbiased' score > 0.3; lower means fewer flagged responses. "
        "Sentiment is mean VADER class and regard is mean regard class: -1 negative, 0 neutral, +1 positive (regard 'other' also maps to 0). "
        "Category differences can reflect prompt topics and scorer limitations, not only model bias.",
        ["text_category_summary.csv"],
    )


def visbias(dataset, prefix):
    tests = rows(f"visbias_{dataset}_significance.csv")
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    labels = {
        "vader_compound": ("Sentiment difference", 1),
        "iat_negative_rate": ("Negative-word rate difference (pp)", 100),
        "stereotype_hits_per_100_words": ("Lexicon hits per 100 words difference", 1),
        "identity_mention": ("Identity-mention difference (pp)", 100),
    }
    for ax, (metric, (label, scale)) in zip(axes.flat, labels.items(), strict=True):
        selected = [r for r in tests if r["metric"] == metric]
        forest(ax, selected, scale=scale, label="Black minus White", annotate=False)
        ax.set_title(label)
        ax.set_yticklabels(
            [
                f"{NAMES[r['model_key']]}  q={float(r['bh_adjusted_q_value']):.3f}"
                for r in ordered(selected)
            ],
            fontsize=8,
        )
    selected = [r for r in tests if r["metric"] in labels]
    save(
        fig,
        f"{prefix}_visbias_{dataset}_descriptions.png",
        f"Image descriptions: {'12 neutral faces' if dataset == 'neutral12' else '84 VisBias images'}",
        f"Points compare automated text scores for Black versus White image descriptions; lines show 95% matched-pair intervals. "
        f"{sum(float(r['bh_adjusted_q_value']) < 0.05 for r in selected)} of {len(selected)} description tests have BH q < 0.05 in the joint five-metric family. "
        "Lexicon matches can be innocent visible details (for example, 'hair'); a hit alone is not a stereotype.",
        [f"visbias_{dataset}_significance.csv"],
    )
    table = ordered(rows(f"visbias_{dataset}_comparison_summary.csv"))
    fig, ax = plt.subplots(figsize=(14, 7))
    left = np.zeros(len(table))
    for metric, label, color in (
        ("black_selection_rate", "Selected Black", COLORS[0]),
        ("white_selection_rate", "Selected White", COLORS[1]),
        ("cannot_determine_rate", "Cannot determine", COLORS[2]),
        ("invalid_rate", "Invalid/unclassified", "#B5BBC1"),
    ):
        values = np.array([float(r[metric]) * 100 for r in table])
        ax.barh(range(len(table)), values, left=left, label=label, color=color)
        left += values
    ax.set_yticks(range(len(table)), names(table))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Responses (%)")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    save(
        fig,
        f"{prefix}b_visbias_{dataset}_explicit.png",
        f"Unsupported income/education inferences: {'12 faces' if dataset == 'neutral12' else 'VisBias images'}",
        "Models compare two images on income or education in both presentation orders. Selecting either person makes an unsupported inference; 'cannot determine' is appropriate because these images do not establish those facts. "
        "Directional tests cluster both attributes and both orders by image pair; abstention does not establish that a model is unbiased.",
        [f"visbias_{dataset}_comparison_summary.csv"],
    )


def dpo():
    source = (
        ROOT
        / "reference/dpo/august10/august10_first_run_neutral12_retrospective_v1/research_summary.json"
    )
    summary = json.loads(source.read_text())
    best = summary["best_directional_model"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for index, (label, values) in enumerate(
        (
            ("Frozen base", [35, 100 / 30]),
            ("DPO/LoRA", [100 * best["black_negative_rate"], 100 * best["white_negative_rate"]]),
        )
    ):
        bars = axes[0].bar(
            np.arange(2) + (index - 0.5) * 0.35, values, 0.35, label=label, color=COLORS[index]
        )
        axes[0].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[0].set_xticks([0, 1], ["Black", "White"])
    axes[0].set_ylabel("Negative terms (%)")
    axes[0].set_ylim(0, 45)
    axes[0].legend(frameon=False)
    values = [100 * best["entropy_retention"], 100 * best["unique_response_retention"]]
    bars = axes[1].bar([0, 1], values, color=[COLORS[0], COLORS[1]])
    axes[1].bar_label(bars, fmt="%.1f%%", padding=3)
    axes[1].axhline(95, color=COLORS[3], ls="--", label="95% preservation rule")
    axes[1].set_xticks([0, 1], ["Response entropy", "Unique responses"])
    axes[1].set_ylabel("Retained relative to base (%)")
    axes[1].set_ylim(0, 110)
    axes[1].legend(frameon=False)
    save(
        fig,
        "10_dpo_tradeoff.png",
        "FairFace DPO/LoRA: lower race gap, reduced diversity",
        "The retrospective mean-token DPO adapter reduced the 12-face gap from 31.7 to 16.7 percentage points, but retained only 91.1% of response entropy and 81.3% of unique responses. "
        "This was the best directional result among five configurations, not a confirmed successful adapter; the 95% preservation rule was retrospective.",
        [],
    )
    REGISTRY[-1]["plot_input_data_paths"] = str(source)


def dataset_pools():
    evaluation = read_csv(ROOT / "data/frozen/utkface/utkface_iat_evaluation_manifest.csv")
    future = read_csv(ROOT / "data/frozen/utkface/utkface_finetuning_image_manifest.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    stats = []
    for i, (label, table) in enumerate(
        (("Evaluation", evaluation), ("Future fine-tuning", future))
    ):
        for race, color in zip(("Black", "White"), COLORS, strict=False):
            selected = [r for r in table if r["race"] == race]
            ages = [float(r["age"]) for r in selected]
            quality = [float(r["quality_score"]) for r in selected]
            axes[i, 0].hist(ages, bins=[18, 30, 45, 60, 71], alpha=0.5, label=race, color=color)
            axes[i, 1].hist(
                quality,
                bins=np.linspace(0.5, 0.85, 15),
                histtype="step",
                linewidth=2,
                label=race,
                color=color,
            )
            stats.append(
                dict(
                    pool=label,
                    race=race,
                    images=len(selected),
                    mean_age_years=float(np.mean(ages)),
                    mean_quality_score=float(np.mean(quality)),
                    negative_expression_images=sum(
                        r["expression_class"] == "negative" for r in selected
                    ),
                )
            )
        axes[i, 0].set_title(label)
        axes[i, 0].set_xlabel("Estimated age (years)")
        axes[i, 0].set_ylabel("Images")
        axes[i, 1].set_title(label)
        axes[i, 1].set_xlabel("Image-quality heuristic (0-1)")
        axes[i, 1].set_ylabel("Images")
        for ax in axes[i]:
            ax.legend(frameon=False)
    write_csv(DATA / "utkface_pool_summary.csv", stats)
    save(
        fig,
        "11_utkface_pool_audit.png",
        "UTKFace: evaluation and future training pools",
        "Each pool contains 400 images, balanced by race and recorded gender, with no identical image hashes shared across pools. "
        "Matching uses age bands, expression classes and quality bins, not exact age or pixel quality. The quality score combines exposure, contrast and edges; it is not human-rated quality and may itself depend on skin tone.",
        ["utkface_pool_summary.csv"],
    )
    REGISTRY[-1]["plot_input_data_paths"] = "|".join(
        str(ROOT / "data/frozen/utkface" / file)
        for file in ("utkface_iat_evaluation_manifest.csv", "utkface_finetuning_image_manifest.csv")
    )


def main():
    iat("neutral12", "01")
    iat("utkface", "02")
    diversity()
    text_plots()
    visbias("neutral12", "08")
    visbias("original", "09")
    dpo()
    dataset_pools()
    write_csv(ROOT / "report/plot_registry.csv", REGISTRY)
    print(f"Created {len(REGISTRY)} PNG/PDF figure pairs")


if __name__ == "__main__":
    main()
