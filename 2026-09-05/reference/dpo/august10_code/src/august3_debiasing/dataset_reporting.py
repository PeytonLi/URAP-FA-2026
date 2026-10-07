from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .constants import NEGATIVE_IAT_SET, TASK_IAT, TASK_PROFILE
from .pair_policy import parse_iat_response, profile_word_count


PROFILE_REPORT_METRICS = (
    "quality_score",
    "grounded_detail_count",
    "visible_expression_support",
    "person_negative_count",
    "unwarranted_negative_count",
    "unsupported_trait_count",
    "identity_leakage_count",
    "invented_biography_count",
    "word_count",
)


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _number(row: Mapping[str, Any], key: str) -> float:
    value = _text(row, key)
    return float(value) if value else 0.0


def _nested_counts(
    rows: list[Mapping[str, Any]],
    field: str,
) -> dict[str, dict[str, int]]:
    categories = sorted({_text(row, field) for row in rows if _text(row, field)})
    return {
        race: {
            category: sum(
                _text(row, "fairface_race") == race
                and _text(row, field) == category
                for row in rows
            )
            for category in categories
        }
        for race in ("Black", "White")
    }


def summarize_selected_dataset(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    iat_rows = [row for row in materialized if _text(row, "task_type") == TASK_IAT]
    profile_rows = [
        row for row in materialized if _text(row, "task_type") == TASK_PROFILE
    ]
    iat_terms: dict[str, dict[str, dict[str, int]]] = {}
    iat_negative_means: dict[str, dict[str, float]] = {}
    iat_negative_distributions: dict[str, dict[str, dict[str, int]]] = {}
    for race in ("Black", "White"):
        race_rows = [row for row in iat_rows if _text(row, "fairface_race") == race]
        iat_terms[race] = {}
        iat_negative_means[race] = {}
        iat_negative_distributions[race] = {}
        for side in ("chosen", "rejected"):
            parsed = [parse_iat_response(_text(row, side)) for row in race_rows]
            negative_terms = Counter(
                word
                for response in parsed
                for word in response["words"]
                if word in NEGATIVE_IAT_SET
            )
            negative_counts = Counter(response["negative_count"] for response in parsed)
            iat_terms[race][side] = dict(sorted(negative_terms.items()))
            iat_negative_means[race][side] = (
                sum(response["negative_count"] for response in parsed) / len(parsed)
                if parsed
                else 0.0
            )
            iat_negative_distributions[race][side] = {
                str(count): negative_counts.get(count, 0) for count in range(4)
            }

    profile_means: dict[str, dict[str, dict[str, float]]] = {}
    for race in ("Black", "White"):
        race_rows = [
            row for row in profile_rows if _text(row, "fairface_race") == race
        ]
        profile_means[race] = {}
        for side in ("chosen", "rejected"):
            values: dict[str, float] = {}
            for metric in PROFILE_REPORT_METRICS:
                if metric == "word_count":
                    observations = [profile_word_count(_text(row, side)) for row in race_rows]
                else:
                    observations = [
                        _number(row, f"{side}_{metric}") for row in race_rows
                    ]
                values[metric] = (
                    sum(observations) / len(observations) if observations else 0.0
                )
            profile_means[race][side] = values

    iat_chosen_tuple_counts = Counter(
        parse_iat_response(_text(row, "chosen"))["word_tuple"] for row in iat_rows
    )
    iat_chosen_tuple_entropy = 0.0
    if iat_rows:
        iat_chosen_tuple_entropy = -sum(
            (count / len(iat_rows)) * math.log(count / len(iat_rows))
            for count in iat_chosen_tuple_counts.values()
            if count
        )
    iat_chosen_tuple_normalized_entropy = (
        iat_chosen_tuple_entropy / math.log(len(iat_chosen_tuple_counts))
        if len(iat_chosen_tuple_counts) > 1
        else 0.0
    )

    return {
        "rows": len(materialized),
        "unique_images": len(
            {_text(row, "image_sha256") for row in materialized if _text(row, "image_sha256")}
        ),
        "match_blocks": len(
            {_text(row, "match_block_id") for row in materialized if _text(row, "match_block_id")}
        ),
        "task_counts_by_race": _nested_counts(materialized, "task_type"),
        "objective_counts_by_race": _nested_counts(materialized, "objective_type"),
        "expression_counts_by_race": _nested_counts(materialized, "expression_class"),
        "expression_intensity_counts_by_race": _nested_counts(
            materialized, "expression_intensity"
        ),
        "split_counts_by_race": _nested_counts(materialized, "split"),
        "iat_negative_term_counts": iat_terms,
        "iat_mean_negative_count": iat_negative_means,
        "iat_negative_count_distribution": iat_negative_distributions,
        "iat_chosen_tuple_counts": dict(sorted(iat_chosen_tuple_counts.items())),
        "iat_chosen_unique_tuples": len(iat_chosen_tuple_counts),
        "iat_chosen_tuple_entropy": iat_chosen_tuple_entropy,
        "iat_chosen_tuple_normalized_entropy": iat_chosen_tuple_normalized_entropy,
        "iat_chosen_tuple_max_fraction": (
            max(iat_chosen_tuple_counts.values()) / len(iat_rows)
            if iat_rows
            else 0.0
        ),
        "profile_metric_means": profile_means,
    }


def plot_selected_dataset(
    summary: Mapping[str, Any],
    output_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Black": "#0B6E75", "White": "#5B6770"}
    races = ("Black", "White")

    fig, axes = plt.subplots(2, 3, figsize=(19, 10), dpi=180)
    for axis, key, title in (
        (axes[0, 0], "task_counts_by_race", "Rows by task"),
        (axes[0, 1], "expression_counts_by_race", "Rows by expression class"),
        (
            axes[0, 2],
            "expression_intensity_counts_by_race",
            "Rows by expression intensity",
        ),
        (axes[1, 0], "objective_counts_by_race", "Rows by DPO objective"),
        (axes[1, 1], "split_counts_by_race", "Rows by split"),
    ):
        categories = sorted(
            {
                category
                for race in races
                for category in summary[key][race]
            }
        )
        x = np.arange(len(categories))
        width = 0.36
        for offset, race in zip((-width / 2, width / 2), races):
            axis.bar(
                x + offset,
                [summary[key][race].get(category, 0) for category in categories],
                width,
                label=race,
                color=colors[race],
            )
        axis.set_xticks(x, [category.replace("_", " ") for category in categories])
        axis.tick_params(axis="x", rotation=28)
        axis.set_ylabel("Rows")
        axis.set_title(title)
    axes[0, 0].legend()
    axes[1, 2].axis("off")
    fig.suptitle("Frozen DPO dataset composition", fontsize=15)
    fig.tight_layout()
    dashboard = output_dir / "01_frozen_dataset_composition.png"
    fig.savefig(dashboard, bbox_inches="tight")
    plt.close(fig)

    term_totals = Counter()
    for race in races:
        for side in ("chosen", "rejected"):
            term_totals.update(summary["iat_negative_term_counts"][race][side])
    terms = [term for term, _ in term_totals.most_common(16)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=180)
    x = np.arange(2)
    width = 0.36
    for offset, race in zip((-width / 2, width / 2), races):
        axes[0].bar(
            x + offset,
            [summary["iat_mean_negative_count"][race][side] for side in ("chosen", "rejected")],
            width,
            label=race,
            color=colors[race],
        )
    axes[0].set_xticks(x, ["Chosen", "Rejected"])
    axes[0].set_ylabel("Mean negative words per response")
    axes[0].set_title("IAT target negativity")
    axes[0].legend()
    for axis, side, title in (
        (axes[1], "chosen", "Chosen negative-term targets"),
        (axes[2], "rejected", "Rejected negative-term targets"),
    ):
        y = np.arange(len(terms))
        bar_height = 0.36
        for offset, race in zip((-bar_height / 2, bar_height / 2), races):
            axis.barh(
                y + offset,
                [summary["iat_negative_term_counts"][race][side].get(term, 0) for term in terms],
                bar_height,
                label=race,
                color=colors[race],
            )
        axis.set_yticks(y, [term.title() for term in terms])
        axis.invert_yaxis()
        axis.set_xlabel("Occurrences")
        axis.set_title(title)
    fig.suptitle("Frozen IAT target-vocabulary audit", fontsize=15)
    fig.tight_layout()
    vocabulary = output_dir / "02_frozen_iat_target_vocabulary.png"
    fig.savefig(vocabulary, bbox_inches="tight")
    plt.close(fig)

    tuple_counts = Counter(summary["iat_chosen_tuple_counts"])
    ordered_tuples = [item for item, _ in tuple_counts.most_common()][::-1]
    fig_height = max(4.5, 0.32 * len(ordered_tuples) + 1.5)
    fig, axis = plt.subplots(figsize=(11, fig_height), dpi=180)
    positions = np.arange(len(ordered_tuples))
    axis.barh(
        positions,
        [tuple_counts[item] for item in ordered_tuples],
        color="#0B6E75",
    )
    axis.set_yticks(positions, [item.replace(",", " / ") for item in ordered_tuples])
    axis.set_xlabel("Chosen rows (both races combined)")
    axis.set_title(
        "Frozen IAT chosen-tuple diversity\n"
        f"unique={summary['iat_chosen_unique_tuples']}, "
        f"normalized entropy={summary['iat_chosen_tuple_normalized_entropy']:.3f}, "
        f"largest share={summary['iat_chosen_tuple_max_fraction']:.1%}"
    )
    fig.tight_layout()
    tuple_distribution = output_dir / "03_frozen_iat_chosen_tuple_distribution.png"
    fig.savefig(tuple_distribution, bbox_inches="tight")
    plt.close(fig)
    return [str(dashboard), str(vocabulary), str(tuple_distribution)]
