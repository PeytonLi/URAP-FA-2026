from __future__ import annotations

import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .constants import NEGATIVE_IAT_SET
from .pair_policy import identity_hits, profile_word_count, tokens

RACES = ("Black", "White")
EXPRESSION_CLASSES = ("negative", "neutral", "positive", "ambiguous")


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _nested_counts(
    rows: Iterable[Mapping[str, Any]],
    field: str,
    *,
    categories: Iterable[str] | None = None,
) -> dict[str, dict[str, int]]:
    materialized = list(rows)
    values = list(categories or sorted({_text(row, field) for row in materialized}))
    return {
        race: {
            value: sum(
                _text(row, "fairface_race") == race
                and _text(row, field).lower() == value.lower()
                for row in materialized
            )
            for value in values
        }
        for race in RACES
    }


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_pretraining_inputs(
    annotation_rows: Iterable[Mapping[str, Any]],
    matched_rows: Iterable[Mapping[str, Any]],
    teacher_rows: Iterable[Mapping[str, Any]],
    *,
    expected_teacher_jobs: int | None = None,
) -> dict[str, Any]:
    annotations = [dict(row) for row in annotation_rows]
    matched = [dict(row) for row in matched_rows]
    teacher = [dict(row) for row in teacher_rows]
    metadata_by_hash = {_text(row, "image_sha256"): row for row in matched}

    status_by_race = _nested_counts(annotations, "expression_annotation_status")
    annotation_totals = Counter(_text(row, "fairface_race") for row in annotations)
    annotation_valid_rates = {
        race: _rate(status_by_race[race].get("ok", 0), annotation_totals[race])
        for race in RACES
    }

    match_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matched:
        match_groups[_text(row, "sampling_match_id")].append(row)
    match_violations = []
    stratum_fields = (
        "fairface_gender",
        "age_bucket",
        "expression_class",
        "expression_fine",
        "expression_intensity",
    )
    for match_id, rows in match_groups.items():
        races = sorted(_text(row, "fairface_race") for row in rows)
        if not match_id or races != ["Black", "White"]:
            match_violations.append(f"{match_id or '<missing>'}:race_membership")
            continue
        if any(
            len({_text(row, field) for row in rows}) != 1 for field in stratum_fields
        ):
            match_violations.append(f"{match_id}:stratum_mismatch")

    enriched_teacher = []
    unknown_teacher_hashes = set()
    for row in teacher:
        digest = _text(row, "image_sha256")
        metadata = metadata_by_hash.get(digest)
        if metadata is None:
            unknown_teacher_hashes.add(digest)
            enriched_teacher.append({**row, "fairface_race": "Unknown"})
        else:
            enriched_teacher.append(
                {
                    **row,
                    "fairface_race": _text(metadata, "fairface_race"),
                    "fairface_gender": _text(metadata, "fairface_gender"),
                    "expression_class": _text(metadata, "expression_class").lower(),
                    "expression_fine": _text(metadata, "expression_fine").lower(),
                }
            )

    teacher_status = _nested_counts(enriched_teacher, "generation_status")
    teacher_totals = Counter(_text(row, "fairface_race") for row in enriched_teacher)
    teacher_valid_rates = {
        race: _rate(teacher_status[race].get("ok", 0), teacher_totals[race])
        for race in RACES
    }
    teacher_retry_rates = {
        race: _rate(
            sum(
                int(float(_text(row, "generation_attempt") or 0)) > 1
                for row in enriched_teacher
                if _text(row, "fairface_race") == race
            ),
            teacher_totals[race],
        )
        for race in RACES
    }
    teacher_ok = [
        row for row in enriched_teacher if _text(row, "generation_status") == "ok"
    ]
    teacher_ok_by_hash = Counter(_text(row, "image_sha256") for row in teacher_ok)
    image_coverage_counts = {
        race: Counter(
            str(teacher_ok_by_hash[_text(row, "image_sha256")])
            for row in matched
            if _text(row, "fairface_race") == race
        )
        for race in RACES
    }
    covered_matches = sum(
        all(teacher_ok_by_hash[_text(row, "image_sha256")] > 0 for row in rows)
        for rows in match_groups.values()
    )
    teacher_word_counts = {
        race: [
            profile_word_count(_text(row, "response"))
            for row in teacher_ok
            if _text(row, "fairface_race") == race
        ]
        for race in RACES
    }
    teacher_identity_counts = {
        race: sum(
            bool(identity_hits(_text(row, "response")))
            for row in teacher_ok
            if _text(row, "fairface_race") == race
        )
        for race in RACES
    }
    validation_issues = Counter(
        issue
        for row in enriched_teacher
        for issue in _text(row, "validation_issues").split("|")
        if issue
    )
    teacher_negative_terms = {
        race: Counter(
            token
            for row in teacher_ok
            if _text(row, "fairface_race") == race
            for token in tokens(_text(row, "response"))
            if token in NEGATIVE_IAT_SET
        )
        for race in RACES
    }
    teacher_negative_counts = {
        race: [
            sum(token in NEGATIVE_IAT_SET for token in tokens(_text(row, "response")))
            for row in teacher_ok
            if _text(row, "fairface_race") == race
        ]
        for race in RACES
    }
    teacher_negative_by_expression = {
        race: {
            expression: _mean(
                sum(
                    token in NEGATIVE_IAT_SET
                    for token in tokens(_text(row, "response"))
                )
                for row in teacher_ok
                if _text(row, "fairface_race") == race
                and _text(row, "expression_class") == expression
            )
            for expression in EXPRESSION_CLASSES[:-1]
        }
        for race in RACES
    }
    expected = (
        expected_teacher_jobs if expected_teacher_jobs is not None else len(teacher)
    )

    return {
        "annotation": {
            "images": len(annotations),
            "race_counts": dict(sorted(annotation_totals.items())),
            "status_counts_by_race": status_by_race,
            "valid_rate_by_race": annotation_valid_rates,
            "expression_counts_by_race": _nested_counts(
                annotations,
                "expression_class",
                categories=EXPRESSION_CLASSES,
            ),
            "fine_expression_counts_by_race": _nested_counts(
                annotations, "expression_fine"
            ),
        },
        "matched": {
            "images": len(matched),
            "matches": len(match_groups),
            "race_counts": dict(
                sorted(Counter(_text(row, "fairface_race") for row in matched).items())
            ),
            "gender_counts_by_race": _nested_counts(matched, "fairface_gender"),
            "expression_counts_by_race": _nested_counts(
                matched,
                "expression_class",
                categories=EXPRESSION_CLASSES[:-1],
            ),
            "fine_expression_counts_by_race": _nested_counts(
                matched, "expression_fine"
            ),
            "expression_intensity_counts_by_race": _nested_counts(
                matched, "expression_intensity"
            ),
            "mean_expression_confidence_by_race": {
                race: _mean(
                    float(_text(row, "expression_confidence"))
                    for row in matched
                    if _text(row, "fairface_race") == race
                )
                for race in RACES
            },
            "exact_match_violations": match_violations,
        },
        "teacher": {
            "expected_jobs": expected,
            "processed_jobs": len(teacher),
            "complete": len(teacher) == expected,
            "unknown_image_hashes": sorted(unknown_teacher_hashes),
            "status_counts_by_race": teacher_status,
            "valid_rate_by_race": teacher_valid_rates,
            "accepted_profiles_per_image_counts_by_race": {
                race: dict(
                    sorted(
                        image_coverage_counts[race].items(),
                        key=lambda item: int(item[0]),
                    )
                )
                for race in RACES
            },
            "image_coverage_rate_by_race": {
                race: _rate(
                    sum(
                        count
                        for accepted, count in image_coverage_counts[race].items()
                        if int(accepted) > 0
                    ),
                    sum(image_coverage_counts[race].values()),
                )
                for race in RACES
            },
            "complete_matches_with_profile_coverage": covered_matches,
            "total_matches_for_profile_coverage": len(match_groups),
            "retry_rate_by_race": teacher_retry_rates,
            "mean_generation_attempt_by_race": {
                race: _mean(
                    float(_text(row, "generation_attempt") or 0)
                    for row in enriched_teacher
                    if _text(row, "fairface_race") == race
                )
                for race in RACES
            },
            "mean_profile_words_by_race": {
                race: _mean(teacher_word_counts[race]) for race in RACES
            },
            "profile_word_range_by_race": {
                race: (
                    [min(teacher_word_counts[race]), max(teacher_word_counts[race])]
                    if teacher_word_counts[race]
                    else []
                )
                for race in RACES
            },
            "identity_leakage_records_by_race": teacher_identity_counts,
            "negative_lexicon_mean_count_by_race": {
                race: _mean(teacher_negative_counts[race]) for race in RACES
            },
            "negative_lexicon_profile_rate_by_race": {
                race: _rate(
                    sum(count > 0 for count in teacher_negative_counts[race]),
                    len(teacher_negative_counts[race]),
                )
                for race in RACES
            },
            "negative_lexicon_mean_by_expression_and_race": teacher_negative_by_expression,
            "negative_lexicon_term_counts_by_race": {
                race: dict(sorted(teacher_negative_terms[race].items()))
                for race in RACES
            },
            "unique_response_fraction_by_race": {
                race: _rate(
                    len(
                        {
                            _text(row, "response")
                            for row in teacher_ok
                            if _text(row, "fairface_race") == race
                        }
                    ),
                    sum(_text(row, "fairface_race") == race for row in teacher_ok),
                )
                for race in RACES
            },
            "validation_issue_counts": dict(sorted(validation_issues.items())),
            "token_usage": {
                key: sum(int(row.get(key) or 0) for row in teacher)
                for key in (
                    "prompt_tokens",
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "thought_tokens",
                    "reasoning_output_tokens",
                    "total_tokens",
                )
            },
        },
    }


def plot_pretraining_inputs(
    summary: Mapping[str, Any],
    output_dir: Path,
) -> list[str]:
    cache_dir = Path(tempfile.gettempdir()) / "august3-matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Black": "#0B6E75", "White": "#5B6770"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    for axis, section, title in (
        (axes[0], "annotation", "Annotated candidate pool"),
        (axes[1], "matched", "Expression-verified pilot pool"),
    ):
        counts = summary[section]["expression_counts_by_race"]
        categories = [
            category
            for category in EXPRESSION_CLASSES
            if any(category in counts[race] for race in RACES)
        ]
        x = np.arange(len(categories))
        width = 0.36
        for offset, race in zip((-width / 2, width / 2), RACES):
            total = sum(counts[race].values())
            axis.bar(
                x + offset,
                [_rate(counts[race].get(item, 0), total) * 100 for item in categories],
                width,
                label=race,
                color=colors[race],
            )
        axis.set_xticks(x, [item.title() for item in categories])
        axis.set_ylabel("Share within race (%)")
        axis.set_title(title)
        axis.set_ylim(bottom=0)
    axes[0].legend()
    fig.suptitle("Visible-expression distribution before and after exact matching")
    fig.tight_layout()
    expression_path = output_dir / "01_expression_pool_and_matching.png"
    fig.savefig(expression_path, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), dpi=180)
    matched = summary["matched"]
    for axis, key, title in (
        (axes[0], "gender_counts_by_race", "Gender labels"),
        (axes[1], "expression_counts_by_race", "Expression classes"),
        (axes[2], "fine_expression_counts_by_race", "Fine expressions"),
        (
            axes[3],
            "expression_intensity_counts_by_race",
            "Expression intensity",
        ),
    ):
        counts = matched[key]
        categories = sorted({category for race in RACES for category in counts[race]})
        x = np.arange(len(categories))
        width = 0.36
        for offset, race in zip((-width / 2, width / 2), RACES):
            axis.bar(
                x + offset,
                [counts[race].get(item, 0) for item in categories],
                width,
                label=race,
                color=colors[race],
            )
        axis.set_xticks(x, [item.title() for item in categories])
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylabel("Images")
        axis.set_title(title)
    axes[0].legend()
    fig.suptitle(
        f"Matched FairFace pilot: {matched['images']} images, "
        f"{matched['matches']} Black/White pairs"
    )
    fig.tight_layout()
    balance_path = output_dir / "02_matched_pool_balance.png"
    fig.savefig(balance_path, bbox_inches="tight")
    plt.close(fig)

    teacher = summary["teacher"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 5), dpi=180)
    metrics = (
        ("valid_rate_by_race", "Valid responses (%)", 100),
        ("image_coverage_rate_by_race", "Images with a valid profile (%)", 100),
        ("retry_rate_by_race", "Responses requiring retry (%)", 100),
        ("mean_profile_words_by_race", "Mean profile words", 1),
        ("identity_leakage_records_by_race", "Accepted identity leaks", 1),
    )
    for axis, (key, title, scale) in zip(axes, metrics):
        values = [teacher[key][race] * scale for race in RACES]
        axis.bar(RACES, values, color=[colors[race] for race in RACES])
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        for index, value in enumerate(values):
            label = f"{value:.1f}" if isinstance(value, float) else str(value)
            axis.text(index, value, label, ha="center", va="bottom")
    fig.suptitle(
        "Independent grounded-profile teacher quality "
        f"({teacher['processed_jobs']}/{teacher['expected_jobs']} jobs)"
    )
    fig.tight_layout()
    teacher_path = output_dir / "03_teacher_generation_quality.png"
    fig.savefig(teacher_path, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=180)
    expressions = list(EXPRESSION_CLASSES[:-1])
    x = np.arange(len(expressions))
    width = 0.36
    for offset, race in zip((-width / 2, width / 2), RACES):
        axes[0].bar(
            x + offset,
            [
                teacher["negative_lexicon_mean_by_expression_and_race"][race][item]
                for item in expressions
            ],
            width,
            label=race,
            color=colors[race],
        )
    axes[0].set_xticks(x, [item.title() for item in expressions])
    axes[0].set_ylabel("Mean IAT-negative tokens per profile")
    axes[0].set_title("Mechanical rate by matched expression")
    axes[0].legend()
    expression_values = [
        teacher["negative_lexicon_mean_by_expression_and_race"][race][expression]
        for race in RACES
        for expression in expressions
    ]
    if not any(expression_values):
        axes[0].set_ylim(0, 1)
        axes[0].text(
            0.5,
            0.52,
            "0.00 for both races\nin every expression stratum",
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    totals = Counter()
    for race in RACES:
        totals.update(teacher["negative_lexicon_term_counts_by_race"][race])
    terms = [term for term, _ in totals.most_common(12)]
    if terms:
        positions = np.arange(len(terms))
        for offset, race in zip((-width / 2, width / 2), RACES):
            axes[1].barh(
                positions + offset,
                [
                    teacher["negative_lexicon_term_counts_by_race"][race].get(term, 0)
                    for term in terms
                ],
                width,
                label=race,
                color=colors[race],
            )
        axes[1].set_yticks(positions, [term.title() for term in terms])
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Occurrences in accepted profiles")
    else:
        axes[1].set_axis_off()
        axes[1].text(
            0.5,
            0.52,
            "No exact IAT-negative terms\nfound in accepted profiles",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=13,
        )
    axes[1].set_title("IAT-negative vocabulary screen")
    fig.suptitle(
        "Teacher negative-language audit (lexical safeguard, not emotion accuracy)"
    )
    fig.tight_layout()
    negativity_path = output_dir / "04_teacher_negative_language.png"
    fig.savefig(negativity_path, bbox_inches="tight")
    plt.close(fig)
    return [
        str(expression_path),
        str(balance_path),
        str(teacher_path),
        str(negativity_path),
    ]
