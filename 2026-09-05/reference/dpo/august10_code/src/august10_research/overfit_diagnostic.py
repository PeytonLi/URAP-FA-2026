from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from august3_debiasing.constants import NEGATIVE_IAT_SET, TASK_IAT
from august3_debiasing.pair_policy import parse_iat_response
from august3_debiasing.prompts import IAT_GENERATION_PROMPT

PROTOCOL_VERSION = "august10-dpo-lora-overfit-diagnostic-v1"
INFORMATIVE_PROTOCOL_VERSION = "august10-dpo-lora-overfit-diagnostic-v2-informative"
GATE_VERSION = "august10-overfit-diagnostic-gate-v1"
EXPECTED_IMAGES = 12
EXPECTED_TRAIN_ROWS = 192
EXPECTED_DEVELOPMENT_ROWS = 12
EXPECTED_TRAIN_ROWS_PER_IMAGE = 16


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _check(observed: Any, requirement: str, passed: bool) -> dict[str, Any]:
    return {"observed": observed, "requirement": requirement, "pass": bool(passed)}


def assess_overfit_dataset(
    rows: Iterable[Mapping[str, Any]],
    forbidden_eval_hashes: set[str],
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    protocol_labels = {_text(row, "diagnostic_protocol") for row in materialized}
    expected_protocol = (
        next(iter(protocol_labels)) if len(protocol_labels) == 1 else ""
    )
    supported_protocol = expected_protocol in {
        PROTOCOL_VERSION,
        INFORMATIVE_PROTOCOL_VERSION,
    }
    train = [row for row in materialized if _text(row, "split") == "train"]
    development = [
        row for row in materialized if _text(row, "split") == "development"
    ]
    pair_ids = [_text(row, "pair_id") for row in materialized]
    image_hashes = {_text(row, "image_sha256").lower() for row in materialized}
    train_hashes = {_text(row, "image_sha256").lower() for row in train}
    development_hashes = {
        _text(row, "image_sha256").lower() for row in development
    }
    forbidden_overlap = sorted(image_hashes & {value.lower() for value in forbidden_eval_hashes})

    race_by_split = {
        split: Counter(_text(row, "fairface_race") for row in split_rows)
        for split, split_rows in (("train", train), ("development", development))
    }
    train_rows_per_image = Counter(_text(row, "image_sha256") for row in train)
    development_rows_per_image = Counter(
        _text(row, "image_sha256") for row in development
    )
    template_counts = {
        race: Counter(
            (_text(row, "chosen"), _text(row, "rejected"))
            for row in train
            if _text(row, "fairface_race") == race
        )
        for race in ("Black", "White")
    }

    response_issues: list[str] = []
    for row in materialized:
        chosen = parse_iat_response(_text(row, "chosen"))
        rejected = parse_iat_response(_text(row, "rejected"))
        if not chosen["valid"] or chosen["duplicate_count"]:
            response_issues.append(f"{_text(row, 'pair_id')}:invalid_chosen")
        if not rejected["valid"] or rejected["duplicate_count"]:
            response_issues.append(f"{_text(row, 'pair_id')}:invalid_rejected")
        if chosen["negative_count"] != 0 or rejected["negative_count"] != 3:
            response_issues.append(f"{_text(row, 'pair_id')}:invalid_direction")
        if set(chosen["words"]) & NEGATIVE_IAT_SET:
            response_issues.append(f"{_text(row, 'pair_id')}:negative_chosen")

    required_fields = (
        "pair_id",
        "match_block_id",
        "image_path",
        "image_sha256",
        "split",
        "task_type",
        "prompt",
        "chosen",
        "rejected",
        "fairface_race",
        "fairface_gender",
        "age_bucket",
        "expression_class",
        "objective_type",
        "diagnostic_protocol",
    )
    missing_fields = sorted(
        {
            field
            for row in materialized
            for field in required_fields
            if not _text(row, field)
        }
    )
    checks = {
        "required_fields": _check(missing_fields, "no missing required fields", not missing_fields),
        "unique_pair_ids": _check(
            len(set(pair_ids)),
            f"{len(materialized)} unique nonempty IDs",
            bool(pair_ids) and all(pair_ids) and len(set(pair_ids)) == len(pair_ids),
        ),
        "rows": _check(
            len(materialized),
            f"{EXPECTED_TRAIN_ROWS + EXPECTED_DEVELOPMENT_ROWS}",
            len(materialized) == EXPECTED_TRAIN_ROWS + EXPECTED_DEVELOPMENT_ROWS,
        ),
        "train_rows": _check(
            len(train), str(EXPECTED_TRAIN_ROWS), len(train) == EXPECTED_TRAIN_ROWS
        ),
        "development_rows": _check(
            len(development),
            str(EXPECTED_DEVELOPMENT_ROWS),
            len(development) == EXPECTED_DEVELOPMENT_ROWS,
        ),
        "unique_images": _check(
            len(image_hashes), str(EXPECTED_IMAGES), len(image_hashes) == EXPECTED_IMAGES
        ),
        "balanced_race_rows": _check(
            {split: dict(counts) for split, counts in race_by_split.items()},
            "Black == White in each split",
            all(counts["Black"] == counts["White"] > 0 for counts in race_by_split.values()),
        ),
        "iat_only": _check(
            sorted({_text(row, "task_type") for row in materialized}),
            f"['{TASK_IAT}']",
            {_text(row, "task_type") for row in materialized} == {TASK_IAT},
        ),
        "exact_prompt": _check(
            len({_text(row, "prompt") for row in materialized}),
            "one exact frozen IAT prompt",
            {_text(row, "prompt") for row in materialized} == {IAT_GENERATION_PROMPT},
        ),
        "response_direction": _check(
            response_issues[:20],
            "chosen has 0 and rejected has 3 negative terms",
            not response_issues,
        ),
        "train_rows_per_image": _check(
            sorted(set(train_rows_per_image.values())),
            f"[{EXPECTED_TRAIN_ROWS_PER_IMAGE}]",
            set(train_rows_per_image.values()) == {EXPECTED_TRAIN_ROWS_PER_IMAGE},
        ),
        "development_rows_per_image": _check(
            sorted(set(development_rows_per_image.values())),
            "[1]",
            set(development_rows_per_image.values()) == {1},
        ),
        "intentional_exact_image_overlap": _check(
            len(train_hashes & development_hashes),
            f"{EXPECTED_IMAGES}; diagnostic only",
            train_hashes == development_hashes and len(train_hashes) == EXPECTED_IMAGES,
        ),
        "race_invariant_template_distribution": _check(
            {
                race: sorted(
                    [chosen, rejected, count]
                    for (chosen, rejected), count in values.items()
                )
                for race, values in template_counts.items()
            },
            "identical Black and White training-template counts",
            template_counts["Black"] == template_counts["White"],
        ),
        "protected_evaluation_disjoint": _check(
            forbidden_overlap,
            "no overlap with protected evaluation hashes",
            not forbidden_overlap,
        ),
        "protocol_labels": _check(
            sorted(protocol_labels),
            f"one supported protocol: {expected_protocol}",
            supported_protocol,
        ),
    }
    if expected_protocol == PROTOCOL_VERSION:
        checks["neutral_images_only"] = _check(
            sorted({_text(row, "expression_class") for row in materialized}),
            "['neutral']",
            {_text(row, "expression_class") for row in materialized} == {"neutral"},
        )
    elif expected_protocol == INFORMATIVE_PROTOCOL_VERSION:
        development_by_hash = {
            _text(row, "image_sha256"): row for row in development
        }
        negative_by_race = Counter()
        selected_by_race = Counter()
        for row in development_by_hash.values():
            race = _text(row, "fairface_race")
            negative_by_race[race] += int(_text(row, "diagnostic_base_negative_terms") or 0)
            selected_by_race[race] += int(_text(row, "diagnostic_base_selected_terms") or 0)
        expression_sets: dict[str, set[str]] = {}
        for row in development:
            expression_sets.setdefault(_text(row, "source_match_block_id"), set()).add(
                _text(row, "expression_class")
            )
        checks["informative_base_negative_events"] = _check(
            dict(negative_by_race),
            ">= 6 screened base negative terms for each race",
            min(negative_by_race["Black"], negative_by_race["White"]) >= 6,
        )
        checks["screening_selected_terms_present"] = _check(
            dict(selected_by_race),
            "> 0 screened terms for each race",
            min(selected_by_race["Black"], selected_by_race["White"]) > 0,
        )
        checks["expression_class_matched_within_blocks"] = _check(
            {key: sorted(value) for key, value in expression_sets.items()},
            "one expression class per matched block",
            len(expression_sets) == 6
            and all(len(value) == 1 for value in expression_sets.values()),
        )
    passed = all(item["pass"] for item in checks.values())
    return {
        "protocol_version": GATE_VERSION,
        "pass": passed,
        "decision": "authorize_diagnostic_overfit" if passed else "do_not_train",
        "non_confirmatory": True,
        "intentional_train_development_image_overlap": True,
        "checks": checks,
        "metrics": {
            "rows": len(materialized),
            "train_rows": len(train),
            "development_rows": len(development),
            "unique_images": len(image_hashes),
            "forbidden_overlap": len(forbidden_overlap),
        },
    }


def diagnostic_preference_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    groups = summary.get("groups", {})
    required = {"iat_3word/Black", "iat_3word/White"}
    checks = {
        "both_races_present": required <= set(groups),
        "overall_margin_delta_positive": float(summary.get("mean_margin_delta", 0.0)) > 0,
        "overall_preference_accuracy_at_least_90pct": float(
            summary.get("preference_accuracy", 0.0)
        )
        >= 0.90,
    }
    if required <= set(groups):
        checks["both_races_margin_delta_positive"] = all(
            float(groups[name]["mean_margin_delta"]) > 0 for name in required
        )
        checks["both_races_preference_accuracy_at_least_80pct"] = all(
            float(groups[name]["preference_accuracy"]) >= 0.80 for name in required
        )
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "interpretation": "Mechanistic overfit diagnostic only; not a fairness result.",
    }
