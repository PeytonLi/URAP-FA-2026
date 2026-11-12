from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .constants import EXPRESSION_PAIR_MAXIMUM_CUE_DIFFERENCE
from .expression_annotation import EXPRESSION_CLASSES
from .pair_policy import identity_hits

SCORE_NAMES = ("intensity", "brow_activity", "eye_activity", "mouth_activity")
PAIR_JUDGMENT_KEYS = {
    "first_expression_class",
    "second_expression_class",
    "first_intensity",
    "second_intensity",
    "first_brow_activity",
    "second_brow_activity",
    "first_eye_activity",
    "second_eye_activity",
    "first_mouth_activity",
    "second_mouth_activity",
    "equivalent_for_fairness_training",
    "confidence",
    "visible_difference_summary",
}


def validate_pair_judgment(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != PAIR_JUDGMENT_KEYS:
        raise ValueError(
            "expression-pair keys differ; "
            f"missing={sorted(PAIR_JUDGMENT_KEYS - set(value))}, "
            f"extra={sorted(set(value) - PAIR_JUDGMENT_KEYS)}"
        )
    result: dict[str, Any] = {}
    for side in ("first", "second"):
        expression_class = str(value[f"{side}_expression_class"]).strip().lower()
        if expression_class not in EXPRESSION_CLASSES:
            raise ValueError(f"invalid {side}_expression_class")
        result[f"{side}_expression_class"] = expression_class
        for name in SCORE_NAMES:
            try:
                raw = float(value[f"{side}_{name}"])
                parsed = int(raw)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{side}_{name} must be an integer in [0, 4]"
                ) from exc
            if not math.isfinite(raw) or raw != parsed or not 0 <= parsed <= 4:
                raise ValueError(f"{side}_{name} must be an integer in [0, 4]")
            result[f"{side}_{name}"] = parsed
    equivalent = value["equivalent_for_fairness_training"]
    if not isinstance(equivalent, bool):
        raise ValueError("equivalent_for_fairness_training must be boolean")
    try:
        confidence = float(value["confidence"])
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be in [0, 1]") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    summary = str(value["visible_difference_summary"]).strip()
    if not summary or len(summary) > 400 or len(summary.split()) > 40:
        raise ValueError(
            "visible_difference_summary must contain 1 to 40 words and at most 400 characters"
        )
    if identity_hits(summary):
        raise ValueError("visible_difference_summary contains identity language")
    result.update(
        {
            "equivalent_for_fairness_training": equivalent,
            "confidence": confidence,
            "visible_difference_summary": summary,
        }
    )
    if equivalent:
        if (
            result["first_expression_class"] != result["second_expression_class"]
            or result["first_expression_class"] == "ambiguous"
        ):
            raise ValueError("equivalent pair must have the same non-ambiguous class")
        if any(
            abs(result[f"first_{name}"] - result[f"second_{name}"])
            > EXPRESSION_PAIR_MAXIMUM_CUE_DIFFERENCE
            for name in SCORE_NAMES
        ):
            raise ValueError("equivalent pair has a facial-cue score difference above 1")
    return result


def reverse_to_canonical(value: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_pair_judgment(value)
    result = dict(validated)
    for field in ("expression_class", *SCORE_NAMES):
        result[f"first_{field}"] = validated[f"second_{field}"]
        result[f"second_{field}"] = validated[f"first_{field}"]
    return result


def aggregate_pair_judgments(
    normal: Mapping[str, Any],
    reversed_order: Mapping[str, Any],
    *,
    expected_expression_class: str,
    minimum_confidence: float = 0.70,
) -> tuple[dict[str, Any], list[str]]:
    first = validate_pair_judgment(normal)
    second = reverse_to_canonical(reversed_order)
    issues: list[str] = []
    if not first["equivalent_for_fairness_training"]:
        issues.append("normal_order_not_equivalent")
    if not second["equivalent_for_fairness_training"]:
        issues.append("reversed_order_not_equivalent")
    if min(first["confidence"], second["confidence"]) < minimum_confidence:
        issues.append("low_pair_equivalence_confidence")
    expected = expected_expression_class.strip().lower()
    for order_name, value in (("normal", first), ("reversed", second)):
        for side in ("first", "second"):
            if value[f"{side}_expression_class"] != expected:
                issues.append(f"{order_name}_{side}_class_differs_from_individual_label")
    for side in ("first", "second"):
        for name in SCORE_NAMES:
            if abs(first[f"{side}_{name}"] - second[f"{side}_{name}"]) > 1:
                issues.append(f"order_unstable:{side}_{name}")
    max_cue_difference = max(
        abs(first[f"first_{name}"] - first[f"second_{name}"])
        for name in SCORE_NAMES
    )
    if max_cue_difference > EXPRESSION_PAIR_MAXIMUM_CUE_DIFFERENCE:
        issues.append("normal_order_cue_difference_above_1")
    reverse_max = max(
        abs(second[f"first_{name}"] - second[f"second_{name}"])
        for name in SCORE_NAMES
    )
    if reverse_max > EXPRESSION_PAIR_MAXIMUM_CUE_DIFFERENCE:
        issues.append("reversed_order_cue_difference_above_1")
    summary = {
        "minimum_confidence": min(first["confidence"], second["confidence"]),
        "max_cue_difference": max(max_cue_difference, reverse_max),
        "normal_scores": {
            side: {name: first[f"{side}_{name}"] for name in SCORE_NAMES}
            for side in ("first", "second")
        },
        "reversed_canonical_scores": {
            side: {name: second[f"{side}_{name}"] for name in SCORE_NAMES}
            for side in ("first", "second")
        },
    }
    return summary, sorted(set(issues))
