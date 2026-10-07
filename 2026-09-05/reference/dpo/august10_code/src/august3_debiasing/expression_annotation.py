from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from typing import Any, Mapping


EXPRESSION_CLASSES = {"positive", "neutral", "negative", "ambiguous"}
EXPRESSION_ANNOTATION_POLICY_VERSION = "august3-visible-expression-v2-intensity"
FINE_EXPRESSIONS = {
    "happy",
    "neutral",
    "sad",
    "angry",
    "fear",
    "disgust",
    "surprise",
    "other",
}


def stratified_annotation_subset(
    rows: list[Mapping[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Take a stable round-robin prefix across race/gender intersections."""
    materialized = [dict(row) for row in rows]
    if limit <= 0 or limit >= len(materialized):
        return materialized
    grouped: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for row in materialized:
        key = (
            str(row.get("fairface_race", "")).strip(),
            str(row.get("fairface_gender", "")).strip(),
        )
        grouped[key].append(row)
    keys = sorted(grouped)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for key in keys:
            if grouped[key]:
                selected.append(grouped[key].popleft())
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    return selected


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("expression response contains no JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expression response must be a JSON object")
    return value


def validate_expression_annotation(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "expression_class",
        "expression_fine",
        "expression_intensity",
        "confidence",
        "visible_cues",
    }
    if set(value) != required:
        raise ValueError(
            f"expression annotation keys differ; missing={sorted(required - set(value))}, "
            f"extra={sorted(set(value) - required)}"
        )
    expression_class = str(value["expression_class"]).strip().lower()
    expression_fine = str(value["expression_fine"]).strip().lower()
    expression_intensity = int(value["expression_intensity"])
    if isinstance(value["expression_intensity"], float) and not value[
        "expression_intensity"
    ].is_integer():
        raise ValueError("expression_intensity must be an integer")
    confidence = float(value["confidence"])
    visible_cues = str(value["visible_cues"]).strip()
    if expression_class not in EXPRESSION_CLASSES:
        raise ValueError("invalid expression_class")
    if expression_fine not in FINE_EXPRESSIONS:
        raise ValueError("invalid expression_fine")
    if not 0 <= expression_intensity <= 3:
        raise ValueError("expression_intensity must be in [0, 3]")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if not visible_cues or len(visible_cues) > 300:
        raise ValueError("visible_cues must contain 1 to 300 characters")
    if expression_class == "ambiguous" and expression_fine != "other":
        raise ValueError("ambiguous annotations must use fine label other")
    if expression_fine == "happy" and expression_class != "positive":
        raise ValueError("happy fine label must use positive class")
    if expression_fine == "neutral" and expression_class != "neutral":
        raise ValueError("neutral fine label must use neutral class")
    if expression_fine in {"sad", "angry", "fear", "disgust"} and expression_class != "negative":
        raise ValueError("negative fine label must use negative class")
    if expression_class in {"positive", "negative"} and expression_intensity == 0:
        raise ValueError("positive and negative classes require nonzero intensity")
    if expression_class == "neutral" and expression_intensity > 1:
        raise ValueError("neutral class intensity must be 0 or 1")
    return {
        "expression_class": expression_class,
        "expression_fine": expression_fine,
        "expression_intensity": expression_intensity,
        "confidence": confidence,
        "visible_cues": visible_cues,
    }
