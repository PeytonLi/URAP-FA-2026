from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PROFILE_EXACT_FIELDS = (
    "chosen_visible_expression_support",
    "rejected_visible_expression_support",
    "chosen_person_negative_count",
    "rejected_person_negative_count",
    "chosen_unwarranted_negative_count",
    "rejected_unwarranted_negative_count",
    "chosen_unsupported_trait_count",
    "rejected_unsupported_trait_count",
    "chosen_identity_leakage_count",
    "rejected_identity_leakage_count",
    "chosen_invented_biography_count",
    "rejected_invented_biography_count",
)

BASE_FIELDS = (
    "task_type",
    "fairface_gender",
    "age_bucket",
    "expression_class",
    "expression_fine",
    "expression_intensity",
    "expression_pair_equivalent",
    "expression_pair_min_confidence",
    "expression_pair_max_cue_difference",
    "expression_pair_audit_version",
    "objective_type",
)


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = _text(row, key)
    return int(float(value)) if value else 0


def _tuple(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z]+", text.lower()))


def _current_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    base = tuple(_text(row, field).lower() for field in BASE_FIELDS)
    if _text(row, "task_type") == "iat_3word":
        return (
            *base,
            _tuple(_text(row, "chosen")),
            _tuple(_text(row, "rejected")),
            _integer(row, "chosen_unwarranted_negative_count"),
            _integer(row, "rejected_unwarranted_negative_count"),
            _integer(row, "chosen_visible_expression_support"),
            _integer(row, "rejected_visible_expression_support"),
        )
    return base + tuple(_integer(row, field) for field in PROFILE_EXACT_FIELDS)


def _original_pair_key(row: Mapping[str, Any], *, require_objective: bool) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        _text(row, "sampling_match_id"),
        _text(row, "task_type"),
    )
    if require_objective:
        key += (_text(row, "objective_type"),)
    if _text(row, "task_type") == "iat_3word":
        key += (_tuple(_text(row, "chosen")), _tuple(_text(row, "rejected")))
    return key


def _count_blocks(
    rows: list[dict[str, str]],
    key_function: Any,
) -> tuple[int, Counter[str]]:
    groups: dict[tuple[Any, ...], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: {"Black": [], "White": []}
    )
    for row in rows:
        race = _text(row, "fairface_race")
        if race in {"Black", "White"}:
            groups[key_function(row)][race].append(row)

    task_counts: Counter[str] = Counter()
    total = 0
    for grouped in groups.values():
        count = min(len(grouped["Black"]), len(grouped["White"]))
        if count:
            task = _text((grouped["Black"] or grouped["White"])[0], "task_type")
            task_counts[task] += count
            total += count
    return total, task_counts


def analyze_selection(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    current_blocks, current_tasks = _count_blocks(materialized, _current_signature)
    bilateral_blocks, bilateral_tasks = _count_blocks(
        materialized,
        lambda row: _original_pair_key(row, require_objective=False),
    )
    structural_blocks, structural_tasks = _count_blocks(
        materialized,
        lambda row: _original_pair_key(row, require_objective=True),
    )
    row_counts = Counter(_text(row, "fairface_race") for row in materialized)
    task_rows = Counter(_text(row, "task_type") for row in materialized)

    return {
        "protocol_version": "august10-selection-bottleneck-analysis-v1",
        "eligible_rows": len(materialized),
        "eligible_rows_by_race": dict(sorted(row_counts.items())),
        "eligible_rows_by_task": dict(sorted(task_rows.items())),
        "current_exact_signature": {
            "blocks": current_blocks,
            "rows": current_blocks * 2,
            "blocks_by_task": dict(sorted(current_tasks.items())),
        },
        "original_match_bilateral_ceiling": {
            "blocks": bilateral_blocks,
            "rows": bilateral_blocks * 2,
            "blocks_by_task": dict(sorted(bilateral_tasks.items())),
        },
        "recommended_structural_match": {
            "description": (
                "Preserve sampling_match_id and require the same task/objective; "
                "IAT rows must also share chosen and rejected tuples."
            ),
            "blocks": structural_blocks,
            "rows": structural_blocks * 2,
            "blocks_by_task": dict(sorted(structural_tasks.items())),
        },
        "recoverable_blocks_without_new_judge_calls": structural_blocks - current_blocks,
        "retention_fraction_current_vs_structural": (
            current_blocks / structural_blocks if structural_blocks else 0.0
        ),
        "conclusion": (
            "No efficacy training is authorized from this micro pool. Replace exact matching "
            "on noisy judge scores with original-pair structural matching, then audit enough "
            "additional candidates to satisfy the 100-row pilot gate."
        ),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
