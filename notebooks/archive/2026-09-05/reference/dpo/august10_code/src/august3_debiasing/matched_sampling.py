from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .constants import EXPRESSION_CLASSES, TARGET_GENDERS, TARGET_RACES


CORE_FIELDS = {
    "image_id",
    "image_path",
    "image_sha256",
    "fairface_race",
    "fairface_gender",
    "age_bucket",
    "expression_class",
    "expression_fine",
    "expression_intensity",
    "expression_confidence",
}


@dataclass(frozen=True)
class SamplingPolicy:
    target_images: int = 1600
    minimum_expression_confidence: float = 0.70
    seed: str = "august3-sampling-v2-intensity"

    def __post_init__(self) -> None:
        if self.target_images <= 0 or self.target_images % 2:
            raise ValueError("target_images must be a positive even number")
        if not 0.0 <= self.minimum_expression_confidence <= 1.0:
            raise ValueError("minimum_expression_confidence must be in [0, 1]")


@dataclass
class SamplingResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return len(self.rows) == self.report.get("target_images")


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _stable_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _stratum(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        _text(row, "fairface_gender"),
        _text(row, "age_bucket"),
        _text(row, "expression_class").lower(),
        _text(row, "expression_fine").lower(),
        _text(row, "expression_intensity"),
    )


def build_matched_sample(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy: SamplingPolicy | None = None,
    forbidden_hashes: set[str] | None = None,
) -> SamplingResult:
    policy = policy or SamplingPolicy()
    forbidden_hashes = {digest.lower() for digest in (forbidden_hashes or set())}
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("annotated image pool is empty")
    missing = CORE_FIELDS - set(materialized[0])
    if missing:
        raise ValueError(f"annotated image pool is missing columns: {sorted(missing)}")

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    excluded = Counter()
    eligible: list[dict[str, Any]] = []
    for row in materialized:
        image_id = _text(row, "image_id")
        digest = _text(row, "image_sha256").lower()
        if not image_id or image_id in seen_ids:
            raise ValueError(f"image IDs must be nonempty and unique: {image_id!r}")
        if not digest or digest in seen_hashes:
            raise ValueError(f"image hashes must be nonempty and unique: {digest!r}")
        seen_ids.add(image_id)
        seen_hashes.add(digest)
        if digest in forbidden_hashes:
            excluded["frozen_evaluation_overlap"] += 1
            continue
        if _text(row, "fairface_race") not in TARGET_RACES:
            excluded["non_target_race"] += 1
            continue
        if _text(row, "fairface_gender") not in TARGET_GENDERS:
            excluded["invalid_gender_label"] += 1
            continue
        expression_class = _text(row, "expression_class").lower()
        if expression_class not in EXPRESSION_CLASSES:
            excluded["ambiguous_expression"] += 1
            continue
        try:
            confidence = float(_text(row, "expression_confidence"))
        except ValueError:
            excluded["invalid_expression_confidence"] += 1
            continue
        if not math.isfinite(confidence) or confidence < policy.minimum_expression_confidence:
            excluded["low_expression_confidence"] += 1
            continue
        try:
            intensity_value = float(_text(row, "expression_intensity"))
            intensity = int(intensity_value)
        except ValueError:
            excluded["invalid_expression_intensity"] += 1
            continue
        if (
            not math.isfinite(intensity_value)
            or intensity_value != intensity
            or not 0 <= intensity <= 3
        ):
            excluded["invalid_expression_intensity"] += 1
            continue
        eligible.append(row)

    grouped: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"Black": [], "White": []}
    )
    for row in eligible:
        grouped[_stratum(row)][_text(row, "fairface_race")].append(row)

    queues: dict[tuple[str, ...], deque[tuple[dict[str, Any], dict[str, Any]]]] = {}
    available_pairs_by_stratum = {}
    for stratum, race_rows in grouped.items():
        black = sorted(
            race_rows["Black"],
            key=lambda row: _stable_rank(policy.seed, _text(row, "image_id")),
        )
        white = sorted(
            race_rows["White"],
            key=lambda row: _stable_rank(policy.seed, _text(row, "image_id")),
        )
        pairs = list(zip(black, white))
        if pairs:
            queues[stratum] = deque(pairs)
            available_pairs_by_stratum["|".join(stratum)] = len(pairs)

    stratum_order = sorted(
        queues,
        key=lambda value: _stable_rank(policy.seed, "|".join(value)),
    )
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    target_pairs = policy.target_images // 2
    while len(selected) < target_pairs:
        progress = False
        for stratum in stratum_order:
            if queues[stratum]:
                selected.append(queues[stratum].popleft())
                progress = True
                if len(selected) >= target_pairs:
                    break
        if not progress:
            break

    output: list[dict[str, Any]] = []
    for index, (black, white) in enumerate(selected, 1):
        raw = f"{_text(black, 'image_sha256')}|{_text(white, 'image_sha256')}"
        match_id = "aug3-sample-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        for row in (black, white):
            revised = dict(row)
            revised["sampling_match_id"] = match_id
            revised["sampling_order"] = index
            output.append(revised)
    output.sort(key=lambda row: (int(row["sampling_order"]), _text(row, "fairface_race")))

    report = {
        "input_images": len(materialized),
        "eligible_images": len(eligible),
        "target_images": policy.target_images,
        "selected_images": len(output),
        "selected_matches": len(selected),
        "unfilled_images": max(0, policy.target_images - len(output)),
        "excluded_counts": dict(excluded),
        "available_pairs_by_stratum": dict(sorted(available_pairs_by_stratum.items())),
        "selected_race_counts": dict(Counter(_text(row, "fairface_race") for row in output)),
        "selected_gender_counts": dict(Counter(_text(row, "fairface_gender") for row in output)),
        "selected_expression_counts": dict(
            Counter(_text(row, "expression_class").lower() for row in output)
        ),
        "selected_fine_expression_counts": dict(
            Counter(_text(row, "expression_fine").lower() for row in output)
        ),
        "selected_expression_intensity_counts": dict(
            Counter(_text(row, "expression_intensity") for row in output)
        ),
        "minimum_expression_confidence": policy.minimum_expression_confidence,
        "forbidden_hash_count": len(forbidden_hashes),
    }
    return SamplingResult(rows=output, report=report)
