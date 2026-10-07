from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .candidate_pairs import METADATA_REQUIRED, validate_expression_pair_provenance
from .constants import TASK_IAT, TASK_PROFILE, TASK_TYPES
from .pair_policy import (
    IDENTITY_POLICY_VERSION,
    identity_hits,
    parse_iat_response,
    profile_word_count,
)
from .prompts import PROMPT_BY_TASK

GENERATED_RESPONSE_POLICY_VERSION = (
    "august3-generated-response-v2:" + IDENTITY_POLICY_VERSION
)


@dataclass(frozen=True)
class GenerationJob:
    response_id: str
    image_id: str
    image_path: str
    image_sha256: str
    task_type: str
    prompt: str
    sample_index: int
    generation_seed: int


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def complete_match_subset(
    metadata_rows: Iterable[Mapping[str, Any]],
    max_images: int,
) -> list[dict[str, Any]]:
    metadata = [dict(row) for row in metadata_rows]
    if max_images <= 0 or max_images >= len(metadata):
        return metadata
    if max_images % 2:
        raise ValueError(
            "max_images must be even so race-matched blocks remain complete"
        )
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    block_order: list[str] = []
    for row in metadata:
        match_id = _text(row, "sampling_match_id")
        if not match_id:
            raise ValueError("metadata row is missing sampling_match_id")
        if match_id not in blocks:
            block_order.append(match_id)
        blocks[match_id].append(row)
    for match_id, block in blocks.items():
        races = {_text(row, "fairface_race") for row in block}
        if len(block) != 2 or races != {"Black", "White"}:
            raise ValueError(f"incomplete race-matched metadata block: {match_id}")
    selected = [
        row for match_id in block_order[: max_images // 2] for row in blocks[match_id]
    ]
    if len(selected) != max_images:
        raise ValueError("metadata does not contain enough complete blocks")
    return selected


def select_complete_matches(
    metadata_rows: Iterable[Mapping[str, Any]],
    match_ids: Iterable[str],
) -> list[dict[str, Any]]:
    metadata = [dict(row) for row in metadata_rows]
    requested = [str(match_id).strip() for match_id in match_ids]
    if not requested or any(not match_id for match_id in requested):
        raise ValueError("at least one nonempty sampling match ID is required")
    if len(requested) != len(set(requested)):
        raise ValueError("sampling match IDs must be unique")

    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        blocks[_text(row, "sampling_match_id")].append(row)
    missing = [match_id for match_id in requested if match_id not in blocks]
    if missing:
        raise ValueError(f"sampling match IDs are absent from metadata: {missing}")

    selected = []
    for match_id in requested:
        block = blocks[match_id]
        races = {_text(row, "fairface_race") for row in block}
        if len(block) != 2 or races != {"Black", "White"}:
            raise ValueError(f"incomplete race-matched metadata block: {match_id}")
        selected.extend(block)
    return selected


def generated_response_issues(task_type: str, response: str) -> list[str]:
    text = response.strip()
    if not text:
        return ["empty_response"]
    if task_type == TASK_IAT:
        parsed = parse_iat_response(text)
        issues = []
        if not parsed["valid"]:
            issues.append("not_exact_three_word_iat")
        elif parsed["duplicate_count"]:
            issues.append("duplicate_iat_word")
        return issues
    if task_type == TASK_PROFILE:
        issues = []
        words = profile_word_count(text)
        if not 80 <= words <= 140:
            issues.append("profile_length_out_of_range")
        if identity_hits(text):
            issues.append("identity_leakage")
        if not re.search(r"\bthe\s+person\b", text, flags=re.IGNORECASE):
            issues.append("missing_subject_reference")
        return issues
    return ["invalid_task_type"]


def build_generation_jobs(
    metadata_rows: Iterable[Mapping[str, Any]],
    *,
    iat_samples_per_image: int = 32,
    profile_samples_per_image: int = 2,
    seed: int = 8032026,
    tasks: set[str] | None = None,
    provider_prefix: str = "gemma",
) -> list[GenerationJob]:
    metadata = [dict(row) for row in metadata_rows]
    if not metadata:
        raise ValueError("metadata is empty")
    missing = METADATA_REQUIRED - set(metadata[0])
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    tasks = tasks or set(TASK_TYPES)
    if not tasks or not tasks <= TASK_TYPES:
        raise ValueError(f"tasks must be a nonempty subset of {sorted(TASK_TYPES)}")
    sample_counts = {
        TASK_IAT: iat_samples_per_image,
        TASK_PROFILE: profile_samples_per_image,
    }
    if any(sample_counts[task] <= 0 for task in tasks):
        raise ValueError("sample counts must be positive for every requested task")
    if not re.fullmatch(r"[a-z0-9-]+", provider_prefix):
        raise ValueError(
            "provider_prefix must contain only lowercase letters, digits, and hyphens"
        )

    jobs = []
    seen_images: set[str] = set()
    seen_hashes: set[str] = set()
    for row in metadata:
        image_id = _text(row, "image_id")
        digest = _text(row, "image_sha256").lower()
        if not image_id or image_id in seen_images:
            raise ValueError(f"image IDs must be nonempty and unique: {image_id!r}")
        if not digest or digest in seen_hashes:
            raise ValueError(f"image hashes must be nonempty and unique: {digest!r}")
        validate_expression_pair_provenance(row, image_id=image_id)
        seen_images.add(image_id)
        seen_hashes.add(digest)
        for task_type in sorted(tasks):
            for sample_index in range(sample_counts[task_type]):
                response_id = (
                    f"{provider_prefix}-{task_type}-{image_id}-{sample_index:03d}"
                )
                jobs.append(
                    GenerationJob(
                        response_id=response_id,
                        image_id=image_id,
                        image_path=_text(row, "image_path"),
                        image_sha256=digest,
                        task_type=task_type,
                        prompt=PROMPT_BY_TASK[task_type],
                        sample_index=sample_index,
                        generation_seed=stable_seed(seed, response_id),
                    )
                )
    return jobs
