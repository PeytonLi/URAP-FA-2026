from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .constants import (
    EXPRESSION_CLASSES,
    EXPRESSION_PAIR_AUDIT_VERSION,
    EXPRESSION_PAIR_MAXIMUM_CUE_DIFFERENCE,
    EXPRESSION_PAIR_MINIMUM_CONFIDENCE,
    TARGET_GENDERS,
    TARGET_RACES,
    TASK_IAT,
    TASK_PROFILE,
    TASK_TYPES,
)
from .pair_policy import parse_iat_response, prompt_metadata_hits
from .prompts import PROMPT_BY_TASK

METADATA_REQUIRED = {
    "image_id",
    "image_path",
    "image_sha256",
    "sampling_match_id",
    "fairface_race",
    "fairface_gender",
    "age_bucket",
    "expression_class",
    "expression_fine",
    "expression_intensity",
    "expression_pair_equivalent",
    "expression_pair_min_confidence",
    "expression_pair_max_cue_difference",
    "expression_pair_audit_version",
}

RESPONSE_REQUIRED = {
    "response_id",
    "image_sha256",
    "task_type",
    "prompt",
    "response",
    "response_provenance",
}

AGE_BUCKET_ORDER = {
    "0-2": 0,
    "3-9": 1,
    "10-19": 2,
    "20-29": 3,
    "30-39": 4,
    "40-49": 5,
    "50-59": 6,
    "60-69": 7,
    "70+": 8,
}


@dataclass(frozen=True)
class CandidatePolicy:
    max_profile_pairs_per_image: int = 2
    max_iat_contrasts_per_match: int = 3
    require_distinct_profile_provenance: bool = True
    include_controlled_iat: bool = True
    max_controlled_iat_matches: int | None = None
    seed: str = "august3-candidates-v1"

    def __post_init__(self) -> None:
        if (
            self.max_profile_pairs_per_image <= 0
            or self.max_iat_contrasts_per_match < 0
        ):
            raise ValueError(
                "candidate limits must be positive (or zero for observed IAT)"
            )
        if (
            self.max_controlled_iat_matches is not None
            and self.max_controlled_iat_matches <= 0
        ):
            raise ValueError("max_controlled_iat_matches must be positive when set")


@dataclass
class CandidateBuildResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _stable_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _validate_columns(
    rows: list[dict[str, Any]], required: set[str], label: str
) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def validate_expression_pair_provenance(
    row: Mapping[str, Any], *, image_id: str = "<unknown>"
) -> None:
    if _text(row, "expression_pair_equivalent").lower() != "true":
        raise ValueError(f"{image_id}: expression pair was not marked equivalent")
    if _text(row, "expression_pair_audit_version") != EXPRESSION_PAIR_AUDIT_VERSION:
        raise ValueError(
            f"{image_id}: expression pair audit version is missing or stale"
        )
    try:
        confidence = float(_text(row, "expression_pair_min_confidence"))
        cue_difference = float(_text(row, "expression_pair_max_cue_difference"))
    except ValueError as exc:
        raise ValueError(
            f"{image_id}: expression pair audit scores are invalid"
        ) from exc
    if (
        not math.isfinite(confidence)
        or confidence < EXPRESSION_PAIR_MINIMUM_CONFIDENCE
        or confidence > 1.0
    ):
        raise ValueError(f"{image_id}: expression pair confidence is below policy")
    if (
        not math.isfinite(cue_difference)
        or cue_difference < 0
        or cue_difference > EXPRESSION_PAIR_MAXIMUM_CUE_DIFFERENCE
    ):
        raise ValueError(f"{image_id}: expression pair cue difference exceeds policy")


def _validate_metadata(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _validate_columns(rows, METADATA_REQUIRED, "metadata")
    by_hash: dict[str, dict[str, Any]] = {}
    image_ids: set[str] = set()
    for row in rows:
        image_id = _text(row, "image_id")
        digest = _text(row, "image_sha256").lower()
        if not image_id or not digest:
            raise ValueError("metadata image_id and image_sha256 must be nonempty")
        if image_id in image_ids or digest in by_hash:
            raise ValueError(f"duplicate metadata image identity: {image_id}")
        if _text(row, "fairface_race") not in TARGET_RACES:
            raise ValueError(f"{image_id}: only Black and White rows are accepted")
        if _text(row, "fairface_gender") not in TARGET_GENDERS:
            raise ValueError(f"{image_id}: invalid FairFace gender label")
        if _text(row, "expression_class").lower() not in EXPRESSION_CLASSES:
            raise ValueError(f"{image_id}: ambiguous or invalid expression class")
        try:
            intensity_value = float(_text(row, "expression_intensity"))
            intensity = int(intensity_value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{image_id}: invalid expression intensity") from exc
        if (
            not math.isfinite(intensity_value)
            or intensity_value != intensity
            or not 0 <= intensity <= 3
        ):
            raise ValueError(f"{image_id}: invalid expression intensity")
        validate_expression_pair_provenance(row, image_id=image_id)
        image_ids.add(image_id)
        by_hash[digest] = row

    matches: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        matches[_text(row, "sampling_match_id")].append(row)
    for match_id, match_rows in matches.items():
        if not match_id or len(match_rows) != 2:
            raise ValueError(
                f"sampling match {match_id!r} must contain exactly two images"
            )
        if {_text(row, "fairface_race") for row in match_rows} != {"Black", "White"}:
            raise ValueError(
                f"sampling match {match_id!r} must contain Black and White"
            )
        covariate_signatures = {
            (
                _text(row, "fairface_gender"),
                _text(row, "expression_class").lower(),
                _text(row, "expression_fine").lower(),
                _text(row, "expression_pair_equivalent").lower(),
                _text(row, "expression_pair_min_confidence"),
                _text(row, "expression_pair_max_cue_difference"),
                _text(row, "expression_pair_audit_version"),
            )
            for row in match_rows
        }
        if len(covariate_signatures) != 1:
            raise ValueError(f"sampling match {match_id!r} has mismatched strata")
        age_indices = [AGE_BUCKET_ORDER.get(_text(row, "age_bucket")) for row in match_rows]
        if None in age_indices or abs(age_indices[0] - age_indices[1]) > 1:
            raise ValueError(
                f"sampling match {match_id!r} exceeds adjacent age-bucket policy"
            )
        intensities = [int(float(_text(row, "expression_intensity"))) for row in match_rows]
        maximum_difference = int(
            float(_text(match_rows[0], "expression_pair_max_cue_difference"))
        )
        if abs(intensities[0] - intensities[1]) > maximum_difference:
            raise ValueError(
                f"sampling match {match_id!r} exceeds expression-intensity policy"
            )
    return by_hash


def _validate_responses(
    rows: list[dict[str, Any]],
    metadata_by_hash: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    _validate_columns(rows, RESPONSE_REQUIRED, "responses")
    response_ids: set[str] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        response_id = _text(row, "response_id")
        digest = _text(row, "image_sha256").lower()
        task_type = _text(row, "task_type")
        if not response_id or response_id in response_ids:
            raise ValueError(
                f"response IDs must be nonempty and unique: {response_id!r}"
            )
        if digest not in metadata_by_hash:
            raise ValueError(f"{response_id}: response image is absent from metadata")
        if task_type not in TASK_TYPES:
            raise ValueError(f"{response_id}: invalid task type {task_type!r}")
        if _text(row, "prompt") != PROMPT_BY_TASK[task_type]:
            raise ValueError(
                f"{response_id}: prompt differs from the frozen task prompt"
            )
        if prompt_metadata_hits(_text(row, "prompt")):
            raise ValueError(f"{response_id}: prompt exposes protected metadata")
        if not _text(row, "response") or not _text(row, "response_provenance"):
            raise ValueError(f"{response_id}: response and provenance are required")
        response_ids.add(response_id)
        grouped[(digest, task_type)].append(row)
    return grouped


def _candidate_sides(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    pair_id: str,
    seed: str,
) -> dict[str, str]:
    if _stable_rank(seed, pair_id).endswith(tuple("02468ace")):
        first, second = left, right
    else:
        first, second = right, left
    return {
        "candidate_1": _text(first, "response"),
        "candidate_2": _text(second, "response"),
        "candidate_1_provenance": _text(first, "response_provenance"),
        "candidate_2_provenance": _text(second, "response_provenance"),
        "candidate_1_response_id": _text(first, "response_id"),
        "candidate_2_response_id": _text(second, "response_id"),
    }


def _common_fields(metadata: Mapping[str, Any], task_type: str) -> dict[str, Any]:
    return {
        "image_path": _text(metadata, "image_path"),
        "image_sha256": _text(metadata, "image_sha256").lower(),
        "split": "train",
        "task_type": task_type,
        "prompt": PROMPT_BY_TASK[task_type],
        "fairface_race": _text(metadata, "fairface_race"),
        "fairface_gender": _text(metadata, "fairface_gender"),
        "age_bucket": _text(metadata, "age_bucket"),
        "expression_class": _text(metadata, "expression_class").lower(),
        "expression_fine": _text(metadata, "expression_fine").lower(),
        "expression_intensity": _text(metadata, "expression_intensity"),
        "expression_confidence": _text(metadata, "expression_confidence"),
        "expression_pair_equivalent": _text(
            metadata, "expression_pair_equivalent"
        ).lower(),
        "expression_pair_min_confidence": _text(
            metadata, "expression_pair_min_confidence"
        ),
        "expression_pair_max_cue_difference": _text(
            metadata, "expression_pair_max_cue_difference"
        ),
        "expression_pair_audit_version": _text(
            metadata, "expression_pair_audit_version"
        ),
        "sampling_match_id": _text(metadata, "sampling_match_id"),
        "source_image_id": _text(metadata, "image_id"),
    }


def _deduplicate_responses(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (_text(row, "response"), _text(row, "response_provenance"))
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _build_profile_candidates(
    metadata_rows: list[dict[str, Any]],
    responses: Mapping[tuple[str, str], list[dict[str, Any]]],
    policy: CandidatePolicy,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for metadata in metadata_rows:
        digest = _text(metadata, "image_sha256").lower()
        profile_rows = _deduplicate_responses(responses.get((digest, TASK_PROFILE), []))
        possible = []
        for left_index, left in enumerate(profile_rows):
            for right in profile_rows[left_index + 1 :]:
                if _text(left, "response") == _text(right, "response"):
                    continue
                if policy.require_distinct_profile_provenance and _text(
                    left, "response_provenance"
                ) == _text(right, "response_provenance"):
                    continue
                raw_id = "|".join(
                    sorted((_text(left, "response_id"), _text(right, "response_id")))
                )
                pair_id = (
                    "profile-" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:20]
                )
                possible.append((pair_id, left, right))
        possible.sort(key=lambda item: _stable_rank(policy.seed, item[0]))
        for pair_id, left, right in possible[: policy.max_profile_pairs_per_image]:
            candidates.append(
                {
                    "pair_id": pair_id,
                    "match_block_id": f"provisional-profile-{_text(metadata, 'image_id')}",
                    **_common_fields(metadata, TASK_PROFILE),
                    **_candidate_sides(left, right, pair_id=pair_id, seed=policy.seed),
                }
            )
    return candidates


def _canonical_iat_by_tuple(
    rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        parsed = parse_iat_response(_text(row, "response"))
        if not parsed["valid"] or parsed["duplicate_count"]:
            continue
        canonical = ", ".join(word.title() for word in parsed["words"])
        normalized = dict(row)
        normalized["response"] = canonical
        current = result.get(parsed["word_tuple"])
        if current is None or _text(normalized, "response_id") < _text(
            current, "response_id"
        ):
            result[parsed["word_tuple"]] = normalized
    return result


def _build_iat_candidates(
    metadata_rows: list[dict[str, Any]],
    responses: Mapping[tuple[str, str], list[dict[str, Any]]],
    policy: CandidatePolicy,
) -> list[dict[str, Any]]:
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        by_match[_text(row, "sampling_match_id")].append(row)

    candidates: list[dict[str, Any]] = []
    for match_id, match_rows in sorted(by_match.items()):
        first, second = sorted(match_rows, key=lambda row: _text(row, "fairface_race"))
        first_pool = _canonical_iat_by_tuple(
            responses.get((_text(first, "image_sha256").lower(), TASK_IAT), [])
        )
        second_pool = _canonical_iat_by_tuple(
            responses.get((_text(second, "image_sha256").lower(), TASK_IAT), [])
        )
        shared = sorted(set(first_pool) & set(second_pool))
        contrasts = []
        for low_tuple in shared:
            low_count = parse_iat_response(first_pool[low_tuple]["response"])[
                "negative_count"
            ]
            for high_tuple in shared:
                high_count = parse_iat_response(first_pool[high_tuple]["response"])[
                    "negative_count"
                ]
                if low_count >= high_count:
                    continue
                contrast_id = f"{match_id}|{low_tuple}|{high_tuple}"
                contrasts.append(
                    (contrast_id, low_tuple, high_tuple, high_count - low_count)
                )
        contrasts.sort(
            key=lambda item: (
                -item[3],
                _stable_rank(policy.seed, item[0]),
            )
        )
        for contrast_index, (contrast_id, low_tuple, high_tuple, _) in enumerate(
            contrasts[: policy.max_iat_contrasts_per_match]
        ):
            provisional_id = (
                "provisional-iat-"
                + hashlib.sha256(contrast_id.encode("utf-8")).hexdigest()[:16]
            )
            for metadata, pool in ((first, first_pool), (second, second_pool)):
                pair_id = f"iat-{hashlib.sha256((contrast_id + '|' + _text(metadata, 'image_id')).encode('utf-8')).hexdigest()[:20]}"
                candidates.append(
                    {
                        "pair_id": pair_id,
                        "match_block_id": provisional_id,
                        **_common_fields(metadata, TASK_IAT),
                        **_candidate_sides(
                            pool[low_tuple],
                            pool[high_tuple],
                            pair_id=f"{provisional_id}-{contrast_index}",
                            seed=policy.seed,
                        ),
                    }
                )
    return candidates


CONTROLLED_IAT_CONTRASTS = {
    "positive": (
        ("Smiling, Happy, Joyful", "Smiling, Happy, Sadness"),
        ("Cheerful, Glad, Delight", "Cheerful, Glad, Grief"),
        ("Laughing, Joyous, Pleasure", "Laughing, Joyous, Pain"),
    ),
    "neutral": (
        ("Friend, Pleasing, Excellent", "Friend, Pleasing, Pain"),
        ("Friendship, Appealing, Glad", "Friendship, Appealing, Sadness"),
        ("Cheer, Delight, Pleasing", "Cheer, Delight, Grief"),
    ),
    "sad": (
        ("Sadness, Friend, Pleasing", "Sadness, Grief, Pleasing"),
        ("Grief, Friend, Appealing", "Grief, Pain, Appealing"),
        ("Sadness, Friendship, Glad", "Sadness, Tragic, Glad"),
    ),
    "angry": (
        ("Angry, Friend, Pleasing", "Angry, Annoy, Pleasing"),
        ("Annoy, Friend, Appealing", "Annoy, Hate, Appealing"),
        ("Angry, Friendship, Glad", "Angry, Detest, Glad"),
    ),
    "disgust": (
        ("Disgust, Friend, Pleasing", "Disgust, Nasty, Pleasing"),
        ("Disgust, Friendship, Glad", "Disgust, Gross, Glad"),
        ("Nasty, Friend, Appealing", "Nasty, Hate, Appealing"),
    ),
    "pain": (
        ("Pain, Friend, Pleasing", "Pain, Hurtful, Pleasing"),
        ("Pain, Friendship, Glad", "Pain, Tragic, Glad"),
        ("Hurtful, Friend, Appealing", "Hurtful, Pain, Appealing"),
    ),
    "fear": (
        ("Pain, Friend, Pleasing", "Pain, Horrible, Pleasing"),
        ("Negative, Friendship, Glad", "Negative, Horrific, Glad"),
        ("Awful, Friend, Appealing", "Awful, Disaster, Appealing"),
    ),
    "negative": (
        ("Negative, Friend, Pleasing", "Negative, Horrible, Pleasing"),
        ("Awful, Friendship, Glad", "Awful, Tragic, Glad"),
        ("Pain, Friend, Appealing", "Pain, Horrific, Appealing"),
    ),
}


def _controlled_contrast_key(metadata: Mapping[str, Any]) -> str:
    fine = _text(metadata, "expression_fine").lower()
    aliases = {
        "anger": "angry",
        "anxiety": "fear",
        "worried": "fear",
        "happiness": "positive",
        "happy": "positive",
    }
    key = aliases.get(fine, fine)
    if key in CONTROLLED_IAT_CONTRASTS:
        return key
    expression_class = _text(metadata, "expression_class").lower()
    return (
        expression_class
        if expression_class in {"positive", "neutral", "negative"}
        else "neutral"
    )


def _build_controlled_iat_candidates(
    metadata_rows: list[dict[str, Any]],
    policy: CandidatePolicy,
) -> list[dict[str, Any]]:
    by_match: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata_rows:
        by_match[_text(row, "sampling_match_id")].append(row)
    match_ids = sorted(
        by_match,
        key=lambda match_id: _stable_rank(policy.seed + "-controlled-iat", match_id),
    )
    if policy.max_controlled_iat_matches is not None:
        match_ids = match_ids[: policy.max_controlled_iat_matches]

    candidates = []
    for match_id in match_ids:
        match_rows = by_match[match_id]
        if len(match_rows) != 2:
            continue
        key = _controlled_contrast_key(match_rows[0])
        contrasts = CONTROLLED_IAT_CONTRASTS[key]
        contrast_index = int(
            _stable_rank(policy.seed + "-controlled-iat-tuple", match_id)[:8], 16
        ) % len(contrasts)
        chosen_text, rejected_text = contrasts[contrast_index]
        template_id = f"controlled-iat-v1-{key}-{contrast_index + 1}"
        provisional_id = (
            "provisional-"
            + hashlib.sha256(f"{match_id}|{template_id}".encode("utf-8")).hexdigest()[
                :20
            ]
        )
        chosen_response = {
            "response": chosen_text,
            "response_provenance": "controlled_race_blind_iat_v1",
            "response_id": f"{provisional_id}-lower-negative",
        }
        rejected_response = {
            "response": rejected_text,
            "response_provenance": "controlled_race_blind_iat_v1",
            "response_id": f"{provisional_id}-higher-negative",
        }
        for metadata in sorted(match_rows, key=lambda row: _text(row, "fairface_race")):
            pair_id = (
                "iat-controlled-"
                + hashlib.sha256(
                    f"{provisional_id}|{_text(metadata, 'image_id')}".encode("utf-8")
                ).hexdigest()[:20]
            )
            candidates.append(
                {
                    "pair_id": pair_id,
                    "match_block_id": provisional_id,
                    **_common_fields(metadata, TASK_IAT),
                    **_candidate_sides(
                        chosen_response,
                        rejected_response,
                        pair_id=provisional_id,
                        seed=policy.seed,
                    ),
                    "candidate_construction": "prespecified_race_blind_expression_template",
                    "controlled_template_id": template_id,
                }
            )
    return candidates


def build_candidate_pairs(
    metadata_rows: Iterable[Mapping[str, Any]],
    response_rows: Iterable[Mapping[str, Any]],
    *,
    policy: CandidatePolicy | None = None,
) -> CandidateBuildResult:
    policy = policy or CandidatePolicy()
    metadata = [dict(row) for row in metadata_rows]
    raw_responses = [dict(row) for row in response_rows]
    metadata_by_hash = _validate_metadata(metadata)
    responses = _validate_responses(raw_responses, metadata_by_hash)

    profile = _build_profile_candidates(metadata, responses, policy)
    observed_iat = _build_iat_candidates(metadata, responses, policy)
    controlled_iat = (
        _build_controlled_iat_candidates(metadata, policy)
        if policy.include_controlled_iat
        else []
    )
    iat = observed_iat + controlled_iat
    rows = profile + iat
    pair_counts = Counter(_text(row, "pair_id") for row in rows)
    duplicates = [pair_id for pair_id, count in pair_counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"candidate pair IDs collided: {duplicates[:3]}")
    rows.sort(key=lambda row: (_text(row, "task_type"), _text(row, "pair_id")))

    images_by_task: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        images_by_task[_text(row, "task_type")].add(_text(row, "image_sha256"))
    report = {
        "metadata_images": len(metadata),
        "response_rows": len(raw_responses),
        "candidate_pairs": len(rows),
        "candidate_pairs_by_task": dict(
            Counter(_text(row, "task_type") for row in rows)
        ),
        "iat_candidate_pairs_by_construction": {
            "observed_shared_tuple": len(observed_iat),
            "controlled_race_blind": len(controlled_iat),
        },
        "candidate_images_by_task": {
            task: len(hashes) for task, hashes in sorted(images_by_task.items())
        },
        "profile_pairs_require_distinct_provenance": policy.require_distinct_profile_provenance,
        "iat_construction": (
            "shared exact observed responses plus prespecified demographic-metadata-agnostic "
            "expression-stratified "
            "direct-negative-count contrasts; every candidate remains subject to four-condition judging"
        ),
        "race_metadata_in_prompt": False,
    }
    return CandidateBuildResult(rows=rows, report=report)
