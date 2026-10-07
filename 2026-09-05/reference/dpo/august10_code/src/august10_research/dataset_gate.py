from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

NEGATIVE_IAT_WORDS = {
    "abuse",
    "grief",
    "poison",
    "sadness",
    "pain",
    "despise",
    "failure",
    "nasty",
    "angry",
    "detest",
    "horrible",
    "negative",
    "ugly",
    "dirty",
    "gross",
    "evil",
    "rotten",
    "annoy",
    "disaster",
    "horrific",
    "scorn",
    "awful",
    "disgust",
    "hate",
    "humiliate",
    "selfish",
    "tragic",
    "bothersome",
    "hatred",
    "hurtful",
    "sick",
}

PROFILE_HARM_FIELDS = (
    "identity_leakage_count",
    "unwarranted_negative_count",
    "unsupported_trait_count",
    "invented_biography_count",
)

REQUIRED_COLUMNS = {
    "pair_id",
    "match_block_id",
    "sampling_match_id",
    "image_sha256",
    "split",
    "task_type",
    "fairface_race",
    "expression_class",
    "chosen",
    "rejected",
    "chosen_response_provenance",
    "objective_type",
}


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = _text(row, key)
    return int(float(value)) if value else 0


def _iat_tuple(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z]+", text.lower()))


def _negative_count(words: tuple[str, ...]) -> int:
    return sum(word in NEGATIVE_IAT_WORDS for word in words)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _check(
    checks: dict[str, dict[str, Any]],
    name: str,
    passed: bool,
    observed: Any,
    requirement: str,
) -> None:
    checks[name] = {
        "pass": bool(passed),
        "observed": observed,
        "requirement": requirement,
    }


def assess_dataset(
    rows: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    checks: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    columns = set(materialized[0]) if materialized else set()
    missing = sorted(REQUIRED_COLUMNS - columns)
    _check(checks, "required_columns", not missing, missing, "no required column may be missing")
    if missing or not materialized:
        return {
            "protocol_version": "august10-dataset-gate-v2",
            "pass": False,
            "decision": "do_not_train",
            "checks": checks,
            "issues": ["empty_dataset" if not materialized else "missing_required_columns"],
            "metrics": {"rows": len(materialized)},
        }

    split_rows = Counter(_text(row, "split").lower() for row in materialized)
    race_rows = Counter(_text(row, "fairface_race") for row in materialized)
    task_rows = Counter(_text(row, "task_type") for row in materialized)
    image_counts = Counter(_text(row, "image_sha256") for row in materialized)
    unique_images = len([value for value in image_counts if value])
    blocks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        blocks[_text(row, "match_block_id")].append(row)

    valid_blocks: list[list[dict[str, Any]]] = []
    invalid_blocks: dict[str, list[str]] = {}
    direct_iat_rows = 0
    iat_rows = 0
    profile_rows = 0
    valid_profile_rows = 0
    direct_profile_blocks = 0
    chosen_profile_hashes: Counter[str] = Counter()

    import hashlib

    for block_id, block_rows in blocks.items():
        block_issues: list[str] = []
        if not block_id:
            block_issues.append("missing_match_block_id")
        if len(block_rows) != 2:
            block_issues.append("block_must_have_two_rows")
        if {_text(row, "fairface_race") for row in block_rows} != {"Black", "White"}:
            block_issues.append("block_must_have_one_row_per_race")
        for field in ("split", "task_type", "objective_type", "sampling_match_id"):
            if len({_text(row, field) for row in block_rows}) != 1:
                block_issues.append(f"block_{field}_mismatch")
        if len({_text(row, "image_sha256") for row in block_rows}) != len(block_rows):
            block_issues.append("block_reuses_image")

        task = _text(block_rows[0], "task_type") if block_rows else ""
        if task == "iat_3word":
            chosen_tuples = {_iat_tuple(_text(row, "chosen")) for row in block_rows}
            rejected_tuples = {_iat_tuple(_text(row, "rejected")) for row in block_rows}
            if len(chosen_tuples) != 1 or len(rejected_tuples) != 1:
                block_issues.append("iat_target_tuple_differs_by_race")
            for row in block_rows:
                iat_rows += 1
                chosen = _iat_tuple(_text(row, "chosen"))
                rejected = _iat_tuple(_text(row, "rejected"))
                valid_format = len(chosen) == 3 == len(set(chosen)) and len(rejected) == 3 == len(
                    set(rejected)
                )
                if not valid_format:
                    block_issues.append("iat_invalid_three_distinct_word_format")
                elif _negative_count(chosen) < _negative_count(rejected):
                    direct_iat_rows += 1
                else:
                    block_issues.append("iat_not_direct_negative_count_reduction")
        elif task == "grounded_profile":
            for field in (
                "chosen_response_provenance",
                "rejected_response_provenance",
            ):
                if len({_text(row, field) for row in block_rows}) != 1:
                    block_issues.append(f"profile_block_{field}_mismatch")
            block_direct = True
            for row in block_rows:
                profile_rows += 1
                chosen = _text(row, "chosen")
                chosen_profile_hashes[hashlib.sha256(chosen.lower().encode()).hexdigest()] += 1
                length = _word_count(chosen)
                if not policy["profile_min_words"] <= length <= policy["profile_max_words"]:
                    block_issues.append("profile_chosen_length_invalid")
                chosen_harm = sum(_integer(row, f"chosen_{field}") for field in PROFILE_HARM_FIELDS)
                rejected_harm = sum(
                    _integer(row, f"rejected_{field}") for field in PROFILE_HARM_FIELDS
                )
                if chosen_harm != 0:
                    block_issues.append("profile_chosen_contains_mechanical_harm")
                if chosen_harm >= rejected_harm:
                    block_direct = False
                if (
                    chosen_harm == 0
                    and policy["profile_min_words"] <= length <= policy["profile_max_words"]
                ):
                    valid_profile_rows += 1
            if block_direct:
                direct_profile_blocks += 1
        else:
            block_issues.append("unknown_task_type")

        if block_issues:
            invalid_blocks[block_id or "<missing>"] = sorted(set(block_issues))
        else:
            valid_blocks.append(block_rows)

    block_counts = Counter(_text(rows[0], "split").lower() for rows in valid_blocks)
    task_block_counts = Counter(
        (_text(rows[0], "split").lower(), _text(rows[0], "task_type")) for rows in valid_blocks
    )
    negative_blocks = sum(
        _text(rows[0], "expression_class").lower() == "negative" for rows in valid_blocks
    )
    negative_iat_blocks = sum(
        _text(rows[0], "expression_class").lower() == "negative"
        and _text(rows[0], "task_type") == "iat_3word"
        for rows in valid_blocks
    )
    negative_profile_blocks = sum(
        _text(rows[0], "expression_class").lower() == "negative"
        and _text(rows[0], "task_type") == "grounded_profile"
        for rows in valid_blocks
    )
    negative_blocks_by_split = Counter(
        _text(rows[0], "split").lower()
        for rows in valid_blocks
        if _text(rows[0], "expression_class").lower() == "negative"
    )
    negative_iat_blocks_by_split = Counter(
        _text(rows[0], "split").lower()
        for rows in valid_blocks
        if _text(rows[0], "expression_class").lower() == "negative"
        and _text(rows[0], "task_type") == "iat_3word"
    )
    chosen_iat_tuples = Counter(
        _iat_tuple(_text(rows[0], "chosen"))
        for rows in valid_blocks
        if _text(rows[0], "task_type") == "iat_3word"
    )
    iat_blocks = sum(chosen_iat_tuples.values())
    tuple_cap = max(1, math.ceil(iat_blocks * policy["max_iat_chosen_tuple_fraction"]))
    controlled_rows = sum(
        "controlled_race_blind_iat" in _text(row, "chosen_response_provenance")
        for row in materialized
    )
    train_hashes = {
        _text(row, "image_sha256") for row in materialized if _text(row, "split").lower() == "train"
    }
    dev_hashes = {
        _text(row, "image_sha256")
        for row in materialized
        if _text(row, "split").lower() == "development"
    }
    duplicate_profiles = sum(count - 1 for count in chosen_profile_hashes.values() if count > 1)

    minimum_checks = (
        ("total_rows", len(materialized), policy["min_total_rows"]),
        ("train_rows", split_rows["train"], policy["min_train_rows"]),
        ("development_rows", split_rows["development"], policy["min_development_rows"]),
        ("unique_images", unique_images, policy["min_unique_images"]),
        ("total_blocks", len(valid_blocks), policy["min_total_blocks"]),
        ("train_blocks", block_counts["train"], policy["min_train_blocks"]),
        ("development_blocks", block_counts["development"], policy["min_development_blocks"]),
        (
            "train_iat_blocks",
            task_block_counts[("train", "iat_3word")],
            policy["min_train_iat_blocks"],
        ),
        (
            "train_profile_blocks",
            task_block_counts[("train", "grounded_profile")],
            policy["min_train_profile_blocks"],
        ),
        (
            "development_iat_blocks",
            task_block_counts[("development", "iat_3word")],
            policy["min_development_iat_blocks"],
        ),
        (
            "development_profile_blocks",
            task_block_counts[("development", "grounded_profile")],
            policy["min_development_profile_blocks"],
        ),
        ("negative_expression_blocks", negative_blocks, policy["min_negative_expression_blocks"]),
        ("negative_iat_blocks", negative_iat_blocks, policy.get("min_negative_iat_blocks", 0)),
        (
            "negative_profile_blocks",
            negative_profile_blocks,
            policy.get("min_negative_profile_blocks", 0),
        ),
        (
            "train_negative_expression_blocks",
            negative_blocks_by_split["train"],
            policy.get("min_train_negative_expression_blocks", 0),
        ),
        (
            "train_negative_iat_blocks",
            negative_iat_blocks_by_split["train"],
            policy.get("min_train_negative_iat_blocks", 0),
        ),
        (
            "development_negative_expression_blocks",
            negative_blocks_by_split["development"],
            policy.get("min_development_negative_expression_blocks", 0),
        ),
        (
            "development_negative_iat_blocks",
            negative_iat_blocks_by_split["development"],
            policy.get("min_development_negative_iat_blocks", 0),
        ),
        (
            "unique_iat_chosen_tuples",
            len(chosen_iat_tuples),
            policy["min_unique_iat_chosen_tuples"],
        ),
    )
    for name, observed, required in minimum_checks:
        _check(checks, name, observed >= required, observed, f">= {required}")

    _check(checks, "all_blocks_valid", not invalid_blocks, len(invalid_blocks), "0 invalid blocks")
    _check(
        checks,
        "equal_race_rows",
        race_rows["Black"] == race_rows["White"],
        dict(race_rows),
        "Black == White",
    )
    _check(
        checks,
        "one_row_per_image",
        max(image_counts.values(), default=0) <= policy["max_rows_per_image"],
        max(image_counts.values(), default=0),
        f"<= {policy['max_rows_per_image']}",
    )
    _check(
        checks,
        "train_development_image_disjoint",
        not (train_hashes & dev_hashes),
        len(train_hashes & dev_hashes),
        "0 shared image hashes",
    )
    _check(
        checks,
        "iat_direct_reduction_fraction",
        iat_rows > 0 and direct_iat_rows / iat_rows == 1.0,
        direct_iat_rows / iat_rows if iat_rows else 0.0,
        "1.0",
    )
    _check(
        checks,
        "iat_tuple_cap",
        max(chosen_iat_tuples.values(), default=0) <= tuple_cap,
        {"largest": max(chosen_iat_tuples.values(), default=0), "cap": tuple_cap},
        "largest tuple count <= rounded 15% cap",
    )
    controlled_fraction = controlled_rows / len(materialized)
    _check(
        checks,
        "controlled_iat_fraction",
        controlled_fraction <= policy["max_controlled_iat_fraction_of_all_rows"],
        controlled_fraction,
        f"<= {policy['max_controlled_iat_fraction_of_all_rows']}",
    )
    profile_blocks = (
        task_block_counts[("train", "grounded_profile")]
        + task_block_counts[("development", "grounded_profile")]
    )
    direct_profile_fraction = direct_profile_blocks / profile_blocks if profile_blocks else 0.0
    _check(
        checks,
        "profile_direct_harm_reduction_fraction",
        direct_profile_fraction >= policy["min_profile_direct_harm_reduction_fraction"],
        direct_profile_fraction,
        f">= {policy['min_profile_direct_harm_reduction_fraction']}",
    )
    _check(
        checks,
        "profile_chosen_unique",
        duplicate_profiles == 0,
        duplicate_profiles,
        "0 duplicate chosen profiles",
    )
    _check(
        checks,
        "profile_rows_mechanically_valid",
        valid_profile_rows == profile_rows,
        {"valid": valid_profile_rows, "total": profile_rows},
        "all profile rows",
    )

    failed = [name for name, result in checks.items() if not result["pass"]]
    if invalid_blocks:
        issues.append("invalid_match_blocks")
    issues.extend(failed)
    return {
        "protocol_version": "august10-dataset-gate-v2",
        "pass": not failed,
        "decision": "authorize_short_pilot" if not failed else "do_not_train",
        "checks": checks,
        "issues": sorted(set(issues)),
        "invalid_blocks": invalid_blocks,
        "metrics": {
            "rows": len(materialized),
            "rows_by_split": dict(sorted(split_rows.items())),
            "rows_by_race": dict(sorted(race_rows.items())),
            "rows_by_task": dict(sorted(task_rows.items())),
            "valid_blocks": len(valid_blocks),
            "valid_blocks_by_split": dict(sorted(block_counts.items())),
            "negative_blocks_by_split": dict(sorted(negative_blocks_by_split.items())),
            "negative_iat_blocks_by_split": dict(
                sorted(negative_iat_blocks_by_split.items())
            ),
            "iat_chosen_tuple_counts": {
                "|".join(key): value for key, value in sorted(chosen_iat_tuples.items())
            },
        },
    }
