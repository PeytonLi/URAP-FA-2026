from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .constants import TASK_IAT, TASK_PROFILE
from .pair_policy import (
    PairPolicy,
    parse_iat_response,
    profile_word_count,
    tokens,
    validate_pair,
)


@dataclass(frozen=True)
class DatasetPolicy:
    min_rows: int = 1000
    min_unique_images: int = 400
    min_iat_fraction: float = 0.25
    max_iat_fraction: float = 0.50
    min_iat_negative_count_reduction_fraction: float = 0.80
    max_iat_fixed_substitution_fraction: float = 0.20
    max_iat_chosen_tuple_fraction: float = 0.15
    iat_tuple_balance_min_rows: int = 20
    max_controlled_iat_fraction_of_all_rows: float = 0.30
    max_profile_block_word_count_ratio: float = 1.25
    max_profile_block_grounded_detail_difference: int = 1
    max_profile_block_quality_difference: float = 0.75
    max_profile_objective_fraction: float = 0.40
    profile_objective_balance_min_rows: int = 20
    max_profile_chosen_reuse: int = 1


@dataclass
class DatasetAudit:
    issues: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    invalid_pairs: dict[str, list[str]] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.issues and not self.invalid_pairs

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "issues": self.issues,
            "invalid_pairs": self.invalid_pairs,
            "summary": self.summary,
        }


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = _text(row, key)
    return int(float(value)) if value else 0


def _number(row: Mapping[str, Any], key: str) -> float:
    value = _text(row, key)
    return float(value) if value else 0.0


def _block_signature(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        _text(row, "split").lower(),
        _text(row, "task_type"),
        _text(row, "fairface_gender"),
        _text(row, "age_bucket"),
        _text(row, "expression_class").lower(),
        _text(row, "expression_fine").lower(),
        _text(row, "expression_intensity"),
        _text(row, "expression_pair_equivalent").lower(),
        _text(row, "expression_pair_min_confidence"),
        _text(row, "expression_pair_max_cue_difference"),
        _text(row, "expression_pair_audit_version"),
        _text(row, "objective_type"),
    )


def _audit_iat_block(rows: list[Mapping[str, Any]], issues: list[str]) -> None:
    chosen = [parse_iat_response(_text(row, "chosen")) for row in rows]
    rejected = [parse_iat_response(_text(row, "rejected")) for row in rows]
    if len({item["word_tuple"] for item in chosen}) != 1:
        issues.append("iat_chosen_target_tuple_differs_by_race")
    if len({item["word_tuple"] for item in rejected}) != 1:
        issues.append("iat_rejected_target_tuple_differs_by_race")
    if len({item["negative_count"] for item in chosen}) != 1:
        issues.append("iat_chosen_negative_count_differs_by_race")
    if len({item["negative_count"] for item in rejected}) != 1:
        issues.append("iat_rejected_negative_count_differs_by_race")
    if len({_integer(row, "chosen_unwarranted_negative_count") for row in rows}) != 1:
        issues.append("iat_chosen_unwarranted_count_differs_by_race")
    if len({_integer(row, "rejected_unwarranted_negative_count") for row in rows}) != 1:
        issues.append("iat_rejected_unwarranted_count_differs_by_race")
    if len({_integer(row, "chosen_visible_expression_support") for row in rows}) != 1:
        issues.append("iat_chosen_expression_support_differs_by_race")
    if len({_integer(row, "rejected_visible_expression_support") for row in rows}) != 1:
        issues.append("iat_rejected_expression_support_differs_by_race")


def _audit_profile_block(
    rows: list[Mapping[str, Any]],
    policy: DatasetPolicy,
    issues: list[str],
) -> None:
    for field_name in (
        "chosen_visible_expression_support",
        "rejected_visible_expression_support",
        "chosen_person_negative_count",
        "chosen_unwarranted_negative_count",
        "chosen_unsupported_trait_count",
        "chosen_identity_leakage_count",
        "chosen_invented_biography_count",
        "rejected_person_negative_count",
        "rejected_unwarranted_negative_count",
        "rejected_unsupported_trait_count",
        "rejected_identity_leakage_count",
        "rejected_invented_biography_count",
    ):
        if len({_integer(row, field_name) for row in rows}) != 1:
            issues.append(f"profile_metric_differs_by_race:{field_name}")

    word_counts = [profile_word_count(_text(row, "chosen")) for row in rows]
    if min(word_counts) == 0 or max(word_counts) / min(word_counts) > policy.max_profile_block_word_count_ratio:
        issues.append("profile_chosen_length_differs_by_race")
    grounded = [_integer(row, "chosen_grounded_detail_count") for row in rows]
    if max(grounded) - min(grounded) > policy.max_profile_block_grounded_detail_difference:
        issues.append("profile_grounded_detail_differs_by_race")
    for prefix in ("chosen", "rejected"):
        quality = [_number(row, f"{prefix}_quality_score") for row in rows]
        if max(quality) - min(quality) > policy.max_profile_block_quality_difference:
            issues.append(f"profile_{prefix}_quality_differs_by_race")


def _target_term_counts(rows: Iterable[Mapping[str, Any]], race: str) -> Counter[str]:
    result: Counter[str] = Counter()
    for row in rows:
        if _text(row, "task_type") != TASK_IAT or _text(row, "fairface_race") != race:
            continue
        parsed = parse_iat_response(_text(row, "chosen"))
        result.update(parsed["words"])
    return result


def audit_dataset(
    rows: Iterable[Mapping[str, Any]],
    *,
    pair_policy: PairPolicy | None = None,
    dataset_policy: DatasetPolicy | None = None,
    forbidden_eval_hashes: set[str] | None = None,
) -> DatasetAudit:
    pair_policy = pair_policy or PairPolicy()
    dataset_policy = dataset_policy or DatasetPolicy()
    materialized = list(rows)
    audit = DatasetAudit()

    pair_ids = [_text(row, "pair_id") for row in materialized]
    duplicate_pair_ids = sorted(
        pair_id for pair_id, count in Counter(pair_ids).items() if pair_id and count > 1
    )
    if duplicate_pair_ids:
        audit.issues.append("duplicate_pair_ids")
        audit.summary["duplicate_pair_ids"] = duplicate_pair_ids

    for row in materialized:
        validation = validate_pair(row, pair_policy)
        if not validation.valid:
            audit.invalid_pairs[validation.pair_id] = validation.issues

    row_count = len(materialized)
    unique_hashes = {_text(row, "image_sha256") for row in materialized if _text(row, "image_sha256")}
    task_counts = Counter(_text(row, "task_type") for row in materialized)
    race_counts = Counter(_text(row, "fairface_race") for row in materialized)
    objective_counts = Counter(_text(row, "objective_type") for row in materialized)
    split_counts = Counter(_text(row, "split").lower() for row in materialized)
    chosen_provenance = Counter(_text(row, "chosen_response_provenance") for row in materialized)
    rejected_provenance = Counter(
        _text(row, "rejected_response_provenance") for row in materialized
    )

    if row_count < dataset_policy.min_rows:
        audit.issues.append("too_few_rows")
    if len(unique_hashes) < dataset_policy.min_unique_images:
        audit.issues.append("too_few_unique_images")
    if race_counts.get("Black", 0) != race_counts.get("White", 0):
        audit.issues.append("race_row_counts_not_equal")

    iat_count = task_counts.get(TASK_IAT, 0)
    iat_fraction = iat_count / row_count if row_count else 0.0
    if not dataset_policy.min_iat_fraction <= iat_fraction <= dataset_policy.max_iat_fraction:
        audit.issues.append("iat_task_fraction_out_of_range")
    iat_objectives = Counter(
        _text(row, "objective_type")
        for row in materialized
        if _text(row, "task_type") == TASK_IAT
    )
    direct_fraction = (
        iat_objectives.get("negative_count_reduction", 0) / iat_count if iat_count else 0.0
    )
    fixed_fraction = (
        iat_objectives.get("unsupported_negative_substitution", 0) / iat_count
        if iat_count
        else 0.0
    )
    if direct_fraction < dataset_policy.min_iat_negative_count_reduction_fraction:
        audit.issues.append("insufficient_direct_iat_negative_count_reduction")
    if fixed_fraction > dataset_policy.max_iat_fixed_substitution_fraction:
        audit.issues.append("too_many_iat_fixed_count_substitutions")
    iat_chosen_tuples = Counter(
        parse_iat_response(_text(row, "chosen"))["word_tuple"]
        for row in materialized
        if _text(row, "task_type") == TASK_IAT
    )
    iat_tuple_row_cap = (
        2
        * math.ceil(
            (iat_count / 2) * dataset_policy.max_iat_chosen_tuple_fraction
        )
        if iat_count >= dataset_policy.iat_tuple_balance_min_rows
        else iat_count
    )
    if any(count > iat_tuple_row_cap for count in iat_chosen_tuples.values()):
        audit.issues.append("iat_chosen_tuple_overrepresented")
    iat_tuple_entropy = 0.0
    if iat_count:
        iat_tuple_entropy = -sum(
            (count / iat_count) * math.log(count / iat_count)
            for count in iat_chosen_tuples.values()
            if count
        )
    iat_tuple_normalized_entropy = (
        iat_tuple_entropy / math.log(len(iat_chosen_tuples))
        if len(iat_chosen_tuples) > 1
        else 0.0
    )
    controlled_iat_rows = sum(
        _text(row, "task_type") == TASK_IAT
        and (
            _text(row, "chosen_response_provenance").startswith("controlled_race_blind_iat")
            or _text(row, "rejected_response_provenance").startswith(
                "controlled_race_blind_iat"
            )
        )
        for row in materialized
    )
    controlled_iat_fraction = controlled_iat_rows / row_count if row_count else 0.0
    if controlled_iat_fraction > dataset_policy.max_controlled_iat_fraction_of_all_rows + 1e-12:
        audit.issues.append("controlled_iat_fraction_exceeds_overall_cap")

    profile_count = task_counts.get(TASK_PROFILE, 0)
    profile_chosen_signatures = Counter(
        " ".join(tokens(_text(row, "chosen")))
        for row in materialized
        if _text(row, "task_type") == TASK_PROFILE
    )
    profile_duplicate_targets = {
        hashlib.sha256(signature.encode("utf-8")).hexdigest(): count
        for signature, count in profile_chosen_signatures.items()
        if count > dataset_policy.max_profile_chosen_reuse
    }
    if profile_duplicate_targets:
        audit.issues.append("profile_chosen_response_reused")
    profile_objectives = Counter(
        _text(row, "objective_type")
        for row in materialized
        if _text(row, "task_type") == TASK_PROFILE
    )
    profile_objective_fractions = {
        objective: count / profile_count
        for objective, count in profile_objectives.items()
        if profile_count
    }
    if profile_count >= dataset_policy.profile_objective_balance_min_rows and any(
        fraction > dataset_policy.max_profile_objective_fraction
        for fraction in profile_objective_fractions.values()
    ):
        audit.issues.append("profile_objective_overrepresented")

    blocks: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in materialized:
        blocks[_text(row, "match_block_id")].append(row)
    block_issues: dict[str, list[str]] = {}
    for block_id, block_rows in blocks.items():
        issues: list[str] = []
        if not block_id:
            issues.append("missing_match_block_id")
        if len(block_rows) != 2:
            issues.append("block_must_have_exactly_two_rows")
        races = {_text(row, "fairface_race") for row in block_rows}
        if races != {"Black", "White"}:
            issues.append("block_must_have_one_black_and_one_white_row")
        if len({_block_signature(row) for row in block_rows}) != 1:
            issues.append("block_strata_do_not_match")
        if len({_text(row, "image_sha256") for row in block_rows}) != len(block_rows):
            issues.append("block_reuses_image_content")
        if len(block_rows) == 2 and len({_text(row, "task_type") for row in block_rows}) == 1:
            if _text(block_rows[0], "task_type") == TASK_IAT:
                _audit_iat_block(block_rows, issues)
            elif _text(block_rows[0], "task_type") == TASK_PROFILE:
                _audit_profile_block(block_rows, dataset_policy, issues)
        if issues:
            block_issues[block_id or "<missing>"] = issues
    if block_issues:
        audit.issues.append("invalid_match_blocks")

    image_splits: dict[str, set[str]] = defaultdict(set)
    for row in materialized:
        image_splits[_text(row, "image_sha256")].add(_text(row, "split").lower())
    split_leaks = sorted(digest for digest, splits in image_splits.items() if len(splits) > 1)
    if split_leaks:
        audit.issues.append("image_content_crosses_train_development_split")

    eval_overlap: list[str] = []
    if forbidden_eval_hashes is not None:
        eval_overlap = sorted(unique_hashes & forbidden_eval_hashes)
        if eval_overlap:
            audit.issues.append("training_evaluation_image_overlap")

    black_terms = _target_term_counts(materialized, "Black")
    white_terms = _target_term_counts(materialized, "White")
    if black_terms != white_terms:
        audit.issues.append("iat_chosen_target_vocabulary_differs_by_race")

    chosen_negative_by_race: dict[str, int] = {}
    for race in ("Black", "White"):
        chosen_negative_by_race[race] = sum(
            parse_iat_response(_text(row, "chosen"))["negative_count"]
            for row in materialized
            if _text(row, "task_type") == TASK_IAT
            and _text(row, "fairface_race") == race
        ) + sum(
            _integer(row, "chosen_person_negative_count")
            for row in materialized
            if _text(row, "task_type") == TASK_PROFILE
            and _text(row, "fairface_race") == race
        )
    if chosen_negative_by_race.get("Black") != chosen_negative_by_race.get("White"):
        audit.issues.append("chosen_target_negativity_differs_by_race")

    audit.summary.update(
        {
            "rows": row_count,
            "unique_images": len(unique_hashes),
            "task_counts": dict(task_counts),
            "race_counts": dict(race_counts),
            "objective_counts": dict(objective_counts),
            "split_counts": dict(split_counts),
            "chosen_response_provenance_counts": dict(chosen_provenance),
            "rejected_response_provenance_counts": dict(rejected_provenance),
            "iat_fraction": iat_fraction,
            "iat_objective_counts": dict(iat_objectives),
            "iat_direct_negative_count_reduction_fraction": direct_fraction,
            "iat_fixed_substitution_fraction": fixed_fraction,
            "iat_chosen_tuple_counts": dict(sorted(iat_chosen_tuples.items())),
            "iat_chosen_unique_tuples": len(iat_chosen_tuples),
            "iat_chosen_tuple_row_cap": iat_tuple_row_cap,
            "iat_chosen_tuple_entropy": iat_tuple_entropy,
            "iat_chosen_tuple_normalized_entropy": iat_tuple_normalized_entropy,
            "controlled_iat_rows": controlled_iat_rows,
            "controlled_iat_fraction_of_all_rows": controlled_iat_fraction,
            "profile_objective_counts": dict(profile_objectives),
            "profile_objective_fractions": profile_objective_fractions,
            "profile_chosen_unique_responses": len(profile_chosen_signatures),
            "profile_chosen_unique_fraction": (
                len(profile_chosen_signatures) / profile_count if profile_count else 0.0
            ),
            "profile_duplicate_target_key": "sha256_of_normalized_chosen_response",
            "profile_duplicate_chosen_targets": profile_duplicate_targets,
            "match_blocks": len(blocks),
            "invalid_match_blocks": block_issues,
            "split_leak_hashes": split_leaks,
            "evaluation_overlap_hashes": eval_overlap,
            "chosen_negative_targets_by_race": chosen_negative_by_race,
            "black_iat_chosen_term_counts": dict(sorted(black_terms.items())),
            "white_iat_chosen_term_counts": dict(sorted(white_terms.items())),
        }
    )
    return audit
