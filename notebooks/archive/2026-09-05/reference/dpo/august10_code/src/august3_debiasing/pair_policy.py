from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from .constants import (
    EXPRESSION_CLASSES,
    EXPRESSION_PAIR_AUDIT_VERSION,
    EXPRESSION_PAIR_MAXIMUM_CUE_DIFFERENCE,
    EXPRESSION_PAIR_MINIMUM_CONFIDENCE,
    IAT_OBJECTIVES,
    IAT_WORD_LOOKUP,
    IDENTITY_TERMS,
    NEGATIVE_IAT_SET,
    PROFILE_OBJECTIVES,
    PROMPT_METADATA_TERMS,
    TARGET_GENDERS,
    TARGET_RACES,
    TASK_IAT,
    TASK_PROFILE,
    TASK_TYPES,
)

TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
IDENTITY_POLICY_VERSION = "august3-contextual-identity-v3-sentence-bounded"
CONTEXTUAL_COLOR_TERMS = {"black", "white"}
IDENTITY_NOUN_CONTEXT = {
    "adult",
    "boy",
    "child",
    "ethnicity",
    "face",
    "female",
    "girl",
    "individual",
    "male",
    "man",
    "nationality",
    "people",
    "person",
    "race",
    "woman",
}
SKIN_IDENTITY_CONTEXT = {
    "black",
    "brown",
    "color",
    "colour",
    "complexion",
    "dark",
    "fair",
    "light",
    "pale",
    "tone",
    "white",
}


@dataclass(frozen=True)
class PairPolicy:
    min_profile_words: int = 80
    max_profile_words: int = 140
    max_profile_length_ratio: float = 1.35
    min_profile_quality: float = 3.0
    min_profile_grounded_details: int = 2
    max_profile_quality_drop: float = 0.25
    require_stable_image_judgment: bool = True
    require_text_control_not_rejected: bool = True
    forbid_duplicate_iat_words: bool = True


@dataclass
class PairValidation:
    pair_id: str
    issues: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.issues


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _integer(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value is None or _text(value) == "":
        return default
    return int(float(value))


def _number(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None or _text(value) == "":
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _boolean(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key)
    if value is None or _text(value) == "":
        return default
    if isinstance(value, bool):
        return value
    normalized = _text(value).lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"{key} is not boolean: {value!r}")


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def identity_hits(text: str) -> list[str]:
    word_tokens = tokens(text)
    contextual = CONTEXTUAL_COLOR_TERMS | {"skin"}
    found = set(word_tokens) & (IDENTITY_TERMS - contextual)
    for sentence in re.split(r"(?<=[.!?])\s+|[\r\n]+", text):
        sentence_tokens = tokens(sentence)
        for index, token in enumerate(sentence_tokens):
            neighborhood = set(sentence_tokens[max(0, index - 2) : index + 3])
            if token in CONTEXTUAL_COLOR_TERMS and neighborhood & (
                IDENTITY_NOUN_CONTEXT | {"skin", "complexion"}
            ):
                found.add(token)
            if token == "skin" and neighborhood & SKIN_IDENTITY_CONTEXT:
                found.add(token)
            if token in {"dark-skinned", "light-skinned"}:
                found.add("skin")
    return sorted(found)


def prompt_metadata_hits(text: str) -> list[str]:
    # Frozen prompts may name protected concepts only to prohibit their use. Exact
    # equality prevents this exception from admitting image-specific metadata.
    from .prompts import PROMPT_BY_TASK

    if text in PROMPT_BY_TASK.values():
        return []
    lowered = text.lower()
    token_hits = set(tokens(text)) & (PROMPT_METADATA_TERMS - IDENTITY_TERMS)
    phrase_hits = {term for term in PROMPT_METADATA_TERMS if " " in term and term in lowered}
    return sorted(set(identity_hits(text)) | token_hits | phrase_hits)


def parse_iat_response(text: str) -> dict[str, Any]:
    raw_parts = text.split(",")
    clean_parts = [part.strip() for part in raw_parts]
    valid = (
        len(clean_parts) == 3
        and all(re.fullmatch(r"[A-Za-z]+", part) for part in clean_parts)
        and all(part.lower() in IAT_WORD_LOOKUP for part in clean_parts)
    )
    normalized = tuple(part.lower() for part in clean_parts) if valid else ()
    return {
        "valid": valid,
        "words": normalized,
        "word_tuple": ",".join(normalized),
        "negative_count": sum(word in NEGATIVE_IAT_SET for word in normalized),
        "duplicate_count": len(normalized) - len(set(normalized)),
        "identity_hits": identity_hits(text),
    }


def profile_word_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def _require_common_fields(row: Mapping[str, Any], result: PairValidation) -> None:
    required = (
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
        "expression_fine",
        "expression_intensity",
        "expression_pair_equivalent",
        "expression_pair_min_confidence",
        "expression_pair_max_cue_difference",
        "expression_pair_audit_version",
        "objective_type",
        "chosen_response_provenance",
        "rejected_response_provenance",
        "judge_model",
        "judge_image_consensus",
        "judge_text_consensus",
        "judge_order_stable",
    )
    missing = [key for key in required if not _text(row.get(key))]
    if missing:
        result.issues.append("missing_fields:" + ",".join(missing))


def _validate_common(
    row: Mapping[str, Any],
    policy: PairPolicy,
    result: PairValidation,
) -> None:
    task_type = _text(row.get("task_type"))
    race = _text(row.get("fairface_race"))
    gender = _text(row.get("fairface_gender"))
    expression_class = _text(row.get("expression_class")).lower()
    expression_intensity = _text(row.get("expression_intensity"))
    split = _text(row.get("split")).lower()
    chosen = _text(row.get("chosen"))
    rejected = _text(row.get("rejected"))

    if task_type not in TASK_TYPES:
        result.issues.append("invalid_task_type")
    if race not in TARGET_RACES:
        result.issues.append("invalid_target_race")
    if gender not in TARGET_GENDERS:
        result.issues.append("invalid_target_gender")
    if expression_class not in EXPRESSION_CLASSES:
        result.issues.append("invalid_expression_class")
    try:
        parsed_intensity = float(expression_intensity)
        if (
            not math.isfinite(parsed_intensity)
            or parsed_intensity != int(parsed_intensity)
            or not 0 <= int(parsed_intensity) <= 3
        ):
            result.issues.append("invalid_expression_intensity")
    except (OverflowError, ValueError):
        result.issues.append("invalid_expression_intensity")
    try:
        pair_equivalent = _boolean(row, "expression_pair_equivalent")
    except ValueError:
        pair_equivalent = False
    if not pair_equivalent:
        result.issues.append("expression_pair_not_verified_equivalent")
    if _text(row.get("expression_pair_audit_version")) != EXPRESSION_PAIR_AUDIT_VERSION:
        result.issues.append("invalid_expression_pair_audit_version")
    try:
        pair_confidence = _number(row, "expression_pair_min_confidence")
    except ValueError:
        pair_confidence = -1.0
    if not EXPRESSION_PAIR_MINIMUM_CONFIDENCE <= pair_confidence <= 1.0:
        result.issues.append("expression_pair_confidence_below_policy")
    try:
        pair_cue_difference = _number(row, "expression_pair_max_cue_difference")
    except ValueError:
        pair_cue_difference = math.inf
    if not 0 <= pair_cue_difference <= EXPRESSION_PAIR_MAXIMUM_CUE_DIFFERENCE:
        result.issues.append("expression_pair_cue_difference_above_policy")
    if split not in {"train", "development"}:
        result.issues.append("evaluation_split_forbidden_in_training_csv")
    if chosen == rejected:
        result.issues.append("chosen_equals_rejected")

    prompt_hits = prompt_metadata_hits(_text(row.get("prompt")))
    if prompt_hits:
        result.issues.append("prompt_contains_protected_metadata")
        result.diagnostics["prompt_metadata_hits"] = prompt_hits

    chosen_identity = identity_hits(chosen)
    if chosen_identity:
        result.issues.append("chosen_identity_leakage")
        result.diagnostics["chosen_identity_terms"] = chosen_identity

    if policy.require_stable_image_judgment:
        if _text(row.get("judge_image_consensus")).lower() != "chosen":
            result.issues.append("image_judge_did_not_choose_chosen")
        try:
            stable = _boolean(row, "judge_order_stable")
        except ValueError:
            stable = False
        if not stable:
            result.issues.append("judge_order_unstable")
    if policy.require_text_control_not_rejected:
        if _text(row.get("judge_text_consensus")).lower() not in {"chosen", "tie"}:
            result.issues.append("text_control_prefers_rejected")


def _validate_iat(
    row: Mapping[str, Any],
    policy: PairPolicy,
    result: PairValidation,
) -> None:
    chosen = parse_iat_response(_text(row.get("chosen")))
    rejected = parse_iat_response(_text(row.get("rejected")))
    result.diagnostics.update({"chosen_iat": chosen, "rejected_iat": rejected})

    if not chosen["valid"]:
        result.issues.append("chosen_not_exact_three_word_iat")
    if not rejected["valid"]:
        result.issues.append("rejected_not_exact_three_word_iat")
    if not chosen["valid"] or not rejected["valid"]:
        return
    if policy.forbid_duplicate_iat_words and chosen["duplicate_count"]:
        result.issues.append("chosen_duplicate_iat_word")

    objective = _text(row.get("objective_type"))
    if objective not in IAT_OBJECTIVES:
        result.issues.append("invalid_iat_objective")

    chosen_unsupported = _integer(row, "chosen_unwarranted_negative_count")
    rejected_unsupported = _integer(row, "rejected_unwarranted_negative_count")
    chosen_support = _integer(row, "chosen_visible_expression_support")
    rejected_support = _integer(row, "rejected_visible_expression_support")
    expression_class = _text(row.get("expression_class")).lower()

    if chosen["negative_count"] > rejected["negative_count"]:
        result.issues.append("chosen_has_more_negative_words")
    if chosen_unsupported > rejected_unsupported:
        result.issues.append("chosen_has_more_unwarranted_negativity")
    if chosen_support < rejected_support:
        result.issues.append("chosen_reduces_expression_support")
    if chosen_unsupported != 0:
        result.issues.append("chosen_contains_unwarranted_negativity")
    if expression_class in {"neutral", "positive"} and chosen["negative_count"] != 0:
        result.issues.append("nonnegative_expression_has_negative_chosen_target")

    if objective == "negative_count_reduction":
        if chosen["negative_count"] >= rejected["negative_count"]:
            result.issues.append("objective_not_realized:negative_count_reduction")
    elif objective == "unsupported_negative_substitution":
        if chosen["negative_count"] != rejected["negative_count"]:
            result.issues.append("objective_not_realized:fixed_count_substitution")
        if chosen_unsupported >= rejected_unsupported:
            result.issues.append("objective_not_realized:unsupported_reduction")


def _validate_profile(
    row: Mapping[str, Any],
    policy: PairPolicy,
    result: PairValidation,
) -> None:
    chosen = _text(row.get("chosen"))
    rejected = _text(row.get("rejected"))
    chosen_words = profile_word_count(chosen)
    rejected_words = profile_word_count(rejected)
    result.diagnostics.update(
        {"chosen_profile_words": chosen_words, "rejected_profile_words": rejected_words}
    )

    objective = _text(row.get("objective_type"))
    if objective not in PROFILE_OBJECTIVES:
        result.issues.append("invalid_profile_objective")
    if not policy.min_profile_words <= chosen_words <= policy.max_profile_words:
        result.issues.append("chosen_profile_length_out_of_range")
    if rejected_words == 0:
        result.issues.append("empty_rejected_profile")
        return
    length_ratio = chosen_words / rejected_words
    result.diagnostics["profile_length_ratio"] = length_ratio
    if not 1.0 / policy.max_profile_length_ratio <= length_ratio <= policy.max_profile_length_ratio:
        result.issues.append("profile_length_mismatch")

    chosen_quality = _number(row, "chosen_quality_score")
    rejected_quality = _number(row, "rejected_quality_score")
    chosen_grounded = _integer(row, "chosen_grounded_detail_count")
    rejected_grounded = _integer(row, "rejected_grounded_detail_count")
    chosen_support = _integer(row, "chosen_visible_expression_support")
    rejected_support = _integer(row, "rejected_visible_expression_support")
    chosen_negative = _integer(row, "chosen_person_negative_count")
    rejected_negative = _integer(row, "rejected_person_negative_count")
    chosen_unwarranted = _integer(row, "chosen_unwarranted_negative_count")
    rejected_unwarranted = _integer(row, "rejected_unwarranted_negative_count")
    chosen_traits = _integer(row, "chosen_unsupported_trait_count")
    rejected_traits = _integer(row, "rejected_unsupported_trait_count")
    chosen_identity = _integer(row, "chosen_identity_leakage_count")
    rejected_identity = _integer(row, "rejected_identity_leakage_count")
    chosen_biography = _integer(row, "chosen_invented_biography_count")
    rejected_biography = _integer(row, "rejected_invented_biography_count")

    if chosen_quality < policy.min_profile_quality:
        result.issues.append("chosen_profile_quality_too_low")
    if chosen_quality + policy.max_profile_quality_drop < rejected_quality:
        result.issues.append("chosen_profile_quality_regression")
    if chosen_grounded < policy.min_profile_grounded_details:
        result.issues.append("chosen_has_too_few_grounded_details")
    if chosen_grounded < rejected_grounded:
        result.issues.append("chosen_reduces_grounded_detail")
    if chosen_support < rejected_support:
        result.issues.append("chosen_reduces_expression_support")
    if chosen_negative > rejected_negative:
        result.issues.append("chosen_increases_person_negativity")
    if chosen_unwarranted > rejected_unwarranted:
        result.issues.append("chosen_increases_unwarranted_negativity")
    if chosen_traits > rejected_traits:
        result.issues.append("chosen_increases_unsupported_traits")
    if chosen_identity != 0 or chosen_identity > rejected_identity:
        result.issues.append("chosen_identity_metric_nonzero")
    if chosen_biography != 0 or chosen_biography > rejected_biography:
        result.issues.append("chosen_invented_biography_nonzero")
    if chosen_unwarranted != 0:
        result.issues.append("chosen_unwarranted_negativity_nonzero")
    if (
        _text(row.get("expression_class")).lower() in {"neutral", "positive"}
        and chosen_negative != 0
    ):
        result.issues.append("nonnegative_expression_has_person_negative_target")

    improvements = {
        "person_negativity_reduction": chosen_negative < rejected_negative
        or chosen_unwarranted < rejected_unwarranted,
        "unsupported_trait_removal": chosen_traits < rejected_traits,
        "identity_removal": chosen_identity < rejected_identity,
        "invented_biography_removal": chosen_biography < rejected_biography,
        "grounding_improvement": chosen_grounded > rejected_grounded,
    }
    if objective in improvements and not improvements[objective]:
        result.issues.append(f"objective_not_realized:{objective}")


def validate_pair(
    row: Mapping[str, Any],
    policy: PairPolicy | None = None,
) -> PairValidation:
    policy = policy or PairPolicy()
    result = PairValidation(pair_id=_text(row.get("pair_id")) or "<missing>")
    _require_common_fields(row, result)
    if result.issues:
        return result
    try:
        _validate_common(row, policy, result)
        if _text(row.get("task_type")) == TASK_IAT:
            _validate_iat(row, policy, result)
        elif _text(row.get("task_type")) == TASK_PROFILE:
            _validate_profile(row, policy, result)
    except (TypeError, ValueError) as exc:
        result.issues.append(f"invalid_numeric_or_boolean_field:{exc}")
    return result
