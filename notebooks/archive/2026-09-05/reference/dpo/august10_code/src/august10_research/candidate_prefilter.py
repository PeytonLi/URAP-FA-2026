from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

POSITIVE_IAT_WORDS = {
    "love",
    "cheer",
    "friend",
    "pleasure",
    "adore",
    "cheerful",
    "friendship",
    "joyful",
    "smiling",
    "cherish",
    "excellent",
    "glad",
    "joyous",
    "spectacular",
    "appealing",
    "delight",
    "excitement",
    "laughing",
    "attractive",
    "fabulous",
    "glorious",
    "pleasing",
    "beautiful",
    "fantastic",
    "happy",
    "lovely",
    "terrific",
    "celebrate",
    "enjoy",
    "magnificent",
}

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

IAT_WORDS = POSITIVE_IAT_WORDS | NEGATIVE_IAT_WORDS
TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?")
IDENTITY_TERMS = {
    "african",
    "african-american",
    "asian",
    "black",
    "boy",
    "caucasian",
    "ethnicity",
    "female",
    "gender",
    "girl",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "hispanic",
    "indian",
    "latina",
    "latino",
    "male",
    "man",
    "nationality",
    "race",
    "racial",
    "she",
    "skin",
    "white",
    "woman",
}
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
class PrefilterPolicy:
    min_profile_words: int = 80
    max_profile_words: int = 140
    max_profile_length_ratio: float = 1.35


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


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


def parse_iat_response(text: str) -> dict[str, Any]:
    parts = [part.strip().lower() for part in text.split(",")]
    valid = (
        len(parts) == 3
        and all(re.fullmatch(r"[A-Za-z]+", part) for part in parts)
        and all(part in IAT_WORDS for part in parts)
    )
    words = tuple(parts) if valid else ()
    return {
        "valid": valid,
        "words": words,
        "negative_count": sum(word in NEGATIVE_IAT_WORDS for word in words),
        "duplicate_count": len(words) - len(set(words)),
        "identity_hits": identity_hits(text),
    }


def profile_word_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))


def mechanical_exclusion_reasons(
    row: Mapping[str, Any],
    policy: PrefilterPolicy | None = None,
) -> list[str]:
    """Return only failures that cannot be repaired by a multimodal judgment."""

    policy = policy or PrefilterPolicy()
    first = _text(row, "candidate_1")
    second = _text(row, "candidate_2")
    reasons: list[str] = []
    if not _text(row, "pair_id"):
        reasons.append("missing_pair_id")
    if not first or not second:
        reasons.append("empty_candidate")
        return reasons
    if first == second:
        reasons.append("identical_candidates")
    if _text(row, "fairface_race") not in {"Black", "White"}:
        reasons.append("unsupported_race")
    if _text(row, "expression_pair_equivalent").lower() != "true":
        reasons.append("expression_pair_not_equivalent")

    task = _text(row, "task_type")
    if task == "iat_3word":
        first_iat = parse_iat_response(first)
        second_iat = parse_iat_response(second)
        if not first_iat["valid"] or not second_iat["valid"]:
            reasons.append("iat_candidate_format_invalid")
            return sorted(set(reasons))
        if first_iat["negative_count"] == second_iat["negative_count"]:
            reasons.append("iat_has_no_direct_negative_reduction")
            return sorted(set(reasons))
        lower = (
            first_iat
            if first_iat["negative_count"] < second_iat["negative_count"]
            else second_iat
        )
        if lower["duplicate_count"]:
            reasons.append("iat_lower_negative_candidate_has_duplicates")
        if lower["identity_hits"]:
            reasons.append("iat_lower_negative_candidate_has_identity_leakage")
        if (
            _text(row, "expression_class").lower() in {"neutral", "positive"}
            and lower["negative_count"] != 0
        ):
            reasons.append("iat_nonnegative_expression_target_remains_negative")
    elif task == "grounded_profile":
        first_words = profile_word_count(first)
        second_words = profile_word_count(second)
        if not first_words or not second_words:
            reasons.append("empty_profile")
            return sorted(set(reasons))
        ratio = max(first_words, second_words) / min(first_words, second_words)
        if ratio > policy.max_profile_length_ratio:
            reasons.append("profile_length_ratio_exceeds_policy")
        possible_chosen = (
            policy.min_profile_words <= first_words <= policy.max_profile_words
            and not identity_hits(first)
        ) or (
            policy.min_profile_words <= second_words <= policy.max_profile_words
            and not identity_hits(second)
        )
        if not possible_chosen:
            reasons.append("profile_has_no_identity_safe_in_range_candidate")
    else:
        reasons.append("unsupported_task_type")
    return sorted(set(reasons))
