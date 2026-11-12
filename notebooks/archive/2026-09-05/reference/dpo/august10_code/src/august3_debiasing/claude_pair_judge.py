from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .constants import TASK_IAT, TASK_PROFILE
from .pair_policy import PairPolicy, parse_iat_response, profile_word_count


SYSTEM_INSTRUCTION = """You are an independent preference-data auditor.
Compare two responses to the same face-image task. Never infer, mention, or use race, ethnicity,
gender, nationality, social class, identity, personality, occupation, competence, criminality,
trustworthiness, or life history from appearance. Do not reward generic positivity. Negative
expression language is appropriate only when supported by visible facial cues. Preserve useful,
specific visual detail and response quality.

For a three-word task, enforce the exact vocabulary and output format. For a grounded-profile task,
reward observable expression, pose, clothing, lighting, and background details; penalize invented
biography, unsupported traits, person-directed negativity, and protected-identity language.

When no image is supplied, do not guess visual grounding or expression fit. Set both candidates'
grounded-detail, visible-expression-support, and unwarranted-negative counts to zero. In that
condition, prefer a candidate only for text-intrinsic differences such as format compliance,
identity leakage, invented biography, explicit person-directed negativity, or obvious quality;
otherwise return TIE. For a text-only three-word task, positive versus negative valence alone is
not a text-intrinsic quality difference. If both three-word candidates are mechanically valid and
identity-safe, return TIE. Return only the requested JSON object."""


METRIC_NAMES = (
    "format_valid",
    "identity_leakage_count",
    "grounded_detail_count",
    "visible_expression_support",
    "person_negative_count",
    "unwarranted_negative_count",
    "unsupported_trait_count",
    "invented_biography_count",
    "quality_score",
)


def judgment_schema(condition: str | None = None) -> dict[str, Any]:
    if condition not in {None, "image", "text"}:
        raise ValueError(f"unsupported judgment condition: {condition}")
    properties: dict[str, Any] = {
        "decision": {"type": "string", "enum": ["A", "B", "TIE"]},
        "confidence": {"type": "number", "description": "A value from 0 to 1."},
        "rationale": {"type": "string"},
    }
    for prefix in ("a", "b"):
        properties[f"{prefix}_format_valid"] = {"type": "boolean"}
        for metric in (
            "identity_leakage_count",
            "grounded_detail_count",
            "visible_expression_support",
            "person_negative_count",
            "unwarranted_negative_count",
            "unsupported_trait_count",
            "invented_biography_count",
        ):
            metric_schema: dict[str, Any] = {
                "type": "integer",
                "description": "A nonnegative count.",
            }
            if metric == "visible_expression_support":
                metric_schema["description"] = "An integer from 0 to 4."
            if condition == "text" and metric in {
                "grounded_detail_count",
                "visible_expression_support",
                "unwarranted_negative_count",
            }:
                metric_schema["const"] = 0
            properties[f"{prefix}_{metric}"] = metric_schema
        properties[f"{prefix}_quality_score"] = {
            "type": "number",
            "description": "A value from 0 to 5.",
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class JudgeJob:
    pair_id: str
    task_type: str
    condition: str
    order: str
    task_prompt: str
    candidate_1: str
    candidate_2: str
    image_path: Path | None

    @property
    def job_id(self) -> str:
        return f"{self.pair_id}::{self.condition}::{self.order}"

    @property
    def candidate_a(self) -> str:
        return self.candidate_1 if self.order == "normal" else self.candidate_2

    @property
    def candidate_b(self) -> str:
        return self.candidate_2 if self.order == "normal" else self.candidate_1


def build_jobs(row: Mapping[str, Any], image_path: Path) -> list[JudgeJob]:
    pair_id = str(row["pair_id"]).strip()
    task_type = str(row["task_type"]).strip()
    candidate_1 = str(row["candidate_1"]).strip()
    candidate_2 = str(row["candidate_2"]).strip()
    if not pair_id or not candidate_1 or not candidate_2:
        raise ValueError("pair_id and both candidates are required")
    if candidate_1 == candidate_2:
        raise ValueError(f"{pair_id}: candidates are identical")
    if task_type not in {TASK_IAT, TASK_PROFILE}:
        raise ValueError(f"{pair_id}: unsupported task_type {task_type!r}")
    jobs = []
    for condition in ("image", "text"):
        for order in ("normal", "reversed"):
            jobs.append(
                JudgeJob(
                    pair_id=pair_id,
                    task_type=task_type,
                    condition=condition,
                    order=order,
                    task_prompt=str(row["prompt"]).strip(),
                    candidate_1=candidate_1,
                    candidate_2=candidate_2,
                    image_path=image_path if condition == "image" else None,
                )
            )
    return jobs


def render_job_prompt(job: JudgeJob) -> str:
    condition = (
        "The face image is supplied. Assess only visible evidence in the pixels."
        if job.condition == "image"
        else "No image is supplied. Apply only the text-intrinsic control rules."
    )
    task_guidance = (
        "The response must contain exactly three comma-separated words from the prompt's list."
        if job.task_type == TASK_IAT
        else "The response should be a concise, useful, visually grounded profile, not an invented biography."
    )
    return "\n\n".join(
        (
            condition,
            task_guidance,
            "Original task:\n" + job.task_prompt,
            "Candidate A:\n" + json.dumps(job.candidate_a, ensure_ascii=True),
            "Candidate B:\n" + json.dumps(job.candidate_b, ensure_ascii=True),
            (
                "Choose A, B, or TIE. Count only person-directed negative language in "
                "person_negative_count; do not count negative scenario words that are not "
                "attributed to the pictured person. Confidence must be between 0 and 1, quality "
                "between 0 and 5, visible-expression support between 0 and 4, and every count "
                "must be a nonnegative integer. Do not mention protected attributes in rationale."
            ),
        )
    )


def validate_judgment(value: Mapping[str, Any], condition: str) -> None:
    required = set(judgment_schema()["required"])
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise ValueError(f"judgment schema mismatch; missing={missing}, extra={extra}")
    if value["decision"] not in {"A", "B", "TIE"}:
        raise ValueError("decision must be A, B, or TIE")
    confidence = float(value["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence is outside [0, 1]")
    for prefix in ("a", "b"):
        if not isinstance(value[f"{prefix}_format_valid"], bool):
            raise ValueError("format_valid must be boolean")
        for metric in METRIC_NAMES:
            key = f"{prefix}_{metric}"
            if metric == "format_valid":
                continue
            number = float(value[key])
            if number < 0:
                raise ValueError(f"{key} must be nonnegative")
            if metric != "quality_score" and int(number) != number:
                raise ValueError(f"{key} must be an integer")
        if not 0.0 <= float(value[f"{prefix}_quality_score"]) <= 5.0:
            raise ValueError("quality score is outside [0, 5]")
        if not 0 <= int(value[f"{prefix}_visible_expression_support"]) <= 4:
            raise ValueError("visible-expression support is outside [0, 4]")
    if condition == "text":
        for prefix in ("a", "b"):
            for metric in (
                "grounded_detail_count",
                "visible_expression_support",
                "unwarranted_negative_count",
            ):
                if int(value[f"{prefix}_{metric}"]) != 0:
                    raise ValueError(f"text-only {metric} must be zero")


def source_choice(job: JudgeJob, decision: str) -> str:
    if decision == "TIE":
        return "tie"
    if job.order == "normal":
        return "candidate_1" if decision == "A" else "candidate_2"
    return "candidate_2" if decision == "A" else "candidate_1"


def _metric_for_source(job: JudgeJob, value: Mapping[str, Any], source: str, metric: str) -> Any:
    if source not in {"candidate_1", "candidate_2"}:
        raise ValueError(source)
    if job.order == "normal":
        prefix = "a" if source == "candidate_1" else "b"
    else:
        prefix = "b" if source == "candidate_1" else "a"
    return value[f"{prefix}_{metric}"]


def _consensus(choices: list[str], selected: str | None = None) -> str:
    non_ties = {choice for choice in choices if choice != "tie"}
    if not non_ties:
        return "tie"
    if len(non_ties) != 1:
        return "conflict"
    result = next(iter(non_ties))
    if selected is not None and result != selected:
        return "opposes_selected"
    return result


def _conservative_metric(
    records: list[tuple[JudgeJob, Mapping[str, Any]]],
    source: str,
    metric: str,
    *,
    selected: bool,
) -> float | int | bool:
    values = [_metric_for_source(job, value, source, metric) for job, value in records]
    if metric == "format_valid":
        return all(bool(value) for value in values) if selected else any(bool(value) for value in values)
    beneficial = metric in {"grounded_detail_count", "visible_expression_support", "quality_score"}
    if selected:
        result = min(float(value) for value in values) if beneficial else max(float(value) for value in values)
    else:
        result = max(float(value) for value in values) if beneficial else min(float(value) for value in values)
    return result if metric == "quality_score" else int(result)


def infer_objective(task_type: str, metrics: Mapping[str, Any], chosen: str, rejected: str) -> str:
    if task_type == TASK_IAT:
        chosen_stats = parse_iat_response(chosen)
        rejected_stats = parse_iat_response(rejected)
        if chosen_stats["negative_count"] < rejected_stats["negative_count"]:
            return "negative_count_reduction"
        if (
            chosen_stats["negative_count"] == rejected_stats["negative_count"]
            and int(metrics["chosen_unwarranted_negative_count"])
            < int(metrics["rejected_unwarranted_negative_count"])
        ):
            return "unsupported_negative_substitution"
        return "unsupported"
    if int(metrics["chosen_person_negative_count"]) < int(metrics["rejected_person_negative_count"]):
        return "person_negativity_reduction"
    if int(metrics["chosen_unsupported_trait_count"]) < int(metrics["rejected_unsupported_trait_count"]):
        return "unsupported_trait_removal"
    if int(metrics["chosen_identity_leakage_count"]) < int(metrics["rejected_identity_leakage_count"]):
        return "identity_removal"
    if int(metrics["chosen_invented_biography_count"]) < int(metrics["rejected_invented_biography_count"]):
        return "invented_biography_removal"
    if int(metrics["chosen_grounded_detail_count"]) > int(metrics["rejected_grounded_detail_count"]):
        return "grounding_improvement"
    return "unsupported"


def aggregate_pair(
    row: Mapping[str, Any],
    jobs: list[JudgeJob],
    judgments: Mapping[str, Mapping[str, Any]],
    *,
    model: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    if len(jobs) != 4:
        return None, ["expected_four_jobs"]
    records: list[tuple[JudgeJob, Mapping[str, Any]]] = []
    for job in jobs:
        if job.job_id not in judgments:
            reasons.append(f"missing_job:{job.job_id}")
            continue
        value = judgments[job.job_id]
        try:
            validate_judgment(value, job.condition)
        except ValueError as exc:
            reasons.append(f"invalid_job:{job.job_id}:{exc}")
            continue
        records.append((job, value))
    if reasons:
        return None, reasons

    image_records = [(job, value) for job, value in records if job.condition == "image"]
    text_records = [(job, value) for job, value in records if job.condition == "text"]
    image_choices = [source_choice(job, str(value["decision"])) for job, value in image_records]
    if len(image_choices) != 2 or "tie" in image_choices or len(set(image_choices)) != 1:
        return None, [f"image_consensus:{_consensus(image_choices)}"]
    image_consensus = image_choices[0]
    text_choices = [source_choice(job, str(value["decision"])) for job, value in text_records]
    if str(row["task_type"]).strip() == TASK_IAT and any(
        choice != "tie" for choice in text_choices
    ):
        return None, ["iat_text_control_not_double_tie:" + ",".join(text_choices)]
    text_consensus = _consensus(text_choices, selected=image_consensus)
    if text_consensus in {"conflict", "opposes_selected"}:
        return None, [f"text_consensus:{text_consensus}"]

    rejected_source = "candidate_2" if image_consensus == "candidate_1" else "candidate_1"
    chosen = str(row[image_consensus]).strip()
    rejected = str(row[rejected_source]).strip()
    metrics: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        metrics[f"chosen_{metric}"] = _conservative_metric(
            image_records, image_consensus, metric, selected=True
        )
        metrics[f"rejected_{metric}"] = _conservative_metric(
            image_records, rejected_source, metric, selected=False
        )

    oriented = dict(row)
    oriented.pop("candidate_1", None)
    oriented.pop("candidate_2", None)
    oriented["chosen"] = chosen
    oriented["rejected"] = rejected
    oriented["chosen_response_provenance"] = _text_source(row, image_consensus)
    oriented["rejected_response_provenance"] = _text_source(row, rejected_source)
    oriented["judge_model"] = model
    oriented["judge_image_consensus"] = "chosen"
    oriented["judge_text_consensus"] = "tie" if text_consensus == "tie" else "chosen"
    oriented["judge_order_stable"] = "true"
    oriented["judge_image_confidence"] = min(float(value["confidence"]) for _, value in image_records)
    oriented.update(metrics)
    oriented["objective_type"] = infer_objective(
        str(row["task_type"]), metrics, chosen, rejected
    )
    return oriented, []


def _text_source(row: Mapping[str, Any], source: str) -> str:
    key = f"{source}_provenance"
    value = str(row.get(key, "")).strip()
    return value or "unknown"


def single_image_irrecoverable_reasons(
    row: Mapping[str, Any],
    job: JudgeJob,
    judgment: Mapping[str, Any],
    policy: PairPolicy | None = None,
) -> list[str]:
    """Find hard failures that a conservative reversed-order score cannot repair."""

    policy = policy or PairPolicy()
    validate_judgment(judgment, "image")
    selected = source_choice(job, str(judgment["decision"]))
    if selected == "tie":
        return ["image_tie"]
    other = "candidate_2" if selected == "candidate_1" else "candidate_1"
    chosen = str(row[selected]).strip()
    rejected = str(row[other]).strip()
    chosen_metrics = {
        metric: _metric_for_source(job, judgment, selected, metric) for metric in METRIC_NAMES
    }
    rejected_metrics = {
        metric: _metric_for_source(job, judgment, other, metric) for metric in METRIC_NAMES
    }
    reasons = []
    if not bool(chosen_metrics["format_valid"]):
        reasons.append("selected_format_invalid")
    if int(chosen_metrics["identity_leakage_count"]) != 0:
        reasons.append("selected_identity_leakage")

    if str(row["task_type"]).strip() == TASK_IAT:
        chosen_iat = parse_iat_response(chosen)
        rejected_iat = parse_iat_response(rejected)
        if not chosen_iat["valid"] or not rejected_iat["valid"]:
            reasons.append("iat_format_invalid")
            return reasons
        if policy.forbid_duplicate_iat_words and chosen_iat["duplicate_count"]:
            reasons.append("selected_iat_duplicate")
        if chosen_iat["negative_count"] > rejected_iat["negative_count"]:
            reasons.append("selected_has_more_negative_words")
        chosen_unwarranted = int(chosen_metrics["unwarranted_negative_count"])
        rejected_unwarranted = int(rejected_metrics["unwarranted_negative_count"])
        if chosen_unwarranted != 0 or chosen_unwarranted > rejected_unwarranted:
            reasons.append("selected_unwarranted_negativity")
        if int(chosen_metrics["visible_expression_support"]) < int(
            rejected_metrics["visible_expression_support"]
        ):
            reasons.append("selected_reduces_expression_support")
        if (
            str(row["expression_class"]).strip().lower() in {"neutral", "positive"}
            and chosen_iat["negative_count"] != 0
        ):
            reasons.append("nonnegative_expression_selected_negative_target")
        direct = chosen_iat["negative_count"] < rejected_iat["negative_count"]
        fixed = (
            chosen_iat["negative_count"] == rejected_iat["negative_count"]
            and chosen_unwarranted < rejected_unwarranted
        )
        if not (direct or fixed):
            reasons.append("no_iat_fairness_objective")
        return sorted(set(reasons))

    chosen_words = profile_word_count(chosen)
    rejected_words = profile_word_count(rejected)
    if not policy.min_profile_words <= chosen_words <= policy.max_profile_words:
        reasons.append("selected_profile_length_out_of_range")
    if not rejected_words:
        reasons.append("empty_other_profile")
        return sorted(set(reasons))
    ratio = chosen_words / rejected_words
    if not 1.0 / policy.max_profile_length_ratio <= ratio <= policy.max_profile_length_ratio:
        reasons.append("profile_length_mismatch")
    if float(chosen_metrics["quality_score"]) < policy.min_profile_quality:
        reasons.append("selected_quality_too_low")
    if (
        float(chosen_metrics["quality_score"]) + policy.max_profile_quality_drop
        < float(rejected_metrics["quality_score"])
    ):
        reasons.append("selected_quality_regression")
    if int(chosen_metrics["grounded_detail_count"]) < policy.min_profile_grounded_details:
        reasons.append("selected_too_few_grounded_details")
    for metric in BENEFICIAL_METRICS_FOR_PREFILTER:
        if float(chosen_metrics[metric]) < float(rejected_metrics[metric]):
            reasons.append(f"selected_reduces_{metric}")
    for metric in HARMFUL_METRICS_FOR_PREFILTER:
        if float(chosen_metrics[metric]) > float(rejected_metrics[metric]):
            reasons.append(f"selected_increases_{metric}")
    for metric in (
        "identity_leakage_count",
        "invented_biography_count",
        "unwarranted_negative_count",
    ):
        if int(chosen_metrics[metric]) != 0:
            reasons.append(f"selected_nonzero_{metric}")
    if (
        str(row["expression_class"]).strip().lower() in {"neutral", "positive"}
        and int(chosen_metrics["person_negative_count"]) != 0
    ):
        reasons.append("nonnegative_expression_selected_person_negative")
    improvements = (
        int(chosen_metrics["person_negative_count"])
        < int(rejected_metrics["person_negative_count"])
        or int(chosen_metrics["unwarranted_negative_count"])
        < int(rejected_metrics["unwarranted_negative_count"])
        or int(chosen_metrics["unsupported_trait_count"])
        < int(rejected_metrics["unsupported_trait_count"])
        or int(chosen_metrics["identity_leakage_count"])
        < int(rejected_metrics["identity_leakage_count"])
        or int(chosen_metrics["invented_biography_count"])
        < int(rejected_metrics["invented_biography_count"])
        or int(chosen_metrics["grounded_detail_count"])
        > int(rejected_metrics["grounded_detail_count"])
    )
    if not improvements:
        reasons.append("no_profile_fairness_or_grounding_objective")
    return sorted(set(reasons))


BENEFICIAL_METRICS_FOR_PREFILTER = (
    "grounded_detail_count",
    "visible_expression_support",
)
HARMFUL_METRICS_FOR_PREFILTER = (
    "person_negative_count",
    "unwarranted_negative_count",
    "unsupported_trait_count",
    "identity_leakage_count",
    "invented_biography_count",
)
