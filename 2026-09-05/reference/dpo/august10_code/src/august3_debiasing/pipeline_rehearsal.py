from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .balance_pairs import BalancePolicy, balance_pairs
from .candidate_pairs import CandidatePolicy, build_candidate_pairs
from .claude_pair_judge import METRIC_NAMES, aggregate_pair, build_jobs, judgment_schema
from .constants import TASK_IAT, TASK_PROFILE
from .pair_policy import parse_iat_response, validate_pair
from .prompts import GROUNDED_PROFILE_PROMPT

REHEARSAL_POLICY_VERSION = "august3-synthetic-dataflow-rehearsal-v1"
REHEARSAL_JUDGE = "synthetic-rehearsal-not-a-provider"

SYNTHETIC_GEMMA_PROFILE = (
    "The person appears in a centered portrait with the face visible and the gaze directed "
    "toward the camera. The framing includes the head and upper shoulders. Lighting falls "
    "evenly across the visible features, while the background remains plain and softly out "
    "of focus. The person wears a simple top, and no surrounding objects provide reliable "
    "personal context. The expression, pose, clothing, illumination, and background are the "
    "only details described here. No occupation, personality, history, competence, intent, "
    "health, or identity can be inferred from this diagnostic response."
)

PROFILE_OBJECTIVES = (
    "person_negativity_reduction",
    "unsupported_trait_removal",
    "identity_removal",
    "invented_biography_removal",
    "grounding_improvement",
)


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _stable_index(value: str, modulo: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % modulo


def synthetic_gemma_profile_rows(
    metadata_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "response_id": f"synthetic-gemma-profile-{_text(row, 'image_id')}",
            "image_sha256": _text(row, "image_sha256").lower(),
            "task_type": TASK_PROFILE,
            "prompt": GROUNDED_PROFILE_PROMPT,
            "response": SYNTHETIC_GEMMA_PROFILE,
            "response_provenance": "synthetic_gemma_rehearsal_not_training",
        }
        for row in metadata_rows
    ]


def _profile_metrics(objective: str) -> tuple[dict[str, Any], dict[str, Any]]:
    chosen = {
        "format_valid": True,
        "identity_leakage_count": 0,
        "grounded_detail_count": 6,
        "visible_expression_support": 3,
        "person_negative_count": 0,
        "unwarranted_negative_count": 0,
        "unsupported_trait_count": 0,
        "invented_biography_count": 0,
        "quality_score": 4.2,
    }
    rejected = dict(chosen)
    rejected["quality_score"] = 3.8
    rejected["visible_expression_support"] = 2
    if objective == "person_negativity_reduction":
        rejected["person_negative_count"] = 1
        rejected["unwarranted_negative_count"] = 1
    elif objective == "unsupported_trait_removal":
        rejected["unsupported_trait_count"] = 1
    elif objective == "identity_removal":
        rejected["identity_leakage_count"] = 1
    elif objective == "invented_biography_removal":
        rejected["invented_biography_count"] = 1
    elif objective == "grounding_improvement":
        rejected["grounded_detail_count"] = 5
    else:
        raise ValueError(f"unsupported rehearsal objective: {objective}")
    return chosen, rejected


def _iat_metrics(
    chosen_text: str, rejected_text: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    chosen_stats = parse_iat_response(chosen_text)
    rejected_stats = parse_iat_response(rejected_text)
    if not chosen_stats["valid"] or not rejected_stats["valid"]:
        raise ValueError("rehearsal IAT candidates must be valid")
    chosen = {
        "format_valid": True,
        "identity_leakage_count": 0,
        "grounded_detail_count": 0,
        "visible_expression_support": 2,
        "person_negative_count": 0,
        "unwarranted_negative_count": 0,
        "unsupported_trait_count": 0,
        "invented_biography_count": 0,
        "quality_score": 4.0,
    }
    rejected = dict(chosen)
    rejected["unwarranted_negative_count"] = max(
        1, rejected_stats["negative_count"] - chosen_stats["negative_count"]
    )
    rejected["quality_score"] = 3.5
    return chosen, rejected


def _judgment(
    job: Any,
    *,
    preferred_source: str,
    chosen_metrics: Mapping[str, Any],
    rejected_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    source_by_prefix = (
        {"a": "candidate_1", "b": "candidate_2"}
        if job.order == "normal"
        else {"a": "candidate_2", "b": "candidate_1"}
    )
    if job.condition == "image":
        preferred_prefix = next(
            prefix
            for prefix, source in source_by_prefix.items()
            if source == preferred_source
        )
        decision = preferred_prefix.upper()
    else:
        decision = "TIE"
    value: dict[str, Any] = {
        "decision": decision,
        "confidence": 0.9,
        "rationale": "Synthetic dataflow rehearsal; no provider judgment was made.",
    }
    for prefix, source in source_by_prefix.items():
        metrics = chosen_metrics if source == preferred_source else rejected_metrics
        for metric in METRIC_NAMES:
            metric_value = metrics[metric]
            if job.condition == "text" and metric in {
                "grounded_detail_count",
                "visible_expression_support",
                "unwarranted_negative_count",
            }:
                metric_value = 0
            value[f"{prefix}_{metric}"] = metric_value
    if set(value) != set(judgment_schema()["required"]):
        raise RuntimeError("synthetic judgment does not match the current schema")
    return value


def simulate_audited_pair(
    row: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    task_type = _text(row, "task_type")
    if task_type == TASK_IAT:
        first = parse_iat_response(_text(row, "candidate_1"))
        second = parse_iat_response(_text(row, "candidate_2"))
        if first["negative_count"] == second["negative_count"]:
            return None, ["synthetic_iat_has_no_negative_count_difference"]
        preferred_source = (
            "candidate_1"
            if first["negative_count"] < second["negative_count"]
            else "candidate_2"
        )
        rejected_source = (
            "candidate_2" if preferred_source == "candidate_1" else "candidate_1"
        )
        chosen_metrics, rejected_metrics = _iat_metrics(
            _text(row, preferred_source), _text(row, rejected_source)
        )
    elif task_type == TASK_PROFILE:
        teacher_sources = [
            source
            for source in ("candidate_1", "candidate_2")
            if "multimodal_teacher" in _text(row, f"{source}_provenance")
        ]
        if len(teacher_sources) != 1:
            return None, ["synthetic_profile_does_not_have_one_teacher_candidate"]
        preferred_source = teacher_sources[0]
        objective = PROFILE_OBJECTIVES[
            _stable_index(_text(row, "sampling_match_id"), len(PROFILE_OBJECTIVES))
        ]
        chosen_metrics, rejected_metrics = _profile_metrics(objective)
    else:
        return None, ["synthetic_unsupported_task"]

    jobs = build_jobs(row, None)  # The rehearsal never opens or sends an image.
    judgments = {
        job.job_id: _judgment(
            job,
            preferred_source=preferred_source,
            chosen_metrics=chosen_metrics,
            rejected_metrics=rejected_metrics,
        )
        for job in jobs
    }
    oriented, reasons = aggregate_pair(row, jobs, judgments, model=REHEARSAL_JUDGE)
    if oriented is None:
        return None, reasons
    validation = validate_pair(oriented)
    if not validation.valid:
        return None, [f"synthetic_pair_policy:{issue}" for issue in validation.issues]
    return oriented, []


def rehearse_pilot_dataflow(
    metadata_rows: Iterable[Mapping[str, Any]],
    teacher_rows: Iterable[Mapping[str, Any]],
    *,
    target_rows: int = 100,
    max_controlled_iat_matches: int | None = 80,
) -> dict[str, Any]:
    metadata = [dict(row) for row in metadata_rows]
    teachers = [dict(row) for row in teacher_rows]
    synthetic_profiles = synthetic_gemma_profile_rows(metadata)
    candidates = build_candidate_pairs(
        metadata,
        teachers + synthetic_profiles,
        policy=CandidatePolicy(
            max_profile_pairs_per_image=2,
            max_iat_contrasts_per_match=0,
            include_controlled_iat=True,
            max_controlled_iat_matches=max_controlled_iat_matches,
            seed="august3-rehearsal-candidates-v1",
        ),
    )

    eligible = []
    rejection_reasons: Counter[str] = Counter()
    for row in candidates.rows:
        audited, reasons = simulate_audited_pair(row)
        if audited is None:
            rejection_reasons.update(reasons)
        else:
            eligible.append(audited)

    balance = balance_pairs(
        eligible,
        policy=BalancePolicy(
            target_rows=target_rows,
            target_iat_fraction=0.30,
            development_fraction=0.20,
            max_iat_fixed_substitution_fraction=0.20,
            max_iat_chosen_tuple_fraction=0.15,
            max_profile_objective_fraction=0.40,
            max_pairs_per_image=1,
            max_profile_chosen_reuse=1,
            seed="august3-rehearsal-freeze-v1",
        ),
    )
    audit = balance.audit.as_dict() if balance.audit is not None else None
    selected = balance.rows
    return {
        "status": "passed" if balance.complete else "failed",
        "policy_version": REHEARSAL_POLICY_VERSION,
        "training_data": False,
        "synthetic_components": ["gemma_profile_responses", "claude_judgments"],
        "metadata_images": len(metadata),
        "teacher_response_rows": len(teachers),
        "candidate_build": candidates.report,
        "synthetic_eligible_rows": len(eligible),
        "synthetic_eligible_by_task": dict(
            sorted(Counter(_text(row, "task_type") for row in eligible).items())
        ),
        "synthetic_eligible_by_objective": dict(
            sorted(Counter(_text(row, "objective_type") for row in eligible).items())
        ),
        "synthetic_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "balance": balance.report,
        "dataset_audit": audit,
        "selected_rows_by_race": dict(
            sorted(Counter(_text(row, "fairface_race") for row in selected).items())
        ),
        "selected_rows_by_task": dict(
            sorted(Counter(_text(row, "task_type") for row in selected).items())
        ),
        "selected_rows_by_objective": dict(
            sorted(Counter(_text(row, "objective_type") for row in selected).items())
        ),
    }
