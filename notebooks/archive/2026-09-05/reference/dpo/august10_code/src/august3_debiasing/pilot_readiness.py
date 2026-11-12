from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

EXPECTED_GEMMA_PLAN = {
    "protocol_version": "august3-gemma-candidate-generation-v2",
    "status": "validated_dry_run",
    "dry_run": True,
    "images": 160,
    "expected_responses": 1600,
    "tasks": ["grounded_profile", "iat_3word"],
    "model_id": "google/gemma-3-12b-it",
    "model_revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
    "attention_implementation": "sdpa",
    "torch_dtype": "bfloat16",
    "generation_seed": 8032026,
    "samples_per_image": {"grounded_profile": 2, "iat_3word": 8},
    "generation_configs": {
        "grounded_profile": {
            "do_sample": True,
            "max_new_tokens": 240,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "iat_3word": {
            "do_sample": True,
            "max_new_tokens": 30,
            "temperature": 1.0,
            "top_p": 1.0,
        },
    },
    "allow_hf_download": False,
    "allow_unpinned_model": False,
    "job_plan_sha256": "5aa1b74c13b14bc642ff1632cc7058c7883bd3476842f8cc86de0b7de40cf7ff",
}

EXPECTED_CANDIDATE_BUILD_PROTOCOL = "august3-candidate-build-v2"
EXPECTED_CLAUDE_AUDIT_PROTOCOL = (
    "august3-claude-pair-audit-v5-iat-double-text-tie"
)
EXPECTED_BALANCE_PROTOCOL = "august3-balanced-freeze-v2"


def gemma_plan_issues(
    manifest: Mapping[str, Any],
    *,
    metadata_sha256: str,
    pair_manifest_sha256: str,
) -> list[str]:
    expected = {
        **EXPECTED_GEMMA_PLAN,
        "metadata_csv_sha256": metadata_sha256,
        "expression_pair_manifest_sha256": pair_manifest_sha256,
    }
    return [
        f"{key}:expected={value!r}:actual={manifest.get(key)!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]


def claude_provider_stage(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    if manifest is None:
        return {
            "stage": "claude_provider_access",
            "status": "pending",
            "issues": ["one-pair provider preflight has not been supplied"],
        }
    status = str(manifest.get("status", "unknown"))
    if status == "complete" and int(manifest.get("completed_judge_calls", 0)) > 0:
        return {
            "stage": "claude_provider_access",
            "status": "passed",
            "judge_model": manifest.get("judge_model"),
            "backend": manifest.get("backend"),
            "region": manifest.get("vertex_region"),
            "completed_judge_calls": int(manifest.get("completed_judge_calls", 0)),
            "issues": [],
        }
    failure = str(manifest.get("failure_message", ""))
    quota_match = re.search(r"([a-z_]+requests_per_base_model)", failure)
    error_code = (
        "RESOURCE_EXHAUSTED" if "RESOURCE_EXHAUSTED" in failure else "provider_failure"
    )
    return {
        "stage": "claude_provider_access",
        "status": "blocked" if "429" in failure else "failed",
        "judge_model": manifest.get("judge_model"),
        "backend": manifest.get("backend"),
        "region": manifest.get("vertex_region"),
        "completed_judge_calls": int(manifest.get("completed_judge_calls", 0)),
        "error_code": error_code,
        "quota_dimension": quota_match.group(1) if quota_match else None,
        "issues": ["provider preflight did not complete a judgment"],
    }


def readiness_summary(stages: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(stage["stage"]): stage for stage in stages}
    required_for_training = (
        "verified_pair_inputs",
        "teacher_profiles",
        "gemma_generation_plan",
        "gemma_response_pool",
        "claude_provider_access",
        "candidate_build",
        "claude_bulk_audit",
        "frozen_pilot_csv",
    )
    unresolved = [
        name
        for name in required_for_training
        if by_name.get(name, {}).get("status") != "passed"
    ]
    external = [
        name
        for name in ("gemma_response_pool", "claude_provider_access")
        if by_name.get(name, {}).get("status") in {"pending", "blocked"}
    ]
    return {
        "status": (
            "ready_for_training"
            if not unresolved
            else "blocked_external"
            if external
            else "not_ready"
        ),
        "ready_for_training": not unresolved,
        "unresolved_stages": unresolved,
        "external_blockers": external,
        "stages": stages,
    }
