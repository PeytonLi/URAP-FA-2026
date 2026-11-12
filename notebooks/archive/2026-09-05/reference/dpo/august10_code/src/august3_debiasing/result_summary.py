from __future__ import annotations

from typing import Any, Mapping


def _status(value: Mapping[str, Any], expected: str, label: str) -> None:
    if value.get("status") != expected:
        raise ValueError(f"{label} status is {value.get('status')!r}, expected {expected!r}")


def build_final_summary(
    *,
    training_manifest: Mapping[str, Any],
    development_generation_manifest: Mapping[str, Any],
    development_generation_gate: Mapping[str, Any],
    iat_manifest: Mapping[str, Any],
    profile_manifest: Mapping[str, Any],
    base_iat: Mapping[str, Any],
    adapter_iat: Mapping[str, Any],
    base_neutral: Mapping[str, Any],
    adapter_neutral: Mapping[str, Any],
    profile_summary: Mapping[str, Any],
    iat_acceptance: Mapping[str, Any],
    profile_acceptance: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _status(training_manifest, "complete", "training manifest")
    _status(
        development_generation_manifest,
        "complete",
        "development generation manifest",
    )
    _status(iat_manifest, "complete", "IAT evaluation manifest")
    _status(profile_manifest, "complete", "profile evaluation manifest")
    if not profile_manifest.get("confirmatory"):
        raise ValueError("profile evaluation is a preflight, not the complete confirmatory panel")
    if iat_manifest.get("base_inference_rerun") is not False:
        raise ValueError("IAT evaluation did not preserve the immutable base responses")
    if development_generation_manifest.get("development_only") is not True:
        raise ValueError("development generation did not use the development-only protocol")
    if development_generation_manifest.get("confirmatory_evaluation_opened") is not False:
        raise ValueError("development generation opened confirmatory evaluation data")

    training_hashes = dict(training_manifest.get("final_adapter_hashes", {}))
    development_hashes = dict(
        development_generation_manifest.get("adapter_hashes", {})
    )
    iat_hashes = dict(iat_manifest.get("adapter_hashes", {}))
    profile_hashes = dict(profile_manifest.get("adapter_hashes", {}))
    hashes_complete = bool(
        training_hashes and development_hashes and iat_hashes and profile_hashes
    )
    hashes_match = (
        hashes_complete
        and training_hashes == development_hashes == iat_hashes == profile_hashes
    )
    development_gate = training_manifest.get("development_preference_gate", {})
    reference_hashes = iat_manifest.get("verified_reference_hashes", {})
    required_reference_files = {
        "base_gemma3_fairface_iat.csv",
        "base_gemma3_neutral_iat.csv",
        "eval_metadata.csv",
        "neutral_benchmark_metadata.csv",
        "frozen_input_manifest.json",
    }
    reference_checks_verified = (
        isinstance(reference_hashes, Mapping)
        and required_reference_files <= set(reference_hashes)
        and all(
            isinstance(reference_hashes[name], str)
            and len(reference_hashes[name]) == 64
            for name in required_reference_files
        )
    )
    protocol_matched = (
        iat_manifest.get("generation_protocol_mismatches") == []
        and not iat_manifest.get("protocol_mismatch_explicitly_allowed")
    )
    checks = {
        "exact_adapter_hashes_match_all_stages": hashes_match,
        "development_preference_gate_pass": bool(development_gate.get("pass")),
        "development_generation_guardrail_pass": bool(
            development_generation_gate.get("guardrail_pass")
        ),
        "iat_acceptance_pass": bool(iat_acceptance.get("pass")),
        "profile_acceptance_pass": bool(profile_acceptance.get("pass")),
        "profile_panel_confirmatory": bool(profile_manifest.get("confirmatory")),
        "immutable_iat_base_reused": iat_manifest.get("base_inference_rerun") is False,
        "frozen_iat_reference_checks_verified": reference_checks_verified,
        "frozen_iat_generation_protocol_matched": protocol_matched,
        "profile_bootstrap_uses_matched_race_blocks": (
            profile_manifest.get("bootstrap_resampling_unit")
            == "matched_black_white_block"
        ),
    }

    key_metrics: list[dict[str, Any]] = []

    def metric(
        family: str,
        group: str,
        name: str,
        base: float,
        adapter: float,
        direction: str,
    ) -> None:
        key_metrics.append(
            {
                "family": family,
                "group": group,
                "metric": name,
                "base": base,
                "adapter": adapter,
                "adapter_minus_base": adapter - base,
                "desired_direction": direction,
            }
        )

    for race in ("Black", "White"):
        metric(
            "fairface_iat",
            race,
            "negative_rate",
            float(base_iat["race_metrics"][race]["negative_rate"]),
            float(adapter_iat["race_metrics"][race]["negative_rate"]),
            "decrease_or_equal",
        )
        metric(
            "neutral_12_face_iat",
            race,
            "negative_rate",
            float(base_neutral["race_metrics"][race]["negative_rate"]),
            float(adapter_neutral["race_metrics"][race]["negative_rate"]),
            "decrease_or_equal",
        )
    metric(
        "fairface_iat",
        "Black_minus_White",
        "signed_negative_rate_gap",
        float(base_iat["negative_rate_gap_b_minus_w"]),
        float(adapter_iat["negative_rate_gap_b_minus_w"]),
        "toward_zero",
    )
    metric(
        "neutral_12_face_iat",
        "Black_minus_White",
        "signed_negative_rate_gap",
        float(base_neutral["negative_rate_gap_b_minus_w"]),
        float(adapter_neutral["negative_rate_gap_b_minus_w"]),
        "toward_zero",
    )
    for group in ("overall", "Black", "White"):
        profile_group = profile_summary["groups"][group]
        for name, direction in (
            ("quality_score", "preserve_or_increase"),
            ("grounded_detail_count", "preserve_or_increase"),
            ("visible_expression_support", "preserve_or_increase"),
            ("identity_leakage_count", "decrease_or_equal"),
            ("person_negative_count", "decrease_or_equal"),
            ("unwarranted_negative_count", "decrease_or_equal"),
            ("unsupported_trait_count", "decrease_or_equal"),
            ("invented_biography_count", "decrease_or_equal"),
        ):
            metric(
                "grounded_profile",
                group,
                name,
                float(profile_group["base"]["judge_metrics"][name]),
                float(profile_group["adapter"]["judge_metrics"][name]),
                direction,
            )
        for name, direction in (
            ("length_valid_rate", "preserve_or_increase"),
            ("unique_response_rate", "preserve_or_increase"),
            ("distinct_bigram_rate", "preserve_or_increase"),
            ("mechanical_identity_leakage_rate", "decrease_or_equal"),
        ):
            metric(
                "grounded_profile_mechanical",
                group,
                name,
                float(profile_group["base"][name]),
                float(profile_group["adapter"][name]),
                direction,
            )

    summary = {
        "status": "complete",
        "experiment_revision": "2026-08-03",
        "pass": all(checks.values()),
        "checks": checks,
        "adapter_hashes": training_hashes,
        "training": {
            "train_rows": training_manifest.get("train_rows"),
            "development_rows": training_manifest.get("development_rows"),
            "gradient_steps": training_manifest.get("gradient_steps"),
            "optimizer_updates": training_manifest.get("optimizer_updates"),
            "gpu_memory": training_manifest.get("gpu_memory"),
            "development_preference_gate": development_gate,
            "development_generation_gate": development_generation_gate,
        },
        "iat": {
            "acceptance": iat_acceptance,
            "base_signed_gap": base_iat["negative_rate_gap_b_minus_w"],
            "adapter_signed_gap": adapter_iat["negative_rate_gap_b_minus_w"],
            "signed_gap_change": (
                float(adapter_iat["negative_rate_gap_b_minus_w"])
                - float(base_iat["negative_rate_gap_b_minus_w"])
            ),
        },
        "profiles": {
            "acceptance": profile_acceptance,
            "images": profile_summary["rows"],
            "image_preferences": profile_summary["groups"]["overall"][
                "image_preferences"
            ],
            "image_order_stable_rate": profile_summary["groups"]["overall"][
                "image_order_stable_rate"
            ],
        },
    }
    return summary, key_metrics
