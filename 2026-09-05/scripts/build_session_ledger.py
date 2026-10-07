#!/usr/bin/env python3
"""Summarize saved billing snapshots and actual new generation records, offline."""

import json
from datetime import datetime

from common import ROOT, read_csv, write_csv, write_json


def load(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    initial = load("results/preflight/access.json")
    final = load("results/runpod/final_account.json")
    pod = load("runtime/pod.json")
    termination = load("results/runpod/termination_request.json")
    hosted = read_csv(ROOT / "results/hosted_visbias12/responses.csv")
    rows = []
    for key in sorted({r["model_key"] for r in hosted}):
        selected = [r for r in hosted if r["model_key"] == key]
        rows.append(
            dict(
                benchmark="VisBias neutral12",
                model_key=key,
                logical_tasks=len(selected),
                generation_calls=len(selected),
                reported_cost_usd=sum(float(r.get("cost_usd") or 0) for r in selected),
                provider="|".join(sorted({r.get("provider", "") for r in selected})),
                evidence="results/hosted_visbias12/responses.csv",
            )
        )
    gptroot = ROOT / "results/utkface_recovered/gpt_utkface_recovered_20260905"
    calls = [
        json.loads(line) for line in (gptroot / "api_responses.jsonl").read_text().splitlines()
    ]
    gpt = read_csv(gptroot / "responses.csv")
    rows.append(
        dict(
            benchmark="IAT UTKFace400",
            model_key="gpt56_luna",
            logical_tasks=len(gpt),
            generation_calls=len(calls),
            reported_cost_usd=sum(
                float(r["response"].get("usage", {}).get("cost") or 0) for r in calls
            ),
            provider="Azure",
            evidence=str((gptroot / "api_responses.jsonl").relative_to(ROOT)),
        )
    )
    local = load("results/gemma_local/manifest.json")
    probe = load("results/gemma_probe/manifest.json")
    hours = (
        datetime.fromisoformat(termination["requested_at"])
        - datetime.fromisoformat(pod["deployed_at"])
    ).total_seconds() / 3600
    router_delta = final["openrouter"]["total_usage"] - initial["openrouter"]["total_usage"]
    runpod_delta = (
        initial["runpod"]["myself"]["clientBalance"] - final["runpod"]["myself"]["clientBalance"]
    )
    reported = sum(r["reported_cost_usd"] for r in rows)
    result = dict(
        account_snapshot_at=final["checked_at"],
        openrouter_generation_calls=sum(r["generation_calls"] for r in rows),
        openrouter_logical_tasks=sum(r["logical_tasks"] for r in rows),
        openrouter_reported_call_cost_usd=reported,
        openrouter_account_usage_delta_usd=router_delta,
        openrouter_account_minus_reported_usd=router_delta - reported,
        openrouter_remaining_usd=final["openrouter"]["remaining"],
        local_full_run_tasks=local["completed_tasks"],
        local_full_run_generation_calls=local["generation_calls"],
        local_probe_generation_calls=probe["generation_calls"],
        local_valid_full_run_tasks=local["valid_tasks"],
        local_full_run_elapsed_seconds=local["elapsed_seconds"],
        runpod_deployment_to_termination_hours=hours,
        runpod_list_rate_usd_per_hour=pod["cost_per_hour"],
        runpod_walltime_estimate_usd=hours * pod["cost_per_hour"],
        runpod_account_balance_delta_usd=runpod_delta,
        runpod_remaining_usd=final["runpod"]["myself"]["clientBalance"],
        remaining_runpod_storage_spend_usd_per_hour=final["runpod"]["myself"]["currentSpendPerHr"],
        total_observed_account_deltas_usd=router_delta + runpod_delta,
        gpu_terminated=True,
        network_volume_preserved=True,
        new_training=False,
        billing_caveat="Account deltas include delayed billing/other account activity and storage; not a per-run invoice. GPU wall time uses deployment-to-termination-request timestamps, not exact provider billable seconds.",
        no_direct_calls_to=["Anthropic", "OpenAI", "Gemini"],
    )
    write_csv(ROOT / "report/SESSION_CALLS.csv", rows)
    write_json(ROOT / "report/SESSION_LEDGER.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
