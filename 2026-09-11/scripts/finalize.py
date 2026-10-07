"""Read-only API reconciliation and offline generation-contract audit; no model calls."""
import fcntl
import hashlib
import sys

from common import ROOT, json_read, now, read_jsonl, rows, secret, write_json

sys.path.insert(0, str(ROOT / "src"))
from multimodel_bias.constants import IAT_PROMPT
from multimodel_bias.iat import EndpointUnavailableError, ModelSpec, validated_model_snapshot
from multimodel_bias.openrouter import OpenRouterClient


def audit_archives():
    study = json_read(ROOT / "config/study.json")
    keys = ("model_id", "api_revision_hint", "provider_endpoint", "max_tokens",
            "reasoning_effort", "supports_temperature", "supports_top_p")
    results = {}
    for panel, config in study["panels"].items():
        old = json_read(ROOT / ("reference/neutral12_run_manifest.json" if panel == "neutral12"
                                else "reference/models_aug31.json"))
        source_config = json_read(ROOT / ("reference/neutral12_original_config.json" if panel == "neutral12"
                                           else "reference/iat_utkface_aug31.json"))
        contract = json_read(ROOT / f"results/hosted/{panel}/contract.json")
        for key in ("temperature", "top_p", "base_seed", "fallback_enabled", "fallback_max_tokens", "provider", "benchmark_version"):
            assert source_config[key] == contract["config"][key], (panel, key)
        assert contract["prompt"] == IAT_PROMPT
        old_models = {m["key"]: m for m in old["models"]}
        new_models = {m["key"]: m for m in config["models"]}
        images = {r["image_id"]: r for r in rows(ROOT / config["manifest"])}
        archived = rows(ROOT / f"results/hosted/{panel}/archived_responses.csv")
        for row in archived:
            model = new_models[row["model_key"]]
            assert all(model[k] == old_models[row["model_key"]][k] for k in keys)
            assert row["image_sha256"] == images[row["image_id"]]["sha256"]
            assert int(row["iteration"]) <= source_config["repeats"]
            assert row["benchmark_version"] == source_config["benchmark_version"]
            assert row["model_id_requested"] == model["model_id"]
            assert int(row["max_tokens"]) == model["max_tokens"]
            assert row["reasoning_effort"] == model["reasoning_effort"]
            for key in ("temperature", "top_p"):
                assert float(row[key + "_requested"]) == source_config[key]
        results[panel] = dict(archived_slots=len(archived),
                             archived_technical_errors=sum(r["response_source"] == "api_error" for r in archived),
                             model_and_generation_contracts_match=True, image_hashes_match=True,
                             source_repetitions=source_config["repeats"], target_repetitions=config["repeats"])
    original = json_read(ROOT / "reference/neutral12_august24_reuse_audit.json")
    assert original["checks_passed"]["prompt_sha256"] == hashlib.sha256(IAT_PROMPT.encode()).hexdigest()
    return results


def main():
    lock = (ROOT / "results/hosted/worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    archive_audit = audit_archives()
    client = OpenRouterClient(secret("openrouter-api-key"), max_retries=0, timeout_seconds=30)
    post = ROOT / "results/postflight"
    post.mkdir(exist_ok=True)
    write_json(post / "catalog.json", client.model_catalog())
    endpoints = {}
    for config in json_read(ROOT / "config/study.json")["panels"].values():
        for data in config["models"]:
            model = ModelSpec(**data)
            key = model.model_id + "|" + model.provider_endpoint
            if key in endpoints:
                continue
            try:
                endpoints[key] = validated_model_snapshot(client, [model])[0]
            except EndpointUnavailableError as exc:
                endpoints[key] = dict(status=exc.status, reason=str(exc))
    write_json(post / "routes.json", endpoints)
    calls = read_jsonl(ROOT / "results/hosted/calls.jsonl")
    session = json_read(ROOT / "results/hosted/session.json")
    credits = client.credits()
    session.update(credits_latest=credits, checked_at=now(),
                   account_usage_delta=credits["total_usage"]-session["credits_initial"]["total_usage"],
                   api_calls=len(calls), reported_cost_usd=sum(float(r.get("cost_usd", 0)) for r in calls),
                   errors=sum(bool(r.get("error")) for r in calls))
    for call in calls:
        if call.get("error"):
            continue
        response = call["response"]
        config = json_read(ROOT / "config/study.json")["panels"][call["panel"]]
        model = next(m for m in config["models"] if m["key"] == call["model_key"])
        assert response["model"] in {model["model_id"], model["api_revision_hint"]}
        assert call["provider_endpoint"] == model["provider_endpoint"]
        assert call["request_parameters"]["provider"]["allow_fallbacks"] is False
        assert call["request_parameters"]["provider"]["only"] == [model["provider_endpoint"]]
    write_json(ROOT / "results/hosted/session.json", session)
    write_json(post / "audit.json", dict(checked_at=now(), archived=archive_audit,
               new_call_route_contracts_match=True, catalog_revision_checks_passed=True,
               api_call_attempts=len(calls), successful_model_calls=sum(not r.get("error") for r in calls),
               unknown_outcome_calls=sum(r.get("error") == "unknown_call_outcome" for r in calls),
               account_minus_ledger_usd=session["account_usage_delta"]-session["reported_cost_usd"],
               note="No new inference. Unavailable routes are recorded, not replaced. Hosted weight bytes cannot be independently verified."))
    print(session)


if __name__ == "__main__":
    main()
