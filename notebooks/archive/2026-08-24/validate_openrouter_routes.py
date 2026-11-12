#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from multimodel_bias.iat import load_model_specs, validated_model_snapshot
from multimodel_bias.openrouter import OpenRouterClient
from multimodel_bias.secrets import get_openrouter_api_key


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed on OpenRouter model-alias or provider-route drift."
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("config/models_explicit_aug24.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/compatibility/openrouter_route_snapshot_20260824.json"),
    )
    args = parser.parse_args()

    config_document, models = load_model_specs(args.model_config)
    client = OpenRouterClient(get_openrouter_api_key(), max_retries=1)
    credits = client.credits()
    catalog = {entry.get("id"): entry for entry in client.model_catalog()}
    endpoint_documents = {
        model.model_id: client.get(f"/models/{model.model_id}/endpoints").get("data", {})
        for model in models
    }
    snapshot = {
        "captured_at": datetime.now(UTC).isoformat(),
        "model_config": str(args.model_config),
        "catalog_snapshot_date_declared": config_document.get("catalog_snapshot_date"),
        "provider_fallback_allowed": False,
        "data_collection": "deny",
        "zero_data_retention_required": True,
        "credits": credits,
        "key_metadata": client.key_metadata(),
        "catalog_entries": {model.model_id: catalog.get(model.model_id) for model in models},
        "endpoint_documents": endpoint_documents,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        snapshot["models"] = validated_model_snapshot(client, models)
        snapshot["validation_status"] = "passed"
    except ValueError as exc:
        snapshot["validation_status"] = "failed"
        snapshot["validation_error"] = str(exc)
        args.output.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    args.output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "validated_models": len(models),
                "remaining_credit_usd": credits["remaining"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
