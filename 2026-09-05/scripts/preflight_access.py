#!/usr/bin/env python3
"""Read-only credential/route/stock checks. No inference or deployment."""

from __future__ import annotations

import getpass
import json
import ssl
import subprocess
import sys
import urllib.request
import urllib.error
import re

import certifi

from common import ROOT, utcnow, write_json

sys.path.insert(0, str(ROOT / "src"))
from multimodel_bias.iat import load_model_specs, validated_model_snapshot
from multimodel_bias.openrouter import OpenRouterClient


def secret(service):
    result = subprocess.run(
        ["security", "find-generic-password", "-a", getpass.getuser(), "-s", service, "-w"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def runpod_request(path, *, query=None, payload=None, method="GET", variables=None):
    key = secret("runpod-api-key")
    url = "https://api.runpod.io/graphql" if query else "https://rest.runpod.io/v1" + path
    body = {"query": query} if query else payload
    if query and variables is not None:
        body["variables"] = variables
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body else None,
        method="POST" if query else method,
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "User-Agent": "Research-RunPod-ReadOnly-Check/1.0",
        },
    )
    try:
        with urllib.request.urlopen(
            req, context=ssl.create_default_context(cafile=certifi.where()), timeout=30
        ) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            doc = json.loads(raw)
            if isinstance(doc, list):
                detail = str(
                    [
                        {k: item.get(k) for k in ("code", "path", "message")}
                        for item in doc
                        if isinstance(item, dict)
                    ]
                )
            else:
                detail = str(doc.get("error", doc.get("message", "No error message")))
        except (ValueError, AttributeError):
            detail = re.sub(r"<[^>]+>", " ", raw)[:500]
        detail = re.sub(r"ssh-ed25519\s+\S+", "[PUBLIC_KEY_REDACTED]", detail)
        raise RuntimeError(
            f"RunPod HTTP {exc.code}: {detail.replace(key, '[REDACTED]')[:500]}"
        ) from None
    except Exception as exc:
        raise RuntimeError(f"RunPod {method} failed ({type(exc).__name__})") from None


def main():
    client = OpenRouterClient(secret("openrouter-api-key"), max_retries=0)
    _, models = load_model_specs(ROOT / "config/models_aug31.json")
    catalog = client.model_catalog()
    write_json(ROOT / "results/preflight/openrouter_catalog.json", catalog)
    routes = []
    for model in models:
        try:
            snapshot = validated_model_snapshot(client, [model])[0]
            routes.append({"model_key": model.key, "status": "metadata_valid", **snapshot})
        except Exception as exc:
            routes.append(
                {"model_key": model.key, "status": "unavailable", "error": str(exc)[:600]}
            )
    write_json(ROOT / "results/preflight/openrouter_routes.json", routes)
    balance = client.credits()
    account = runpod_request("", query="query { myself { clientBalance currentSpendPerHr } }")
    volumes = runpod_request("/networkvolumes")
    volume = [v for v in volumes if v["id"] == "zecocmq95n"]
    stock = runpod_request(
        "",
        query='query { gpuTypes(input: {id:"NVIDIA H100 80GB HBM3"}) '
        "{ id memoryInGb lowestPrice(input: {gpuCount:1,secureCloud:true,"
        'dataCenterId:"US-NE-1",supportPublicIp:true}) '
        "{ stockStatus uninterruptablePrice } } }",
    )
    summary = {
        "checked_at": utcnow(),
        "openrouter": balance,
        "runpod": account.get("data"),
        "network_volume": volume,
        "gpu_stock": stock.get("data"),
        "routes": [{k: r[k] for k in ("model_key", "status")} for r in routes],
    }
    write_json(ROOT / "results/preflight/access.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
