"""Read-only Prime catalogs; never provision compute, upload data or launch runs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import study

MODELS = ("google/gemma-4-26B-A4B-it", "Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3.5-9B")
QUERIES = {
    "shared_models": ("train", "models", "--output", "json"),
    "dedicated_cached_models": ("train", "models", "--fft-only", "--output", "json"),
    "training_options": ("train", "configs", "--output", "json"),
    "h100_single_gpu": ("availability", "list", "--gpu-type", "H100_80GB", "--gpu-count", "1",
                        "--output", "json"),
    "h200_single_gpu": ("availability", "list", "--gpu-type", "H200_141GB", "--gpu-count", "1",
                        "--output", "json"),
    "b200_single_gpu": ("availability", "list", "--gpu-type", "B200_180GB", "--gpu-count", "1",
                        "--output", "json"),
}


def read_catalog(cli, query, *, runner=subprocess.run):
    if query not in QUERIES:
        raise ValueError("Only the allowlisted read-only catalog queries are permitted")
    command = ["rtk", "proxy", str(cli), "--plain", *QUERIES[query]]
    environment = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    try:
        result = runner(command, capture_output=True, text=True, encoding="utf-8",
                        env=environment, timeout=60, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "query": list(QUERIES[query])}
    except OSError as exc:
        return {"status": "local_execution_error", "error_type": type(exc).__name__,
                "query": list(QUERIES[query])}
    if result.returncode:
        # Do not retain uncontrolled error bodies, which can contain account
        # information or request details. Only classify obvious HTTP status codes.
        error = result.stdout + result.stderr
        statuses = [code for code in ("401", "403", "404", "429", "500", "502", "503")
                    if code in error]
        return {"status": "request_failed", "exit_code": result.returncode,
                "http_status_candidates": statuses, "query": list(QUERIES[query])}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"status": "invalid_json", "query": list(QUERIES[query])}
    return {"status": "ok", "query": list(QUERIES[query]), "payload": payload}


def summarize(catalogs):
    shared = catalogs.get("shared_models", {})
    names = {row["name"] for row in shared.get("payload", {}).get("models", [])}
    dedicated = catalogs.get("dedicated_cached_models", {})
    cached = {row["name"] for row in dedicated.get("payload", {}).get("available_fft_models", [])}
    matches = []
    for model in MODELS:
        matches.append({"model": model,
                        "listed_in_shared_catalog": model in names if shared.get("status") == "ok" else None,
                        "listed_in_dedicated_cache_catalog": model in cached
                        if dedicated.get("status") == "ok" else None,
                        "image_sft_verified": False, "image_orpo_verified": False})
    hardware = {}
    for name in ("h100_single_gpu", "h200_single_gpu", "b200_single_gpu"):
        response = catalogs.get(name, {})
        rows = response.get("payload", {}).get("gpu_resources", [])
        # Quotes are not capacity reservations, feasibility tests or invoices.
        available = [row for row in rows if row.get("stock_status") == "Available"]
        prices = [row["price_value"] for row in available
                  if isinstance(row.get("price_value"), (int, float))]
        hardware[name] = {"status": response.get("status"), "resource_count": len(rows),
                          "available_resource_count": len(available),
                          "minimum_available_quoted_price_usd_per_hour": min(prices) if prices else None}
    return {"models": matches, "gpu_quotes": hardware,
            "catalog_membership_is_not_algorithm_or_image_support": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    user_directory = Path(os.environ.get("USERPROFILE", str(Path.home())))
    parser.add_argument("--cli", type=Path, default=user_directory / ".local/bin/prime.exe")
    args = parser.parse_args()
    if not args.cli.is_file():
        raise ValueError("Working Prime CLI executable not found; no credentials or PATH are modified")
    catalogs = {name: read_catalog(args.cli, name) for name in QUERIES}
    report = {"checked_at": study.now(), "cli": str(args.cli.resolve()),
              "catalogs": catalogs, "summary": summarize(catalogs),
              "training_performed": False, "optimizer_updates": 0,
              "billable_resources_created": False, "images_uploaded": False,
              "spending_authorization_evaluated": False,
              "scope": "Read-only catalogs; no target-policy or backend feasibility approval"}
    receipt_name = "catalogs_" + report["checked_at"].replace(":", "").replace("-", "").replace("+", "_") + ".json"
    receipt = study.ROOT / "results/prime" / receipt_name
    study.save_json(receipt, report)
    report["immutable_receipt"] = str(receipt.relative_to(study.ROOT))
    study.save_json(study.ROOT / "results/prime_preflight.json", report)
    print(json.dumps({"checked_at": report["checked_at"], "summary": report["summary"]}, indent=2))
    # Preserve partial evidence rather than treating a hidden dedicated endpoint
    # error as an empty catalog or claiming a successful full preflight.
    return 0 if all(row["status"] == "ok" for row in catalogs.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
