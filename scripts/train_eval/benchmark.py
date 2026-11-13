"""Reuse the existing Fireworks benchmark; never train on the 12 test images."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import study

# Reuse the old benchmark without shadowing this study's sibling report.py.
sys.path.append(str(study.REPO / "2026-09-11/scripts"))
import fireworks_gemma4 as fw
from multimodel_bias.iat import image_data_url

MODELS = {
    "gemma4": ("gemma-4-26b-a4b-it", "Gemma 4 26B A4B IT"),
    "gemma3_12b": ("gemma-3-12b-it", "Gemma 3 12B IT"),
    "qwen3vl": ("qwen3-vl-8b-instruct", "Qwen3-VL 8B Instruct"),
    "qwen35": ("qwen3p5-9b", "Qwen3.5 9B"),
}


def wait_for_deployment(deployment):
    deadline = time.monotonic() + 1200
    previous = None
    while time.monotonic() < deadline:
        state = study.cli_json("deployment", "get", deployment)
        state.pop("created_by", None)
        study.save_json(study.ROOT / "results/deployments" / (deployment.rsplit("/", 1)[1] + "_latest.json"), state)
        if state["state"] != previous:
            print(f"{deployment}: {state['state']}", flush=True)
            previous = state["state"]
        if state["state"] == "READY":
            return
        if state["state"] in ("FAILED", "DELETING", "DELETED"):
            raise RuntimeError(f"Deployment entered {state['state']}; do not restart it automatically")
        time.sleep(30)
    raise TimeoutError("Deployment did not become ready in the bounded 20-minute wait")


def cleanup_owned_deployment(deployment, name):
    receipt_path = study.ROOT / f"results/deployments/{name}_created.json"
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    current = study.cli_json("deployment", "get", deployment)
    if original["name"] != deployment or current["base_model"] != original["base_model"]:
        raise RuntimeError("Cleanup refused: deployment differs from this study's creation receipt")
    # These owned benchmark deployments have finished serving. The normal delete
    # guard blocks any deployment used in the last hour, even after our run exits.
    result = subprocess.run([str(study.FIRECTL), "deployment", "delete", deployment, "--ignore-checks"],
                            capture_output=True, text=True, timeout=90)
    receipt = {"requested_at": study.now(), "deployment": deployment,
               "delete_acknowledged": result.returncode == 0, "exit_code": result.returncode}
    study.save_json(study.ROOT / f"results/deployments/{name}_cleanup.json", receipt)
    if result.returncode:
        raise RuntimeError("Study deployment deletion failed; billable resource requires attention")
    print(f"Cleanup requested for study deployment {deployment}", flush=True)


def probe(selected, image_dir, deployment=None):
    key, _ = study.fireworks_auth()
    client = fw.OpenRouterClient(key, base_url=fw.FIREWORKS_BASE_URL,
                                max_retries=0, timeout_seconds=45)
    stimuli = fw.load_stimuli(study.IAT12_MANIFEST, image_dir=image_dir)
    receipts = []
    for name in selected:
        model_id, display = MODELS[name]
        row = {"checked_at": study.now(), "model_key": name, "display_name": display,
               "base_model": "accounts/fireworks/models/" + model_id,
               "model_id": deployment or "accounts/fireworks/models/" + model_id,
               "image_sha256": stimuli[0].sha256, "training_performed": False}
        try:
            response = client.chat_completion(fw.fireworks_payload(
                row["model_id"], image_data_url(stimuli[0].path), fw.run_config(1),
                17083027, fallback=False))
            row["response"] = fw._safe_response_snapshot(response)
            row["endpoint_usable"] = bool(response.get("choices"))
            print(f"{display}: HTTP success; image request accepted", flush=True)
        except Exception as exc:
            row["endpoint_usable"] = False
            row["error"] = str(exc)
            print(f"{display}: {exc}", flush=True)
        receipts.append(row)
    suffix = "dedicated" if deployment else "serverless"
    study.save_json(study.ROOT / f"results/probes/{'-'.join(selected)}_{suffix}.json", receipts)
    return receipts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "smoke", "run"))
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--image-dir", type=Path, default=study.IAT12_IMAGE_DIR)
    parser.add_argument("--deployment", help="Dedicated deployment resource for exactly one selected model")
    parser.add_argument("--wait-for-ready", action="store_true")
    parser.add_argument("--cleanup", action="store_true", help="Delete only the deployment in this study's creation receipt")
    args = parser.parse_args()
    if args.deployment and (len(args.models) != 1 or not args.deployment.startswith("accounts/")
                           or "/deployments/" not in args.deployment):
        parser.error("--deployment requires one --models selection and a full deployment resource")
    if (args.wait_for_ready or args.cleanup) and not args.deployment:
        parser.error("--wait-for-ready / --cleanup require a dedicated deployment")
    try:
        if args.wait_for_ready:
            wait_for_deployment(args.deployment)
        receipts = probe(args.models, args.image_dir, args.deployment)
        if args.action == "probe":
            return 0 if all(row["endpoint_usable"] for row in receipts) else 2
        key, _ = study.fireworks_auth()
        environment = {**os.environ, "FIREWORKS_API_KEY": key, "PYTHONUNBUFFERED": "1"}
        for receipt in receipts:
            if not receipt["endpoint_usable"]:
                print(f"Skipping unusable endpoint: {receipt['display_name']}", flush=True)
                continue
            name = receipt["model_key"]
            command = [sys.executable, str(study.REPO / "2026-09-11/scripts/fireworks_gemma4.py"),
                       args.action, "--image-dir", str(args.image_dir), "--model", receipt["model_id"],
                       "--base-model", receipt["base_model"], "--model-key", name + "_base",
                       "--display-name", receipt["display_name"] + " (Fireworks base)",
                       "--run-id", name + "_base_20261003", "--output-root", str(study.ROOT / "results/benchmarks")]
            subprocess.run(command, env=environment, check=True)
        return 0 if all(row["endpoint_usable"] for row in receipts) else 2
    finally:
        if args.cleanup:
            cleanup_owned_deployment(args.deployment, args.models[0])


if __name__ == "__main__":
    raise SystemExit(main())
