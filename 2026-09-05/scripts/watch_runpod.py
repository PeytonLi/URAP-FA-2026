#!/usr/bin/env python3
"""Monitor our pod; pull and verify results before releasing compute, never the volume."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time

from common import ROOT, sha, utcnow, write_json
from preflight_access import runpod_request

REMOTE = "/workspace/research-evidence-september5"


def connection():
    state = json.loads((ROOT / "runtime/pod.json").read_text())
    response = runpod_request(
        "",
        query='query { pod(input: {podId:"'
        + state["pod_id"]
        + '"}) { id desiredStatus runtime { ports { ip isIpPublic '
        "privatePort publicPort type } } } }",
    )
    pod = response["data"]["pod"]
    ports = (pod.get("runtime") or {}).get("ports") or []
    port = next(p for p in ports if p["isIpPublic"] and p["privatePort"] == 22)
    ssh = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={ROOT}/runtime/known_hosts",
        "-p",
        str(port["publicPort"]),
        "-i",
        str(ROOT.home() / ".ssh/id_ed25519"),
    ]
    return ssh, f"root@{port['ip']}"


def sync_and_verify(ssh, target):
    hashes = subprocess.run(
        ssh + [target, f"python3 {REMOTE}/scripts/remote_artifact_hashes.py"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    records = json.loads(hashes.stdout)
    if not records:
        raise ValueError("No remote artifacts to verify")
    for folder in ("results", "runtime"):
        destination = ROOT / ("runtime/remote" if folder == "runtime" else "results")
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                shlex.join(ssh),
                f"{target}:{REMOTE}/{folder}/",
                str(destination) + "/",
            ],
            check=True,
            timeout=180,
        )
    for row in records:
        relative = row["path"]
        local = ROOT / (
            relative.replace("runtime/", "runtime/remote/", 1)
            if relative.startswith("runtime/")
            else relative
        )
        row["local_path"] = str(local.relative_to(ROOT))
        row["matches"] = local.is_file() and sha(local) == row["sha256"]
    receipt = {
        "checked_at": utcnow(),
        "all_match": all(r["matches"] for r in records),
        "files": records,
    }
    write_json(ROOT / "results/runpod/download_verification.json", receipt)
    if not receipt["all_match"]:
        raise ValueError("Remote artifact changed during transfer; retry after worker stops")
    return receipt


def main():
    state = json.loads((ROOT / "runtime/pod.json").read_text())
    while True:
        elapsed = time.time() - state["deployed_epoch"]
        try:
            ssh, target = connection()
            check = subprocess.run(
                ssh + [target, f"test -f {REMOTE}/runtime/run_exit.json"], timeout=30
            )
            if check.returncode == 0:
                receipt = sync_and_verify(ssh, target)
                print(f"Verified {len(receipt['files'])} remote artifacts", flush=True)
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/runpod_control.py"),
                        "terminate",
                        "--results-verified",
                    ],
                    check=True,
                )
                break
            if elapsed > 10200:
                subprocess.run(ssh + [target, f"touch {REMOTE}/STOP_REQUESTED"], timeout=30)
                print("GPU time ceiling reached; requested stop at next task boundary", flush=True)
            print(f"RunPod elapsed {elapsed / 60:.1f} minutes; worker not finished", flush=True)
        except Exception as exc:
            print(f"Watch check: {type(exc).__name__}: {str(exc)[:250]}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
