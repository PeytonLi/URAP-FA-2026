"""Own-pod lifecycle with bounded spend; never delete the existing network volume."""
import argparse
import json
import shlex
import ssl
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

import certifi

from common import ROOT, json_read, now, secret, sha, write_json

STATE = ROOT / "runtime/pod.json"
REMOTE = "/workspace/research-evaluation-september11"


def api(path="", query=None, variables=None, method="GET"):
    url = "https://api.runpod.io/graphql" if query else "https://rest.runpod.io/v1" + path
    data = json.dumps(dict(query=query, variables=variables or {})).encode() if query else None
    req = urllib.request.Request(url, data=data, method="POST" if query else method,
          headers={"Authorization": "Bearer " + secret("runpod-api-key"), "Content-Type": "application/json",
                   "User-Agent": "Research-RunPod/1.0"})
    with urllib.request.urlopen(req, context=ssl.create_default_context(cafile=certifi.where()), timeout=30) as response:
        raw = response.read()
        result = json.loads(raw) if raw else {}
    if isinstance(result, dict) and result.get("errors"):
        raise RuntimeError(str([e.get("message") for e in result["errors"]]))
    return result


def ssh_options(state):
    return ["ssh", "-i", str(Path.home() / ".ssh/id_ed25519"), "-p", str(state["port"]),
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=" + str(ROOT / "runtime/known_hosts")]


def remote(state, command, timeout=60):
    return subprocess.run(ssh_options(state)+["root@"+state["ip"], command], check=True,
                          capture_output=True, text=True, timeout=timeout).stdout


def sync(state, download=False):
    if download:
        archive = ROOT / "runtime/download.tar.gz"
        with archive.open("wb") as f:
            subprocess.run(ssh_options(state)+["root@"+state["ip"], f"tar -czf - -C {REMOTE} results/local"],
                           stdout=f, check=True, timeout=300)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                target=(ROOT / member.name).resolve()
                if not target.is_relative_to(ROOT / "results/local") or not (member.isfile() or member.isdir()):
                    raise ValueError("Unsafe archive member")
            # Python 3.11.0 lacks extraction filters; validated regular files and
            # directories above exclude path traversal, links and special files.
            tar.extractall(ROOT)
    else:
        archive=ROOT / "runtime/upload.tar.gz"
        excluded={".venv","runtime","results",".git","__pycache__",".pytest_cache",".ruff_cache"}
        with tarfile.open(archive,"w:gz") as tar:
            for path in sorted(ROOT.rglob("*")):
                relative=path.relative_to(ROOT)
                if path.is_file() and not set(relative.parts)&excluded:
                    tar.add(path,arcname=str(relative),recursive=False)
        remote(state, f"mkdir -p {REMOTE}")
        with archive.open("rb") as f:
            subprocess.run(ssh_options(state)+["root@"+state["ip"],f"tar --no-same-owner -xzf - -C {REMOTE}"],
                           stdin=f, check=True, timeout=300)


def terminate(state, reason):
    api("/pods/"+state["pod_id"], method="DELETE")
    pods = api("/pods")
    volumes = api("/networkvolumes")
    receipt = dict(terminated_at=now(), reason=reason, pod_id=state["pod_id"],
                  pod_absent=not any(p["id"] == state["pod_id"] for p in pods),
                  network_volume_preserved=any(v["id"] == state["network_volume_id"] for v in volumes),
                  elapsed_seconds=time.time()-state["deployed_epoch"],
                  estimated_compute_usd=(time.time()-state["deployed_epoch"])*state["cost_per_hour"]/3600,
                  account=api(query="query { myself { clientBalance currentSpendPerHr } }")["data"])
    write_json(ROOT / "results/runpod/cleanup.json", receipt)
    state["terminated"] = True
    write_json(STATE, state)
    print(json.dumps(receipt), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("deploy", "connect", "status", "sync", "download", "watch", "terminate"))
    args = p.parse_args()
    cfg = json_read(ROOT / "config/study.json")
    if args.action == "deploy":
        if STATE.exists():
            raise ValueError("Existing own-pod state: do not deploy twice")
        volume = next(v for v in api("/networkvolumes") if v["id"] == cfg["network_volume_id"])
        assert volume["dataCenterId"] == cfg["data_center_id"]
        account = api(query="query { myself { clientBalance currentSpendPerHr } }")["data"]["myself"]
        if account["clientBalance"] < cfg["runpod_max_usd"]:
            raise ValueError("Insufficient Runpod credit")
        name = "urap-evaluation-september11"
        if any(p.get("name") == name for p in api("/pods")):
            raise ValueError("Existing pod with study name")
        payload = dict(name=name, cloudType="SECURE", gpuCount=1, gpuTypeId=cfg["gpu_type_id"],
                       dataCenterId=cfg["data_center_id"], networkVolumeId=cfg["network_volume_id"],
                       volumeMountPath="/workspace", containerDiskInGb=25, volumeInGb=0,
                       imageName="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", startSsh=True,
                       ports="22/tcp", supportPublicIp=True,
                       env=[dict(key="PUBLIC_KEY", value=(Path.home()/".ssh/id_ed25519.pub").read_text().strip())])
        result = api(query="mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input:$input) { id desiredStatus costPerHr } }",
                     variables={"input":payload})["data"]["podFindAndDeployOnDemand"]
        state = dict(pod_id=result["id"], deployed_at=now(), deployed_epoch=time.time(),
                     network_volume_id=cfg["network_volume_id"], cost_per_hour=float(result["costPerHr"]),
                     starting_account=account, terminated=False)
        state["hard_deadline_epoch"] = state["deployed_epoch"] + min(cfg["runpod_max_seconds"],
                                               (cfg["runpod_max_usd"]-0.30)*3600/state["cost_per_hour"])
        write_json(STATE, state)
        write_json(ROOT / "results/runpod/deployment.json", state)
        print(json.dumps(state), flush=True)
        return
    state = json_read(STATE)
    if state["terminated"]:
        print("Own compute already terminated")
        return
    if args.action == "connect":
        for _ in range(90):
            pod = api(query='query { pod(input: {podId:"'+state["pod_id"]+'"}) { id desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } } } }')["data"]["pod"]
            ports = (pod.get("runtime") or {}).get("ports", [])
            route = next((r for r in ports if r["privatePort"] == 22 and r["isIpPublic"]), None)
            if route:
                state.update(ip=route["ip"], port=route["publicPort"])
                scan = subprocess.run(["ssh-keyscan", "-p", str(state["port"]), state["ip"]], capture_output=True, text=True, timeout=30)
                if scan.returncode == 0 and scan.stdout:
                    (ROOT / "runtime/known_hosts").write_text(scan.stdout)
                    write_json(STATE, state)
                    print(remote(state, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"))
                    return
            time.sleep(5)
        raise TimeoutError("Pod SSH did not become ready")
    if args.action == "sync":
        sync(state)
    elif args.action == "download":
        sync(state, True)
    elif args.action == "status":
        print(remote(state, f"tail -n 5 {REMOTE}/runtime/local.log; cat {REMOTE}/results/local/manifest.json 2>/dev/null", timeout=30))
    elif args.action == "terminate":
        terminate(state, "manual_complete_or_budget_stop")
    elif args.action == "watch":
        while time.time() < state["hard_deadline_epoch"]:
            try:
                progress = remote(state, f"test -f {REMOTE}/runtime/local_exit.json && cat {REMOTE}/runtime/local_exit.json || true")
                if progress.strip():
                    write_json(ROOT / "results/runpod/worker_exit.json", json.loads(progress))
                    sync(state, True)
                    # Verify a stable remote snapshot after process exit, not a moving live file.
                    listing = remote(state, f"cd {shlex.quote(REMOTE)} && find results/local -type f -exec sha256sum {{}} +")
                    check = {line.split(None,1)[1]:line.split(None,1)[0] for line in listing.splitlines()}
                    matches = all(sha(ROOT / path) == digest for path,digest in check.items())
                    write_json(ROOT / "results/runpod/download_verification.json", dict(all_match=matches, files=check, checked_at=now()))
                    if not matches:
                        raise ValueError("Remote download checksum mismatch")
                    terminate(state, "worker_finished_and_results_verified")
                    return
            except Exception as exc:
                write_json(ROOT / "results/runpod/watch_warning.json", dict(type=type(exc).__name__, checked_at=now()))
            time.sleep(30)
        try:
            remote(state, f"touch {REMOTE}/STOP_REQUESTED")
            time.sleep(10)
            sync(state, True)
        finally:
            terminate(state, "hard_spend_deadline; network-volume artifacts preserved")


if __name__ == "__main__":
    main()
