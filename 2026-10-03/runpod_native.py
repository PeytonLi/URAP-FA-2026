"""Extras-only benchmark run on a rented RunPod GPU with the same billing guards
as prime_native.py: spending ceiling, dead-man's switch, recovered-and-verified
outputs, and unconditional pod deletion on every exit path.

Used instead of Prime after the wallet there ran dry. Everything provider-neutral
(package_inputs, resume validation, remote worker, recovery format) is reused from
prime_native so `runpod_native.py launch --resume-from <dir>` behaves identically.
"""
import argparse
import json
import math
import os
import signal
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath

import curated
import hub_backup
import iat_prompts
import study
from prime_native import (ATTEMPTS, DEAD_MANS_MINUTES, command, extras_only_plan, package_inputs,
                          resume_results, retry, transfer_hub_token)

API = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
# Inference-only run: the 12B model in bf16 is ~24GB, so any modern GPU with
# >=48GB VRAM works. On-demand secure-cloud pods only, cheapest first.
MIN_VRAM_GB = 48
MAX_HOURLY_USD = 2.0
SPEND_CEILING_USD = 4.0
BALANCE_RESERVE_USD = 0.50
# Wall time the ceiling must buy: v4 ORPO (~4.4 h) + setup + four IAT evals.
MIN_RUN_HOURS = 6.5
IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
WATCHDOG_KEEPALIVE = "/tmp/urap_launcher_keepalive"
WATCHDOG_STALE_SECONDS = 45 * 60


def runpod_key():
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not key:
        raise ValueError("RUNPOD_API_KEY is not set")
    return key


def api(method, path, body=None, graphql=False, timeout=30):
    url = (GRAPHQL if graphql else API + path)
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + runpod_key())
    req.add_header("Content-Type", "application/json")
    # rest.runpod.io's edge 403s the default Python-urllib user-agent (curl works).
    req.add_header("User-Agent", "curl/8.5.0")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw_body = response.read()
        # 204 No Content (e.g. DELETE /pods/{id}) has no body to parse.
        payload = json.loads(raw_body) if raw_body.strip() else {}
    if graphql and payload.get("errors"):
        raise ValueError("RunPod GraphQL error: " + json.dumps(payload["errors"])[:400])
    return payload


def balance():
    payload = api("POST", None, graphql=True, body={
        "query": "query{myself{clientBalance currentSpendPerHr}}"})
    return float(payload["data"]["myself"]["clientBalance"])


def gpu_offers():
    payload = api("POST", None, graphql=True, body={
        "query": "query{gpuTypes{id displayName memoryInGb securePrice "
                 "lowestPrice(input:{gpuCount:1,secureCloud:true})"
                 "{uninterruptablePrice stockStatus}}}"})
    offers = []
    for gpu in payload["data"]["gpuTypes"]:
        low = gpu.get("lowestPrice") or {}
        price = low.get("uninterruptablePrice") or gpu.get("securePrice")
        if (gpu.get("memoryInGb") or 0) < MIN_VRAM_GB or not isinstance(price, (int, float)):
            continue
        if (not math.isfinite(price) or price <= 0 or price > MAX_HOURLY_USD
                or SPEND_CEILING_USD / price < MIN_RUN_HOURS):
            continue
        stock = (low.get("stockStatus") or "").lower()
        if stock in ("none", "unavailable"):
            continue  # an out-of-stock offer can only fail the create call
        offers.append({"price": price, "stock": stock, "gpu": gpu})
    rank = {"high": 0, "medium": 1, "low": 2}
    offers.sort(key=lambda o: (o["price"], rank.get(o["stock"], 3)))
    return offers


def get_pod(pod_id):
    return api("GET", "/pods/" + pod_id, timeout=30)


def list_pods():
    return api("GET", "/pods", timeout=30)


def terminate_pod(pod_id):
    return api("DELETE", "/pods/" + pod_id, timeout=30)


def launch(resume_from=None, attach_pod=None, attach_key=None, extras_only=None, prompt_sweep=False):
    if prompt_sweep and (extras_only is not None or resume_from is None):
        raise ValueError("Prompt sweep needs a saved checkpoint and cannot run extras")
    if extras_only is not None and not resume_from:
        raise ValueError("--extras-only requires --resume-from with a verified checkpoint")
    if resume_from:
        resumed = resume_results(resume_from)  # Validate before cloud calls or purchases.
        if extras_only is not None:
            extras_only_plan(extras_only, resumed[0])
        if prompt_sweep:
            iat_prompts.sweep_plan(resumed[0])
    curated.training_rows(study.dataset_config())  # Fail before any billable provisioning or upload.
    hub_api, hub_account = hub_backup.connect()  # No paid pod without a usable Hub login.
    wallet_before = balance()
    if wallet_before <= BALANCE_RESERVE_USD:
        raise ValueError(f"RunPod balance ${wallet_before:.2f} below reserve ${BALANCE_RESERVE_USD:.2f}")
    spending_ceiling = min(SPEND_CEILING_USD, wallet_before - BALANCE_RESERVE_USD)
    if spending_ceiling < SPEND_CEILING_USD:
        raise ValueError("RunPod balance cannot fund the full run; no paid resource created")
    attached = None
    if attach_pod:
        # Drive a pod that already exists (created outside this launcher, e.g. a
        # manual probe while stock flapped) through the identical flow. The pod
        # must carry the matching PUBLIC_KEY for attach_key.
        attached = get_pod(attach_pod)
        if not attached.get("id"):
            raise ValueError("attach target pod not found: " + str(attach_pod))
        price = float(attached.get("costPerHr") or 0)
        gpu_id = (attached.get("machine") or {}).get("gpuTypeId") or "attached"
        if (not math.isfinite(price) or price <= 0 or price > MAX_HOURLY_USD
                or spending_ceiling / price < MIN_RUN_HOURS):
            raise ValueError(f"Attached pod costs ${price}/hr — outside the $0-{MAX_HOURLY_USD} guard")
        offers = [{"price": price, "stock": "attached", "gpu": {"id": gpu_id}}]
    else:
        # Capacity flickers; poll up to 30 min for an eligible on-demand offer.
        poll_deadline = time.monotonic() + 30 * 60
        while True:
            offers = gpu_offers()
            checked_at = study.now()
            capacity = {"checked_at": checked_at, "provider": "runpod",
                        "availability": [{"gpu": o["gpu"]["id"], "price": o["price"],
                                          "stock": o["stock"]} for o in offers],
                        "wallet_usd": wallet_before,
                        "spending_ceiling_usd": spending_ceiling, "paid_resources_created": False}
            study.save_json(study.ROOT / "results" / "prime" /
                            ("runpod_capacity_" + checked_at.replace(":", "") + ".json"), capacity)
            if offers:
                break
            if time.monotonic() >= poll_deadline:
                raise ValueError("No eligible on-demand RunPod GPU offer; no paid resource created")
            time.sleep(60)
    offer_index = 0
    offer = offers[offer_index]
    price = float(offer["price"])
    gpu_id = offer["gpu"]["id"]
    stamp = study.now().replace(":", "").replace("-", "").replace("+", "_")
    folder = ATTEMPTS / stamp
    folder.mkdir(parents=True, exist_ok=False)
    archive = folder / "inputs.tar.gz"
    bundle = package_inputs(archive, resume_from=resume_from, extras_only=extras_only,
                            prompt_sweep=prompt_sweep)
    study.save_json(folder / "input_checksums.json", bundle)
    if attach_key:
        key = Path(attach_key)
        if not key.is_file():
            raise ValueError("attach key not found: " + str(attach_key))
    else:
        key = folder / "pod_key"
        command(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "urap-native-pilot", "-f", key],
                capture_output=True, timeout=30)
    public = key.with_suffix(".pub").read_text().strip()
    name = "urap-balanced-pilot-" + stamp.split(".")[0].lower()
    if attached:
        name = str(attached.get("name") or name)  # keep reconcile-by-name working
        injected = (attached.get("env") or {}).get("PUBLIC_KEY") or ""
        if public.split()[:2] != injected.split()[:2]:
            raise ValueError("attach key does not match the pod's injected PUBLIC_KEY")
    body = {"name": name, "imageName": IMAGE, "gpuTypeIds": [gpu_id],
            "cloudType": "SECURE", "computeType": "GPU", "gpuCount": 1,
            "containerDiskInGb": 60, "volumeInGb": 0,
            "ports": ["22/tcp"], "supportPublicIp": True,
            "interruptible": False,
            "env": {"PUBLIC_KEY": public}}
    receipt = {"started_at": study.now(), "provider": "runpod", "pod_name": name,
               "quote": {"gpu": gpu_id, "price_hr": price, "stock": offer["stock"],
                         "image": IMAGE},
               "quoted_hourly_usd": price, "spending_ceiling_usd": spending_ceiling,
               "wallet_before_usd": wallet_before, "input_bundle": bundle,
               "requested_conditions": ([f"gemma3_12b_{m}_prompt_{p}"
                                         for p in iat_prompts.PROMPT_IDS for m in ("base", "orpo")]
                                        if prompt_sweep else
                                        ["gemma3_12b_" + m + "_extras" for m in extras_only]
                                        if extras_only is not None else ["gemma3_12b_orpo"]),
               "status": "provisioning", "optimizer_updates_verified": 0,
               "hugging_face_account": hub_account, "hub_backups_required": True}
    study.save_json(folder / "launch.json", receipt)
    owned, ssh_args, remote, host = None, None, None, None
    scp_args = None
    worker_pid = None
    started = time.monotonic()
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(
            SystemExit(f"launcher received signal {signum}")))
    # RunPod stock flaps minute-to-minute and a listed offer can still reject
    # the create call ("no instances currently available"). Cycle the whole
    # offer list, then refresh it and keep retrying until the deadline.
    provision_deadline = time.monotonic() + 60 * 60
    try:
        while True:
            try:
                raw = attached if attached else api("POST", "/pods", body=body)
                break
            except urllib.error.HTTPError as exc:
                # Out-of-stock or host-constraint rejections: try the next
                # eligible offer rather than dying with capacity still around.
                offer_index += 1
                if offer_index >= len(offers):
                    if time.monotonic() >= provision_deadline:
                        raise
                    receipt.update(offer_rounds=receipt.get("offer_rounds", 0) + 1,
                                   last_offer_error=str(exc)[:200])
                    study.save_json(folder / "launch.json", receipt)
                    time.sleep(60)
                    offers = gpu_offers()
                    if not offers:
                        continue  # nothing listed this minute; poll again
                    offer_index = 0
                offer = offers[offer_index]
                price = float(offer["price"])
                gpu_id = offer["gpu"]["id"]
                body["gpuTypeIds"] = [gpu_id]
                receipt.update(quote={"gpu": gpu_id, "price_hr": price,
                                      "stock": offer["stock"], "image": IMAGE},
                               quoted_hourly_usd=price,
                               offer_retries=offer_index,
                               last_offer_error=str(exc)[:200])
                study.save_json(folder / "launch.json", receipt)
        study.save_json(folder / "pod_created.json", raw)
        owned = raw.get("id")
        if not owned:
            raise ValueError("Successful pod response missing ID; inspect private ownership receipt")
        receipt.update(pod_id=owned, status="waiting_for_ssh")
        study.save_json(folder / "launch.json", receipt)
        # Billing starts at creation; unbilled provisioning retries must not eat run time.
        deadline = time.monotonic() + min(MIN_RUN_HOURS * 3600, spending_ceiling / price * 3600)
        ready_deadline = time.monotonic() + 900
        while time.monotonic() < ready_deadline:
            try:
                current = get_pod(owned)
            except Exception:
                time.sleep(15)
                continue
            status = str(current.get("desiredStatus") or "")
            if status not in ("", "RUNNING", "CREATED"):
                raise RuntimeError("Owned pod failed before worker start: " + status)
            if current.get("name") != name:
                raise ValueError("Owned pod identity mismatch")
            charged = current.get("costPerHr")
            if charged is not None and float(charged) > price + 0.05:
                raise ValueError("Created pod costs more than the checked quote")
            public_ip = current.get("publicIp")
            port = (current.get("portMappings") or {}).get("22")
            if public_ip and port:
                common = ["-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                          "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                          "-o", "StrictHostKeyChecking=accept-new",
                          "-o", "UserKnownHostsFile=" + str(folder / "known_hosts")]
                host = "root@" + public_ip
                ssh_args = ["ssh", *common, "-p", str(port), host]
                scp_args = ["scp", *common, "-P", str(port)]
                try:
                    command([*ssh_args, "true"], capture_output=True, timeout=30)
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    pass  # sshd still booting; keep polling
            time.sleep(15)
        else:
            raise TimeoutError("Owned pod SSH never became usable within 15 minutes")
        remote = "/tmp/" + name
        command([*ssh_args, "mkdir", "-p", remote], capture_output=True, timeout=30)
        # Resumed checkpoints make the bundle large; allow ~200 KB/s on slow pod links.
        upload_timeout = 180 + Path(archive).stat().st_size // 200_000
        retry(lambda: command([*scp_args, archive, host + ":" + remote + "/inputs.tar.gz"],
                              capture_output=True, timeout=upload_timeout), attempts=3, delay=10)
        command([*ssh_args, "tar", "-xzf", remote + "/inputs.tar.gz", "-C", remote],
                capture_output=True, timeout=300)
        hub_token_path = transfer_hub_token(ssh_args, remote, hub_api.token)
        receipt.update(status="running_remote_worker", remote_workspace=remote,
                       deadline_monotonic_seconds=deadline,
                       estimated_max_gpu_charge_usd=(deadline - started) / 3600 * price)
        study.save_json(folder / "launch.json", receipt)
        detached = command([*ssh_args, "env", "HF_TOKEN_PATH=" + hub_token_path,
                            "python3", remote + "/2026-10-03/remote_native.py", "--detach"],
                           capture_output=True, timeout=30)
        worker_pid = json.loads(detached.stdout)["worker_pid"]
        if not isinstance(worker_pid, int) or isinstance(worker_pid, bool) or worker_pid <= 0:
            raise ValueError("Owned remote worker PID missing")
        receipt["remote_worker_pid"] = worker_pid
        study.save_json(folder / "launch.json", receipt)
        # Dead-man's switch, container-safe: RunPod pods have no systemd, so a
        # detached watchdog polls a keepalive file the monitor touches; on
        # staleness it DELETEs its own pod via the injected RUNPOD_API_KEY.
        watchdog = (
            "setsid bash -c 'while :; do sleep 60; "
            "if [ $(( $(date +%s) - $(stat -c %Y " + WATCHDOG_KEEPALIVE +
            " 2>/dev/null || echo 0) )) -gt " + str(WATCHDOG_STALE_SECONDS) + " ]; then "
            "curl -s -m 30 -X DELETE https://rest.runpod.io/v1/pods/\"$RUNPOD_POD_ID\" "
            "-H \"Authorization: Bearer $RUNPOD_API_KEY\" >/dev/null 2>&1; fi; done' "
            ">/dev/null 2>&1 &")
        try:
            command([*ssh_args, "date +%s > " + WATCHDOG_KEEPALIVE + "; " + watchdog],
                    capture_output=True, timeout=30)
            receipt["dead_mans_switch_armed"] = "api_watchdog"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            receipt["dead_mans_switch_armed"] = False
        last_wallet = 0.0
        last_liveness = 0.0
        last_deadman = time.monotonic()
        state = {}
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Owned pod wall-time/spending guard reached")
            if receipt.get("dead_mans_switch_armed") and \
                    time.monotonic() - last_deadman >= DEAD_MANS_MINUTES * 60 // 4:
                try:
                    command([*ssh_args, "date +%s > " + WATCHDOG_KEEPALIVE],
                            capture_output=True, timeout=30)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    pass  # keepalive only gets staler on failure — watchdog stays armed
                last_deadman = time.monotonic()
            if time.monotonic() - last_liveness >= 300:
                try:
                    current = retry(lambda: get_pod(owned), attempts=3, delay=5)
                except Exception as exc:
                    receipt["monitor_warning"] = "pod_liveness_unavailable:" + type(exc).__name__
                    study.save_json(folder / "launch.json", receipt)
                else:
                    if str(current.get("desiredStatus")) not in ("RUNNING", "CREATED", ""):
                        raise RuntimeError("Owned pod died during monitoring: " +
                                           str(current.get("desiredStatus")))
                last_liveness = time.monotonic()
            if time.monotonic() - last_wallet >= 60:
                try:
                    latest = retry(balance, attempts=3, delay=5)
                except Exception as exc:
                    receipt["monitor_warning"] = "wallet_api_unavailable:" + type(exc).__name__
                    study.save_json(folder / "launch.json", receipt)
                else:
                    receipt["wallet_latest_usd"] = latest
                    receipt["wallet_checked_at"] = study.now()
                    study.save_json(folder / "launch.json", receipt)
                    if wallet_before - latest >= spending_ceiling or latest < BALANCE_RESERVE_USD:
                        raise TimeoutError("RunPod credit reserve reached")
                last_wallet = time.monotonic()
            try:
                observed = command([*ssh_args, "python3", remote + "/2026-10-03/remote_native.py",
                                    "--status", worker_pid], capture_output=True, timeout=30)
                state = json.loads(observed.stdout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                    json.JSONDecodeError) as exc:
                receipt["monitor_warning"] = "worker_observation_unavailable:" + type(exc).__name__
                study.save_json(folder / "launch.json", receipt)
            else:
                receipt["last_worker_observation"] = state
                study.save_json(folder / "launch.json", receipt)
                if not state["worker_running"] and not state["child_running"]:
                    break
            time.sleep(10)
        complete = state["summary"].get("status") == "training_and_benchmarks_complete_reports_pending"
        receipt.update(remote_exit_code=0 if complete else 1,
                       status="worker_complete_reports_pending" if complete else "remote_failed")
        study.save_json(folder / "launch.json", receipt)
    except BaseException as exc:
        receipt.update(status="incomplete", error_type=type(exc).__name__)
        if owned is None:
            matches = []
            for _ in range(6):
                try:
                    matches = [p for p in list_pods() if p.get("name") == name]
                except Exception:
                    matches = []
                if matches:
                    break
                time.sleep(10)
            if len(matches) == 1:
                owned = matches[0]["id"]
                receipt["pod_id"] = owned
            elif len(matches) > 1:
                receipt["cleanup_requires_attention"] = True
                receipt["ambiguous_owned_pods"] = [p["id"] for p in matches]
        if owned and ssh_args and remote:
            try:
                command([*ssh_args, "python3", remote + "/2026-10-03/remote_native.py", "--stop"],
                        capture_output=True, timeout=30)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
            if worker_pid:
                for _ in range(9):
                    try:
                        observed = command([*ssh_args, "python3",
                                            remote + "/2026-10-03/remote_native.py",
                                            "--status", worker_pid], capture_output=True, timeout=30)
                        state = json.loads(observed.stdout)
                        if not state["worker_running"] and not state["child_running"]:
                            break
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                            json.JSONDecodeError):
                        pass
                    time.sleep(10)
        raise
    finally:
        try:
            if owned and ssh_args and receipt.get("remote_workspace"):
                try:
                    command([*ssh_args, "test", "-f",
                        remote + "/2026-10-03/data/private/remote_outputs_checksums.json"],
                        capture_output=True, timeout=30)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    retry(lambda: command([*ssh_args, "python3",
                        remote + "/2026-10-03/remote_native.py", "--package"],
                        capture_output=True, timeout=180), attempts=2, delay=10)
                retry(lambda: command([*scp_args, host + ":" + remote +
                        "/2026-10-03/data/private/remote_outputs.tar.gz", folder],
                        capture_output=True, timeout=600), attempts=3, delay=10)
                retry(lambda: command([*scp_args, host + ":" + remote +
                        "/2026-10-03/data/private/remote_outputs_checksums.json", folder],
                        capture_output=True, timeout=30), attempts=3, delay=5)
                hashes = json.loads((folder / "remote_outputs_checksums.json").read_text())
                recovered = folder / "remote_outputs.tar.gz"
                if study.sha(recovered) != hashes["archive_sha256"]:
                    raise ValueError("Recovered archive checksum mismatch")
                with tarfile.open(recovered) as archive:
                    if any(m.issym() or m.islnk() for m in archive.getmembers()):
                        raise ValueError("Recovered outputs must not contain symlinks")
                    archive.extractall(folder / "recovered", filter="data")
                if any(PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts
                       for p in hashes["files"]):
                    raise ValueError("Unsafe recovered checksum path")
                if any(study.sha(folder / "recovered" / p) != h
                       for p, h in hashes["files"].items()):
                    raise ValueError("Recovered checkpoint/evaluation file checksum mismatch")
                receipt["outputs_recovered_and_verified"] = True
                receipt["recovery_checksums"] = hashes
                # Count only runs trained on this pod, not checkpoints shipped in for reuse.
                runs = [json.loads(path.read_text(encoding="utf-8")) for path in
                        (folder / "recovered/data/private/native_runs").glob("*/run.json")
                        if "2026-10-03/" + path.relative_to(folder / "recovered").as_posix()
                        not in bundle["files"]]
                receipt["optimizer_updates_verified"] = sum(run["optimizer_updates"] for run in runs)
                receipt["completed_training_conditions"] = sum(
                    run["status"] == "training_complete_evaluation_pending" for run in runs)
        except BaseException as exc:
            receipt["recovery_error_type"] = type(exc).__name__
            if isinstance(exc, subprocess.CalledProcessError):
                study.save_json(folder / "recovery_error.json", {
                    "error_type": type(exc).__name__, "returncode": exc.returncode,
                    "stderr": exc.stderr.decode("utf-8", errors="replace")
                        if isinstance(exc.stderr, bytes) else exc.stderr})
        finally:
            if owned:
                try:
                    current = get_pod(owned)
                    if current.get("name", name) != name:
                        receipt["cleanup_name_mismatch"] = True
                except Exception:
                    pass
                try:
                    retry(lambda: terminate_pod(owned), attempts=4, delay=5)
                except Exception as exc:
                    receipt.update(cleanup_requires_attention=True,
                                   cleanup_error_type=type(exc).__name__,
                                   cleanup_error=str(exc))
                else:
                    receipt["termination_requested_at"] = study.now()
                    try:
                        for _ in range(20):
                            try:
                                current = get_pod(owned)
                            except Exception:
                                receipt["termination_verified"] = True  # pod record gone
                                break
                            if str(current.get("desiredStatus")) not in ("RUNNING", "CREATED"):
                                receipt["termination_verified"] = True
                                break
                            time.sleep(10)
                    except Exception:
                        pass
                    if not receipt.get("termination_verified"):
                        receipt["cleanup_requires_attention"] = True
            receipt["finished_at"] = study.now()
            receipt["elapsed_seconds"] = time.monotonic() - started
            study.save_json(folder / "launch.json", receipt)
            print(json.dumps({k: receipt.get(k) for k in ("status", "pod_id", "remote_exit_code",
                "outputs_recovered_and_verified", "termination_verified", "error_type")}), flush=True)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch",))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--attach-pod", help="existing pod id to drive instead of creating one")
    parser.add_argument("--attach-key", type=Path,
                        help="private key matching the attached pod's injected PUBLIC_KEY")
    parser.add_argument("--extras-only", type=lambda text: text.split(","),
                        help="comma list (base,orpo): skip training/IAT, run only DT/BOLD/VisBias")
    parser.add_argument("--prompt-sweep", action="store_true",
                        help="evaluate only prompts 3,4,5 on base and saved ORPO; never train")
    args = parser.parse_args()
    result = launch(resume_from=args.resume_from,
                    attach_pod=args.attach_pod, attach_key=args.attach_key,
                    extras_only=args.extras_only, prompt_sweep=args.prompt_sweep)
    raise SystemExit(0 if result["status"] == "worker_complete_reports_pending"
                     and result.get("outputs_recovered_and_verified")
                     and result.get("termination_verified") else 1)
