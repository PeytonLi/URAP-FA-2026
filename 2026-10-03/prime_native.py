"""One owned H100/H200 pod, credit/time guard, output recovery and verified cleanup."""
import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

import curated
import pilot
import hub_backup
import study

ATTEMPTS = study.ROOT / "data/private/prime_native"


def terminated(pods, owned):
    """UNKNOWN is not proof of shutdown; require absence plus terminated history."""
    return (not any(p.id == owned for p in pods.list().data)
            and any(p.id == owned and p.terminated_at for p in pods.history().data))


def command(args, **kwargs):
    prefix = ["rtk", "proxy"] if shutil.which("rtk") else []
    return subprocess.run([*prefix, *map(str, args)], check=True, **kwargs)


def transfer_hub_token(ssh_args, remote, token):
    """SSH stdin, owner-only file outside the source/output bundles; never argv."""
    path = remote + "/.hf-write-token"
    program = ("import os,sys; p=" + repr(path) + "; "
               "fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); "
               "f=os.fdopen(fd,'w'); f.write(sys.stdin.read()); f.close()")
    try:
        command([*ssh_args, "python3", "-c", shlex.quote(program)], input=token,
                text=True, capture_output=True, timeout=30)
    except Exception as exc:
        raise RuntimeError("Secure Hugging Face credential transfer failed (" + type(exc).__name__ + ")") from None
    return path


def retry(operation, attempts=12, delay=5):
    """Retry transient provider-control-plane failures before giving up cleanup."""
    error = None
    for _ in range(attempts):
        try:
            return operation()
        except Exception as exc:
            error = exc
            time.sleep(delay)
    raise error


DEAD_MANS_MINUTES = 45  # pod-side shutdown horizon; launcher keeps postponing


def pod_ssh_connection(pods, owned, current):
    """Some providers update get_status before the detailed pod record."""
    connection = current.ssh_connection
    if not connection:
        statuses = pods.get_status([owned])
        connection = next((item.ssh_connection for item in statuses
                           if item.pod_id == owned and item.ssh_connection), None)
    return connection


def safe_hourly_quote(offer):
    if (offer.get("gpuCount") != 1 or offer.get("gpuMemory", 0) < 80
            or offer.get("stockStatus") != "Available" or offer.get("isSpot") is True
            or offer.get("prepaidTime") or offer["prices"].get("isVariable") is True
            or offer["prices"].get("currency") != "USD"):
        raise ValueError("Fixed-price available single-GPU offer without prepayment required")
    price = offer["prices"].get("onDemand")
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        raise ValueError("Usable fixed hourly price missing")
    for resource in ("disk", "vcpu", "memory"):
        spec = offer[resource]
        if spec.get("pricePerUnit") and spec.get("defaultIncludedInPrice") is not True:
            count, rate = spec.get("defaultCount"), spec["pricePerUnit"]
            if (isinstance(count, bool) or not isinstance(count, (int, float)) or count < 0
                    or isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate < 0
                    or not math.isfinite(count) or not math.isfinite(rate)):
                raise ValueError("Additional resource cost is not known and finite")
            price += rate * count
    return price


def resume_results(recovered):
    recovered = recovered.resolve()
    if not recovered.is_relative_to(ATTEMPTS.resolve()) or recovered.name != "recovered":
        raise ValueError("Resume source must be an owned attempt's recovered output directory")
    import native_report
    training, conditions = native_report.verified_results(recovered, require_all=False)
    if not training and not conditions:
        raise ValueError("No completed verified outputs to resume")
    return training, conditions


def package_inputs(destination, resume_from=None):
    resumed = resume_results(resume_from) if resume_from else ({}, {})
    config = study.dataset_config()
    if not config:
        raise ValueError("Active curated dataset is required")
    rows = curated.training_rows(config)
    paths = set()
    for name in ("study.py", "curated.py", "review_targets.py", "draft_targets.py", "pilot.py",
                 "objectives.py", "hub_backup.py", "native_train.py", "native_benchmark.py", "remote_native.py",
                 "benchmark.py", "vision.py", "curated_vision_check.py", "qwen_vision_check.py",
                 "gemma3_vision_check.py", "native_extra_benchmarks.py",
                 "native_report.py", "report.py"):
        paths.add(study.ROOT / name)
    paths.update((study.ROOT / "config").glob("*.json"))
    paths.update((study.REPO / "2026-09-11/src/multimodel_bias").glob("*.py"))
    paths.add(study.REPO / "2026-09-11/scripts/fireworks_gemma4.py")
    paths.add(study.REPO / "2026-09-11/data/iat12/manifest.csv")
    # DecodingTrust + BOLD + VisBias inputs (frozen, sha-pinned under SOURCE.json).
    paths.update((study.REPO / "2026-09-05/data/bold").glob("*"))
    paths.update((study.REPO / "2026-09-05/data/decodingtrust").glob("*"))
    paths.update((study.REPO / "2026-09-05/data/frozen/visbias").glob("*"))
    visbias_normalized = study.REPO / "2026-09-05/data/private/visbias/normalized"
    if not visbias_normalized.is_dir():
        raise ValueError("VisBias normalized images missing: " + str(visbias_normalized))
    paths.update(visbias_normalized.rglob("*.jpg"))
    paths.add(study.ROOT / "tests/test_native_train.py")
    paths.add(study.ROOT / "tests/test_hub_backup.py")
    config = study.dataset_config()
    paths.add(study.REPO / config["manifest_path"])
    if pilot.FROZEN.is_file():
        paths.add(pilot.FROZEN)
    for name in ("draft_targets.json", "codex_direct_reviews.json", "ai_review_decisions.json", "human_review_decisions.json"):
        path = study.ROOT / "data/private/target_review" / name
        if path.exists():
            paths.add(path)
    targets = study.REPO / config["targets_path"]
    if not targets.is_file():
        raise ValueError("Approved targets file missing: " + config["targets_path"])
    paths.add(targets)
    paths.update(study.REPO / config["asset_root"] / row["image_path"] for row in rows)
    stimuli = study.read_csv(study.REPO / "2026-09-11/data/iat12/manifest.csv")
    checksums = {}
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(paths):
            if path.is_symlink() or not path.resolve().is_relative_to(study.REPO):
                raise ValueError("Unsafe input bundle file")
            name = path.relative_to(study.REPO).as_posix()
            archive.add(path, arcname=name)
            checksums[name] = study.sha(path)
        benchmark_dir = study.ROOT / "data/private/benchmark_images"
        for stimulus in stimuli:
            path = benchmark_dir / stimulus["filename"]
            if not path.is_file() or study.sha(path) != stimulus["sha256"]:
                raise ValueError("Reference benchmark JPEG hash mismatch: " + stimulus["filename"])
            name = "2026-10-03/data/private/benchmark_images/" + path.name
            archive.add(path, arcname=name)
            checksums[name] = study.sha(path)
        for _, checkpoint in resumed[0].values():
            saved = [checkpoint.parent / "run.json", *checkpoint.iterdir()]
            backup_receipt = checkpoint.parent / "hub_backup.json"
            if backup_receipt.is_file():
                saved.append(backup_receipt)
            for path in saved:
                if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ATTEMPTS.resolve()):
                    raise ValueError("Unsafe recovered checkpoint file")
                name = "2026-10-03/data/private/native_runs/" + path.relative_to(checkpoint.parent.parent).as_posix()
                archive.add(path, arcname=name)
                checksums[name] = study.sha(path)
        for _, _, folder, _ in resumed[1].values():
            for path in folder.rglob("*"):
                if path.is_symlink() or not path.resolve().is_relative_to(ATTEMPTS.resolve()):
                    raise ValueError("Unsafe recovered benchmark file")
                if path.is_file():
                    name = "2026-10-03/results/native_benchmarks/" + path.relative_to(folder.parent).as_posix()
                    archive.add(path, arcname=name)
                    checksums[name] = study.sha(path)
    return {"files": checksums, "archive_sha256": study.sha(destination),
            "training_images": sum(1 for r in rows if r["split"] == "train"),
            "development_images": sum(1 for r in rows if r["split"] == "development"),
            "benchmark_images": len(stimuli), "credentials_included": False,
            "benchmark_training_allowed": False, "resumed_training_conditions": len(resumed[0]),
            "resumed_benchmark_conditions": len(resumed[1])}


def launch(resume_from=None):
    if resume_from:
        resume_results(resume_from)  # Validate before cloud calls or purchases.
    from prime_cli.core import APIClient
    from prime_cli.api.pods import PodsClient
    from prime_cli.api.wallet import WalletClient
    curated.training_rows(study.dataset_config())  # Fail before any billable provisioning or upload.
    hub_api, hub_account = hub_backup.connect()  # No paid pod without a usable Hub login.
    client = APIClient()
    request = client.request
    # The installed SDK explicitly passes timeout=None, disabling its HTTP
    # client's default. Bound every request so the credit guard cannot hang.
    client.request = lambda method, endpoint, **kw: request(
        method, endpoint, **{**kw, "timeout": kw.get("timeout") or 30})
    pods = PodsClient(client)
    wallet = WalletClient(client).get(limit=1, team_id=client.config.team_id)
    cap = json.loads((study.ROOT / "config/prime_budget.json").read_text())["total_spending_cap"]
    if wallet.currency != "USD" or wallet.balance_usd < 25 or cap != 100:
        raise ValueError("Expected authorized $100 cap and usable USD credits")
    if client.config.team_id:
        raise ValueError("This authorization uses the personal credit wallet, not an unrelated team")
    # Leave a substantial reserve for asynchronous billing, artifact retrieval,
    # termination delays and a failed local monitor. Never enable credit top-ups.
    spending_ceiling = min(40.0, wallet.balance_usd - 4.0)
    catalog = {gpu: client.get("/availability/gpus", params={"gpu_type": gpu, "gpu_count": 1,
               "page": 1, "page_size": 100}) for gpu in ("H100_80GB", "H200_141GB")}
    checked_at = study.now()
    capacity = {"checked_at": checked_at, "availability": catalog, "wallet_usd": wallet.balance_usd,
                "spending_ceiling_usd": spending_ceiling, "paid_resources_created": False}
    study.save_json(study.ROOT / "results/prime" / ("native_capacity_" + checked_at.replace(":", "") + ".json"), capacity)
    offers = [offer for result in catalog.values() for offer in result.get("items", [])]
    candidates = []
    for offer in offers:
        try:
            price = safe_hourly_quote(offer)
        except ValueError:
            continue
        if price <= 6.0 and "ubuntu_22_cuda_12" in offer.get("images", []):
            candidates.append((price, offer))
    if not candidates:
        raise ValueError("No eligible fixed-price H100/H200 launch offer; no paid resource created")
    price, offer = min(candidates, key=lambda item: item[0])
    stamp = study.now().replace(":", "").replace("-", "").replace("+", "_")
    folder = ATTEMPTS / stamp
    folder.mkdir(parents=True, exist_ok=False)
    archive = folder / "inputs.tar.gz"
    bundle = package_inputs(archive, resume_from=resume_from)
    study.save_json(folder / "input_checksums.json", bundle)
    key = folder / "pod_key"
    command(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "urap-native-pilot", "-f", key],
            capture_output=True, timeout=30)
    public = key.with_suffix(".pub").read_text().strip()
    registered = client.request("POST", "/ssh_keys/", json={"name": "URAP native pilot " + stamp, "publicKey": public})
    key_id = registered.get("id")
    if not key_id:
        raise ValueError("Uploaded SSH key ID missing; no GPU created")
    name = "urap-balanced-pilot-" + stamp.split(".")[0].lower()
    body = {"pod": {"name": name, "cloudId": offer["cloudId"], "gpuType": offer["gpuType"],
                    "socket": offer["socket"], "gpuCount": 1, "image": "ubuntu_22_cuda_12",
                    "dataCenterId": offer.get("dataCenter"), "country": offer.get("country"),
                    "security": offer.get("security"), "sshKeyId": key_id, "autoRestart": False,
                    "maxPrice": price}, "provider": {"type": offer["provider"]}, "disks": [],
                    "sharedWithTeam": False}
    receipt = {"started_at": study.now(), "pod_name": name, "quote": offer, "quoted_hourly_usd": price,
               "spending_ceiling_usd": spending_ceiling, "wallet_before_usd": wallet.balance_usd,
               "credit_top_up_enabled_by_this_work": False, "input_bundle": bundle,
               "requested_conditions": ["gemma3_12b_sft", "gemma3_12b_orpo"], "status": "provisioning",
               "optimizer_updates_verified": 0, "ssh_key_id": key_id,
               "hugging_face_account": hub_account, "hub_backups_required": True}
    study.save_json(folder / "launch.json", receipt)
    owned, ssh_args, remote = None, None, None
    worker_pid = None
    started = time.monotonic()
    # SIGTERM's default disposition kills without running finally — convert it
    # so the cleanup path below always gets to delete the owned pod.
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(
        SystemExit(f"launcher received signal {signum}")))
    try:
        # Keep the raw successful creation response before SDK validation, so
        # unexpected schema drift cannot lose ownership of a billable pod.
        raw = client.request("POST", "/pods", json=body)
        study.save_json(folder / "pod_created.json", raw)
        owned = raw.get("id")
        if not owned:
            raise ValueError("Successful pod response missing ID; inspect private ownership receipt")
        receipt.update(pod_id=owned, status="waiting_for_ssh")
        study.save_json(folder / "launch.json", receipt)
        deadline = started + min(16 * 3600, spending_ceiling / price * 3600)
        ready_deadline = time.monotonic() + 900
        while time.monotonic() < ready_deadline:
            try:
                current = pods.get(owned)
            except Exception:
                time.sleep(15)
                continue
            if current.name != name or current.gpu_count != 1:
                raise ValueError("Owned pod identity mismatch")
            if current.price_hr is not None and current.price_hr > price + 0.01:
                raise ValueError("Created pod costs more than the checked all-in quote")
            try:
                connection = pod_ssh_connection(pods, owned, current)
            except Exception:
                connection = None
            if isinstance(connection, list):
                connection = next((value for value in connection if value), None)
            if connection:
                parts = shlex.split(connection)
                host = next((p for p in parts if re.fullmatch(r"[a-zA-Z0-9_-]+@[a-zA-Z0-9.-]+", p)), None)
                port = parts[parts.index("-p") + 1] if "-p" in parts else "22"
                if not host or not port.isdigit():
                    raise ValueError("Unrecognized provider SSH connection format")
                common = ["-i", str(key), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                          "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                          "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=" + str(folder / "known_hosts")]
                ssh_args = ["ssh", *common, "-p", port, host]
                scp_args = ["scp", *common, "-P", port]
                try:
                    command([*ssh_args, "true"], capture_output=True, timeout=30)
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    pass  # sshd still booting; keep polling
            if current.status in ("STOPPED", "TERMINATED", "FAILED", "ERROR"):
                raise RuntimeError("Owned pod failed before training")
            time.sleep(15)
        else:
            raise TimeoutError("Owned pod SSH never became usable within 15 minutes")
        remote = "/tmp/" + name
        command([*ssh_args, "mkdir", "-p", remote], capture_output=True, timeout=30)
        retry(lambda: command([*scp_args, archive, host + ":" + remote + "/inputs.tar.gz"],
                              capture_output=True, timeout=180), attempts=3, delay=10)
        command([*ssh_args, "tar", "-xzf", remote + "/inputs.tar.gz", "-C", remote], capture_output=True, timeout=60)
        hub_token_path = transfer_hub_token(ssh_args, remote, hub_api.token)
        receipt.update(status="running_remote_worker", remote_workspace=remote, deadline_monotonic_seconds=deadline,
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
        # Dead-man's switch: a pod-side shutdown the monitor keeps postponing.
        # If the launcher dies (SIGKILL, host reboot, VM sleep), the last
        # scheduled shutdown still fires and billing stops on the pod itself.
        command([*ssh_args, "shutdown -h +" + str(DEAD_MANS_MINUTES)],
                capture_output=True, timeout=30)
        last_wallet = 0.0
        last_liveness = 0.0
        last_deadman = time.monotonic()
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Owned pod wall-time/spending guard reached")
            if time.monotonic() - last_deadman >= DEAD_MANS_MINUTES * 60 // 4:
                try:
                    command([*ssh_args, "shutdown -c >/dev/null 2>&1 || true; "
                                        "shutdown -h +" + str(DEAD_MANS_MINUTES)],
                            capture_output=True, timeout=30)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    pass  # the previously scheduled shutdown remains as backstop
                last_deadman = time.monotonic()
            if time.monotonic() - last_liveness >= 300:
                try:
                    current = retry(lambda: pods.get(owned), attempts=3, delay=5)
                except Exception as exc:
                    receipt["monitor_warning"] = "pod_liveness_unavailable:" + type(exc).__name__
                    study.save_json(folder / "launch.json", receipt)
                else:
                    if current.status in ("STOPPED", "TERMINATED", "FAILED", "ERROR"):
                        raise RuntimeError("Owned pod died during monitoring: " + str(current.status))
                last_liveness = time.monotonic()
            if time.monotonic() - last_wallet >= 60:
                try:
                    balance = retry(lambda: WalletClient(client).get(limit=1), attempts=3, delay=5)
                except Exception as exc:
                    receipt["monitor_warning"] = "wallet_api_unavailable:" + type(exc).__name__
                    study.save_json(folder / "launch.json", receipt)
                else:
                    receipt["wallet_latest_usd"] = balance.balance_usd
                    receipt["wallet_checked_at"] = study.now()
                    study.save_json(folder / "launch.json", receipt)
                    if wallet.balance_usd - balance.balance_usd >= spending_ceiling or balance.balance_usd < 4:
                        raise TimeoutError("Prime credit reserve reached")
                last_wallet = time.monotonic()
            try:
                observed = command([*ssh_args, "python3", remote + "/2026-10-03/remote_native.py",
                                    "--status", worker_pid], capture_output=True, timeout=30)
                state = json.loads(observed.stdout)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
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
            # A timed-out POST may still have created the named pod. Reconcile
            # ownership with retries (indexing lag can hide a fresh pod); never
            # retry provisioning as a purchase.
            matches = []
            for _ in range(6):
                try:
                    matches = [p for p in pods.list().data if p.name == name]
                except Exception:
                    matches = []
                if matches:
                    break
                time.sleep(10)
            if len(matches) == 1:
                owned = matches[0].id
                receipt["pod_id"] = owned
            elif len(matches) > 1:
                receipt["cleanup_requires_attention"] = True
                receipt["ambiguous_owned_pods"] = [p.id for p in matches]
        if owned and ssh_args and remote:
            command([*ssh_args, "python3", remote + "/2026-10-03/remote_native.py", "--stop"],
                           capture_output=True, timeout=30)
            if worker_pid:
                for _ in range(9):
                    try:
                        observed = command([*ssh_args, "python3", remote + "/2026-10-03/remote_native.py",
                                            "--status", worker_pid], capture_output=True, timeout=30)
                        state = json.loads(observed.stdout)
                        if not state["worker_running"] and not state["child_running"]:
                            break
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
                        pass
                    time.sleep(10)
        raise
    finally:
        try:
            if owned and ssh_args and receipt.get("remote_workspace"):
                # Package partial work as well as complete runs before deleting.
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
                if any(PurePosixPath(p).is_absolute() or ".." in PurePosixPath(p).parts for p in hashes["files"]):
                    raise ValueError("Unsafe recovered checksum path")
                if any(study.sha(folder / "recovered" / p) != h for p, h in hashes["files"].items()):
                    raise ValueError("Recovered checkpoint/evaluation file checksum mismatch")
                receipt["outputs_recovered_and_verified"] = True
                receipt["recovery_checksums"] = hashes
                runs = [json.loads(path.read_text(encoding="utf-8")) for path in
                        (folder / "recovered/data/private/native_runs").glob("*/run.json")]
                receipt["optimizer_updates_verified"] = sum(run["optimizer_updates"] for run in runs)
                receipt["completed_training_conditions"] = sum(
                    run["status"] == "training_complete_evaluation_pending" for run in runs)
        except BaseException as exc:
            receipt["recovery_error_type"] = type(exc).__name__
            if isinstance(exc, subprocess.CalledProcessError):
                study.save_json(folder / "recovery_error.json", {
                    "error_type": type(exc).__name__, "returncode": exc.returncode,
                    "stderr": exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr})
        finally:
            if owned:
                # Delete by provider-issued pod ID unconditionally; a get() or
                # name mismatch must never block it (billing-leak audit fixes).
                try:
                    current = pods.get(owned)
                    if getattr(current, "name", name) != name:
                        receipt["cleanup_name_mismatch"] = True
                except Exception:
                    pass
                try:
                    retry(lambda: pods.delete(owned), attempts=4, delay=5)
                except Exception as exc:
                    receipt.update(cleanup_requires_attention=True,
                                   cleanup_error_type=type(exc).__name__,
                                   cleanup_error=str(exc))
                else:
                    receipt["termination_requested_at"] = study.now()
                    try:
                        for _ in range(20):
                            if retry(lambda: terminated(pods, owned), attempts=3, delay=3):
                                receipt["termination_verified"] = True
                                break
                            time.sleep(10)
                    except Exception:
                        pass
                    if not receipt.get("termination_verified"):
                        receipt["cleanup_requires_attention"] = True
            if key_id:
                try:
                    client.request("DELETE", "/ssh_keys/" + str(key_id))
                except Exception:
                    pass  # non-billing cruft at worst
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
    args = parser.parse_args()
    result = launch(resume_from=args.resume_from)
    raise SystemExit(0 if result["status"] == "worker_complete_reports_pending"
                     and result.get("outputs_recovered_and_verified") and result.get("termination_verified") else 1)
