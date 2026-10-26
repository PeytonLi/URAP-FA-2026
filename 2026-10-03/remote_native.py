"""Owned-pod worker: ORPO-only training, then matched benchmarks."""
import argparse
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTIVE = ROOT / "data/private/remote_active.json"
STOP = ROOT / "data/private/remote_stop"


def owned_pid_running(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        return (str(ROOT) + "/").encode() in (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return False


def worker_status(pid):
    path = ROOT / "data/private/remote_summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    active = json.loads(ACTIVE.read_text()) if ACTIVE.exists() else {}
    return {"summary": summary, "worker_running": owned_pid_running(pid),
            "child_running": owned_pid_running(active.get("child_pid"))}


def detach():
    log = ROOT / "data/private/remote_worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        child = subprocess.Popen([sys.executable, str(ROOT / "remote_native.py")],
                                 stdin=subprocess.DEVNULL, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    return {"worker_pid": child.pid}


def verify_resume(python, output):
    import study
    probe = ("import native_report,native_train,study; "
             "t,b=native_report.verified_results(study.ROOT,require_all=False); "
             "v={n:native_train.version(n) for n in ('torch','transformers','peft','accelerate')}; "
             "assert all(r['versions']==v for r,c in t.values()), 'Recovered runtime mismatch'; "
             "study.save_json(study.ROOT/'data/private/verified_resume_plan.json', "
             "{'training':list(t),'benchmarks':list(b)})")
    child = subprocess.Popen([python, "-c", probe], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
    study.save_json(ACTIVE, {"worker_pid": os.getpid(), "child_pid": child.pid, "phase": "resume_verification"})
    if child.wait(timeout=120):
        raise RuntimeError("Pinned-runtime resume verification failed; see retained worker log")
    plan = json.loads((ROOT / "data/private/verified_resume_plan.json").read_text())
    return {tuple(key) for key in plan["training"]}, {tuple(key) for key in plan["benchmarks"]}


def bootstrap_uv(root):
    """Use venv when present; fall back to uv's official unmanaged installer."""
    bootstrap = root / "runtime/bootstrap"
    try:
        subprocess.run([sys.executable, "-m", "venv", str(bootstrap)], check=True, timeout=120)
        subprocess.run([str(bootstrap / "bin/python"), "-m", "pip", "install", "uv==0.11.25"],
                       check=True, timeout=180)
        return bootstrap / "bin/uv"
    except (subprocess.CalledProcessError, FileNotFoundError):
        installer = root / "runtime/uv-install.sh"
        installer.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request("https://astral.sh/uv/install.sh",
                                         headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            installer.write_bytes(response.read())
        uv_dir = root / "runtime/uv-bin"
        uv_dir.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "UV_UNMANAGED_INSTALL": str(uv_dir),
                       "UV_NO_MODIFY_PATH": "1"}
        subprocess.run(["sh", str(installer)], env=environment, check=True, timeout=180)
        return uv_dir / "uv"


def stop():
    STOP.parent.mkdir(parents=True, exist_ok=True)
    STOP.touch()
    if ACTIVE.exists():
        active = json.loads(ACTIVE.read_text())
        if active.get("worker_pid") and active.get("child_pid"):
            try:
                cmdline = Path("/proc") / str(active["child_pid"]) / "cmdline"
                if str(ROOT).encode() not in cmdline.read_bytes():
                    raise ValueError("Stop refused: PID is not an owned worker child")
                os.kill(active["child_pid"], signal.SIGINT)
            except (ProcessLookupError, FileNotFoundError):
                pass


def main():
    import study
    started = time.monotonic()
    token_path = os.environ.get("HF_TOKEN_PATH")
    if token_path:
        token = Path(token_path).read_text().strip()
        os.environ["HF_TOKEN"] = token
        cache = Path.home() / ".cache/huggingface"
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / "token"
        if not target.exists():
            fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(fd, token.encode())
            os.close(fd)
    summary = {"started_at": study.now(), "cohort": "full", "conditions": [],
               "status": "setup", "gpu_check_passed": False}
    outputs = ROOT / "data/private/remote_summary.json"
    worker_log = None
    study.save_json(outputs, summary)
    try:
        venv = ROOT / "runtime/native-venv"
        if venv.exists():
            raise ValueError("Fresh owned-pod runtime required")
        # Ubuntu 22's system Python is 3.10; the benchmark requires Python 3.11.
        uv = bootstrap_uv(ROOT)
        environment = {**os.environ, "UV_PYTHON_INSTALL_DIR": str(ROOT / "runtime/python"),
                       "UV_CACHE_DIR": str(ROOT / "runtime/uv-cache")}
        subprocess.run([str(uv), "venv", "--python", "3.11.15", "--managed-python",
                        "--seed", str(venv)], env=environment, check=True, timeout=300)
        python = str(venv / "bin/python")
        subprocess.run([python, "-c", "import sys; assert sys.version_info[:3] == (3, 11, 15)"], check=True)
        # Default PyPI Torch uses CUDA 13, newer than the observed H100 driver.
        # Use the official CUDA 12.8 build; native bases/adapters share this runtime.
        cuda_installation = subprocess.Popen([python, "-m", "pip", "install", "--disable-pip-version-check",
            "torch==2.11.0", "torchvision==0.26.0", "--index-url", "https://download.pytorch.org/whl/cu128"])
        study.save_json(ACTIVE, {"worker_pid": os.getpid(), "child_pid": cuda_installation.pid, "phase": "cuda_installation"})
        if cuda_installation.wait(timeout=1800):
            raise RuntimeError("Driver-compatible CUDA runtime installation failed")
        packages = ["torch==2.11.0", "torchvision==0.26.0", "transformers==5.14.0",
                    "peft==0.21.0", "accelerate==1.15.0", "huggingface_hub==1.33.0",
                    "Pillow==11.3.0", "matplotlib", "numpy", "pandas", "scipy",
                    "detoxify==0.5.2", "vaderSentiment==3.3.2"]
        installation = subprocess.Popen([python, "-m", "pip", "install", "--disable-pip-version-check", *packages])
        study.save_json(ACTIVE, {"worker_pid": os.getpid(), "child_pid": installation.pid, "phase": "installation"})
        if installation.wait(timeout=3600):
            raise RuntimeError("Remote dependency installation failed")
        subprocess.run([python, str(ROOT / "tests/test_native_train.py")], check=True, timeout=120)
        subprocess.run([python, str(ROOT / "tests/test_hub_backup.py")], check=True, timeout=120)
        subprocess.run([python, str(ROOT / "hub_backup.py"), "auth"], check=True, timeout=60)
        # Extra-benchmark data: the 09-11 modules resolve data dirs relative to
        # their own tree; point them at the bundled 09-05 frozen data.
        data_root = ROOT.parent / "2026-09-11" / "data"
        for name, target in (("bold", "../../2026-09-05/data/bold"),
                             ("decodingtrust", "../../2026-09-05/data/decodingtrust"),
                             ("visbias", "../../2026-09-05/data/frozen/visbias")):
            link = data_root / name
            if not link.exists():
                link.symlink_to(target)
        gpu = subprocess.run([python, "-c", "import torch,json; assert torch.cuda.is_available(); "
            "assert torch.cuda.is_bf16_supported(); print(json.dumps({'gpu':torch.cuda.get_device_name(0),"
            "'memory_gib':torch.cuda.get_device_properties(0).total_memory/2**30,'cuda':torch.version.cuda}))"], check=True)
        summary.update(status="running", gpu_check_passed=gpu.returncode == 0)
        study.save_json(outputs, summary)
        worker_log = (ROOT / "data/private/remote_worker.log").open("a", encoding="utf-8")
        # The bootstrap parent is system Python; scientific imports belong in the pinned venv.
        ready_training, ready_benchmarks = verify_resume(python, worker_log)
        for model in ("gemma3_12b",):
            for method in ("orpo",):
                if STOP.exists():
                    raise InterruptedError("Owned run stopped before the next condition")
                if (model, method) in ready_training:
                    summary["conditions"].append({"model": model, "method": method,
                                                  "exit_code": 0, "reused_verified_checkpoint": True})
                    study.save_json(outputs, summary)
                    continue
                command = [python, "-u", str(ROOT / "native_train.py"), "train", "--cohort", "full",
                           "--model", model, "--method", method, "--max-seconds", "10800"]
                # Keep child output on the pod so a dropped SSH monitor cannot
                # turn a valid optimizer run into BrokenPipeError.
                child = subprocess.Popen(command, cwd=ROOT.parent, stdout=worker_log,
                                         stderr=subprocess.STDOUT)
                study.save_json(ACTIVE, {"worker_pid": os.getpid(), "child_pid": child.pid,
                                        "phase": model + "_" + method})
                code = child.wait()
                summary["conditions"].append({"model": model, "method": method, "exit_code": code})
                study.save_json(outputs, summary)
                if code:
                    raise RuntimeError("Actual training failed; do not silently substitute a condition")
            evaluations = [(method, development)
                           for development in ((True, False) if study.dataset_config().get(
                               "dataset_id") in study.MULTITRIPLE_DATASETS else (False,))
                           for method in ("base", "orpo")]
            for method, development in evaluations:
                if STOP.exists():
                    raise InterruptedError("Owned run stopped before evaluation")
                if not development and (model, method) in ready_benchmarks:
                    summary["conditions"].append({"model": model, "benchmark": method,
                                                  "exit_code": 0, "reused_verified_benchmark": True})
                    study.save_json(outputs, summary)
                    continue
                command = [python, "-u", str(ROOT / "native_benchmark.py"), "--model", model,
                           "--method", method, "--image-dir", str(ROOT / "data/private/benchmark_images"),
                           "--max-seconds", "3600"]
                if development:
                    command.append("--development")
                if method != "base":
                    folders = list((ROOT / "data/private/native_runs").glob(model + "_" + method + "_*"))
                    if len(folders) != 1:
                        raise ValueError("Exactly one fresh training run per condition required")
                    command.extend(["--checkpoint", str(folders[0] / "epoch_3")])
                child = subprocess.Popen(command, cwd=ROOT.parent, stdout=worker_log,
                                         stderr=subprocess.STDOUT)
                study.save_json(ACTIVE, {"worker_pid": os.getpid(), "child_pid": child.pid,
                                        "phase": model + "_" + method + (
                                            "_development" if development else "_benchmark")})
                code = child.wait()
                summary["conditions"].append({"model": model, "benchmark": method,
                                              "development": development, "exit_code": code})
                study.save_json(outputs, summary)
                if code:
                    raise RuntimeError("Native benchmark failed; retain partial records")
            for method in (() if study.dataset_config().get("benchmark_scope") == "iat_only"
                           else ("base", "orpo")):
                if STOP.exists():
                    raise InterruptedError("Owned run stopped before extra evaluation")
                command = [python, "-u", str(ROOT / "native_extra_benchmarks.py"), "--model", model,
                           "--method", method, "--dt-samples-per-regime", "96",
                           "--bold-sample-count", "192"]
                if method != "base":
                    folders = list((ROOT / "data/private/native_runs").glob(model + "_" + method + "_*"))
                    if len(folders) != 1:
                        raise ValueError("Exactly one fresh training run per condition required")
                    command.extend(["--checkpoint", str(folders[0] / "epoch_3")])
                child = subprocess.Popen(command, cwd=ROOT.parent, stdout=worker_log,
                                         stderr=subprocess.STDOUT)
                study.save_json(ACTIVE, {"worker_pid": os.getpid(), "child_pid": child.pid,
                                        "phase": model + "_" + method + "_extra_benchmarks"})
                code = child.wait()
                summary["conditions"].append({"model": model, "extra_benchmarks": method,
                                              "exit_code": code})
                study.save_json(outputs, summary)
                if code:
                    raise RuntimeError("Extra benchmark failed; retain partial records")
        summary["status"] = "training_and_benchmarks_complete_reports_pending"
    except BaseException as exc:
        summary["status"] = "incomplete" if STOP.exists() else "failed"
        summary["error_type"] = type(exc).__name__
        summary["error_message"] = str(exc)
        exception_log = worker_log or sys.stderr
        traceback.print_exc(file=exception_log)
        exception_log.flush()  # Archive the failure before the interpreter prints its final traceback.
        raise
    finally:
        if worker_log is not None:
            worker_log.flush()
            worker_log.close()
        summary["finished_at"] = study.now()
        summary["elapsed_seconds"] = time.monotonic() - started
        study.save_json(outputs, summary)
        package_outputs()


def package_outputs():
    import study
    # Only owned outputs. No cached base weights, source images or credentials.
    archive = ROOT / "data/private/remote_outputs.tar.gz"
    files = {}
    with tarfile.open(archive, "w:gz", compresslevel=1) as handle:
        for relative in ("data/private/native_runs", "results/native_benchmarks",
                         "results/extra_benchmarks",
                         "data/private/remote_summary.json", "data/private/remote_worker.log"):
            path = ROOT / relative
            if path.exists():
                handle.add(path, arcname=relative)
                for file in ([path] if path.is_file() else path.rglob("*")):
                    if file.is_file():
                        files[file.relative_to(ROOT).as_posix()] = study.sha(file)
    study.save_json(ROOT / "data/private/remote_outputs_checksums.json", {"archive_sha256": study.sha(archive),
                                                                        "files": files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--stop", action="store_true")
    action.add_argument("--package", action="store_true")
    action.add_argument("--detach", action="store_true")
    action.add_argument("--status", type=int)
    args = parser.parse_args()
    if args.stop:
        stop()
    elif args.package:
        package_outputs()
    elif args.detach:
        print(json.dumps(detach()), flush=True)
    elif args.status is not None:
        print(json.dumps(worker_status(args.status)), flush=True)
    else:
        main()
