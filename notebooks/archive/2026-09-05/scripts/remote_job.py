#!/usr/bin/env python3
"""Launch a detached, time-bounded inference worker; capture exit status on the volume."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("launch", "monitor"))
    p.add_argument("--probe", action="store_true")
    args = p.parse_args()
    label = "probe" if args.probe else "run"
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    if args.action == "launch":
        if (runtime / f"{label}_pid.json").exists():
            raise ValueError("A job already exists; do not launch duplicates")
        command = [sys.executable, __file__, "monitor"] + (["--probe"] if args.probe else [])
        with (runtime / f"{label}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=log,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
            )
        state = {"pid": process.pid, "started_epoch": time.time(), "probe": args.probe}
        (runtime / f"{label}_pid.json").write_text(json.dumps(state, indent=2))
        print(json.dumps(state))
        return
    model = (
        "/workspace/.hf/hub/models--google--gemma-3-12b-it/snapshots/"
        "96b6f1eccf38110c56df3a15bffe176da04bfd80"
    )
    command = [
        sys.executable,
        "-u",
        str(ROOT / "scripts/run_local_gemma.py"),
        "--model-dir",
        model,
        "--max-seconds",
        "9900",
    ]
    if args.probe:
        command.append("--probe")
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        MPLCONFIGDIR=str(ROOT / ".matplotlib"),
        XDG_CACHE_HOME=str(runtime / "cache"),
    )
    started = time.time()
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, timeout=10200)
        status = {"exit_code": result.returncode, "timed_out": False}
    except subprocess.TimeoutExpired:
        status = {"exit_code": None, "timed_out": True}
    status.update(started_epoch=started, finished_epoch=time.time(), command=command)
    (runtime / f"{label}_exit.json").write_text(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
