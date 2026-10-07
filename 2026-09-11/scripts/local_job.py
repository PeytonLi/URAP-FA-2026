"""Detached inference supervisor; final status stays on the network volume."""
import argparse
import os
import subprocess
import sys
import time

from common import ROOT, json_read, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("launch", "monitor"))
    p.add_argument("--deadline-epoch", type=float, required=True)
    args = p.parse_args()
    runtime = ROOT / "runtime"
    runtime.mkdir(exist_ok=True)
    if args.action == "launch":
        if (runtime / "local_pid.json").exists():
            raise ValueError("Worker already launched")
        with (runtime / "local.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, __file__, "monitor", "--deadline-epoch", str(args.deadline_epoch)],
                    cwd=ROOT, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(runtime / "local_pid.json", dict(pid=proc.pid, started_epoch=time.time()))
        return
    cfg = json_read(ROOT / "config/study.json")
    model = "/workspace/.hf/hub/models--google--gemma-3-12b-it/snapshots/"+cfg["gemma_revision"]
    command = [sys.executable, "-u", "-B", str(ROOT / "scripts/local_evaluate.py"), "--model-dir", model,
               "--deadline-epoch", str(args.deadline_epoch)]
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
               HF_HUB_DISABLE_TELEMETRY="1", HF_HOME="/tmp/qwen-cache")
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, timeout=max(1,args.deadline_epoch-time.time()+120))
        status = dict(exit_code=result.returncode, timed_out=False)
    except subprocess.TimeoutExpired:
        status = dict(exit_code=None, timed_out=True)
    write_json(runtime / "local_exit.json", dict(status, finished_epoch=time.time()))


if __name__ == "__main__":
    main()
