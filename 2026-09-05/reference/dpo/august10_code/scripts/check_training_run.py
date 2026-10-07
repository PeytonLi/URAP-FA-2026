#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.run_control import pid_is_alive, tail_text


TERMINAL_STATUSES = {"complete", "failed", "supervisor_failed", "orphaned_or_failed"}


def gpu_snapshot() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return []
    result = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            result.append(
                {
                    "name": parts[0],
                    "memory_used_mib": int(parts[1]),
                    "memory_total_mib": int(parts[2]),
                    "utilization_percent": int(parts[3]),
                    "uuid": parts[4],
                }
            )
    return result


def snapshot(receipt_path: Path, tail_lines: int) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    output_dir = Path(receipt["output_dir"])
    training_manifest_path = output_dir / "training_manifest.json"
    training_manifest = (
        json.loads(training_manifest_path.read_text(encoding="utf-8"))
        if training_manifest_path.is_file()
        else None
    )
    supervisor_alive = pid_is_alive(receipt.get("supervisor_pid"))
    child_alive = pid_is_alive(receipt.get("child_pid"))
    status = str(receipt.get("status", "unknown"))
    if status in {"launching", "running"} and not supervisor_alive and not child_alive:
        status = "orphaned_or_failed"
    return {
        "status": status,
        "receipt_status": receipt.get("status"),
        "supervisor_pid": receipt.get("supervisor_pid"),
        "supervisor_alive": supervisor_alive,
        "child_pid": receipt.get("child_pid"),
        "child_alive": child_alive,
        "training_manifest_status": (
            training_manifest.get("status") if training_manifest else "missing"
        ),
        "progress": training_manifest.get("progress") if training_manifest else None,
        "development_preference_summary": (
            training_manifest.get("development_preference_summary")
            if training_manifest
            else None
        ),
        "gpu": gpu_snapshot(),
        "training_log_tail": tail_text(receipt_path.parent / "training.log", tail_lines),
        "receipt": str(receipt_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a supervised August 3 training run.")
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--tail-lines", type=int, default=25)
    parser.add_argument("--watch-seconds", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.tail_lines < 0 or args.watch_seconds < 0 or args.poll_seconds <= 0:
        raise ValueError("tail/watch values must be nonnegative and poll seconds must be positive")
    receipt_path = args.receipt.resolve()
    deadline = time.time() + args.watch_seconds
    while True:
        current = snapshot(receipt_path, args.tail_lines)
        print(json.dumps(current, indent=2, sort_keys=True), flush=True)
        if current["status"] in TERMINAL_STATUSES or time.time() >= deadline:
            return 0 if current["status"] != "orphaned_or_failed" else 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
