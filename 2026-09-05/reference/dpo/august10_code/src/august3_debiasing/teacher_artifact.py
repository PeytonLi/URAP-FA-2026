from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def teacher_task_types(manifest: Mapping[str, Any]) -> list[str]:
    tasks = manifest.get("tasks")
    if isinstance(tasks, list):
        return [str(task).strip() for task in tasks]
    task = str(manifest.get("task", "")).strip()
    return [task] if task else []


def teacher_output_csv(manifest_path: Path, manifest: Mapping[str, Any]) -> Path:
    raw = str(manifest.get("output_csv", "")).strip()
    if not raw:
        raise ValueError("teacher manifest does not name its output CSV")
    name = Path(raw).name
    if not name or Path(name).suffix.lower() != ".csv":
        raise ValueError("teacher manifest output is not a CSV")
    return manifest_path.resolve().parent / name
