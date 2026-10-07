from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


RESERVED_TRAINING_ARGS = {
    "--frozen-csv",
    "--balance-report",
    "--repo-root",
    "--output-dir",
    "--base-model-local-dir",
    "--model-id",
    "--model-revision",
    "--forbidden-eval-hashes",
    "--allow-hf-download",
    "--allow-unpinned-model",
    "--attn-implementation",
    "--no-gradient-checkpointing",
    "--model-smoke-test",
    "--dry-run",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def research_code_manifest(repo_root: Path) -> dict[str, Any]:
    """Hash every file that can change the training program or environment."""

    root = repo_root.resolve()
    paths = [root / "pyproject.toml", root / "requirements-runpod.txt"]
    paths.extend((root / "src").rglob("*.py"))
    paths.extend((root / "scripts").glob("*.py"))
    files = sorted(
        {path.resolve() for path in paths if path.is_file()},
        key=lambda path: str(path.relative_to(root)),
    )
    if not files:
        raise FileNotFoundError(f"no training source files found under {root}")
    hashes = {
        str(path.relative_to(root)): sha256_file(path)
        for path in files
    }
    aggregate = hashlib.sha256()
    for relative_path, digest in hashes.items():
        aggregate.update(relative_path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "schema_version": 1,
        "file_count": len(hashes),
        "tree_sha256": aggregate.hexdigest(),
        "files": hashes,
    }


def adapter_artifact_hashes(adapter_dir: Path) -> dict[str, str]:
    config = adapter_dir / "adapter_config.json"
    if not config.is_file():
        raise FileNotFoundError(f"adapter_config.json is missing from {adapter_dir}")
    weight_paths = sorted(adapter_dir.glob("adapter_model*.safetensors")) + sorted(
        adapter_dir.glob("adapter_model*.bin")
    )
    if not weight_paths:
        raise FileNotFoundError(f"adapter weights are missing from {adapter_dir}")
    paths = [config, *weight_paths]
    return {str(path.relative_to(adapter_dir)): sha256_file(path) for path in paths}


def atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def is_within(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


def parse_gpu_query(output: str) -> list[dict[str, Any]]:
    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"unexpected nvidia-smi query row: {line!r}")
        memory_text = parts[1].lower().replace("mib", "").strip()
        gpus.append(
            {
                "name": parts[0],
                "memory_total_mib": int(memory_text),
                "memory_total_gib": int(memory_text) / 1024,
                "uuid": parts[2],
            }
        )
    return gpus


def validate_gpu_inventory(
    gpus: list[dict[str, Any]],
    *,
    expected_count: int,
    expected_name_substring: str,
    minimum_memory_gib: float,
) -> None:
    if len(gpus) != expected_count:
        raise ValueError(f"expected {expected_count} visible GPU(s), found {len(gpus)}")
    for gpu in gpus:
        if expected_name_substring.lower() not in str(gpu["name"]).lower():
            raise ValueError(
                f"GPU {gpu['name']!r} does not contain required name {expected_name_substring!r}"
            )
        if float(gpu["memory_total_gib"]) < minimum_memory_gib:
            raise ValueError(
                f"GPU {gpu['name']!r} has {gpu['memory_total_gib']:.2f} GiB; "
                f"at least {minimum_memory_gib:.2f} GiB is required"
            )


def validate_model_smoke_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("status") != "validated_model_smoke_test":
        raise ValueError("training preflight did not complete the model smoke test")
    if set(manifest.get("model_smoke_test_covered_tasks", [])) != {
        "iat_3word",
        "grounded_profile",
    }:
        raise ValueError("training preflight did not cover both task types")
    runtime = manifest.get("runtime_environment") or {}
    if not runtime.get("cuda_available") or int(runtime.get("cuda_device_count", 0)) < 1:
        raise ValueError("training preflight did not execute with CUDA")
    records = manifest.get("model_smoke_test") or []
    if len(records) != 2 or any("dpo_loss" not in record for record in records):
        raise ValueError("training preflight is missing gradient smoke-test records")
    memory = manifest.get("model_smoke_test_gpu_memory") or {}
    if float(memory.get("peak_allocated_gib", 0.0)) <= 0.0:
        raise ValueError("training preflight did not record peak allocated GPU memory")
    if float(memory.get("peak_reserved_gib", 0.0)) <= 0.0:
        raise ValueError("training preflight did not record peak reserved GPU memory")


def validate_extra_training_args(args: Iterable[str]) -> list[str]:
    values = list(args)
    conflicts = sorted(
        {
            value.split("=", 1)[0]
            for value in values
            if value.split("=", 1)[0] in RESERVED_TRAINING_ARGS
        }
    )
    if conflicts:
        raise ValueError(f"reserved training arguments cannot be overridden: {conflicts}")
    return values


def build_training_command(
    *,
    python_executable: str,
    training_script: Path,
    frozen_csv: Path,
    balance_report: Path,
    repo_root: Path,
    output_dir: Path,
    base_model_local_dir: Path,
    model_revision: str,
    extra_args: Iterable[str] = (),
    dry_run: bool = False,
) -> list[str]:
    extras = validate_extra_training_args(extra_args)
    command = [
        python_executable,
        str(training_script.resolve()),
        "--frozen-csv",
        str(frozen_csv.resolve()),
        "--balance-report",
        str(balance_report.resolve()),
        "--repo-root",
        str(repo_root.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--base-model-local-dir",
        str(base_model_local_dir.resolve()),
        "--model-revision",
        model_revision,
        "--attn-implementation",
        "sdpa",
        *extras,
    ]
    if dry_run:
        command.extend(("--model-smoke-test", "--dry-run"))
    return command


def pid_is_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def tail_text(path: Path, lines: int = 25) -> list[str]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()[-lines:]
