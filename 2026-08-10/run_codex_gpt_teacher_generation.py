#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.codex_cli_provider import (
    DEFAULT_CODEX_MODEL,
    PROFILE_OUTPUT_SCHEMA,
    CodexCliProvider,
)
from august3_debiasing.constants import TASK_PROFILE
from august3_debiasing.expression_pair_artifact import verify_expression_pair_artifact
from august3_debiasing.response_generation import (
    GENERATED_RESPONSE_POLICY_VERSION,
    GenerationJob,
    build_generation_jobs,
    complete_match_subset,
    generated_response_issues,
)

PROTOCOL_VERSION = "august3-codex-gpt-teacher-v1"
SYSTEM_INSTRUCTION = """You are producing an independent multimodal candidate response for a preference-data study.
Use only visible evidence. Do not describe skin color, skin tone, complexion, race, ethnicity, gender, or nationality. Do not assign identity categories or infer personal background, biography, occupation, personality, competence, intent, health, criminality, or trustworthiness. Do not make the output more positive merely to appear fair, and do not add negative language unsupported by the visible expression. Follow the requested output format exactly. Metadata labels are deliberately unavailable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_image(root: Path, raw_path: str, expected: str) -> Path:
    candidate = Path(raw_path)
    path = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"image path escapes repository root: {raw_path}")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"missing image or SHA-256 mismatch: {raw_path}")
    return path


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            response_id = str(record["response_id"])
            if response_id in records:
                raise ValueError(f"duplicate checkpoint response at line {line_number}")
            records[response_id] = record
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def limit_pending_jobs(
    pending: list[tuple[int, GenerationJob]], max_new_responses: int
) -> list[tuple[int, GenerationJob]]:
    if max_new_responses < 0:
        raise ValueError("--max-new-responses cannot be negative")
    if max_new_responses == 0:
        return pending
    return pending[:max_new_responses]


def manifest_mismatches(
    existing: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    mismatches = []
    for key, value in expected.items():
        if existing.get(key) == value:
            continue
        if key == "expression_pair_manifest" and existing.get(
            "expression_pair_manifest_sha256"
        ) == expected.get("expression_pair_manifest_sha256"):
            continue
        mismatches.append(key)
    return mismatches


def job_input_sha256(job: GenerationJob, args: argparse.Namespace) -> str:
    return sha256_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "response_id": job.response_id,
            "image_sha256": job.image_sha256,
            "task_type": job.task_type,
            "prompt": job.prompt,
            "selection_seed": job.generation_seed,
            "validation_attempts": args.validation_attempts,
            "generated_response_policy_version": GENERATED_RESPONSE_POLICY_VERSION,
            "system_instruction": SYSTEM_INSTRUCTION,
            "output_schema": PROFILE_OUTPUT_SCHEMA,
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "service_tier": args.service_tier,
        }
    )


def generate_teacher_response(
    job: GenerationJob,
    image_path: Path,
    args: argparse.Namespace,
    provider: CodexCliProvider,
) -> dict[str, Any]:
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
        "total_tokens": 0,
    }
    attempt_records = []
    last_response = ""
    for attempt in range(1, args.validation_attempts + 1):
        try:
            generated = provider.generate_profile(
                system=SYSTEM_INSTRUCTION,
                prompt=job.prompt,
                image_path=image_path,
            )
        except RuntimeError as exc:
            attempt_records.append(
                {
                    "attempt": attempt,
                    "provider_error": str(exc)[:4000],
                    "validation_issues": ["provider_error"],
                }
            )
            continue
        for key in usage:
            usage[key] += int(getattr(generated, key))
        last_response = generated.text
        issues = generated_response_issues(job.task_type, generated.text)
        attempt_records.append(
            {
                "attempt": attempt,
                "response_sha256": hashlib.sha256(
                    generated.text.encode("utf-8")
                ).hexdigest(),
                "response_preview": generated.text[:300],
                "validation_issues": issues,
                "provider_thread_id": generated.thread_id,
                "provider_model": generated.model,
                "usage": {
                    key: int(getattr(generated, key)) for key in usage
                },
            }
        )
        if not issues:
            return {
                "response_id": job.response_id,
                "image_sha256": job.image_sha256,
                "task_type": job.task_type,
                "prompt": job.prompt,
                "response": generated.text,
                "response_provenance": f"{args.model}_codex_cli_multimodal_teacher",
                "sample_index": job.sample_index,
                "selection_seed": job.generation_seed,
                "generation_seed_supported": False,
                "generation_attempt": attempt,
                "generation_status": "ok",
                "validation_issues": "",
                "job_input_sha256": job_input_sha256(job, args),
                "provider_thread_id": generated.thread_id,
                "provider_model": generated.model,
                "reasoning_effort": args.reasoning_effort,
                "service_tier": args.service_tier,
                "attempt_records": attempt_records,
                **usage,
            }
    return {
        "response_id": job.response_id,
        "image_sha256": job.image_sha256,
        "task_type": job.task_type,
        "prompt": job.prompt,
        "response": last_response,
        "response_provenance": f"{args.model}_codex_cli_multimodal_teacher",
        "sample_index": job.sample_index,
        "selection_seed": job.generation_seed,
        "generation_seed_supported": False,
        "generation_attempt": len(attempt_records),
        "generation_status": "failed",
        "validation_issues": "|".join(
            attempt_records[-1].get("validation_issues", [])
            if attempt_records
            else ["provider_error"]
        ),
        "attempt_records": attempt_records,
        "job_input_sha256": job_input_sha256(job, args),
        "reasoning_effort": args.reasoning_effort,
        "service_tier": args.service_tier,
        **usage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate image-grounded GPT teacher profiles through ChatGPT-authenticated "
            "Codex CLI runs."
        )
    )
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--expression-pair-manifest", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--reasoning-effort", default="ultra")
    parser.add_argument(
        "--service-tier", choices=("standard", "fast"), default="fast"
    )
    parser.add_argument("--profile-samples-per-image", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--provider-prefix", default="gpt56-sol-codex")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument(
        "--max-new-responses",
        type=int,
        default=0,
        help=(
            "Stop after checkpointing this many new jobs; zero processes all "
            "remaining jobs. This operational limit may change between resumes."
        ),
    )
    parser.add_argument("--validation-attempts", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=8032026)
    parser.add_argument("--auth-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_workers <= 16:
        raise ValueError("--max-workers must be in [1, 16]")
    if args.validation_attempts <= 0:
        raise ValueError("--validation-attempts must be positive")
    if args.profile_samples_per_image <= 0:
        raise ValueError("--profile-samples-per-image must be positive")
    if args.max_new_responses < 0:
        raise ValueError("--max-new-responses cannot be negative")
    root = args.repo_root.resolve()
    metadata_csv = args.metadata_csv.resolve()
    pair_manifest_path = args.expression_pair_manifest.resolve()
    pair_manifest = verify_expression_pair_artifact(metadata_csv, pair_manifest_path)
    metadata = read_csv(metadata_csv)
    if args.max_images > 0:
        metadata = complete_match_subset(metadata, args.max_images)
    jobs = build_generation_jobs(
        metadata,
        profile_samples_per_image=args.profile_samples_per_image,
        seed=args.seed,
        tasks={TASK_PROFILE},
        provider_prefix=args.provider_prefix,
    )
    images = {
        job.image_sha256: resolve_image(root, job.image_path, job.image_sha256)
        for job in jobs
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "codex_gpt_teacher_responses.jsonl"
    manifest_path = args.output_dir / "codex_gpt_teacher_manifest.json"
    output_csv = args.output_dir / "codex_gpt_teacher_response_pool.csv"
    try:
        cli_version = subprocess.run(
            ["codex", "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("the Codex CLI is unavailable") from exc
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "metadata_csv": str(metadata_csv),
        "metadata_csv_sha256": sha256_file(metadata_csv),
        "expression_pair_manifest": str(pair_manifest_path),
        "expression_pair_manifest_sha256": sha256_file(pair_manifest_path),
        "expression_pair_policy_version": pair_manifest["policy_version"],
        "images": len(images),
        "expected_responses": len(jobs),
        "selected_image_ids_sha256": hashlib.sha256(
            "\n".join(row["image_id"] for row in metadata).encode("utf-8")
        ).hexdigest(),
        "selection_strategy": (
            "complete_match_prefix" if args.max_images > 0 else "complete_input"
        ),
        "task": TASK_PROFILE,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "service_tier": args.service_tier,
        "authentication_mode": "chatgpt_managed_codex",
        "codex_cli_version": cli_version,
        "seed": args.seed,
        "profile_samples_per_image": args.profile_samples_per_image,
        "max_workers": args.max_workers,
        "validation_attempts": args.validation_attempts,
        "timeout_seconds": args.timeout_seconds,
        "generated_response_policy_version": GENERATED_RESPONSE_POLICY_VERSION,
        "race_or_gender_metadata_in_prompt": False,
        "system_instruction_sha256": hashlib.sha256(
            SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "output_schema_sha256": sha256_json(PROFILE_OUTPUT_SCHEMA),
        "sampling_seed_supported": False,
        "dry_run": args.dry_run,
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = manifest_mismatches(existing_manifest, manifest)
        if mismatches:
            raise ValueError(f"existing Codex teacher run differs for: {mismatches}")
        if existing_manifest.get("status") == "complete":
            if not output_csv.is_file() or existing_manifest.get(
                "output_csv_sha256"
            ) != sha256_file(output_csv):
                raise ValueError("completed Codex teacher CSV differs from its manifest")
            print(json.dumps(existing_manifest, indent=2, sort_keys=True))
            return 0
    if args.dry_run:
        manifest["status"] = "validated_dry_run"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    checkpoint = load_checkpoint(checkpoint_path)
    expected_ids = {job.response_id for job in jobs}
    extras = set(checkpoint) - expected_ids
    if extras:
        raise ValueError(
            f"Codex checkpoint has unexpected response IDs: {sorted(extras)[:3]}"
        )
    provider = CodexCliProvider(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        service_tier=args.service_tier,
        timeout_seconds=args.timeout_seconds,
        auth_json=args.auth_json,
    )
    pending: list[tuple[int, GenerationJob]] = []
    for position, job in enumerate(jobs, 1):
        if job.response_id in checkpoint:
            if checkpoint[job.response_id].get("job_input_sha256") != job_input_sha256(
                job, args
            ):
                raise ValueError(
                    f"Codex checkpoint provenance mismatch: {job.response_id}"
                )
            continue
        pending.append((position, job))
    pending = limit_pending_jobs(pending, args.max_new_responses)
    manifest.update(
        {
            "status": "generating",
            "checkpointed_responses_at_start": len(checkpoint),
            "invocation_new_response_limit": args.max_new_responses,
            "invocation_planned_responses": len(pending),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    completed_count = len(checkpoint)

    def store(position: int, job: GenerationJob, record: dict[str, Any]) -> None:
        nonlocal completed_count
        append_checkpoint(checkpoint_path, record)
        checkpoint[job.response_id] = record
        completed_count += 1
        print(
            f"Codex GPT teacher response {completed_count}/{len(jobs)} "
            f"(input {position}): {job.response_id}",
            flush=True,
        )

    if args.max_workers == 1:
        for position, job in pending:
            store(
                position,
                job,
                generate_teacher_response(
                    job, images[job.image_sha256], args, provider
                ),
            )
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    generate_teacher_response,
                    job,
                    images[job.image_sha256],
                    args,
                    provider,
                ): (position, job)
                for position, job in pending
            }
            for future in as_completed(futures):
                position, job = futures[future]
                store(position, job, future.result())

    all_records = [
        checkpoint[job.response_id]
        for job in jobs
        if job.response_id in checkpoint
    ]
    accepted = [row for row in all_records if row.get("generation_status") == "ok"]
    write_csv(
        output_csv,
        [
            {key: value for key, value in row.items() if key != "attempt_records"}
            for row in accepted
        ],
    )
    run_complete = len(all_records) == len(jobs)
    manifest.update(
        {
            "status": "complete" if run_complete else "partial",
            "completed_responses": len(accepted),
            "processed_jobs": len(all_records),
            "remaining_responses": len(jobs) - len(all_records),
            "generation_status_counts": dict(
                Counter(row.get("generation_status", "unknown") for row in all_records)
            ),
            "output_csv": str(output_csv.resolve()),
            "output_csv_sha256": sha256_file(output_csv),
            "input_tokens": sum(int(row.get("input_tokens", 0)) for row in all_records),
            "cached_input_tokens": sum(
                int(row.get("cached_input_tokens", 0)) for row in all_records
            ),
            "output_tokens": sum(
                int(row.get("output_tokens", 0)) for row in all_records
            ),
            "reasoning_output_tokens": sum(
                int(row.get("reasoning_output_tokens", 0)) for row in all_records
            ),
            "total_tokens": sum(int(row.get("total_tokens", 0)) for row in all_records),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
