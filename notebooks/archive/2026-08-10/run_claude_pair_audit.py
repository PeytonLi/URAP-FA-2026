#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

CLAUDE_AUDIT_PROTOCOL_VERSION = "august3-claude-pair-audit-v5-iat-double-text-tie"

from august3_debiasing.claude_pair_judge import (
    SYSTEM_INSTRUCTION,
    aggregate_pair,
    build_jobs,
    judgment_schema,
    render_job_prompt,
    single_image_irrecoverable_reasons,
    source_choice,
    validate_judgment,
)
from august3_debiasing.claude_provider import (
    CLAUDE_EFFORT_LEVELS,
    DEFAULT_CLAUDE_MODEL,
    ClaudeProvider,
)
from august3_debiasing.constants import TASK_IAT, TASK_PROFILE
from august3_debiasing.pair_policy import (
    IDENTITY_POLICY_VERSION,
    prompt_metadata_hits,
    validate_pair,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_image(root: Path, raw: str) -> Path:
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Image path escapes repository root: {raw}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_candidates(rows: list[dict[str, str]], root: Path) -> dict[str, Path]:
    required = {
        "pair_id",
        "match_block_id",
        "image_path",
        "image_sha256",
        "split",
        "task_type",
        "prompt",
        "candidate_1",
        "candidate_2",
        "candidate_1_provenance",
        "candidate_2_provenance",
        "fairface_race",
        "fairface_gender",
        "age_bucket",
        "expression_class",
        "expression_fine",
        "expression_intensity",
    }
    if not rows:
        raise ValueError("Candidate CSV is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Candidate CSV is missing columns: {sorted(missing)}")
    pair_ids = [row["pair_id"].strip() for row in rows]
    if len(set(pair_ids)) != len(pair_ids) or not all(pair_ids):
        raise ValueError("Candidate pair IDs must be nonempty and unique")
    images: dict[str, Path] = {}
    for row in rows:
        pair_id = row["pair_id"].strip()
        hits = prompt_metadata_hits(row["prompt"])
        if hits:
            raise ValueError(f"{pair_id}: prompt exposes protected metadata: {hits}")
        image = resolve_image(root, row["image_path"])
        if sha256_file(image) != row["image_sha256"].strip().lower():
            raise ValueError(f"{pair_id}: image SHA-256 does not match CSV")
        build_jobs(row, image)
        images[pair_id] = image
    return images


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            job_id = str(record["job_id"])
            if job_id in records:
                raise ValueError(
                    f"Duplicate checkpoint job at line {line_number}: {job_id}"
                )
            records[job_id] = record
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a resumable four-way Claude audit of mixed DPO candidates."
    )
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_CLAUDE_MODEL)
    parser.add_argument("--effort", choices=CLAUDE_EFFORT_LEVELS, default="high")
    parser.add_argument("--backend", choices=("direct", "vertex"), default="direct")
    parser.add_argument("--vertex-project-id")
    parser.add_argument("--vertex-region", default="global")
    parser.add_argument("--provider-max-attempts", type=int, default=4)
    parser.add_argument("--task-type", choices=(TASK_PROFILE, TASK_IAT))
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.provider_max_attempts <= 0:
        raise ValueError("--provider-max-attempts must be positive")

    root = args.repo_root.resolve()
    candidate_csv = args.candidate_csv.resolve()
    candidate_rows = read_csv(candidate_csv)
    rows = [
        row
        for row in candidate_rows
        if args.task_type is None or row.get("task_type", "").strip() == args.task_type
    ]
    if args.max_pairs is not None:
        if args.max_pairs <= 0:
            raise ValueError("--max-pairs must be positive")
        rows = rows[: args.max_pairs]
    if not rows:
        raise ValueError("candidate selection is empty after task filtering")
    images = validate_candidates(rows, root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "claude_judgments.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)

    all_jobs = [
        job for row in rows for job in build_jobs(row, images[row["pair_id"].strip()])
    ]
    rows_by_pair = {row["pair_id"].strip(): row for row in rows}
    schema_hashes = {
        condition: hashlib.sha256(
            json.dumps(judgment_schema(condition), sort_keys=True).encode("utf-8")
        ).hexdigest()
        for condition in ("image", "text")
    }
    job_input_hashes = {
        job.job_id: hashlib.sha256(
            json.dumps(
                {
                    "rendered_prompt": render_job_prompt(job),
                    "image_sha256": (
                        rows_by_pair[job.pair_id]["image_sha256"].strip().lower()
                        if job.image_path is not None
                        else None
                    ),
                    "judgment_schema_sha256": schema_hashes[job.condition],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        for job in all_jobs
    }
    manifest: dict[str, Any] = {
        "protocol_version": CLAUDE_AUDIT_PROTOCOL_VERSION,
        "candidate_csv": str(candidate_csv),
        "candidate_csv_sha256": sha256_file(candidate_csv),
        "candidate_csv_rows": len(candidate_rows),
        "task_filter": args.task_type,
        "candidate_pairs": len(rows),
        "maximum_judge_calls": len(all_jobs),
        "judge_model": args.model,
        "judge_effort": args.effort,
        "backend": args.backend,
        "vertex_region": args.vertex_region if args.backend == "vertex" else None,
        "provider_max_attempts": args.provider_max_attempts,
        "system_instruction_sha256": hashlib.sha256(
            SYSTEM_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "judgment_schema_sha256_by_condition": schema_hashes,
        "identity_policy_version": IDENTITY_POLICY_VERSION,
        "race_metadata_in_judge_prompt": False,
        "conditions": [
            "image_normal",
            "image_reversed",
            "text_normal",
            "text_reversed",
        ],
        "dry_run": args.dry_run,
    }
    manifest_path = args.output_dir / "claude_pair_audit_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        invariant_keys = (
            "protocol_version",
            "candidate_csv_sha256",
            "candidate_csv_rows",
            "task_filter",
            "candidate_pairs",
            "maximum_judge_calls",
            "judge_model",
            "judge_effort",
            "backend",
            "vertex_region",
            "provider_max_attempts",
            "system_instruction_sha256",
            "judgment_schema_sha256_by_condition",
            "identity_policy_version",
            "race_metadata_in_judge_prompt",
            "conditions",
            "dry_run",
        )
        mismatches = [
            key
            for key in invariant_keys
            if existing_manifest.get(key) != manifest.get(key)
        ]
        if mismatches:
            raise ValueError(f"existing Claude audit differs for: {mismatches}")
    if args.dry_run:
        manifest["status"] = "validated_dry_run"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    manifest["status"] = "judging"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provider = ClaudeProvider(
        model=args.model,
        backend=args.backend,
        vertex_project_id=args.vertex_project_id,
        vertex_region=args.vertex_region,
        max_attempts=args.provider_max_attempts,
        effort=args.effort,
    )
    expected_job_ids = {job.job_id for job in all_jobs}
    extra_checkpoint_jobs = set(checkpoint) - expected_job_ids
    if extra_checkpoint_jobs:
        raise ValueError(
            f"checkpoint contains jobs outside this candidate slice: {sorted(extra_checkpoint_jobs)[:3]}"
        )

    def complete_job(job):
        if job.job_id in checkpoint:
            record = checkpoint[job.job_id]
            if record.get("job_input_sha256") != job_input_hashes[job.job_id]:
                raise ValueError(f"checkpoint input provenance mismatch: {job.job_id}")
            validate_judgment(record["value"], job.condition)
            return record
        try:
            response = provider.generate_json(
                system=SYSTEM_INSTRUCTION,
                prompt=render_job_prompt(job),
                schema=judgment_schema(job.condition),
                image_path=job.image_path,
            )
        except RuntimeError as exc:
            manifest.update(
                {
                    "status": "failed",
                    "failed_job_id": job.job_id,
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:2000],
                    "completed_judge_calls": len(checkpoint),
                }
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise
        record = {
            "job_id": job.job_id,
            "pair_id": job.pair_id,
            "condition": job.condition,
            "order": job.order,
            "job_input_sha256": job_input_hashes[job.job_id],
            "value": response.value,
            "model": response.model,
            "response_id": response.response_id,
            "stop_reason": response.stop_reason,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }
        append_checkpoint(checkpoint_path, record)
        checkpoint[job.job_id] = record
        try:
            validate_judgment(record["value"], job.condition)
        except ValueError as exc:
            manifest.update(
                {
                    "status": "failed",
                    "failed_job_id": job.job_id,
                    "failure_type": type(exc).__name__,
                    "failure_message": str(exc)[:2000],
                    "completed_judge_calls": len(checkpoint),
                }
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise
        return record

    early_exclusions: dict[str, list[str]] = {}
    for position, row in enumerate(rows, 1):
        pair_id = row["pair_id"].strip()
        jobs = {
            (job.condition, job.order): job for job in build_jobs(row, images[pair_id])
        }
        image_normal = jobs[("image", "normal")]
        normal_record = complete_job(image_normal)
        normal_value = normal_record["value"]
        normal_reasons = single_image_irrecoverable_reasons(
            row, image_normal, normal_value
        )
        if normal_reasons:
            early_exclusions[pair_id] = [
                f"image_normal_irrecoverable:{reason}" for reason in normal_reasons
            ]
            continue
        selected_source = source_choice(image_normal, str(normal_value["decision"]))

        image_reversed = jobs[("image", "reversed")]
        reversed_record = complete_job(image_reversed)
        reversed_value = reversed_record["value"]
        reversed_choice = source_choice(image_reversed, str(reversed_value["decision"]))
        if reversed_choice != selected_source:
            early_exclusions[pair_id] = [
                f"image_order_unstable:{selected_source}_vs_{reversed_choice}"
            ]
            continue
        reversed_reasons = single_image_irrecoverable_reasons(
            row, image_reversed, reversed_value
        )
        if reversed_reasons:
            early_exclusions[pair_id] = [
                f"image_reversed_irrecoverable:{reason}" for reason in reversed_reasons
            ]
            continue

        text_normal = jobs[("text", "normal")]
        text_normal_record = complete_job(text_normal)
        validate_judgment(text_normal_record["value"], "text")
        text_normal_choice = source_choice(
            text_normal, str(text_normal_record["value"]["decision"])
        )
        if row["task_type"].strip() == "iat_3word" and text_normal_choice != "tie":
            early_exclusions[pair_id] = [
                f"iat_text_normal_not_tie:{text_normal_choice}"
            ]
            continue
        if text_normal_choice not in {"tie", selected_source}:
            early_exclusions[pair_id] = [
                f"text_normal_opposes_selected:{text_normal_choice}"
            ]
            continue

        text_reversed = jobs[("text", "reversed")]
        complete_job(text_reversed)
        print(
            f"Claude staged audit {position}/{len(rows)}: {pair_id}; calls={len(checkpoint)}",
            flush=True,
        )

    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in rows:
        pair_id = row["pair_id"].strip()
        if pair_id in early_exclusions:
            exclusions.append(
                {"pair_id": pair_id, "reasons": "|".join(early_exclusions[pair_id])}
            )
            continue
        jobs = build_jobs(row, images[pair_id])
        values = {
            job.job_id: checkpoint[job.job_id]["value"]
            for job in jobs
            if job.job_id in checkpoint
        }
        models = {
            str(checkpoint[job.job_id].get("model", args.model))
            for job in jobs
            if job.job_id in checkpoint
        }
        oriented, reasons = aggregate_pair(
            row,
            jobs,
            values,
            model="|".join(sorted(models)) or args.model,
        )
        if oriented is not None:
            validation = validate_pair(oriented)
            reasons.extend(validation.issues)
        if oriented is None or reasons:
            exclusions.append({"pair_id": pair_id, "reasons": "|".join(reasons)})
        else:
            eligible.append(oriented)

    eligible_path = args.output_dir / "claude_judged_pairs_eligible.csv"
    exclusions_path = args.output_dir / "claude_pair_exclusions.csv"
    write_csv(eligible_path, eligible)
    write_csv(exclusions_path, exclusions)
    manifest.update(
        {
            "status": "complete",
            "completed_judge_calls": len(checkpoint),
            "calls_saved_by_staging": len(all_jobs) - len(checkpoint),
            "eligible_pairs": len(eligible),
            "excluded_pairs": len(exclusions),
            "eligible_csv": str(eligible_path.resolve()),
            "eligible_csv_sha256": sha256_file(eligible_path),
            "exclusions_csv": str(exclusions_path.resolve()),
            "exclusions_csv_sha256": sha256_file(exclusions_path),
            "checkpoint_jsonl_sha256": sha256_file(checkpoint_path),
            "input_tokens": sum(
                int(record.get("input_tokens", 0)) for record in checkpoint.values()
            ),
            "output_tokens": sum(
                int(record.get("output_tokens", 0)) for record in checkpoint.values()
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
