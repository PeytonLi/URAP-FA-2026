#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.constants import TASK_IAT, TASK_PROFILE
from august3_debiasing.expression_pair_artifact import verify_expression_pair_artifact
from august3_debiasing.prompts import prompt_sha256
from august3_debiasing.response_generation import GenerationJob, build_generation_jobs

DEFAULT_MODEL_ID = "google/gemma-3-12b-it"
DEFAULT_MODEL_REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"
GEMMA_RESPONSE_PROTOCOL_VERSION = "august3-gemma-candidate-generation-v2"


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def resolve_image(root: Path, raw_path: str, expected_sha256: str) -> Path:
    candidate = Path(raw_path)
    path = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"image path escapes repository root: {raw_path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"image SHA-256 mismatch for {raw_path}: {actual}")
    return path


def resolve_job_images(root: Path, jobs: list[GenerationJob]) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for job in jobs:
        if job.image_sha256 not in images:
            images[job.image_sha256] = resolve_image(
                root, job.image_path, job.image_sha256
            )
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
            response_id = str(record["response_id"])
            if response_id in records:
                raise ValueError(
                    f"duplicate checkpoint response at line {line_number}: {response_id}"
                )
            records[response_id] = record
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def extract_generated_text(output: Any) -> str:
    if not isinstance(output, list) or not output:
        raise ValueError(f"unexpected pipeline output: {type(output).__name__}")
    value = output[0].get("generated_text")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for message in reversed(value):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content.strip()
    raise ValueError("pipeline output contains no assistant text")


def generation_config(task_type: str, args: argparse.Namespace) -> dict[str, Any]:
    if task_type == TASK_IAT:
        return {
            "max_new_tokens": args.iat_max_new_tokens,
            "do_sample": True,
            "temperature": args.iat_temperature,
            "top_p": args.iat_top_p,
        }
    if task_type == TASK_PROFILE:
        return {
            "max_new_tokens": args.profile_max_new_tokens,
            "do_sample": True,
            "temperature": args.profile_temperature,
            "top_p": args.profile_top_p,
        }
    raise ValueError(f"unsupported generation task: {task_type}")


def generation_kwargs(job: GenerationJob, args: argparse.Namespace) -> dict[str, Any]:
    return generation_config(job.task_type, args)


def validate_generation_args(args: argparse.Namespace, tasks: set[str]) -> None:
    for task_type, samples, temperature, top_p, max_new_tokens in (
        (
            TASK_IAT,
            args.iat_samples_per_image,
            args.iat_temperature,
            args.iat_top_p,
            args.iat_max_new_tokens,
        ),
        (
            TASK_PROFILE,
            args.profile_samples_per_image,
            args.profile_temperature,
            args.profile_top_p,
            args.profile_max_new_tokens,
        ),
    ):
        if task_type not in tasks:
            continue
        if samples <= 0:
            raise ValueError(f"{task_type} samples per image must be positive")
        if temperature <= 0:
            raise ValueError(f"{task_type} temperature must be positive")
        if not 0 < top_p <= 1:
            raise ValueError(f"{task_type} top-p must be in (0, 1]")
        if max_new_tokens <= 0:
            raise ValueError(f"{task_type} max-new-tokens must be positive")


def job_plan_sha256(jobs: list[GenerationJob], args: argparse.Namespace) -> str:
    plan = [job_input_sha256(job, args) for job in jobs]
    return hashlib.sha256(
        json.dumps(plan, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def generation_manifest_settings(
    metadata: list[dict[str, str]],
    jobs: list[GenerationJob],
    tasks: set[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "protocol_version": GEMMA_RESPONSE_PROTOCOL_VERSION,
        "generation_method": "transformers_image_text_to_text_pipeline",
        "generation_seed": args.seed,
        "samples_per_image": {
            task_type: (
                args.iat_samples_per_image
                if task_type == TASK_IAT
                else args.profile_samples_per_image
            )
            for task_type in sorted(tasks)
        },
        "generation_configs": {
            task_type: generation_config(task_type, args) for task_type in sorted(tasks)
        },
        "prompt_sha256": {
            task_type: prompt_sha256(task_type) for task_type in sorted(tasks)
        },
        "selected_image_ids_sha256": hashlib.sha256(
            "\n".join(row["image_id"] for row in metadata).encode("utf-8")
        ).hexdigest(),
        "job_plan_sha256": job_plan_sha256(jobs, args),
        "torch_dtype": "bfloat16",
        "device_map": "auto",
        "allow_hf_download": args.allow_hf_download,
        "allow_unpinned_model": args.allow_unpinned_model,
    }


def job_input_sha256(job: GenerationJob, args: argparse.Namespace) -> str:
    value = {
        "response_id": job.response_id,
        "image_sha256": job.image_sha256,
        "task_type": job.task_type,
        "prompt": job.prompt,
        "generation_seed": job.generation_seed,
        "generation_kwargs": generation_kwargs(job, args),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "attention_implementation": args.attn_implementation,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoProcessor, pipeline

    local_dir = (
        Path(args.base_model_local_dir).expanduser()
        if args.base_model_local_dir
        else None
    )
    if local_dir and local_dir.is_dir():
        source: str | Path = local_dir.resolve()
        source_kind = "local_snapshot"
        if not args.allow_unpinned_model and source.name != args.model_revision:
            raise ValueError(
                f"local model snapshot {source.name!r} does not match pinned revision {args.model_revision}"
            )
        common = {"local_files_only": True}
    else:
        if not args.allow_hf_download:
            raise FileNotFoundError(
                "Pinned local Gemma snapshot is unavailable; pass --base-model-local-dir or explicitly allow download"
            )
        source = args.model_id
        source_kind = "huggingface_revision"
        common = {"revision": args.model_revision}

    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, **common)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        **common,
    }
    classes = [
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
    ]
    errors = []
    model = None
    for model_class in classes:
        if model_class is None:
            continue
        try:
            model = model_class.from_pretrained(source, **model_kwargs)
            break
        except Exception as exc:
            errors.append(f"{model_class.__name__}: {exc}")
    if model is None:
        raise RuntimeError(
            "could not load Gemma multimodal model: " + " | ".join(errors)
        )
    model.eval()
    try:
        pipe = pipeline(
            "image-text-to-text",
            model=model,
            processor=processor,
            torch_dtype=torch.bfloat16,
        )
    except Exception:
        pipe = pipeline(
            "image-text-to-text",
            model=model,
            tokenizer=getattr(processor, "tokenizer", None),
            image_processor=getattr(processor, "image_processor", None),
            torch_dtype=torch.bfloat16,
        )
    return torch, processor, model, pipe, str(source), source_kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a resumable, provenance-preserving Gemma response pool."
    )
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--expression-pair-manifest", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--task", choices=("all", TASK_IAT, TASK_PROFILE), default="all"
    )
    parser.add_argument("--iat-samples-per-image", type=int, default=8)
    parser.add_argument("--profile-samples-per-image", type=int, default=2)
    parser.add_argument("--seed", type=int, default=8032026)
    parser.add_argument("--iat-temperature", type=float, default=1.0)
    parser.add_argument("--iat-top-p", type=float, default=1.0)
    parser.add_argument("--iat-max-new-tokens", type=int, default=30)
    parser.add_argument("--profile-temperature", type=float, default=0.7)
    parser.add_argument("--profile-top-p", type=float, default=0.9)
    parser.add_argument("--profile-max-new-tokens", type=int, default=240)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--base-model-local-dir",
        default=os.environ.get(
            "BASE_MODEL_LOCAL_DIR", os.environ.get("GEMMA_LOCAL_DIR", "")
        ),
    )
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "eager"), default="sdpa"
    )
    parser.add_argument("--allow-hf-download", action="store_true")
    parser.add_argument("--allow-unpinned-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    metadata_csv = args.metadata_csv.resolve()
    pair_manifest_path = args.expression_pair_manifest.resolve()
    pair_manifest = verify_expression_pair_artifact(metadata_csv, pair_manifest_path)
    metadata = read_csv(metadata_csv)
    tasks = {TASK_IAT, TASK_PROFILE} if args.task == "all" else {args.task}
    validate_generation_args(args, tasks)
    jobs = build_generation_jobs(
        metadata,
        iat_samples_per_image=args.iat_samples_per_image,
        profile_samples_per_image=args.profile_samples_per_image,
        seed=args.seed,
        tasks=tasks,
    )
    images = resolve_job_images(root, jobs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "gemma_responses.jsonl"
    manifest_path = args.output_dir / "gemma_response_manifest.json"
    output_csv = args.output_dir / "gemma_response_pool.csv"
    base_manifest = {
        "metadata_csv": str(metadata_csv),
        "metadata_csv_sha256": sha256_file(metadata_csv),
        "expression_pair_manifest": str(pair_manifest_path),
        "expression_pair_manifest_sha256": sha256_file(pair_manifest_path),
        "expression_pair_policy_version": pair_manifest["policy_version"],
        "images": len(images),
        "expected_responses": len(jobs),
        "tasks": sorted(tasks),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "attention_implementation": args.attn_implementation,
        "race_or_gender_metadata_in_prompt": False,
        "dry_run": args.dry_run,
        **generation_manifest_settings(metadata, jobs, tasks, args),
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        invariant_keys = tuple(base_manifest)
        mismatches = [
            key
            for key in invariant_keys
            if existing_manifest.get(key) != base_manifest.get(key)
        ]
        if mismatches:
            raise ValueError(f"existing Gemma generation differs for: {mismatches}")
        if existing_manifest.get("status") == "complete":
            if not output_csv.is_file() or existing_manifest.get(
                "output_csv_sha256"
            ) != sha256_file(output_csv):
                raise ValueError(
                    "completed Gemma response CSV differs from its manifest"
                )
            print(json.dumps(existing_manifest, indent=2, sort_keys=True))
            return 0
    if args.dry_run:
        base_manifest["status"] = "validated_dry_run"
        manifest_path.write_text(
            json.dumps(base_manifest, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(base_manifest, indent=2, sort_keys=True))
        return 0

    checkpoint = load_checkpoint(checkpoint_path)
    expected_ids = {job.response_id for job in jobs}
    extras = set(checkpoint) - expected_ids
    if extras:
        raise ValueError(
            f"Gemma checkpoint has unexpected response IDs: {sorted(extras)[:3]}"
        )
    base_manifest["status"] = "generating"
    manifest_path.write_text(
        json.dumps(base_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch, processor, model, pipe, source, source_kind = load_model(args)
    from PIL import Image

    started = time.time()
    for position, job in enumerate(jobs, 1):
        if job.response_id in checkpoint:
            if checkpoint[job.response_id].get("job_input_sha256") != job_input_sha256(
                job, args
            ):
                raise ValueError(
                    f"Gemma checkpoint provenance mismatch: {job.response_id}"
                )
            continue
        seed_everything(job.generation_seed, torch)
        with Image.open(images[job.image_sha256]) as source_image:
            image = source_image.convert("RGB").copy()
        with torch.inference_mode():
            output = pipe(
                image,
                text=job.prompt,
                return_full_text=False,
                generate_kwargs=generation_kwargs(job, args),
            )
        response = extract_generated_text(output)
        if not response:
            raise RuntimeError(f"Gemma returned an empty response: {job.response_id}")
        record = {
            "response_id": job.response_id,
            "image_sha256": job.image_sha256,
            "task_type": job.task_type,
            "prompt": job.prompt,
            "response": response,
            "response_provenance": "base_gemma_3_12b_it_observed",
            "sample_index": job.sample_index,
            "generation_seed": job.generation_seed,
            "job_input_sha256": job_input_sha256(job, args),
        }
        append_checkpoint(checkpoint_path, record)
        checkpoint[job.response_id] = record
        print(f"Gemma response {position}/{len(jobs)}: {job.response_id}", flush=True)

    ordered = [checkpoint[job.response_id] for job in jobs]
    write_csv(output_csv, ordered)
    base_manifest.update(
        {
            "status": "complete",
            "model_source": source,
            "model_source_kind": source_kind,
            "completed_responses": len(ordered),
            "output_csv": str(output_csv.resolve()),
            "output_csv_sha256": sha256_file(output_csv),
            "wall_seconds": time.time() - started,
            "gpu_peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
            "gpu_peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
        }
    )
    manifest_path.write_text(json.dumps(base_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(base_manifest, indent=2, sort_keys=True))

    del pipe, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
