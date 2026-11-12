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

from august3_debiasing.pair_policy import prompt_metadata_hits
from august3_debiasing.prompts import GROUNDED_PROFILE_PROMPT, prompt_sha256
from august3_debiasing.response_generation import stable_seed
from august3_debiasing.run_control import adapter_artifact_hashes


DEFAULT_MODEL_ID = "google/gemma-3-12b-it"
DEFAULT_MODEL_REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"
REQUIRED_METADATA = {
    "image_id",
    "profile_eval_match_id",
    "filename",
    "image_path",
    "image_sha256",
    "fairface_race",
    "fairface_gender",
    "age_band",
    "expression_class",
    "expression_fine",
}


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


def resolve_image(root: Path, raw: str, expected_sha256: str) -> Path:
    candidate = Path(raw)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"image path escapes asset root: {raw}")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256.lower():
        raise ValueError(f"image SHA-256 mismatch for {raw}: {actual}")
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
                raise ValueError(
                    f"duplicate profile response at line {line_number}: {response_id}"
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


def seed_everything(seed: int, torch: Any) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pipeline(args: argparse.Namespace):
    import torch
    import transformers
    from transformers import AutoProcessor, pipeline

    local_dir = Path(args.base_model_local_dir).expanduser() if args.base_model_local_dir else None
    if local_dir and local_dir.is_dir():
        source: str | Path = local_dir.resolve()
        source_kind = "local_snapshot"
        if not args.allow_unpinned_model and source.name != args.model_revision:
            raise ValueError("local Gemma snapshot does not match the pinned revision")
        common = {"local_files_only": True}
    else:
        if not args.allow_hf_download:
            raise FileNotFoundError("pinned local Gemma snapshot is required")
        source = args.model_id
        source_kind = "huggingface_revision"
        common = {"revision": args.model_revision}
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, **common)
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        **common,
    }
    model = None
    errors = []
    for model_class in (
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
    ):
        if model_class is None:
            continue
        try:
            model = model_class.from_pretrained(source, **load_kwargs)
            break
        except Exception as exc:
            errors.append(f"{model_class.__name__}: {exc}")
    if model is None:
        raise RuntimeError("could not load Gemma: " + " | ".join(errors))
    if args.condition == "adapter":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
        model.eval()
        model = model.merge_and_unload(safe_merge=True)
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
    reference = ROOT_DIR / "reference" / "frozen_base"
    parser = argparse.ArgumentParser(
        description="Generate paired base or adapter profiles on the frozen profile panel."
    )
    parser.add_argument(
        "--metadata-csv", type=Path, default=reference / "profile_eval_metadata.csv"
    )
    parser.add_argument("--asset-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--condition", choices=("base", "adapter"), required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--base-model-local-dir",
        default=os.environ.get("BASE_MODEL_LOCAL_DIR", os.environ.get("GEMMA_LOCAL_DIR", "")),
    )
    parser.add_argument("--allow-hf-download", action="store_true")
    parser.add_argument("--allow-unpinned-model", action="store_true")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--generation-seed", type=int, default=8032026)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if (args.condition == "adapter") != (args.adapter_dir is not None):
        raise ValueError("--adapter-dir is required only when --condition adapter is used")
    if not 0 < args.temperature or not 0 < args.top_p <= 1:
        raise ValueError("temperature and top-p must be positive; top-p must not exceed 1")
    if prompt_metadata_hits(GROUNDED_PROFILE_PROMPT):
        raise RuntimeError("frozen profile prompt contains protected metadata")

    metadata_path = args.metadata_csv.resolve()
    metadata = read_csv(metadata_path)
    if not metadata:
        raise ValueError("profile evaluation metadata is empty")
    missing = REQUIRED_METADATA - set(metadata[0])
    if missing:
        raise ValueError(f"profile evaluation metadata is missing columns: {sorted(missing)}")
    image_ids = [row["image_id"].strip() for row in metadata]
    image_hashes = [row["image_sha256"].strip().lower() for row in metadata]
    if len(set(image_ids)) != len(image_ids) or len(set(image_hashes)) != len(image_hashes):
        raise ValueError("profile evaluation image IDs and hashes must be unique")
    images = {
        row["image_id"].strip(): resolve_image(
            args.asset_root.resolve(), row["image_path"], row["image_sha256"]
        )
        for row in metadata
    }
    resolved_adapter = args.adapter_dir.resolve() if args.adapter_dir else None
    frozen_adapter_hashes = (
        adapter_artifact_hashes(resolved_adapter) if resolved_adapter else {}
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.condition}_profile_responses.jsonl"
    output_csv = args.output_dir / f"{args.condition}_profile_responses.csv"
    manifest_path = args.output_dir / f"{args.condition}_profile_generation_manifest.json"
    invariant_manifest = {
        "condition": args.condition,
        "metadata_csv": str(metadata_path),
        "metadata_csv_sha256": sha256_file(metadata_path),
        "images": len(metadata),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "attention_implementation": args.attn_implementation,
        "generation_method": "transformers_image_text_to_text_pipeline",
        "generation_seed": args.generation_seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "prompt_sha256": prompt_sha256("grounded_profile"),
        "adapter_dir": str(resolved_adapter) if resolved_adapter else None,
        "adapter_hashes": frozen_adapter_hashes,
        "protected_metadata_in_prompt": False,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected in invariant_manifest.items():
            if existing.get(key) != expected:
                raise ValueError(f"existing profile manifest differs for {key}")
        if existing.get("status") == "complete":
            if not output_csv.is_file() or existing.get("output_csv_sha256") != sha256_file(
                output_csv
            ):
                raise ValueError("completed profile output differs from its manifest")
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0
    if args.dry_run:
        manifest = {**invariant_manifest, "status": "validated_dry_run", "dry_run": True}
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    checkpoint = load_checkpoint(checkpoint_path)
    expected_ids = {f"profile-eval-{row['image_id'].strip()}" for row in metadata}
    extras = set(checkpoint) - expected_ids
    if extras:
        raise ValueError(f"profile checkpoint has unexpected response IDs: {sorted(extras)[:3]}")
    torch, processor, model, pipe, source, source_kind = load_pipeline(args)
    from PIL import Image

    started = time.time()
    for position, row in enumerate(metadata, 1):
        image_id = row["image_id"].strip()
        response_id = f"profile-eval-{image_id}"
        generation_seed = stable_seed(args.generation_seed, image_id)
        if response_id in checkpoint:
            existing = checkpoint[response_id]
            if (
                existing.get("image_sha256") != row["image_sha256"].strip().lower()
                or int(existing.get("generation_seed", -1)) != generation_seed
                or existing.get("condition") != args.condition
            ):
                raise ValueError(f"checkpoint provenance mismatch for {response_id}")
            continue
        seed_everything(generation_seed, torch)
        with Image.open(images[image_id]) as source_image:
            image = source_image.convert("RGB").copy()
        with torch.inference_mode():
            output = pipe(
                image,
                text=GROUNDED_PROFILE_PROMPT,
                return_full_text=False,
                generate_kwargs={
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": True,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
            )
        record = {
            "response_id": response_id,
            "condition": args.condition,
            "image_id": image_id,
            "filename": row["filename"].strip(),
            "image_sha256": row["image_sha256"].strip().lower(),
            "task_type": "grounded_profile",
            "prompt": GROUNDED_PROFILE_PROMPT,
            "response": extract_generated_text(output),
            "response_provenance": (
                "pinned_base_gemma_3_12b_it"
                if args.condition == "base"
                else "august3_fresh_lora_merged"
            ),
            "generation_seed": generation_seed,
            "fairface_race": row["fairface_race"].strip(),
            "fairface_gender": row["fairface_gender"].strip(),
            "age_band": row["age_band"].strip(),
            "expression_class": row["expression_class"].strip(),
            "expression_fine": row["expression_fine"].strip(),
        }
        append_checkpoint(checkpoint_path, record)
        checkpoint[response_id] = record
        print(f"{args.condition} profile {position}/{len(metadata)}: {image_id}", flush=True)

    ordered = [checkpoint[f"profile-eval-{row['image_id'].strip()}"] for row in metadata]
    write_csv(output_csv, ordered)
    manifest = {
        **invariant_manifest,
        "status": "complete",
        "dry_run": False,
        "model_source": source,
        "model_source_kind": source_kind,
        "completed_responses": len(ordered),
        "output_csv": str(output_csv.resolve()),
        "output_csv_sha256": sha256_file(output_csv),
        "wall_seconds": time.time() - started,
        "gpu_peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
        "gpu_peak_reserved_gib": (
            torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))

    del pipe, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
