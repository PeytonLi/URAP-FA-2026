#!/usr/bin/env python3
"""Inference only: pinned base weights, legacy IAT pipeline, explicit task contracts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

from common import ROOT, sha, utcnow, write_csv, write_json

sys.path.insert(0, str(ROOT / "src"))
from multimodel_bias.constants import NEGATIVE_IAT_SET, POSITIVE_IAT_SET
from multimodel_bias.iat import is_exact_three_word_format, parse_three_words
from multimodel_bias.visbias import _parse_comparison

REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"


def extract_text(result):
    value = result[0]["generated_text"]
    if isinstance(value, str):
        return value.strip()
    for message in reversed(value):
        if message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"].strip()
    raise ValueError("No assistant text in pipeline output")


def seed_all(seed, torch):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validate_tasks(tasks):
    keys = [t["task_key"] for t in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate task keys")
    for task in tasks:
        for image, expected in zip(task["image_paths"], task["image_hashes"], strict=True):
            path = (ROOT / image).resolve()
            if not path.is_relative_to(ROOT) or sha(path) != expected:
                raise ValueError(f"Image input drift: {image}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=9900)
    args = parser.parse_args()
    tasks_path = ROOT / "data/frozen/local_tasks.json"
    tasks = json.loads(tasks_path.read_text())
    validate_tasks(tasks)
    if args.probe:
        selected, seen = [], set()
        for task in tasks:
            group = (task["dataset"], task["task_kind"])
            if group not in seen:
                selected.append(task)
                seen.add(group)
        tasks = selected
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tasks": len(tasks),
                    "task_sha256": sha(tasks_path),
                    "model_revision": REVISION,
                    "training": False,
                },
                indent=2,
            )
        )
        return
    if not args.model_dir or args.model_dir.name != REVISION:
        raise ValueError("The local model directory must be the pinned Hugging Face snapshot")
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration, pipeline

    if transformers.__version__ != "4.50.0":
        raise ValueError("This protocol requires Transformers 4.50.0; do not silently upgrade")
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 78e9:
        raise ValueError("One H100 80 GB is required")
    if "H100" not in torch.cuda.get_device_name(0):
        raise ValueError("Unexpected GPU")
    destination = ROOT / "results" / ("gemma_probe" if args.probe else "gemma_local")
    destination.mkdir(parents=True, exist_ok=True)
    record_path = destination / "responses.jsonl"
    if record_path.exists():
        raise ValueError(
            "Existing run is immutable; inspect it before choosing a new run directory"
        )
    started = time.monotonic()
    snapshot_files = [
        p for p in args.model_dir.iterdir() if p.suffix in {".json", ".safetensors", ".model"}
    ]
    hashes = {p.name: sha(p) for p in sorted(snapshot_files)}
    manifest = {
        "started_at": utcnow(),
        "model_id": "google/gemma-3-12b-it",
        "model_revision": REVISION,
        "model_snapshot_path": str(args.model_dir),
        "base_snapshot_hashes": hashes,
        "base_weights_updated": False,
        "adapter_loaded": False,
        "attention_implementation": "eager",
        "task_sha256": sha(tasks_path),
        "planned_tasks": len(tasks),
        "status": "loading",
        "gpu": torch.cuda.get_device_name(0),
        "precision": "bfloat16",
        "environment": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "pillow", "numpy", "accelerate")
        },
    }
    write_json(destination / "manifest.json", manifest)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.requires_grad_(False)
    model.eval()
    pipe = pipeline(
        "image-text-to-text", model=model, processor=processor, torch_dtype=torch.bfloat16
    )
    manifest["model_generation_config"] = model.generation_config.to_dict()
    manifest["status"] = "running"
    write_json(destination / "manifest.json", manifest)
    rows, previous_iat_dataset = [], None
    for index, task in enumerate(tasks):
        if time.monotonic() - started >= args.max_seconds:
            manifest["status"] = "time_budget_stop"
            break
        if (ROOT / "STOP_REQUESTED").exists():
            manifest["status"] = "user_stop"
            break
        tick = time.monotonic()
        images = []
        for name in task["image_paths"]:
            with Image.open(ROOT / name) as image:
                images.append(image.convert("RGB").copy())
        row = {
            k: v
            for k, v in task.items()
            if k not in {"image_paths", "image_hashes", "local_pipeline_prompt"}
        }
        row.update(
            model_key="gemma_local",
            model_display_name="Gemma 3 12B (local pinned)",
            model_id_requested="google/gemma-3-12b-it",
            model_revision=REVISION,
            provider="RunPod H100 / Transformers 4.50.0",
            image_hashes="|".join(task["image_hashes"]),
            created_at=utcnow(),
            status="completed",
            cost_usd=0.0,
            cost_basis="GPU wall time, not per-response API cost",
        )
        try:
            with torch.inference_mode():
                if task["task_kind"] == "iat":
                    if previous_iat_dataset != task["dataset"]:
                        seed_all(task["seed"], torch)
                        previous_iat_dataset = task["dataset"]
                    raw = extract_text(
                        pipe(
                            images[0],
                            text=task["local_pipeline_prompt"],
                            return_full_text=False,
                            generate_kwargs={
                                "max_new_tokens": 30,
                                "do_sample": True,
                                "temperature": task["temperature"],
                                "top_p": task["top_p"],
                            },
                        )
                    )
                    words = parse_three_words(raw)
                    row.update(
                        sampled_raw=raw,
                        fallback_raw="",
                        response_source="sampled",
                        generation_calls=1,
                    )
                    if len(words) != 3:
                        raw = extract_text(
                            pipe(
                                images[0],
                                text=task["local_pipeline_prompt"],
                                return_full_text=False,
                                generate_kwargs={
                                    "max_new_tokens": 20,
                                    "do_sample": False,
                                    "temperature": 1.0,
                                    "top_p": 1.0,
                                    "top_k": 50,
                                },
                            )
                        )
                        words = parse_three_words(raw)
                        row.update(fallback_raw=raw, response_source="fallback", generation_calls=2)
                    row.update(
                        response=", ".join(words),
                        words="|".join(words),
                        valid_three=len(words) == 3,
                        word_count=len(words),
                        negative_count=sum(w in NEGATIVE_IAT_SET for w in words),
                        positive_count=sum(w in POSITIVE_IAT_SET for w in words),
                        exact_format=is_exact_three_word_format(raw, words),
                    )
                    if len(words) != 3:
                        row["status"] = "invalid"
                else:
                    seed_all(task["seed"], torch)
                    content = [{"type": "text", "text": task["prompt"]}]
                    for image_index in range(len(images)):
                        if task["task_kind"] == "comparison":
                            content.append({"type": "text", "text": f"Image {'AB'[image_index]}:"})
                        content.append({"type": "image"})
                    messages = []
                    if task.get("system_prompt"):
                        messages.append(
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": task["system_prompt"]}],
                            }
                        )
                    messages.append({"role": "user", "content": content})
                    prompt = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    inputs = processor(
                        text=prompt,
                        images=images or None,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to(model.device)
                    for key, value in inputs.items():
                        if value.is_floating_point():
                            inputs[key] = value.to(torch.bfloat16)
                    kwargs = {
                        "max_new_tokens": task["max_new_tokens"],
                        "do_sample": task["temperature"] > 0,
                    }
                    if kwargs["do_sample"]:
                        kwargs.update(temperature=task["temperature"], top_p=task["top_p"])
                    generated = model.generate(**inputs, **kwargs)
                    output_tokens = generated[0, inputs["input_ids"].shape[-1] :]
                    text = processor.decode(output_tokens, skip_special_tokens=True).strip()
                    row.update(
                        response=text,
                        prompt_tokens=inputs["input_ids"].shape[-1],
                        completion_tokens=len(output_tokens),
                        generation_calls=1,
                        rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        truncated=len(output_tokens) >= task["max_new_tokens"],
                    )
                    if not text or row["truncated"]:
                        row["status"] = "invalid"
                    if task["task_kind"] == "comparison":
                        answer = _parse_comparison(text)
                        row.update(
                            parsed_answer=answer,
                            selected_race=(
                                task.get("race_a")
                                if answer == "A"
                                else task.get("race_b")
                                if answer == "B"
                                else ""
                            ),
                        )
                        if answer == "UNCLASSIFIED":
                            row["status"] = "invalid"
            row["latency_seconds"] = time.monotonic() - tick
        except Exception as exc:
            row.update(status="runtime_error", error_type=type(exc).__name__, error=str(exc)[:1000])
            manifest["status"] = "runtime_error_stop"
        with record_path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        rows.append(row)
        if index % 10 == 0 or index == len(tasks) - 1 or row["status"] == "runtime_error":
            manifest.update(
                completed_tasks=len(rows),
                elapsed_seconds=time.monotonic() - started,
                peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_gpu_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            )
            write_json(destination / "manifest.json", manifest)
            write_csv(destination / "responses.csv", rows)
            print(
                f"{index + 1}/{len(tasks)} {task['task_kind']} {task['dataset']} "
                f"{row['status']} elapsed={manifest['elapsed_seconds']:.0f}s",
                flush=True,
            )
        if row["status"] == "runtime_error":
            break
    if manifest["status"] == "running":
        manifest["status"] = "complete"
    after = {p.name: sha(p) for p in sorted(snapshot_files)}
    manifest.update(
        finished_at=utcnow(),
        completed_tasks=len(rows),
        valid_tasks=sum(r["status"] == "completed" for r in rows),
        generation_calls=sum(r.get("generation_calls", 0) for r in rows),
        elapsed_seconds=time.monotonic() - started,
        base_hashes_unchanged=after == hashes,
        base_snapshot_hashes_after=after,
    )
    if hashes != after:
        manifest["status"] = "base_integrity_failure"
    write_csv(destination / "responses.csv", rows)
    write_json(destination / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
