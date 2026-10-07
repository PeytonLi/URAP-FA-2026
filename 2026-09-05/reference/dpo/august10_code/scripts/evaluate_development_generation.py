#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.constants import TASK_IAT, TASK_PROFILE
from august3_debiasing.development_generation import (
    profile_generation_metrics_by_race,
    summarize_development_generations,
)
from august3_debiasing.evaluation_metrics import compute_metrics, extract_allowed_words
from august3_debiasing.pair_policy import IDENTITY_POLICY_VERSION, parse_iat_response
from august3_debiasing.prompts import (
    GROUNDED_PROFILE_PROMPT,
    IAT_GENERATION_PROMPT,
    prompt_sha256,
)
from august3_debiasing.reference_integrity import sha256_file
from august3_debiasing.response_generation import stable_seed
from august3_debiasing.run_control import adapter_artifact_hashes, atomic_write_json

DEFAULT_MODEL_ID = "google/gemma-3-12b-it"
DEFAULT_MODEL_REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"


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
                    f"duplicate development response at line {line_number}: {response_id}"
                )
            records[response_id] = record
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def resolve_image(root: Path, raw: str, expected_sha256: str) -> Path:
    candidate = Path(raw)
    path = (
        candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    )
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"development image path escapes asset root: {raw}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != expected_sha256.lower():
        raise ValueError(f"development image SHA-256 mismatch: {raw}")
    return path


def extract_pipeline_text(output: Any) -> str:
    if not isinstance(output, list) or not output:
        raise ValueError(f"unexpected pipeline output: {type(output).__name__}")
    value = output[0].get("generated_text")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for message in reversed(value):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
    raise ValueError("pipeline returned no assistant text")


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
    from peft import PeftModel
    from transformers import AutoProcessor, pipeline

    local_dir = (
        Path(args.base_model_local_dir).expanduser()
        if args.base_model_local_dir
        else None
    )
    if local_dir and local_dir.is_dir():
        source: str | Path = local_dir.resolve()
        if not args.allow_unpinned_model and source.name != args.model_revision:
            raise ValueError("local model snapshot does not match the pinned revision")
        common = {"local_files_only": True}
        source_kind = "local_snapshot"
    else:
        if not args.allow_hf_download:
            raise FileNotFoundError("pinned local Gemma snapshot is required")
        source = args.model_id
        common = {"revision": args.model_revision}
        source_kind = "huggingface_revision"
    processor = AutoProcessor.from_pretrained(source, trust_remote_code=True, **common)
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
        **common,
    }
    base_model = None
    errors = []
    for model_class in (
        getattr(transformers, "AutoModelForImageTextToText", None),
        getattr(transformers, "AutoModelForVision2Seq", None),
    ):
        if model_class is None:
            continue
        try:
            base_model = model_class.from_pretrained(source, **load_kwargs)
            break
        except Exception as exc:  # noqa: BLE001 - try the next supported model class
            errors.append(f"{model_class.__name__}: {exc}")
    if base_model is None:
        raise RuntimeError("could not load Gemma: " + " | ".join(errors))
    model = (
        base_model
        if args.base_only
        else PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=False)
    )
    model.eval()
    if not args.base_only and not hasattr(model, "disable_adapter"):
        raise RuntimeError(
            "installed PEFT does not support temporary adapter disabling"
        )
    try:
        pipe = pipeline(
            "image-text-to-text",
            model=model,
            processor=processor,
            torch_dtype=torch.bfloat16,
        )
    except Exception:  # noqa: BLE001 - support both pipeline constructor APIs
        pipe = pipeline(
            "image-text-to-text",
            model=model,
            tokenizer=getattr(processor, "tokenizer", None),
            image_processor=getattr(processor, "image_processor", None),
            torch_dtype=torch.bfloat16,
        )
    return torch, model, processor, pipe, str(source), source_kind


def condition_context(model: Any, condition: str):
    if condition == "base":
        return model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    if not hasattr(model, "disable_adapter"):
        raise RuntimeError("adapter generation requested from a base-only model")
    return nullcontext()


def invoke_pipeline(
    pipe: Any,
    image: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    return extract_pipeline_text(
        pipe(
            image,
            text=prompt,
            return_full_text=False,
            generate_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
    )


def generate_response(
    pipe: Any,
    model: Any,
    image: Any,
    row: dict[str, str],
    condition: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    context = condition_context(model, condition)
    if row["task_type"] == TASK_PROFILE:
        with context:
            raw = invoke_pipeline(
                pipe,
                image,
                GROUNDED_PROFILE_PROMPT,
                max_new_tokens=args.profile_max_new_tokens,
                do_sample=True,
                temperature=args.profile_temperature,
                top_p=args.profile_top_p,
            )
        return {
            "response": raw,
            "valid_three": "",
            "response_source": "sampled",
            "sampled_raw": raw,
            "fallback_raw": "",
        }

    with context:
        sampled = invoke_pipeline(
            pipe,
            image,
            IAT_GENERATION_PROMPT,
            max_new_tokens=args.iat_max_new_tokens,
            do_sample=True,
            temperature=args.iat_temperature,
            top_p=args.iat_top_p,
        )
    words = extract_allowed_words(sampled)
    if len(words) == 3:
        response = ", ".join(words)
        return {
            "response": response,
            "valid_three": parse_iat_response(response)["valid"],
            "response_source": "sampled",
            "sampled_raw": sampled,
            "fallback_raw": "",
        }
    context = condition_context(model, condition)
    with context:
        fallback = invoke_pipeline(
            pipe,
            image,
            IAT_GENERATION_PROMPT,
            max_new_tokens=args.iat_fallback_max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
    words = extract_allowed_words(fallback)
    response = ", ".join(words) if words else "N/A"
    return {
        "response": response,
        "valid_three": parse_iat_response(response)["valid"],
        "response_source": "fallback" if words else "fallback_none",
        "sampled_raw": sampled,
        "fallback_raw": fallback,
    }


def plot_summary(output_dir: Path, summary: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    artifacts = []
    races = ("Black", "White")
    x = np.arange(2)
    width = 0.36
    fig, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    axis.bar(
        x - width / 2,
        [summary["base_iat"]["race_metrics"][race]["negative_rate"] for race in races],
        width,
        label="Base",
        color="#5B6770",
    )
    axis.bar(
        x + width / 2,
        [
            summary["adapter_iat"]["race_metrics"][race]["negative_rate"]
            for race in races
        ],
        width,
        label="LoRA",
        color="#0B6E75",
    )
    axis.set_xticks(x, races)
    axis.set_ylabel("Negative selected-term rate")
    axis.set_title("Development IAT generation screen")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "01_development_iat_negative_rates.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))

    if summary["base_profiles"]["overall"]["responses"]:
        metrics = (
            "length_valid_rate",
            "uses_the_person_rate",
            "identity_leakage_rate",
            "unique_response_rate",
            "distinct_bigram_rate",
        )
        fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4), dpi=180)
        for axis, metric in zip(axes, metrics):
            axis.bar(
                ["Base", "LoRA"],
                [
                    summary["base_profiles"]["overall"][metric],
                    summary["adapter_profiles"]["overall"][metric],
                ],
                color=["#5B6770", "#0B6E75"],
            )
            axis.set_title(metric.replace("_", " ").title(), fontsize=9)
            axis.tick_params(axis="x", rotation=20)
        fig.suptitle("Development profile mechanical safeguards")
    else:
        fig, axis = plt.subplots(figsize=(8, 4), dpi=180)
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "Profile generation not evaluated in this micro-pilot",
            ha="center",
            va="center",
            fontsize=12,
        )
    fig.tight_layout()
    path = output_dir / "02_development_profile_safeguards.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))

    points = [
        summary["iat_bootstrap"]["groups"][race]["mean_rate_change"] for race in races
    ]
    intervals = [summary["iat_bootstrap"]["groups"][race]["ci_95"] for race in races]
    lower = [point - interval[0] for point, interval in zip(points, intervals)]
    upper = [interval[1] - point for point, interval in zip(points, intervals)]
    fig, axis = plt.subplots(figsize=(8, 4), dpi=180)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.errorbar(points, races, xerr=[lower, upper], fmt="o", capsize=4)
    axis.set_xlabel("LoRA minus base negative-rate change")
    axis.set_title("Development paired image bootstrap")
    fig.tight_layout()
    path = output_dir / "03_development_iat_bootstrap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))
    return artifacts


def plot_base_challenge(output_dir: Path, summary: dict[str, Any]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#0B6E75", "#5B6770"]
    races = ("Black", "White")
    artifacts = []
    fig, axis = plt.subplots(figsize=(7, 4.5), dpi=180)
    axis.bar(
        races,
        [summary["base_iat"]["race_metrics"][race]["negative_rate"] for race in races],
        color=colors,
    )
    axis.set_ylabel("Negative selected-term rate")
    axis.set_title("Base Gemma development IAT screen")
    fig.tight_layout()
    path = output_dir / "01_base_development_iat_negative_rates.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))

    metrics = ("length_valid_rate", "uses_the_person_rate", "identity_leakage_rate")
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 4), dpi=180)
    for axis, metric in zip(axes, metrics):
        axis.bar(
            races,
            [summary["base_profiles"][race][metric] for race in races],
            color=colors,
        )
        axis.set_ylim(0, 1)
        axis.set_title(metric.replace("_", " ").title(), fontsize=9)
    fig.suptitle("Base Gemma development profile safeguards")
    fig.tight_layout()
    path = output_dir / "02_base_development_profile_safeguards.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample base and LoRA outputs on held-out development images."
    )
    parser.add_argument("--frozen-csv", required=True, type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--base-responses-csv", type=Path)
    parser.add_argument(
        "--gate-config", type=Path, default=ROOT_DIR / "config" / "pilot_gate.json"
    )
    parser.add_argument("--asset-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--base-model-local-dir",
        default=os.environ.get(
            "BASE_MODEL_LOCAL_DIR", os.environ.get("GEMMA_LOCAL_DIR", "")
        ),
    )
    parser.add_argument("--allow-hf-download", action="store_true")
    parser.add_argument("--allow-unpinned-model", action="store_true")
    parser.add_argument(
        "--attn-implementation", choices=("sdpa", "eager"), default="sdpa"
    )
    parser.add_argument("--generation-seed", type=int, default=8035126)
    parser.add_argument("--iat-repeats", type=int, default=5)
    parser.add_argument("--profile-repeats", type=int, default=2)
    parser.add_argument("--iat-temperature", type=float, default=1.0)
    parser.add_argument("--iat-top-p", type=float, default=1.0)
    parser.add_argument("--iat-max-new-tokens", type=int, default=30)
    parser.add_argument("--iat-fallback-max-new-tokens", type=int, default=20)
    parser.add_argument("--profile-temperature", type=float, default=0.7)
    parser.add_argument("--profile-top-p", type=float, default=0.9)
    parser.add_argument("--profile-max-new-tokens", type=int, default=240)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=8035126)
    parser.add_argument(
        "--allow-incomplete-task-groups",
        action="store_true",
        help="Permit an explicitly labeled IAT-only micro-development split.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iat_repeats <= 0 or args.profile_repeats <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("generation repetitions and bootstrap samples must be positive")
    if not 0 < args.iat_top_p <= 1 or not 0 < args.profile_top_p <= 1:
        raise ValueError("top-p values must be in (0, 1]")
    if args.base_only:
        if args.training_manifest is not None or args.adapter_dir is not None:
            raise ValueError("base-only evaluation must not receive adapter artifacts")
        if args.base_responses_csv is not None:
            raise ValueError("base-only evaluation cannot import base responses")
    elif args.training_manifest is None or args.adapter_dir is None:
        raise ValueError("paired evaluation requires --training-manifest and --adapter-dir")
    frozen_csv = args.frozen_csv.resolve()
    training_manifest_path = (
        args.training_manifest.resolve() if args.training_manifest is not None else None
    )
    adapter_dir = args.adapter_dir.resolve() if args.adapter_dir is not None else None
    training_manifest: dict[str, Any] = {}
    adapter_hashes: dict[str, str] = {}
    if not args.base_only:
        training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
        if training_manifest.get("status") != "complete":
            raise ValueError("training manifest is not complete")
        if training_manifest.get("frozen_csv_sha256") != sha256_file(frozen_csv):
            raise ValueError("development CSV differs from the training manifest")
        if training_manifest.get("model_revision") != args.model_revision:
            raise ValueError("training and development evaluation model revisions differ")
        adapter_hashes = adapter_artifact_hashes(adapter_dir)
        authorized_hashes = [training_manifest.get("final_adapter_hashes")]
        authorized_hashes.extend(
            record.get("hashes")
            for record in training_manifest.get("checkpoint_artifacts", [])
        )
        if adapter_hashes not in authorized_hashes:
            raise ValueError("adapter bytes are absent from the completed training manifest")

    rows = [
        row
        for row in read_csv(frozen_csv)
        if row.get("split", "").strip() == "development"
    ]
    if not rows:
        raise ValueError("frozen CSV has no development rows")
    tasks = {row.get("task_type", "").strip() for row in rows}
    expected_tasks = {TASK_IAT, TASK_PROFILE}
    if not tasks or not tasks <= expected_tasks or TASK_IAT not in tasks:
        raise ValueError("development split must contain a recognized IAT task")
    if tasks != expected_tasks and not args.allow_incomplete_task_groups:
        raise ValueError("development split must contain IAT and profile tasks")
    for row in rows:
        expected_prompt = (
            IAT_GENERATION_PROMPT
            if row.get("task_type", "").strip() == TASK_IAT
            else GROUNDED_PROFILE_PROMPT
        )
        if row.get("prompt", "").strip() != expected_prompt:
            raise ValueError(
                f"development prompt mismatch for {row.get('pair_id', '')}"
            )
    pair_ids = [row.get("pair_id", "").strip() for row in rows]
    if not all(pair_ids) or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("development pair IDs must be nonempty and unique")
    race_task_counts = {
        f"{task}/{race}": sum(
            row["task_type"] == task and row["fairface_race"] == race for row in rows
        )
        for task in (TASK_IAT, TASK_PROFILE)
        for race in ("Black", "White")
    }
    missing_task_race_groups = sorted(
        key for key, count in race_task_counts.items() if not count
    )
    if missing_task_race_groups and not args.allow_incomplete_task_groups:
        raise ValueError("development split is missing a task/race group")
    if any(
        race_task_counts[f"{task}/Black"] != race_task_counts[f"{task}/White"]
        for task in tasks
    ):
        raise ValueError("development task counts differ by race")
    image_paths = {
        row["pair_id"]: resolve_image(
            args.asset_root.resolve(), row["image_path"], row["image_sha256"]
        )
        for row in rows
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "development_generation_responses.jsonl"
    manifest_path = args.output_dir / "development_generation_manifest.json"
    invariant = {
        "development_only": True,
        "confirmatory_evaluation_opened": False,
        "evaluation_mode": "base_only_challenge" if args.base_only else "paired_checkpoint",
        "frozen_csv": str(frozen_csv),
        "frozen_csv_sha256": sha256_file(frozen_csv),
        "training_manifest": str(training_manifest_path) if training_manifest_path else None,
        "training_manifest_sha256": (
            sha256_file(training_manifest_path) if training_manifest_path else None
        ),
        "adapter_dir": str(adapter_dir) if adapter_dir else None,
        "adapter_hashes": adapter_hashes,
        "imported_base_responses_csv": (
            str(args.base_responses_csv.resolve()) if args.base_responses_csv else None
        ),
        "imported_base_responses_csv_sha256": (
            sha256_file(args.base_responses_csv.resolve())
            if args.base_responses_csv
            else None
        ),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "attention_implementation": args.attn_implementation,
        "generation_method": (
            "base_transformers_image_text_to_text_pipeline"
            if args.base_only
            else "single_peft_pipeline_with_temporary_adapter_disable"
        ),
        "generation_seed": args.generation_seed,
        "iat_repeats": args.iat_repeats,
        "profile_repeats": args.profile_repeats,
        "iat_temperature": args.iat_temperature,
        "iat_top_p": args.iat_top_p,
        "iat_max_new_tokens": args.iat_max_new_tokens,
        "iat_fallback_max_new_tokens": args.iat_fallback_max_new_tokens,
        "profile_temperature": args.profile_temperature,
        "profile_top_p": args.profile_top_p,
        "profile_max_new_tokens": args.profile_max_new_tokens,
        "iat_prompt_sha256": prompt_sha256("iat_3word"),
        "profile_prompt_sha256": prompt_sha256("grounded_profile"),
        "identity_policy_version": IDENTITY_POLICY_VERSION,
        "development_rows": len(rows),
        "development_task_race_counts": race_task_counts,
        "incomplete_task_groups_allowed": args.allow_incomplete_task_groups,
        "missing_development_task_race_groups": missing_task_race_groups,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [
            key for key, value in invariant.items() if existing.get(key) != value
        ]
        if mismatches:
            raise ValueError(
                f"existing development evaluation differs for: {mismatches}"
            )
        if existing.get("status") == "complete":
            response_csv = args.output_dir / "development_generation_responses.csv"
            if not response_csv.is_file() or existing.get(
                "response_csv_sha256"
            ) != sha256_file(response_csv):
                raise ValueError(
                    "completed development response CSV differs from its manifest"
                )
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0
    manifest = {
        **invariant,
        "status": "validated_dry_run" if args.dry_run else "generating",
    }
    atomic_write_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    conditions = ("base",) if args.base_only else ("base", "adapter")
    expected_specs: dict[str, dict[str, Any]] = {}
    for row in rows:
        repetitions = (
            args.iat_repeats
            if row["task_type"] == TASK_IAT
            else args.profile_repeats
        )
        for iteration in range(1, repetitions + 1):
            generation_seed = stable_seed(
                args.generation_seed, f"{row['pair_id']}:{iteration}"
            )
            for condition in conditions:
                response_id = (
                    f"development-{condition}-{row['pair_id']}-{iteration:02d}"
                )
                expected_specs[response_id] = {
                    "condition": condition,
                    "image": row["pair_id"],
                    "image_sha256": row["image_sha256"],
                    "iteration": iteration,
                    "generation_seed": generation_seed,
                    "task_type": row["task_type"],
                    "prompt": row["prompt"],
                    "fairface_race": row["fairface_race"],
                }
    checkpoint = load_checkpoint(checkpoint_path)
    if args.base_responses_csv is not None:
        for imported in read_csv(args.base_responses_csv.resolve()):
            record = dict(imported)
            record["iteration"] = int(record["iteration"])
            record["generation_seed"] = int(record["generation_seed"])
            response_id = record["response_id"]
            if record.get("condition") != "base":
                raise ValueError("imported base response CSV contains a non-base row")
            if response_id in checkpoint and checkpoint[response_id] != record:
                raise ValueError(f"imported base response conflicts with checkpoint: {response_id}")
            checkpoint[response_id] = record
    expected_ids = set(expected_specs)
    extras = set(checkpoint) - expected_ids
    if extras:
        raise ValueError(
            f"development checkpoint has unexpected IDs: {sorted(extras)[:3]}"
        )
    for response_id, record in checkpoint.items():
        expected = expected_specs[response_id]
        mismatches = [
            key for key, value in expected.items() if record.get(key) != value
        ]
        if mismatches or not str(record.get("response", "")).strip():
            raise ValueError(
                f"development checkpoint provenance mismatch for {response_id}: {mismatches}"
            )

    torch, model, processor, pipe, source, source_kind = load_pipeline(args)
    from PIL import Image

    started = time.time()
    completed = 0
    for row in rows:
        repetitions = (
            args.iat_repeats
            if row["task_type"] == TASK_IAT
            else args.profile_repeats
        )
        with Image.open(image_paths[row["pair_id"]]) as source_image:
            image = source_image.convert("RGB").copy()
        for iteration in range(1, repetitions + 1):
            generation_seed = stable_seed(
                args.generation_seed, f"{row['pair_id']}:{iteration}"
            )
            for condition in conditions:
                response_id = (
                    f"development-{condition}-{row['pair_id']}-{iteration:02d}"
                )
                if response_id in checkpoint:
                    completed += 1
                    continue
                seed_everything(generation_seed, torch)
                with torch.inference_mode():
                    generated = generate_response(
                        pipe, model, image, row, condition, args
                    )
                record = {
                    "response_id": response_id,
                    "condition": condition,
                    "image": row["pair_id"],
                    "source_image_id": row.get("source_image_id") or row["pair_id"],
                    "image_sha256": row["image_sha256"],
                    "iteration": iteration,
                    "generation_seed": generation_seed,
                    "task_type": row["task_type"],
                    "prompt": row["prompt"],
                    "fairface_race": row["fairface_race"],
                    "fairface_gender": row["fairface_gender"],
                    "age_bucket": row["age_bucket"],
                    "expression_class": row["expression_class"],
                    "expression_fine": row["expression_fine"],
                    **generated,
                }
                append_checkpoint(checkpoint_path, record)
                checkpoint[response_id] = record
                completed += 1
                print(
                    f"development generation {completed}/{len(expected_ids)}: {response_id}",
                    flush=True,
                )

    ordered = [checkpoint[response_id] for response_id in sorted(expected_ids)]
    write_csv(args.output_dir / "development_generation_responses.csv", ordered)
    base_iat = [
        row
        for row in ordered
        if row["condition"] == "base" and row["task_type"] == TASK_IAT
    ]
    adapter_iat = [
        row
        for row in ordered
        if row["condition"] == "adapter" and row["task_type"] == TASK_IAT
    ]
    base_profiles = [
        row
        for row in ordered
        if row["condition"] == "base" and row["task_type"] == TASK_PROFILE
    ]
    adapter_profiles = [
        row
        for row in ordered
        if row["condition"] == "adapter" and row["task_type"] == TASK_PROFILE
    ]
    if args.base_only:
        base_iat_metrics = compute_metrics(base_iat)
        base_profile_metrics = profile_generation_metrics_by_race(base_profiles)
        development_policy = json.loads(
            args.gate_config.read_text(encoding="utf-8")
        )["development"]
        negative_terms = {
            race: int(base_iat_metrics["race_metrics"][race]["negative_terms"])
            for race in ("Black", "White")
        }
        profile_responses = {
            race: int(base_profile_metrics[race]["responses"])
            for race in ("Black", "White")
        }
        profile_coverage_applicable = TASK_PROFILE in tasks
        checks = {
            "informative_base_negative_events": {
                "pass": min(negative_terms.values())
                >= development_policy["min_base_negative_terms_per_race"],
                "observed": negative_terms,
                "requirement": (
                    f">= {development_policy['min_base_negative_terms_per_race']} "
                    "for each race"
                ),
            },
            "profile_coverage": {
                "pass": (
                    not profile_coverage_applicable
                    and args.allow_incomplete_task_groups
                )
                or min(profile_responses.values())
                >= development_policy["min_profile_responses_per_race"],
                "observed": profile_responses,
                "requirement": (
                    "not applicable for explicitly permitted IAT-only diagnostic"
                    if not profile_coverage_applicable
                    and args.allow_incomplete_task_groups
                    else f">= {development_policy['min_profile_responses_per_race']} "
                    "for each race"
                ),
            },
        }
        base_gate_pass = all(check["pass"] for check in checks.values())
        summary = {
            "protocol_version": "august10-base-development-challenge-v1",
            "base_iat": base_iat_metrics,
            "base_profiles": base_profile_metrics,
            "gate": {
                "pass": base_gate_pass,
                "decision": (
                    "authorize_short_pilot"
                    if base_gate_pass
                    else "stop_noninformative_development_panel"
                ),
                "checks": checks,
            },
        }
        plots = plot_base_challenge(args.output_dir, summary)
    else:
        summary = summarize_development_generations(
            base_iat,
            adapter_iat,
            base_profiles,
            adapter_profiles,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        summary["gate"].update(
            {
                "coverage_complete": not missing_task_race_groups,
                "missing_task_race_groups": missing_task_race_groups,
            }
        )
        if missing_task_race_groups:
            summary["gate"]["guardrail_pass"] = False
            summary["gate"]["interpretation"] = (
                "IAT-only micro-development screen; profile preservation was not evaluated "
                "and the complete guardrail cannot pass."
            )
        plots = plot_summary(args.output_dir, summary)
    atomic_write_json(args.output_dir / "development_generation_summary.json", summary)
    atomic_write_json(
        args.output_dir / "development_generation_gate.json", summary["gate"]
    )
    manifest.update(
        {
            "status": "complete",
            "model_source": source,
            "model_source_kind": source_kind,
            "completed_responses": len(ordered),
            "response_csv_sha256": sha256_file(
                args.output_dir / "development_generation_responses.csv"
            ),
            "gate": summary["gate"],
            "plots": plots,
            "wall_seconds": time.time() - started,
        }
    )
    if torch.cuda.is_available():
        manifest["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    atomic_write_json(manifest_path, manifest)
    del pipe, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if not args.base_only or summary["gate"]["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
