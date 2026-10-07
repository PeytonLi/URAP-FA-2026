#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.constants import IAT_WORD_LOOKUP, NEGATIVE_IAT_WORDS
from august3_debiasing.evaluation_metrics import (
    acceptance_criteria,
    compute_metrics,
    enrich_rows,
    paired_bootstrap,
    validate_response_panel,
)
from august3_debiasing.pair_policy import IDENTITY_POLICY_VERSION
from august3_debiasing.prompts import IAT_GENERATION_PROMPT, prompt_sha256
from august3_debiasing.reference_integrity import (
    verify_manifest_internal_hashes,
    verify_named_reference_files,
)
from august3_debiasing.run_control import adapter_artifact_hashes


DEFAULT_MODEL_ID = "google/gemma-3-12b-it"
DEFAULT_MODEL_REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"
FROZEN_BASE_PROTOCOL = {
    "attention_implementation": "eager",
    "generation_seed": 7192026,
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 30,
    "fallback_max_new_tokens": 20,
}
WORD_RE = re.compile(r"[A-Za-z]+")


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


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_and_verify_metadata(
    metadata_path: Path,
    asset_root: Path,
    expected_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    rows = read_csv(metadata_path)
    resolved = []
    for row in rows:
        filename = row["filename"]
        candidate = Path(row["image_path"])
        image_path = candidate.resolve() if candidate.is_absolute() else (asset_root / candidate).resolve()
        if not image_path.is_relative_to(asset_root.resolve()):
            raise ValueError(f"metadata path escapes asset root: {row['image_path']}")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if filename not in expected_hashes or sha256_file(image_path) != expected_hashes[filename]:
            raise ValueError(f"evaluation image SHA-256 mismatch: {filename}")
        revised = dict(row)
        revised["image_path"] = str(image_path)
        resolved.append(revised)
    return resolved


def seed_everything(seed: int, torch: Any) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_three(raw: str) -> list[str]:
    words = []
    for match in WORD_RE.finditer(raw):
        canonical = IAT_WORD_LOOKUP.get(match.group(0).lower())
        if canonical is None:
            continue
        if not words or words[-1] != canonical:
            words.append(canonical)
        if len(words) == 3:
            break
    return words


def extract_pipeline_text(output: Any) -> str:
    value = output[0]["generated_text"]
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for message in reversed(value):
            if isinstance(message, dict) and message.get("role") == "assistant":
                if isinstance(message.get("content"), str):
                    return message["content"].strip()
    raise ValueError("pipeline returned no assistant text")


def generate_one(pipe: Any, image: Any, args: argparse.Namespace) -> dict[str, Any]:
    sampled = extract_pipeline_text(
        pipe(
            image,
            text=IAT_GENERATION_PROMPT,
            return_full_text=False,
            generate_kwargs={
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            },
        )
    )
    words = extract_three(sampled)
    if len(words) == 3:
        return {
            "response": ", ".join(words),
            "valid_three": True,
            "response_source": "sampled",
            "sampled_raw": sampled,
            "fallback_raw": "",
        }
    fallback = extract_pipeline_text(
        pipe(
            image,
            text=IAT_GENERATION_PROMPT,
            return_full_text=False,
            generate_kwargs={
                "max_new_tokens": args.fallback_max_new_tokens,
                "do_sample": False,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 50,
            },
        )
    )
    words = extract_three(fallback)
    return {
        "response": ", ".join(words) if words else "N/A",
        "valid_three": len(words) == 3,
        "response_source": "fallback" if words else "fallback_none",
        "sampled_raw": sampled,
        "fallback_raw": fallback,
    }


def generate_condition(
    pipe: Any,
    metadata: list[dict[str, Any]],
    repeats: int,
    args: argparse.Namespace,
    torch: Any,
) -> list[dict[str, Any]]:
    from PIL import Image

    images = {}
    for row in metadata:
        with Image.open(row["image_path"]) as source:
            images[row["filename"]] = source.convert("RGB").copy()
    seed_everything(args.generation_seed, torch)
    records = []
    with torch.inference_mode():
        for iteration in range(1, repeats + 1):
            print(f"adapter evaluation iteration {iteration}/{repeats}", flush=True)
            for row in metadata:
                generated = generate_one(pipe, images[row["filename"]], args)
                records.append(
                    {
                        "image": row["filename"],
                        "iteration": iteration,
                        **generated,
                        **{
                            key: row.get(key, "")
                            for key in (
                                "dataset_index",
                                "fairface_race",
                                "fairface_gender",
                                "age",
                                "expression_class",
                                "expression_fine",
                                "expression_confidence",
                                "expression_provenance",
                                "age_band",
                                "quality_score",
                                "quality_bin",
                                "neutral_benchmark",
                            )
                        },
                    }
                )
    return records


def load_adapter_pipeline(args: argparse.Namespace):
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoProcessor, pipeline

    local_dir = Path(args.base_model_local_dir).expanduser() if args.base_model_local_dir else None
    if local_dir and local_dir.is_dir():
        source: str | Path = local_dir.resolve()
        if not args.allow_unpinned_model and source.name != args.model_revision:
            raise ValueError("local model snapshot does not match pinned revision")
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
        except Exception as exc:
            errors.append(f"{model_class.__name__}: {exc}")
    if base_model is None:
        raise RuntimeError("could not load Gemma: " + " | ".join(errors))
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_dir, is_trainable=False)
    peft_model.eval()
    model = peft_model.merge_and_unload(safe_merge=True)
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
    return torch, model, processor, pipe, str(source), source_kind


def plot_outputs(
    output_dir: Path,
    base: dict[str, Any],
    adapter: dict[str, Any],
    neutral_base: dict[str, Any],
    neutral_adapter: dict[str, Any],
    bootstrap: dict[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    artifacts = []
    labels = ["Black", "White"]
    x = np.arange(2)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    for axis, before, after, title in (
        (axes[0], base, adapter, "Frozen FairFace evaluation"),
        (axes[1], neutral_base, neutral_adapter, "12-face neutral safeguard"),
    ):
        axis.bar(
            x - width / 2,
            [before["race_metrics"][race]["negative_rate"] for race in labels],
            width,
            label="Base Gemma 3",
        )
        axis.bar(
            x + width / 2,
            [after["race_metrics"][race]["negative_rate"] for race in labels],
            width,
            label="August 3 LoRA",
        )
        axis.set_xticks(x, labels)
        axis.set_ylabel("Negative selected-term rate")
        axis.set_title(title)
        axis.legend()
    fig.tight_layout()
    path = output_dir / "01_base_vs_adapter_negative_rates.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))

    negative_words = [word.lower() for word in NEGATIVE_IAT_WORDS]
    combined = Counter()
    for metrics in (base, adapter):
        for race in labels:
            combined.update(
                {
                    word: metrics["word_counts"][race].get(word, 0)
                    for word in negative_words
                }
            )
    top = [word for word, _ in combined.most_common(15)][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), dpi=180, sharey=True)
    for axis, race in zip(axes, labels):
        y = np.arange(len(top))
        axis.barh(
            y - width / 2,
            [base["word_counts"][race].get(word, 0) for word in top],
            width,
            label="Base",
        )
        axis.barh(
            y + width / 2,
            [adapter["word_counts"][race].get(word, 0) for word in top],
            width,
            label="LoRA",
        )
        axis.set_yticks(y, [word.title() for word in top])
        axis.set_title(race)
        axis.set_xlabel("Selected count")
        axis.legend()
    fig.suptitle("Negative IAT word distribution")
    fig.tight_layout()
    path = output_dir / "02_negative_word_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))

    fine = sorted(
        {
            key.split("|", 1)[0]
            for key in set(base["expression_strata"]) | set(adapter["expression_strata"])
        }
    )
    base_gaps = []
    adapter_gaps = []
    kept_fine = []
    for expression in fine:
        black_key = f"{expression}|Black"
        white_key = f"{expression}|White"
        if all(
            key in base["expression_strata"] and key in adapter["expression_strata"]
            for key in (black_key, white_key)
        ):
            kept_fine.append(expression)
            base_gaps.append(
                base["expression_strata"][black_key]["negative_rate"]
                - base["expression_strata"][white_key]["negative_rate"]
            )
            adapter_gaps.append(
                adapter["expression_strata"][black_key]["negative_rate"]
                - adapter["expression_strata"][white_key]["negative_rate"]
            )
    fig, axis = plt.subplots(figsize=(12, 5), dpi=180)
    positions = np.arange(len(kept_fine))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.bar(positions - width / 2, base_gaps, width, label="Base")
    axis.bar(positions + width / 2, adapter_gaps, width, label="LoRA")
    axis.set_xticks(positions, kept_fine, rotation=30, ha="right")
    axis.set_ylabel("Signed Black-minus-White negative-rate gap")
    axis.set_title("Race gap by visible-expression stratum")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "03_expression_stratified_signed_gaps.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))

    safeguards = ["valid_rate", "identity_leakage_rate", "response_entropy", "unique_response_rate"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), dpi=180)
    for axis, metric in zip(axes, safeguards):
        axis.bar(["Base", "LoRA"], [base[metric], adapter[metric]])
        axis.set_title(metric.replace("_", " "))
    fig.suptitle("Format, leakage, and diversity safeguards")
    fig.tight_layout()
    path = output_dir / "04_quality_safeguards.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))

    points = [bootstrap["groups"][race]["mean_rate_change"] for race in labels]
    lower = [
        points[index] - bootstrap["groups"][race]["ci_95"][0]
        for index, race in enumerate(labels)
    ]
    upper = [
        bootstrap["groups"][race]["ci_95"][1] - points[index]
        for index, race in enumerate(labels)
    ]
    fig, axis = plt.subplots(figsize=(8, 4), dpi=180)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.errorbar(points, labels, xerr=[lower, upper], fmt="o", capsize=5)
    axis.set_xlabel("Adapter minus base negative-rate change")
    axis.set_title("Paired image-level bootstrap, 95% CI")
    fig.tight_layout()
    path = output_dir / "05_paired_bootstrap_forest.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))

    intersection_keys = sorted(base["intersectional_metrics"])
    fig, axis = plt.subplots(figsize=(10, 5), dpi=180)
    positions = np.arange(len(intersection_keys))
    axis.bar(
        positions - width / 2,
        [base["intersectional_metrics"][key]["negative_rate"] for key in intersection_keys],
        width,
        label="Base",
    )
    axis.bar(
        positions + width / 2,
        [adapter["intersectional_metrics"][key]["negative_rate"] for key in intersection_keys],
        width,
        label="LoRA",
    )
    axis.set_xticks(positions, [key.replace("_", " ").title() for key in intersection_keys])
    axis.set_ylabel("Negative selected-term rate")
    axis.set_title("Intersectional evaluation")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "06_intersectional_negative_rates.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))
    return artifacts


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    reference = root / "reference" / "frozen_base"
    parser = argparse.ArgumentParser(
        description="Evaluate an August 3 LoRA against the immutable pipeline-generated base reference."
    )
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--asset-root", type=Path, default=root)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-fairface-csv", type=Path, default=reference / "base_gemma3_fairface_iat.csv")
    parser.add_argument("--base-neutral-csv", type=Path, default=reference / "base_gemma3_neutral_iat.csv")
    parser.add_argument("--fairface-metadata-csv", type=Path, default=reference / "eval_metadata.csv")
    parser.add_argument("--neutral-metadata-csv", type=Path, default=reference / "neutral_benchmark_metadata.csv")
    parser.add_argument("--frozen-input-manifest", type=Path, default=reference / "frozen_input_manifest.json")
    parser.add_argument("--reference-checksums", type=Path, default=reference / "SHA256SUMS")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--base-model-local-dir",
        default=os.environ.get("BASE_MODEL_LOCAL_DIR", os.environ.get("GEMMA_LOCAL_DIR", "")),
    )
    parser.add_argument("--allow-hf-download", action="store_true")
    parser.add_argument("--allow-unpinned-model", action="store_true")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--generation-seed", type=int, default=7192026)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--fallback-max-new-tokens", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=3719)
    parser.add_argument("--allow-protocol-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = {
        "attention_implementation": args.attn_implementation,
        "generation_seed": args.generation_seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "fallback_max_new_tokens": args.fallback_max_new_tokens,
    }
    protocol_mismatches = [
        key for key, expected in FROZEN_BASE_PROTOCOL.items() if protocol[key] != expected
    ]
    if protocol_mismatches and not args.allow_protocol_mismatch:
        raise ValueError(
            "adapter generation differs from the frozen base protocol: "
            + ", ".join(protocol_mismatches)
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not (args.adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json missing from {args.adapter_dir}")
    reference_hashes = verify_named_reference_files(
        args.reference_checksums.resolve(),
        {
            "base_gemma3_fairface_iat.csv": args.base_fairface_csv.resolve(),
            "base_gemma3_neutral_iat.csv": args.base_neutral_csv.resolve(),
            "eval_metadata.csv": args.fairface_metadata_csv.resolve(),
            "neutral_benchmark_metadata.csv": args.neutral_metadata_csv.resolve(),
            "frozen_input_manifest.json": args.frozen_input_manifest.resolve(),
        },
    )
    frozen_manifest = json.loads(args.frozen_input_manifest.read_text(encoding="utf-8"))
    verify_manifest_internal_hashes(frozen_manifest, reference_hashes)
    fairface_metadata = load_and_verify_metadata(
        args.fairface_metadata_csv,
        args.asset_root.resolve(),
        frozen_manifest["eval_image_sha256"],
    )
    neutral_metadata = load_and_verify_metadata(
        args.neutral_metadata_csv,
        args.asset_root.resolve(),
        frozen_manifest["neutral_image_sha256"],
    )
    base_fairface = read_csv(args.base_fairface_csv)
    base_neutral = read_csv(args.base_neutral_csv)
    fairface_repeats = validate_response_panel(base_fairface, fairface_metadata)
    neutral_repeats = validate_response_panel(base_neutral, neutral_metadata)
    manifest = {
        "status": "validated_dry_run" if args.dry_run else "evaluating",
        "adapter_dir": str(args.adapter_dir.resolve()),
        "adapter_config_sha256": sha256_file(args.adapter_dir / "adapter_config.json"),
        "adapter_hashes": adapter_artifact_hashes(args.adapter_dir.resolve()),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "attention_implementation": args.attn_implementation,
        "generation_method": "transformers_image_text_to_text_pipeline",
        "frozen_base_generation_protocol": FROZEN_BASE_PROTOCOL,
        "generation_protocol_mismatches": protocol_mismatches,
        "protocol_mismatch_explicitly_allowed": args.allow_protocol_mismatch,
        "generation_seed": args.generation_seed,
        "prompt_sha256": prompt_sha256("iat_3word"),
        "identity_policy_version": IDENTITY_POLICY_VERSION,
        "base_inference_rerun": False,
        "reference_checksums": str(args.reference_checksums.resolve()),
        "reference_checksums_sha256": sha256_file(args.reference_checksums),
        "verified_reference_hashes": reference_hashes,
        "base_fairface_csv": str(args.base_fairface_csv.resolve()),
        "base_fairface_sha256": sha256_file(args.base_fairface_csv),
        "base_neutral_csv": str(args.base_neutral_csv.resolve()),
        "base_neutral_sha256": sha256_file(args.base_neutral_csv),
        "fairface_images": len(fairface_metadata),
        "fairface_repeats": fairface_repeats,
        "neutral_images": len(neutral_metadata),
        "neutral_repeats": neutral_repeats,
    }
    write_json(args.output_dir / "evaluation_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    torch, _model, _processor, pipe, source, source_kind = load_adapter_pipeline(args)
    started = time.time()
    adapter_fairface = generate_condition(
        pipe, fairface_metadata, fairface_repeats, args, torch
    )
    adapter_neutral = generate_condition(
        pipe, neutral_metadata, neutral_repeats, args, torch
    )
    write_csv(args.output_dir / "adapter_fairface_responses.csv", enrich_rows(adapter_fairface))
    write_csv(args.output_dir / "adapter_neutral_responses.csv", enrich_rows(adapter_neutral))

    base_metrics = compute_metrics(base_fairface)
    adapter_metrics = compute_metrics(adapter_fairface)
    neutral_base_metrics = compute_metrics(base_neutral)
    neutral_adapter_metrics = compute_metrics(adapter_neutral)
    bootstrap = paired_bootstrap(
        base_fairface,
        adapter_fairface,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    neutral_bootstrap = paired_bootstrap(
        base_neutral,
        adapter_neutral,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 1,
    )
    acceptance = acceptance_criteria(base_metrics, adapter_metrics, bootstrap)
    acceptance["checks"].update(
        {
            "neutral_absolute_race_gap_nonincrease": (
                neutral_adapter_metrics["negative_rate_gap_abs"]
                <= neutral_base_metrics["negative_rate_gap_abs"]
            ),
            "neutral_black_negative_rate_nonincrease": (
                neutral_adapter_metrics["race_metrics"]["Black"]["negative_rate"]
                <= neutral_base_metrics["race_metrics"]["Black"]["negative_rate"]
            ),
            "neutral_validity_drop_at_most_1pp": (
                neutral_adapter_metrics["valid_rate"]
                >= neutral_base_metrics["valid_rate"] - 0.01
            ),
            "neutral_black_bootstrap_mean_nonpositive": (
                neutral_bootstrap["groups"]["Black"]["mean_rate_change"] <= 0.0
            ),
        }
    )
    acceptance["pass"] = all(acceptance["checks"].values())
    for name, value in (
        ("base_fairface_metrics.json", base_metrics),
        ("adapter_fairface_metrics.json", adapter_metrics),
        ("base_neutral_metrics.json", neutral_base_metrics),
        ("adapter_neutral_metrics.json", neutral_adapter_metrics),
        ("fairface_paired_bootstrap.json", bootstrap),
        ("neutral_paired_bootstrap.json", neutral_bootstrap),
        ("acceptance_criteria.json", acceptance),
    ):
        write_json(args.output_dir / name, value)

    word_rows = []
    for model_label, metrics in (("base", base_metrics), ("adapter", adapter_metrics)):
        for race, counts in metrics["word_counts"].items():
            for word, count in counts.items():
                word_rows.append(
                    {"model": model_label, "race": race, "word": word, "count": count}
                )
    write_csv(args.output_dir / "word_counts.csv", word_rows)
    summary_rows = []
    for model_label, metrics in (("base", base_metrics), ("adapter", adapter_metrics)):
        summary_rows.append(
            {
                "model": model_label,
                "black_negative_rate": metrics["race_metrics"]["Black"]["negative_rate"],
                "white_negative_rate": metrics["race_metrics"]["White"]["negative_rate"],
                "signed_gap_b_minus_w": metrics["negative_rate_gap_b_minus_w"],
                "absolute_gap": metrics["negative_rate_gap_abs"],
                "valid_rate": metrics["valid_rate"],
                "identity_leakage_rate": metrics["identity_leakage_rate"],
                "response_entropy": metrics["response_entropy"],
                "unique_responses": metrics["unique_responses"],
            }
        )
    write_csv(args.output_dir / "base_vs_adapter_summary.csv", summary_rows)
    plots = plot_outputs(
        args.output_dir,
        base_metrics,
        adapter_metrics,
        neutral_base_metrics,
        neutral_adapter_metrics,
        bootstrap,
    )
    manifest.update(
        {
            "status": "complete",
            "model_source": source,
            "model_source_kind": source_kind,
            "wall_seconds": time.time() - started,
            "base_metrics": base_metrics,
            "adapter_metrics": adapter_metrics,
            "acceptance": acceptance,
            "plots": plots,
        }
    )
    if torch.cuda.is_available():
        manifest["gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    write_json(args.output_dir / "evaluation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
