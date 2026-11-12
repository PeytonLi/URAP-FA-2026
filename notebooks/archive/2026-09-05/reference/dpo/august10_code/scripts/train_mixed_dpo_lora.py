#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from august3_debiasing.dataset_audit import DatasetPolicy, audit_dataset
from august3_debiasing.run_control import adapter_artifact_hashes
from august3_debiasing.training_core import (
    build_pair_inputs,
    development_preference_gate,
    dpo_loss_value,
    dpo_margin_gradient_coefficient,
    effective_warmup_updates,
    pair_seed,
    sequence_logp,
    set_torch_seed,
)
from august10_research.dataset_gate import assess_dataset
from august10_research.overfit_diagnostic import (
    INFORMATIVE_PROTOCOL_VERSION,
    PROTOCOL_VERSION as DIAGNOSTIC_PROTOCOL_VERSION,
    assess_overfit_dataset,
    diagnostic_preference_gate,
)


DEFAULT_MODEL_ID = "google/gemma-3-12b-it"
DEFAULT_MODEL_REVISION = "96b6f1eccf38110c56df3a15bffe176da04bfd80"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_EXCLUDE_MODULES = r".*vision_tower.*"
SHA256_RE = re.compile(r"\b[0-9a-fA-F]{64}\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_hashes(path: Path) -> set[str]:
    return {
        match.group(0).lower()
        for match in SHA256_RE.finditer(path.read_text(encoding="utf-8"))
    }


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


def runtime_environment(torch: Any) -> dict[str, Any]:
    packages = {}
    for name in (
        "accelerate",
        "datasets",
        "huggingface-hub",
        "numpy",
        "peft",
        "pillow",
        "safetensors",
        "sentencepiece",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available = bool(torch.cuda.is_available())
    return {
        "packages": packages,
        "torch": str(torch.__version__),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cudnn": (
            torch.backends.cudnn.version()
            if hasattr(torch.backends, "cudnn")
            else None
        ),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_gib": (
                    torch.cuda.get_device_properties(index).total_memory / 1024**3
                ),
            }
            for index in range(torch.cuda.device_count() if cuda_available else 0)
        ],
    }


def resolve_training_paths(rows: list[dict[str, str]], root: Path) -> None:
    for row in rows:
        raw = Path(row["image_path"])
        path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"{row['pair_id']}: image path escapes repository root")
        if not path.is_file() or sha256_file(path) != row["image_sha256"].lower():
            raise ValueError(f"{row['pair_id']}: image missing or SHA-256 mismatch")
        row["image_path"] = str(path)


def load_and_verify_balance_report(path: Path, frozen_csv: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "complete" or not report.get("audit", {}).get("valid"):
        raise ValueError("balance report is not complete and valid")
    if report.get("output_csv_sha256") != sha256_file(frozen_csv):
        raise ValueError("frozen CSV hash differs from the balance report")
    return report


def load_and_verify_freeze_manifest(
    path: Path, frozen_csv: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready_to_train":
        raise ValueError("freeze manifest does not authorize training")
    frozen_record = manifest.get("outputs", {}).get("mixed_dpo_pairs_frozen.csv", {})
    if frozen_record.get("sha256") != sha256_file(frozen_csv):
        raise ValueError("frozen CSV hash differs from the freeze manifest")
    gate_path = path.parent / "dataset_gate.json"
    gate_record = manifest.get("outputs", {}).get("dataset_gate.json", {})
    if not gate_path.is_file() or gate_record.get("sha256") != sha256_file(gate_path):
        raise ValueError("dataset gate bytes differ from the freeze manifest")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("pass") or gate.get("decision") != "authorize_short_pilot":
        raise ValueError("saved dataset gate does not authorize training")
    return manifest, gate


def load_and_verify_diagnostic_manifest(
    path: Path, frozen_csv: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ready_for_diagnostic":
        raise ValueError("diagnostic manifest does not authorize training")
    if manifest.get("protocol_version") not in {
        DIAGNOSTIC_PROTOCOL_VERSION,
        INFORMATIVE_PROTOCOL_VERSION,
    }:
        raise ValueError("unsupported diagnostic protocol")
    if manifest.get("non_confirmatory") is not True:
        raise ValueError("diagnostic manifest must be explicitly non-confirmatory")
    if manifest.get("confirmatory_evaluation_opened") is not False:
        raise ValueError("diagnostic manifest opened a protected confirmatory evaluation")
    frozen_record = manifest.get("outputs", {}).get(frozen_csv.name, {})
    if frozen_record.get("sha256") != sha256_file(frozen_csv):
        raise ValueError("diagnostic CSV hash differs from its manifest")
    gate_path = path.parent / "diagnostic_gate.json"
    gate_record = manifest.get("outputs", {}).get(gate_path.name, {})
    if not gate_path.is_file() or gate_record.get("sha256") != sha256_file(gate_path):
        raise ValueError("diagnostic gate bytes differ from the manifest")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("pass") or gate.get("decision") != "authorize_diagnostic_overfit":
        raise ValueError("saved diagnostic gate does not authorize training")
    return manifest, gate


def dataset_policy_from_balance_report(report: dict[str, Any]) -> DatasetPolicy:
    policy = report["policy"]
    target_rows = int(policy["target_rows"])
    iat_fraction = float(policy["target_iat_fraction"])
    max_pairs_per_image = int(policy["max_pairs_per_image"])
    max_fixed = float(policy["max_iat_fixed_substitution_fraction"])
    return DatasetPolicy(
        min_rows=target_rows,
        min_unique_images=math.ceil(target_rows / max_pairs_per_image),
        min_iat_fraction=max(0.0, iat_fraction - 0.001),
        max_iat_fraction=min(1.0, iat_fraction + 0.001),
        min_iat_negative_count_reduction_fraction=1.0 - max_fixed,
        max_iat_fixed_substitution_fraction=max_fixed,
        max_iat_chosen_tuple_fraction=float(policy["max_iat_chosen_tuple_fraction"]),
        iat_tuple_balance_min_rows=int(policy["iat_tuple_balance_min_blocks"]) * 2,
        max_controlled_iat_fraction_of_all_rows=iat_fraction,
        max_profile_block_word_count_ratio=float(policy["max_profile_word_count_ratio"]),
        max_profile_block_grounded_detail_difference=int(
            policy["max_profile_grounded_detail_difference"]
        ),
        max_profile_block_quality_difference=float(policy["max_profile_quality_difference"]),
        max_profile_objective_fraction=float(policy["max_profile_objective_fraction"]),
        max_profile_chosen_reuse=int(policy["max_profile_chosen_reuse"]),
    )


def block_order(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    blocks: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        blocks[row["match_block_id"]].append(row)
    rng = random.Random(seed)
    block_ids = sorted(blocks)
    rng.shuffle(block_ids)
    ordered = []
    for index, block_id in enumerate(block_ids):
        block = sorted(blocks[block_id], key=lambda row: row["fairface_race"])
        if index % 2:
            block.reverse()
        ordered.extend(block)
    return ordered


def load_model(args: argparse.Namespace):
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    local_dir = Path(args.base_model_local_dir).expanduser() if args.base_model_local_dir else None
    if local_dir and local_dir.is_dir():
        source: str | Path = local_dir.resolve()
        source_kind = "local_snapshot"
        if not args.allow_unpinned_model and source.name != args.model_revision:
            raise ValueError("local Gemma snapshot does not match the pinned base revision")
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
    errors = []
    base_model = None
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
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        exclude_modules=LORA_EXCLUDE_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    vision_trainable = [name for name in trainable if "vision_tower" in name]
    if vision_trainable:
        raise RuntimeError(f"vision tower unexpectedly trainable: {vision_trainable[:5]}")
    return torch, processor, model, str(source), source_kind, trainable


def load_reference_cache(path: Path, expected_header: dict[str, Any]) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("header") != expected_header:
        raise ValueError("reference-score cache header does not match this run")
    return {
        pair_id: {"chosen": float(value["chosen"]), "rejected": float(value["rejected"])}
        for pair_id, value in payload.get("scores", {}).items()
    }


def save_reference_cache(
    path: Path,
    header: dict[str, Any],
    scores: dict[str, dict[str, float]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, {"header": header, "scores": scores})
    temporary.replace(path)


def precompute_references(
    model: Any,
    processor: Any,
    rows: list[dict[str, str]],
    cache_path: Path,
    header: dict[str, Any],
    args: argparse.Namespace,
    torch: Any,
) -> dict[str, dict[str, float]]:
    scores = load_reference_cache(cache_path, header)
    model.eval()
    context = model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
    with torch.inference_mode(), context:
        for position, row in enumerate(rows, 1):
            pair_id = row["pair_id"]
            if pair_id in scores:
                continue
            chosen, chosen_len, rejected, rejected_len = build_pair_inputs(processor, row, torch)
            scores[pair_id] = {
                "chosen": float(
                    sequence_logp(
                        model,
                        chosen,
                        chosen_len,
                        max_prompt_tokens=args.max_prompt_tokens,
                        max_response_tokens=args.max_response_tokens,
                        torch=torch,
                        reduction=args.sequence_logp_reduction,
                    ).item()
                ),
                "rejected": float(
                    sequence_logp(
                        model,
                        rejected,
                        rejected_len,
                        max_prompt_tokens=args.max_prompt_tokens,
                        max_response_tokens=args.max_response_tokens,
                        torch=torch,
                        reduction=args.sequence_logp_reduction,
                    ).item()
                ),
            }
            if position % 10 == 0:
                save_reference_cache(cache_path, header, scores)
                print(f"reference scores {position}/{len(rows)}", flush=True)
    save_reference_cache(cache_path, header, scores)
    return scores


def model_smoke_test_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select both task types, ending with the longest profile activation."""

    selected = []
    for task_type in ("iat_3word", "grounded_profile"):
        candidates = [row for row in rows if row["task_type"] == task_type]
        if not candidates:
            raise ValueError(f"model smoke test requires task {task_type}")
        selected.append(
            max(
                candidates,
                key=lambda row: (
                    len(row["chosen"]) + len(row["rejected"]),
                    row["pair_id"],
                ),
            )
        )
    return selected


def run_model_smoke_test(
    model: Any,
    processor: Any,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    torch: Any,
) -> list[dict[str, Any]]:
    """Exercise reference, policy, replay-gradient, and optimizer-state paths."""

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    records = []
    for position, row in enumerate(model_smoke_test_rows(rows), 1):
        chosen, chosen_len, rejected, rejected_len = build_pair_inputs(
            processor, row, torch
        )
        model.eval()
        adapter_context = (
            model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext()
        )
        with torch.inference_mode(), adapter_context:
            reference_chosen = float(
                sequence_logp(
                    model,
                    chosen,
                    chosen_len,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                    torch=torch,
                    reduction=args.sequence_logp_reduction,
                ).item()
            )
            reference_rejected = float(
                sequence_logp(
                    model,
                    rejected,
                    rejected_len,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                    torch=torch,
                    reduction=args.sequence_logp_reduction,
                ).item()
            )
        model.train()
        chosen_seed = pair_seed(args.seed, row["pair_id"], "chosen")
        rejected_seed = pair_seed(args.seed, row["pair_id"], "rejected")
        set_torch_seed(torch, chosen_seed)
        with torch.no_grad():
            policy_chosen = float(
                sequence_logp(
                    model,
                    chosen,
                    chosen_len,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                    torch=torch,
                    reduction=args.sequence_logp_reduction,
                ).item()
            )
        set_torch_seed(torch, rejected_seed)
        with torch.no_grad():
            policy_rejected = float(
                sequence_logp(
                    model,
                    rejected,
                    rejected_len,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                    torch=torch,
                    reduction=args.sequence_logp_reduction,
                ).item()
            )
        coefficient = dpo_margin_gradient_coefficient(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            args.dpo_beta,
        )

        set_torch_seed(torch, chosen_seed)
        replay_chosen = sequence_logp(
            model,
            chosen,
            chosen_len,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            torch=torch,
            reduction=args.sequence_logp_reduction,
        )
        if abs(float(replay_chosen.detach().item()) - policy_chosen) > args.replay_tolerance:
            raise RuntimeError("model smoke-test chosen replay exceeded tolerance")
        (replay_chosen * coefficient).backward()
        del replay_chosen

        set_torch_seed(torch, rejected_seed)
        replay_rejected = sequence_logp(
            model,
            rejected,
            rejected_len,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            torch=torch,
            reduction=args.sequence_logp_reduction,
        )
        if abs(float(replay_rejected.detach().item()) - policy_rejected) > args.replay_tolerance:
            raise RuntimeError("model smoke-test rejected replay exceeded tolerance")
        (replay_rejected * -coefficient).backward()
        del replay_rejected
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        record = {
            "position": position,
            "pair_id": row["pair_id"],
            "task_type": row["task_type"],
            "chosen_characters": len(row["chosen"]),
            "rejected_characters": len(row["rejected"]),
            "dpo_loss": dpo_loss_value(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                args.dpo_beta,
            ),
        }
        if torch.cuda.is_available():
            record.update(
                {
                    "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                }
            )
        records.append(record)
        print(
            f"model smoke test {position}/2: {row['task_type']} {row['pair_id']}",
            flush=True,
        )
    return records


def evaluate_preference_rows(
    model: Any,
    processor: Any,
    rows: list[dict[str, str]],
    references: dict[str, dict[str, float]],
    args: argparse.Namespace,
    torch: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model.eval()
    records = []
    with torch.inference_mode():
        for row in rows[: args.dev_max_pairs]:
            chosen, chosen_len, rejected, rejected_len = build_pair_inputs(processor, row, torch)
            policy_chosen = float(
                sequence_logp(
                    model,
                    chosen,
                    chosen_len,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                    torch=torch,
                    reduction=args.sequence_logp_reduction,
                ).item()
            )
            policy_rejected = float(
                sequence_logp(
                    model,
                    rejected,
                    rejected_len,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                    torch=torch,
                    reduction=args.sequence_logp_reduction,
                ).item()
            )
            reference = references[row["pair_id"]]
            margin_delta = (policy_chosen - policy_rejected) - (
                reference["chosen"] - reference["rejected"]
            )
            records.append(
                {
                    "pair_id": row["pair_id"],
                    "match_block_id": row["match_block_id"],
                    "task_type": row["task_type"],
                    "fairface_race": row["fairface_race"],
                    "objective_type": row["objective_type"],
                    "policy_margin": policy_chosen - policy_rejected,
                    "reference_margin": reference["chosen"] - reference["rejected"],
                    "margin_delta": margin_delta,
                    "preferred": margin_delta > 0,
                    "dpo_loss": dpo_loss_value(
                        policy_chosen,
                        policy_rejected,
                        reference["chosen"],
                        reference["rejected"],
                        args.dpo_beta,
                    ),
                }
            )
    by_group = defaultdict(list)
    for record in records:
        by_group[(record["task_type"], record["fairface_race"])].append(record)
    summary = {
        "rows": len(records),
        "preference_accuracy": (
            sum(record["preferred"] for record in records) / len(records) if records else 0.0
        ),
        "mean_dpo_loss": (
            sum(record["dpo_loss"] for record in records) / len(records) if records else 0.0
        ),
        "mean_margin_delta": (
            sum(record["margin_delta"] for record in records) / len(records)
            if records
            else 0.0
        ),
        "groups": {
            f"{task}/{race}": {
                "rows": len(values),
                "preference_accuracy": sum(item["preferred"] for item in values) / len(values),
                "mean_margin_delta": sum(item["margin_delta"] for item in values) / len(values),
            }
            for (task, race), values in sorted(by_group.items())
        },
    }
    return records, summary


def plot_training_diagnostics(
    output_dir: Path,
    loss_history: list[dict[str, Any]],
    development_records: list[dict[str, Any]],
    development_summary: dict[str, Any],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    artifacts = []
    colors = {"Black": "#0B6E75", "White": "#5B6770"}
    fig, axis = plt.subplots(figsize=(10, 4.5), dpi=180)
    for race in ("Black", "White"):
        rows = [row for row in loss_history if row["fairface_race"] == race]
        axis.scatter(
            [row["gradient_step"] for row in rows],
            [row["loss"] for row in rows],
            s=18,
            alpha=0.75,
            label=race,
            color=colors[race],
        )
    axis.set_xlabel("Gradient step")
    axis.set_ylabel("DPO loss")
    axis.set_title("Training loss by race (diagnostic, not an outcome metric)")
    axis.legend()
    fig.tight_layout()
    path = output_dir / "01_training_loss_by_race.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))

    group_names = sorted(development_summary["groups"])
    x = np.arange(len(group_names))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), dpi=180)
    axes[0].bar(
        x,
        [development_summary["groups"][name]["preference_accuracy"] for name in group_names],
        color=[colors["Black"] if name.endswith("/Black") else colors["White"] for name in group_names],
    )
    axes[0].axhline(0.5, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(x, [name.replace("grounded_profile", "profile") for name in group_names], rotation=30, ha="right")
    axes[0].set_ylabel("Preference accuracy")
    axes[0].set_title("Held-out development preference accuracy")
    axes[1].bar(
        x,
        [development_summary["groups"][name]["mean_margin_delta"] for name in group_names],
        color=[colors["Black"] if name.endswith("/Black") else colors["White"] for name in group_names],
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, [name.replace("grounded_profile", "profile") for name in group_names], rotation=30, ha="right")
    axes[1].set_ylabel("Policy-reference margin change")
    axes[1].set_title("Held-out development DPO margin")
    fig.tight_layout()
    path = output_dir / "02_development_preference_by_group.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    for race in ("Black", "White"):
        rows = [row for row in development_records if row["fairface_race"] == race]
        axes[0].scatter(
            [row["reference_margin"] for row in rows],
            [row["policy_margin"] for row in rows],
            label=race,
            color=colors[race],
            alpha=0.8,
        )
    limits = axes[0].axis()
    low = min(limits[0], limits[2])
    high = max(limits[1], limits[3])
    axes[0].plot([low, high], [low, high], color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Base reference margin")
    axes[0].set_ylabel("LoRA policy margin")
    axes[0].set_title("Development chosen-vs-rejected margins")
    axes[0].legend()
    for race in ("Black", "White"):
        rows = [row for row in loss_history if row["fairface_race"] == race]
        axes[1].scatter(
            [row["gradient_step"] for row in rows],
            [abs(row["gradient_coefficient"]) for row in rows],
            label=race,
            color=colors[race],
            alpha=0.75,
            s=18,
        )
    axes[1].set_xlabel("Gradient step")
    axes[1].set_ylabel("Absolute replay coefficient")
    axes[1].set_title("Gradient signal magnitude")
    axes[1].legend()
    fig.tight_layout()
    path = output_dir / "03_margin_and_gradient_diagnostics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    artifacts.append(str(path))
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fresh Gemma 3 LoRA on the frozen August 10 mixed DPO dataset."
    )
    parser.add_argument("--frozen-csv", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--freeze-manifest", type=Path)
    source.add_argument("--balance-report", type=Path)
    source.add_argument("--diagnostic-manifest", type=Path)
    parser.add_argument(
        "--gate-config", type=Path, default=ROOT_DIR / "config" / "pilot_gate.json"
    )
    parser.add_argument(
        "--forbidden-eval-hashes",
        type=Path,
        default=ROOT_DIR / "reference" / "frozen_base" / "frozen_input_manifest.json",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument(
        "--base-model-local-dir",
        default=os.environ.get("BASE_MODEL_LOCAL_DIR", os.environ.get("GEMMA_LOCAL_DIR", "")),
    )
    parser.add_argument("--allow-hf-download", action="store_true")
    parser.add_argument("--allow-unpinned-model", action="store_true")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--dpo-beta", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=0,
        help="Explicit warmup updates; zero selects 10%% of planned updates capped at 10.",
    )
    parser.add_argument("--max-gradient-steps", type=int, default=0)
    parser.add_argument("--save-every-updates", type=int, default=10)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-response-tokens", type=int, default=384)
    parser.add_argument(
        "--sequence-logp-reduction",
        choices=("sum", "mean"),
        default="sum",
        help="Reduce response-token log probabilities by sum (standard) or mean.",
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--seed", type=int, default=8032026)
    parser.add_argument("--dev-max-pairs", type=int, default=200)
    parser.add_argument("--replay-tolerance", type=float, default=0.05)
    parser.add_argument("--model-smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.gradient_accumulation <= 0:
        raise ValueError("--gradient-accumulation must be positive")
    if args.max_gradient_steps < 0 or args.max_gradient_steps % 2:
        raise ValueError("--max-gradient-steps must be zero or a positive even number")
    if args.warmup_updates < 0:
        raise ValueError("--warmup-updates must be nonnegative")
    if args.dpo_beta <= 0 or args.learning_rate <= 0:
        raise ValueError("DPO beta and learning rate must be positive")
    if args.model_smoke_test and not args.dry_run:
        raise ValueError("--model-smoke-test must be combined with --dry-run")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_csv = args.frozen_csv.resolve()
    forbidden_eval_hashes = read_hashes(args.forbidden_eval_hashes.resolve())
    rows = read_csv(frozen_csv)
    resolve_training_paths(rows, args.repo_root.resolve())
    training_hashes = {row["image_sha256"].lower() for row in rows}
    forbidden_overlap = sorted(training_hashes & forbidden_eval_hashes)
    if forbidden_overlap:
        raise ValueError(
            "training data overlaps frozen evaluation images: "
            + json.dumps(forbidden_overlap[:5])
        )
    source_manifest: dict[str, Any]
    source_manifest_path: Path
    diagnostic_overfit = args.diagnostic_manifest is not None
    if args.diagnostic_manifest is not None:
        source_manifest_path = args.diagnostic_manifest.resolve()
        source_manifest, saved_gate = load_and_verify_diagnostic_manifest(
            source_manifest_path, frozen_csv
        )
        current_gate = assess_overfit_dataset(rows, forbidden_eval_hashes)
        if current_gate != saved_gate:
            raise ValueError("revalidated diagnostic gate differs from the frozen gate")
        dataset_validation = current_gate
        source_protocol = source_manifest.get("protocol_version")
    elif args.freeze_manifest is not None:
        source_manifest_path = args.freeze_manifest.resolve()
        source_manifest, saved_gate = load_and_verify_freeze_manifest(
            source_manifest_path, frozen_csv
        )
        gate_config = json.loads(args.gate_config.read_text(encoding="utf-8"))
        current_gate = assess_dataset(rows, gate_config["dataset"])
        if not current_gate["pass"]:
            raise ValueError(
                "frozen dataset fails the current August 10 gate: "
                + json.dumps(current_gate)
            )
        if current_gate != saved_gate:
            raise ValueError("revalidated dataset gate differs from the frozen gate")
        dataset_validation = current_gate
        source_protocol = source_manifest.get("protocol_version")
    else:
        source_manifest_path = args.balance_report.resolve()
        source_manifest = load_and_verify_balance_report(source_manifest_path, frozen_csv)
        audit = audit_dataset(
            rows,
            dataset_policy=dataset_policy_from_balance_report(source_manifest),
            forbidden_eval_hashes=forbidden_eval_hashes,
        )
        if not audit.valid:
            raise ValueError(
                "frozen dataset fails current audit: " + json.dumps(audit.as_dict())
            )
        dataset_validation = audit.as_dict()
        source_protocol = source_manifest.get("protocol_version")
    train_rows = [row for row in rows if row["split"] == "train"]
    development_rows = [row for row in rows if row["split"] == "development"]
    if not train_rows or not development_rows:
        raise ValueError("both train and development splits must be nonempty")
    ordered_train = block_order(train_rows, args.seed)
    if args.max_gradient_steps > 0:
        ordered_train = ordered_train[: args.max_gradient_steps]
    planned_optimizer_updates = math.ceil(
        len(ordered_train) / args.gradient_accumulation
    )
    warmup_updates = effective_warmup_updates(
        planned_optimizer_updates, args.warmup_updates
    )

    manifest_path = args.output_dir / "training_manifest.json"
    manifest = {
        "status": (
            "validating_model_runtime"
            if args.model_smoke_test
            else "validated_dry_run"
            if args.dry_run
            else "preparing_references"
        ),
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "frozen_csv": str(frozen_csv),
        "frozen_csv_sha256": sha256_file(frozen_csv),
        "dataset_source_manifest": str(source_manifest_path),
        "dataset_source_manifest_sha256": sha256_file(source_manifest_path),
        "dataset_source_protocol": source_protocol,
        "diagnostic_overfit": diagnostic_overfit,
        "confirmatory_evaluation_opened": False,
        "forbidden_eval_hashes": str(args.forbidden_eval_hashes.resolve()),
        "forbidden_eval_hashes_sha256": sha256_file(args.forbidden_eval_hashes),
        "forbidden_eval_hash_count": len(forbidden_eval_hashes),
        "dataset_validation": dataset_validation,
        "train_rows": len(train_rows),
        "development_rows": len(development_rows),
        "effective_gradient_steps": len(ordered_train),
        "planned_optimizer_updates": planned_optimizer_updates,
        "fresh_lora": True,
        "prior_adapter_loaded": False,
        "gradient_method": "exact_coefficient_replay_one_response_graph_at_a_time",
        "logits_scope": "response_prediction_positions_only_via_logits_to_keep",
        "attention_implementation": args.attn_implementation,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "hyperparameters": {
            "dpo_beta": args.dpo_beta,
            "learning_rate": args.learning_rate,
            "gradient_accumulation": args.gradient_accumulation,
            "warmup_updates_requested": args.warmup_updates,
            "warmup_updates_effective": warmup_updates,
            "max_response_tokens": args.max_response_tokens,
            "sequence_logp_reduction": args.sequence_logp_reduction,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
    }
    write_json(manifest_path, manifest)
    if args.dry_run and not args.model_smoke_test:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    run_started = time.time()
    torch, processor, model, model_source, source_kind, trainable = load_model(args)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = True
    manifest.update(
        {
            "model_source": model_source,
            "model_source_kind": source_kind,
            "runtime_environment": runtime_environment(torch),
            "trainable_parameter_tensors": len(trainable),
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
            "trainable_name_sample": trainable[:20],
        }
    )
    write_json(manifest_path, manifest)

    if args.model_smoke_test:
        smoke_records = run_model_smoke_test(
            model,
            processor,
            train_rows + development_rows,
            args,
            torch,
        )
        manifest.update(
            {
                "status": "validated_model_smoke_test",
                "model_smoke_test": smoke_records,
                "model_smoke_test_covered_tasks": sorted(
                    {record["task_type"] for record in smoke_records}
                ),
            }
        )
        if torch.cuda.is_available():
            manifest["model_smoke_test_gpu_memory"] = {
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "total_gib": (
                    torch.cuda.get_device_properties(0).total_memory / 1024**3
                ),
            }
        write_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    reference_header = {
        "frozen_csv_sha256": sha256_file(frozen_csv),
        "model_revision": args.model_revision,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_response_tokens": args.max_response_tokens,
        "sequence_logp_reduction": args.sequence_logp_reduction,
    }
    all_rows = train_rows + development_rows
    references = precompute_references(
        model,
        processor,
        all_rows,
        args.output_dir / "reference_scores.json",
        reference_header,
        args,
        torch,
    )
    manifest["status"] = "training"
    manifest["reference_scores"] = len(references)
    write_json(manifest_path, manifest)

    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    loss_history = []
    optimizer_update = 0
    training_started = time.time()
    checkpoints = []
    checkpoint_artifacts: list[dict[str, Any]] = []
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def apply_update(gradient_step: int, last_loss: float) -> None:
        nonlocal optimizer_update
        multiplier = min(1.0, (optimizer_update + 1) / warmup_updates)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_update += 1
        if args.save_every_updates and optimizer_update % args.save_every_updates == 0:
            target = checkpoint_dir / f"step_{optimizer_update:04d}"
            model.save_pretrained(target)
            checkpoints.append(target.name)
            checkpoint_artifacts.append(
                {
                    "optimizer_update": optimizer_update,
                    "path": str(target),
                    "hashes": adapter_artifact_hashes(target),
                }
            )
        progress = {
            "gradient_steps_completed": gradient_step,
            "gradient_steps_total": len(ordered_train),
            "optimizer_updates_completed": optimizer_update,
            "optimizer_updates_total": math.ceil(
                len(ordered_train) / args.gradient_accumulation
            ),
            "last_loss": last_loss,
            "last_update_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if torch.cuda.is_available():
            progress.update(
                {
                    "gpu_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                }
            )
        manifest["progress"] = progress
        manifest["checkpoint_artifacts"] = checkpoint_artifacts
        write_json(manifest_path, manifest)

    for step, row in enumerate(ordered_train, 1):
        chosen, chosen_len, rejected, rejected_len = build_pair_inputs(processor, row, torch)
        reference = references[row["pair_id"]]
        chosen_seed = pair_seed(args.seed, row["pair_id"], "chosen")
        rejected_seed = pair_seed(args.seed, row["pair_id"], "rejected")

        set_torch_seed(torch, chosen_seed)
        with torch.no_grad():
            policy_chosen = float(
                sequence_logp(
                    model,
                    chosen,
                    chosen_len,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                    torch=torch,
                    reduction=args.sequence_logp_reduction,
                ).item()
            )
        set_torch_seed(torch, rejected_seed)
        with torch.no_grad():
            policy_rejected = float(
                sequence_logp(
                    model,
                    rejected,
                    rejected_len,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_response_tokens=args.max_response_tokens,
                    torch=torch,
                    reduction=args.sequence_logp_reduction,
                ).item()
            )
        coefficient = dpo_margin_gradient_coefficient(
            policy_chosen,
            policy_rejected,
            reference["chosen"],
            reference["rejected"],
            args.dpo_beta,
        )
        loss_value = dpo_loss_value(
            policy_chosen,
            policy_rejected,
            reference["chosen"],
            reference["rejected"],
            args.dpo_beta,
        )

        set_torch_seed(torch, chosen_seed)
        replay_chosen = sequence_logp(
            model,
            chosen,
            chosen_len,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            torch=torch,
            reduction=args.sequence_logp_reduction,
        )
        if abs(float(replay_chosen.detach().item()) - policy_chosen) > args.replay_tolerance:
            raise RuntimeError("chosen replay log-probability exceeded tolerance")
        accumulation_group_start = ((step - 1) // args.gradient_accumulation) * args.gradient_accumulation
        accumulation_divisor = min(
            args.gradient_accumulation,
            len(ordered_train) - accumulation_group_start,
        )
        (replay_chosen * (coefficient / accumulation_divisor)).backward()
        del replay_chosen

        set_torch_seed(torch, rejected_seed)
        replay_rejected = sequence_logp(
            model,
            rejected,
            rejected_len,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            torch=torch,
            reduction=args.sequence_logp_reduction,
        )
        if abs(float(replay_rejected.detach().item()) - policy_rejected) > args.replay_tolerance:
            raise RuntimeError("rejected replay log-probability exceeded tolerance")
        (replay_rejected * (-coefficient / accumulation_divisor)).backward()
        del replay_rejected
        loss_history.append(
            {
                "gradient_step": step,
                "pair_id": row["pair_id"],
                "match_block_id": row["match_block_id"],
                "task_type": row["task_type"],
                "fairface_race": row["fairface_race"],
                "loss": loss_value,
                "policy_margin": policy_chosen - policy_rejected,
                "reference_margin": reference["chosen"] - reference["rejected"],
                "gradient_coefficient": coefficient,
            }
        )
        if step % args.gradient_accumulation == 0:
            apply_update(step, loss_value)
            print(
                f"optimizer update {optimizer_update}; gradient step {step}/{len(ordered_train)}; loss={loss_value:.6f}",
                flush=True,
            )
    if len(ordered_train) % args.gradient_accumulation:
        apply_update(len(ordered_train), loss_history[-1]["loss"])

    final_dir = checkpoint_dir / "final"
    model.save_pretrained(final_dir)
    checkpoints.append("final")
    checkpoint_artifacts.append(
        {
            "optimizer_update": optimizer_update,
            "path": str(final_dir),
            "hashes": adapter_artifact_hashes(final_dir),
        }
    )
    write_csv(args.output_dir / "training_loss_history.csv", loss_history)

    development_records, development_summary = evaluate_preference_rows(
        model,
        processor,
        development_rows,
        references,
        args,
        torch,
    )
    write_csv(args.output_dir / "development_preference_metrics.csv", development_records)
    write_json(args.output_dir / "development_preference_summary.json", development_summary)
    development_gate = (
        diagnostic_preference_gate(development_summary)
        if diagnostic_overfit
        else development_preference_gate(development_summary)
    )
    write_json(args.output_dir / "development_preference_gate.json", development_gate)
    diagnostic_plots = plot_training_diagnostics(
        args.output_dir,
        loss_history,
        development_records,
        development_summary,
    )

    manifest.update(
        {
            "status": "complete",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_seconds": time.time() - run_started,
            "training_wall_seconds": time.time() - training_started,
            "gradient_steps": len(loss_history),
            "optimizer_updates": optimizer_update,
            "checkpoints": checkpoints,
            "checkpoint_artifacts": checkpoint_artifacts,
            "final_adapter": str(final_dir),
            "final_adapter_hashes": adapter_artifact_hashes(final_dir),
            "development_preference_summary": development_summary,
            "development_preference_gate": development_gate,
            "diagnostic_plots": diagnostic_plots,
        }
    )
    if torch.cuda.is_available():
        manifest["gpu_memory"] = {
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
            "total_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
