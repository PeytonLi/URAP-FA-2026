"""Native single-GPU image LoRA SFT/ORPO candidate; no provider provisioning.

Full approved coverage, or the separately frozen authorized pilot, is mandatory.
CPU fixture checks are not real study runs or evidence of full-size GPU feasibility.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from importlib.metadata import version

import curated
import hub_backup
import objectives
import study
from curated_vision_check import continuation_inputs
from qwen_vision_check import MODELS as QWEN_MODELS, checked_inputs
from vision import GEMMA_HF, GEMMA_REVISION

MODELS = {"gemma4": {"model": GEMMA_HF, "revision": GEMMA_REVISION, "enable_thinking": False},
          "gemma3_12b": {"model": "google/gemma-3-12b-it",
                         "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80",
                         "enable_thinking": None},
          **QWEN_MODELS}


def response_logprobs(model, inputs):
    """Next-token scores only at supervised positions; retain native image fields."""
    import torch

    labels, ids = inputs["labels"], inputs["input_ids"]
    if labels.shape != ids.shape or labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("Native microbatch must contain exactly one masked continuation")
    positions = (labels[0] != -100).nonzero().flatten()
    if (not positions.numel() or positions[0] < 1
            or not torch.equal(positions, torch.arange(int(positions[0]), labels.shape[1], device=labels.device))
            or not torch.equal(labels[0, positions], ids[0, positions])):
        raise ValueError("Expected a contiguous response-only suffix with unchanged token IDs")
    if "pixel_values" not in inputs or not inputs["pixel_values"].numel():
        raise ValueError("Text-only training is forbidden")
    device, dtype = next(model.parameters()).device, model.dtype
    forwarded = {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                 for k, v in inputs.items() if k != "labels"}
    # Compute vocabulary logits only at response predictors, not the much longer
    # prompt/image prefix. Never truncate or remap provider score coordinates.
    logits = model(**forwarded, use_cache=False, logits_to_keep=(positions - 1).to(device)).logits
    if logits.ndim != 3 or logits.shape[:2] != (1, positions.numel()):
        raise ValueError("Native response predictor coordinates do not match labels")
    tokens = labels[0, positions].to(device)
    scores = logits[0].float().log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite response log probabilities")
    return scores


def attach_lora(model):
    """Language attention only; no vision, router, expert or embedding updates."""
    import torch
    from peft import LoraConfig, get_peft_model

    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and ".language_model.layers." in name
               and any(part in name for part in (".self_attn.", ".linear_attn."))]
    if not targets:
        raise ValueError("No supported native language-attention LoRA modules found")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0,
                                           revision=getattr(model.config, "_commit_hash", None),
                                           target_modules=targets, bias="none"))
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Only new LoRA parameters may be optimized")
    model.config.text_config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return model, {"target_modules": targets, "trainable_parameters": sum(p.numel() for p in trainable.values()),
                   "base_parameters_frozen": True, "vision_and_experts_frozen": True,
                   "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0}


def batch_update(model, optimizer, examples, method, coefficient=0.1, example_weights=None):
    """Stream microbatches; fixed dataset weights, not batch-normalized weights."""
    import torch

    if method not in ("sft", "orpo") or not examples:
        raise ValueError("A nonempty SFT or ORPO batch is required")
    weights = [1.0] * len(examples) if example_weights is None else example_weights
    if len(weights) != len(examples) or any(
            isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Each image/pair requires one finite positive loss weight")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = []
    weighted_loss = 0.0
    for (chosen, rejected), weight in zip(examples, weights):
        scores = [response_logprobs(model, chosen)]
        if method == "orpo":
            if rejected is None:
                raise ValueError("ORPO requires the image-conditioned rejected branch")
            if not torch.equal(chosen["pixel_values"], rejected["pixel_values"]):
                raise ValueError("Both preference branches must use the same image")
            scores.append(response_logprobs(model, rejected))
        masks = [torch.ones_like(score) for score in scores]
        loss, values = (objectives.sft_loss(masks, scores) if method == "sft" else
                        objectives.orpo_loss(masks, scores, coefficient=coefficient))
        (loss * weight / len(examples)).backward()
        weighted_loss += float(loss.detach()) * weight / len(examples)
        metrics.append(values)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable):
        optimizer.zero_grad(set_to_none=True)
        raise ValueError("Nonfinite adapter gradient; optimizer update refused")
    norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0, error_if_nonfinite=True)
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError("Missing finite, nonzero adapter gradients")
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in trainable):
        raise ValueError("Nonfinite adapter after optimizer update")
    return {k: sum(m[k] for m in metrics) / (1 if k.startswith("batch_") else len(metrics)) for k in metrics[0]} | {
        "logical_batch_size": len(examples), "gradient_norm_before_clip": float(norm),
        "weighted_objective": weighted_loss, "sum_example_weights": sum(weights)}


def image_control(model, inputs):
    """Forward-only control: replace pixels, leaving all text/grid fields fixed."""
    import torch

    model.eval()
    with torch.no_grad():
        scores = response_logprobs(model, inputs)
        control = {**inputs, "pixel_values": torch.zeros_like(inputs["pixel_values"])}
        changed = response_logprobs(model, control)
    difference = float((scores - changed).abs().max())
    if not math.isfinite(difference) or difference <= 1e-7:
        raise ValueError("Pixel-only control did not demonstrate image conditioning")
    return {"control": "zero_pixels_identical_text_and_native_grid", "maximum_response_logprob_change": difference}


def encode_row(processor, model_name, row, asset_root, method):
    image = (asset_root / row["image_path"]).resolve()
    if not image.is_relative_to(asset_root) or not study.image_matches_manifest(row, image):
        raise ValueError("Native training image hash mismatch")
    def render(branch):
        if model_name == "gemma4":
            return continuation_inputs(processor, image, row[branch])[0]
        if model_name == "gemma3_12b":
            from gemma3_vision_check import checked_inputs as g3_inputs
            return g3_inputs(processor, image, row[branch])[0]
        return checked_inputs(processor, image, row[branch],
                              enable_thinking=MODELS[model_name]["enable_thinking"])[0]
    return render("chosen"), render("rejected") if method == "orpo" else None


def approved_cohort(cohort="full"):
    """No implicit filtering; the user-authorized pilot is an explicit option."""
    config = study.dataset_config()
    if not config:
        raise ValueError("Active curated dataset is required")
    if cohort == "resolved-balanced":
        import pilot
        config, rows = pilot.training_rows()
    elif cohort == "full":
        rows = curated.training_rows(config)
    else:
        raise ValueError("Unknown training cohort")
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] == "development"]
    if len(train) != config["split_counts"]["train"] or len(dev) != config["split_counts"]["development"]:
        raise ValueError("The full frozen training/development cohort is required")
    asset_root = (study.REPO / config["asset_root"]).resolve()
    for row in rows:
        image = (asset_root / row["image_path"]).resolve()
        if not image.is_relative_to(asset_root) or not study.image_matches_manifest(row, image):
            raise ValueError("Approved cohort image hash mismatch")
    return config, train, dev, asset_root


def save_checkpoint(model, processor, optimizer, path):
    import torch
    if path.exists():
        raise ValueError("Existing native checkpoint must not be overwritten")
    model.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    return {"path": str(path), "files": {p.name: study.sha(p) for p in path.iterdir() if p.is_file()}}


def train(model_name, method, max_seconds, cohort="full"):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("An explicit positive run-time ceiling is required")
    # Check the complete approved dataset before downloading large weight shards,
    # allocating GPU memory or creating a study run directory.
    config, train_rows, dev_rows, asset_root = approved_cohort(cohort)
    hub_api, hub_account = hub_backup.connect()  # Mandatory before loading study weights.
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Native BF16 training needs a supported CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise ValueError("This candidate supports one visible GPU, not unverified model dispatch or distributed training")
    settings = MODELS[model_name]
    set_seed(0)
    started = time.monotonic()
    run = study.ROOT / "data/private/native_runs" / (model_name + "_" + method + "_" +
          study.now().replace(":", "").replace("-", "").replace("+", "_"))
    run.mkdir(parents=True, exist_ok=False)
    destination = hub_backup.create(run, hub_api, hub_account)
    processor = AutoProcessor.from_pretrained(settings["model"], revision=settings["revision"])
    base = AutoModelForImageTextToText.from_pretrained(
        settings["model"], revision=settings["revision"], dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    model, adapter = attach_lora(base)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0)
    control = image_control(model, encode_row(processor, model_name, train_rows[0], asset_root, method)[0])
    receipt = {"started_at": study.now(), "status": "running", "model": settings["model"],
               "weight_and_processor_revision": settings["revision"], "method": method, "seed": 0,
               "dataset_id": config["dataset_id"], "manifest_sha256": config["manifest_sha256"],
               "targets_sha256": study.sha(study.REPO / config["targets_path"]),
               "target_approval_kinds": dict(Counter(r["reviewer_kind"] for r in train_rows + dev_rows)),
               "targets_are_ground_truth": False,
               "cohort": cohort, "full_cohort_study": cohort == "full",
               "pilot_class_weights": config.get("pilot_class_weights"),
               "pilot_policy_sha256": config.get("pilot_policy_sha256"),
               "loss_weighting": "fixed_dataset_weights_then_logical_batch_mean" if cohort != "full" else "equal_examples",
               "training_images": len(train_rows), "development_images": len(dev_rows),
               "test_images_used": 0, "profile_targets_used": 0, "optimizer_updates": 0,
               "completed_epochs": 0, "requested_epochs": 3, "logical_batch_size": 4,
               "planned_optimizer_updates": 3 * math.ceil(len(train_rows) / 4),
               "max_seconds": max_seconds, "runtime_ceiling_is_not_provider_billing_control": True,
               "optimizer": "AdamW", "learning_rate": 1e-5, "weight_decay": 0,
               "orpo_coefficient": 0.1 if method == "orpo" else None,
               "loss_reduction": objectives.SFT_REDUCTION if method == "sft" else objectives.ORPO_REDUCTION,
               "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
               "gpu": torch.cuda.get_device_name(0), "adapter": adapter, "image_control": control,
               "history": [], "development": [], "checkpoints": [], "matched_base_benchmark_required": True,
               "hugging_face_repo": destination["repo_id"], "hub_backup_required": True}
    study.save_json(run / "run.json", receipt)
    try:
        for epoch in range(3):
            order = random.Random(epoch).sample(train_rows, len(train_rows))
            for offset in range(0, len(order), 4):
                if time.monotonic() - started >= max_seconds:
                    raise TimeoutError("Run-time ceiling reached; this is not a completed study")
                batch = order[offset:offset + 4]
                examples = [encode_row(processor, model_name, row, asset_root, method) for row in batch]
                metrics = batch_update(model, optimizer, examples, method,
                                       example_weights=[r.get("training_weight", 1.0) for r in batch])
                receipt["optimizer_updates"] += 1
                receipt["history"].append({"epoch": epoch + 1, "update": receipt["optimizer_updates"], **metrics})
                study.save_json(run / "run.json", receipt)
                if receipt["optimizer_updates"] % 10 == 0:
                    print(json.dumps(receipt["history"][-1]), flush=True)
            model.eval()
            losses, preferred, groups = [], [], {}
            with torch.no_grad():
                for row in dev_rows:
                    if time.monotonic() - started >= max_seconds:
                        raise TimeoutError("Run-time ceiling reached during development evaluation")
                    chosen, rejected = encode_row(processor, model_name, row, asset_root, "orpo")
                    c, r = response_logprobs(model, chosen).mean(), response_logprobs(model, rejected).mean()
                    losses.append(float(-c))
                    preferred.append(float(c > r))
                    category = row.get("balance_class", "unclassified")
                    group = groups.setdefault(category, {"loss_sum": 0.0, "preferred_sum": 0.0, "images": 0})
                    group["loss_sum"] += losses[-1]
                    group["preferred_sum"] += preferred[-1]
                    group["images"] += 1
            strata = {k: {"response_loss": g["loss_sum"] / g["images"],
                          "chosen_preference_accuracy": g["preferred_sum"] / g["images"],
                          "images": g["images"]} for k, g in groups.items()}
            receipt["development"].append({"epoch": epoch + 1, "response_loss": sum(losses) / len(losses),
                                            "chosen_preference_accuracy": sum(preferred) / len(preferred),
                                            "images": len(dev_rows), "optimizer_updates_on_development": 0,
                                            "by_target_category": strata,
                                            "macro_category_response_loss": sum(g["response_loss"] for g in strata.values()) / len(strata),
                                            "macro_category_preference_accuracy": sum(g["chosen_preference_accuracy"] for g in strata.values()) / len(strata)})
            receipt["checkpoints"].append(save_checkpoint(model, processor, optimizer, run / f"epoch_{epoch + 1}"))
            receipt["completed_epochs"] = epoch + 1
            study.save_json(run / "run.json", receipt)
            hub_backup.publish(run, run / f"epoch_{epoch + 1}")
        if receipt["optimizer_updates"] != receipt["planned_optimizer_updates"]:
            raise ValueError("Full-study optimizer update count is incomplete")
        receipt["status"] = "training_complete_evaluation_pending"
    except BaseException as exc:
        receipt["status"] = "incomplete_runtime_limit" if isinstance(exc, TimeoutError) else "failed"
        receipt["error_type"] = type(exc).__name__  # No uncontrolled credential-bearing error body.
        if receipt["optimizer_updates"] and isinstance(exc, (TimeoutError, KeyboardInterrupt)):
            # Preserve useful partial work, but never label a stopped run complete.
            receipt["checkpoints"].append(save_checkpoint(model, processor, optimizer, run / "incomplete_adapter"))
            study.save_json(run / "run.json", receipt)
            try:
                hub_backup.publish(run, run / "incomplete_adapter")
            except Exception as backup_error:
                receipt["hub_backup_error_type"] = type(backup_error).__name__
        raise
    finally:
        receipt["finished_at"] = study.now()
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
        study.save_json(run / "run.json", receipt)
    print(json.dumps({"run": str(run), "status": receipt["status"], "optimizer_updates": receipt["optimizer_updates"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check-data", "train"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--method", choices=("sft", "orpo"))
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--cohort", choices=("full", "resolved-balanced"), default="full")
    args = parser.parse_args()
    if args.action == "check-data":
        _, training, development, _ = approved_cohort(args.cohort)
        print(json.dumps({"training_images": len(training), "development_images": len(development), "optimizer_updates": 0}))
        return
    if not args.model or not args.method or args.max_seconds is None:
        parser.error("train requires --model, --method and --max-seconds")
    train(args.model, args.method, args.max_seconds, args.cohort)


if __name__ == "__main__":
    main()
