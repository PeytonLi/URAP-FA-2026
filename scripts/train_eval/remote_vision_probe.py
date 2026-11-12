"""Bounded Gemma 4 image-conditioned forward check; never updates weights.

Uses only six already selected, hash-verified training examples. This is not a
replacement for the incomplete 54-image study. A dedicated trainer may be billed.
"""
import json
import argparse
import math
import os
import re
import subprocess
import time
from datetime import timedelta

import study
import vision

JOB_ID = "urap-gemma4-visioncheck-20261003"
BASE = "accounts/fireworks/models/gemma-4-26b-a4b-it"
SHAPE = "accounts/fireworks/trainingShapes/gemma-4-26b-a4b-256k-b200-lora"
REPORT = study.ROOT / "results/gemma4_remote_vision_probe.json"


def jobs():
    return study.cli_json("rlor-trainer-job", "list") or []


def owned_job():
    return next((job for job in jobs() if job["name"].endswith("/" + JOB_ID)), None)


def trainer_base_model(job):
    # Native firectl list nests the model inside training_config; get/REST
    # representations may expose a top-level field.
    return job.get("base_model") or job.get("training_config", {}).get("base_model")


def tensor_values(tensor):
    """Use the SDK's dense conversion, not sparse storage values as positions."""
    if hasattr(tensor, "to_numpy"):
        array = tensor.to_numpy()
        if array.ndim != 1:
            raise ValueError(f"Expected a one-dimensional token tensor; got shape {array.shape}")
        return array.tolist()
    if getattr(tensor, "sparse_crow_indices", None) is not None:
        raise ValueError("Sparse token scores require the SDK dense conversion")
    return tensor.data


def forward_diagnostics(result, datums):
    """Capture dimensions before scoring, including failed alignment checks."""
    diagnostics, private_outputs = [], []
    for index, output in enumerate(result.loss_fn_outputs):
        item = {"index": index, "output_keys": sorted(output)}
        raw = {}
        for key, tensor in output.items():
            values = tensor_values(tensor)
            item[key] = {"shape": getattr(tensor, "shape", None),
                         "storage_values": len(tensor.data), "dense_values": len(values),
                         "sparse": getattr(tensor, "sparse_crow_indices", None) is not None}
            raw[key] = {**item[key], "values": values}
        if index < len(datums):
            datum = datums[index]
            weights = tensor_values(datum.loss_fn_inputs["weights"])
            item.update({"model_input_positions": datum.model_input.length,
                         "target_positions": len(tensor_values(datum.loss_fn_inputs["target_tokens"])),
                         "weight_positions": len(weights),
                         "response_positions": [i for i, value in enumerate(weights) if value > 0]})
        diagnostics.append(item)
        private_outputs.append(raw)
    return diagnostics, private_outputs


def response_scores(result, datums):
    outputs = result.loss_fn_outputs
    if len(outputs) != len(datums):
        raise ValueError("Remote forward response count does not match submitted datums")
    scores = []
    for output, datum in zip(outputs, datums):
        probabilities = tensor_values(output["logprobs"])
        weights = tensor_values(datum.loss_fn_inputs["weights"])
        if len(probabilities) != len(weights):
            raise ValueError(f"Remote token scores and response mask do not align: {len(probabilities)} scores vs {len(weights)} weights")
        score = sum(p * w for p, w in zip(probabilities, weights) if w > 0)
        if not math.isfinite(score):
            raise ValueError("Nonfinite conditional response score")
        scores.append(score)
    return scores


def conditioning_check(scores, repeated_scores, control):
    if not scores or not (len(scores) == len(repeated_scores) == len(control)):
        raise ValueError("Image-conditioning checks need equal nonempty batches")
    if not all(math.isfinite(value) for batch in (scores, repeated_scores, control) for value in batch):
        raise ValueError("Nonfinite image-conditioning score")
    noise = [a - b for a, b in zip(scores, repeated_scores)]
    deltas = [a - b for a, b in zip(scores, control)]
    threshold = max(1e-4, 10 * max(abs(value) for value in noise))
    exceeding = sum(abs(value) > threshold for value in deltas)
    return {"original_mean_response_logprobs": scores,
            "same_input_repeat_mean_response_logprobs": repeated_scores,
            "image_swap_mean_response_logprobs": control,
            "same_input_repeat_deltas": noise, "same_text_image_swap_deltas": deltas,
            "conditioning_threshold": threshold,
            "image_swap_scores_exceeding_threshold": exceeding,
            "image_conditioning_observed": exceeding >= 2}


def main():
    global JOB_ID, REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3, 4, 5), default=1)
    parser.add_argument("--include-image-boundaries", action="store_true",
                        help="Retained attempt-5 counterexample; backend rejected 258 vs 256 image positions")
    args = parser.parse_args()
    if study.dataset_config():
        raise RuntimeError("This archived probe uses the superseded 54-image cohort. Validate the active curated cohort before creating another trainer.")
    if args.include_image_boundaries and args.attempt != 5:
        raise ValueError("The boundary-owning variant is retained only for the recorded attempt-5 diagnostic, not another training route")
    if args.attempt != 1:
        JOB_ID += f"-a{args.attempt}"
        REPORT = REPORT.with_name(f"gemma4_remote_vision_probe_attempt{args.attempt}.json")
    from fireworks.training.sdk import FiretitanServiceClient, FiretitanProvisioningConfig
    from transformers import AutoProcessor

    if REPORT.exists():
        raise RuntimeError("A probe receipt already exists; inspect it instead of automatically creating another trainer")
    if owned_job() is not None:
        raise RuntimeError("The requested trainer ID already exists; refusing to adopt an unverified resource")
    rows = vision.probe_rows(study.ROOT / "data/private/assets")
    processor = AutoProcessor.from_pretrained(vision.GEMMA_HF, revision=vision.GEMMA_REVISION, token=False)
    contexts = [vision.prompt_chunks(processor, study.ROOT / "data/private/assets" / row["image_path"],
                                    include_image_boundaries=args.include_image_boundaries)
                for row in rows]
    original, swapped = [], []
    for index, row in enumerate(rows):
        own_chunks, own_tokens, _ = contexts[index]
        other_chunks, other_tokens, _ = contexts[(index + 2) % len(rows)]
        if own_tokens != other_tokens:
            raise ValueError("Image-swap control must leave every text token and token position unchanged")
        for branch in ("chosen", "rejected"):
            original.append(vision.response_datum(processor, own_chunks, own_tokens, row[branch])[0])
            swapped.append(vision.response_datum(processor, other_chunks, other_tokens, row[branch])[0])
    key, _ = study.fireworks_auth()
    if not key:
        raise RuntimeError("No Fireworks API key available")
    report = {"started_at": study.now(), "job_id": JOB_ID, "logical_base_model": BASE,
              "requested_training_shape": SHAPE, "processor_revision": vision.GEMMA_REVISION,
              "scope": "Image-bearing forward-only modality validation; not the full SFT/ORPO study",
              "source_pairs_sha256": study.sha(study.ROOT / "data/pairs.csv"),
              "pair_ids": [row["pair_id"] for row in rows], "training_split_only": True,
              "image_boundaries_in_chunk": args.include_image_boundaries,
              "image_spans": [context[2] for context in contexts],
              "benchmark_images_used": 0, "optimizer_updates": 0, "checkpoint_promoted": False,
              "backend_image_support_verified": False, "status": "starting",
              "cleanup_verified": False, "deployment_created": False}
    study.save_json(REPORT, report)
    os.environ.setdefault("FIREWORKS_ACCOUNT_ID", "peli")
    config = FiretitanProvisioningConfig(
        base_model=BASE, tokenizer_model=vision.GEMMA_HF,
        max_lora_rank=8, training_shape_id=SHAPE, trainer_job_id=JOB_ID,
        create_deployment=False, reference_required=False,
        max_context_length=4096, learning_rate=1e-5, seed=0,
        trainer_timeout_s=600, trainer_pending_timeout_s=600,
        inactivity_timeout=timedelta(minutes=10), cleanup_trainer_on_close=True, use_reservation=False,
        display_name="URAP Gemma 4 image forward check; no optimizer updates",
    )
    service = FiretitanServiceClient(api_key=key, managed_config=config)
    try:
        print("Requesting one dedicated Gemma 4 trainer for forward-only validation; no inference deployment", flush=True)
        client = service.create_training_client(BASE, lora_rank=8, lora_alpha=16, seed=0)
        report.update({"status": "forward_checks", "trainer_job_id": service.trainer_job_id,
                       "accelerator_type": service.accelerator_type,
                       "accelerator_count": service.accelerator_count,
                       "transport": client.comms})
        study.save_json(REPORT, report)
        result = client.forward(original, "cross_entropy").result(timeout=120)
        dimensions, private_outputs = forward_diagnostics(result, original)
        report["original_forward_dimensions"] = dimensions
        report["forward_transport_returned"] = True
        study.save_json(REPORT, report)
        # No images, API credentials, or model-input chunks in this private receipt.
        # Preserve numerical evidence even if strict alignment validation fails.
        study.save_json(study.ROOT / "data/private" / f"gemma4_forward_attempt{args.attempt}.json",
                        {"job_id": JOB_ID, "outputs": private_outputs, "dimensions": dimensions})
        scores = response_scores(result, original)
        repeated_result = client.forward(original, "cross_entropy").result(timeout=120)
        report["same_input_repeat_forward_dimensions"] = forward_diagnostics(repeated_result, original)[0]
        study.save_json(REPORT, report)
        repeated_scores = response_scores(repeated_result, original)
        swapped_result = client.forward(swapped, "cross_entropy").result(timeout=120)
        report["swap_forward_dimensions"] = forward_diagnostics(swapped_result, swapped)[0]
        study.save_json(REPORT, report)
        control = response_scores(swapped_result, swapped)
        report.update(conditioning_check(scores, repeated_scores, control))
        report.update({"real_image_chunks_accepted": True, "status": "forward_checks_complete"})
        report["backend_image_support_verified"] = report["image_conditioning_observed"]
        if not report["backend_image_support_verified"]:
            raise RuntimeError("Image chunks were accepted but image conditioning was not demonstrated")
        print("Image-bearing forward checks completed; response scores change when only the image changes", flush=True)
    except Exception as exc:
        # Error bodies can contain image payloads or transport internals. Do not
        # copy them into a tracked report or user output.
        details = re.sub(r"fw_[A-Za-z0-9_-]+", "[redacted key]", str(exc))
        details = re.sub(r"data:[^;]+;base64,[A-Za-z0-9+/=]+", "[omitted image payload]", details)
        report.update({"status": "failed", "error_type": type(exc).__name__,
                       "http_status": getattr(exc, "status_code", None), "sanitized_error": details[:1024]})
        print(f"Forward check failed: {type(exc).__name__}; inspecting the owned trainer state", flush=True)
    finally:
        try:
            service.close()
        except Exception as exc:
            report["sdk_cleanup_error_type"] = type(exc).__name__
        # SDK cleanup may fail before it constructs a managed handle. Inspect and
        # delete only our exact job, after checking its model identity.
        current = owned_job()
        if current is not None:
            report["last_owned_trainer_state"] = current.get("state")
            if trainer_base_model(current) != BASE:
                report["cleanup_identity_mismatch"] = True
            else:
                result = subprocess.run([str(study.FIRECTL), "rlor-trainer-job", "delete", JOB_ID],
                                        capture_output=True, text=True, timeout=60)
                report["native_delete_acknowledged"] = result.returncode == 0
        # Deletion acknowledgment is not proof of disappearance. Give this exact
        # owned job a bounded window to finish asynchronous cleanup.
        cleanup_deadline = time.monotonic() + 45
        while owned_job() is not None and time.monotonic() < cleanup_deadline:
            time.sleep(3)
        report["cleanup_verified"] = owned_job() is None
        report["finished_at"] = study.now()
        study.save_json(REPORT, report)
    print(json.dumps({key: report[key] for key in ("status", "backend_image_support_verified", "optimizer_updates", "cleanup_verified")}, indent=2))
    return 0 if report["backend_image_support_verified"] and report["cleanup_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
