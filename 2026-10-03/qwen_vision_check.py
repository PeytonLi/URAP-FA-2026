"""Processor-only Qwen checks on the frozen training pool; no weights or labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from importlib.metadata import version

import curated
import study
from curated_vision_check import PROBE_RESPONSES_NOT_LABELS, probe_blocks, response_inputs

# Public model repository revisions resolved on October 3, 2026. These pin
# processor/template assets, not Fireworks's immutable serving-weight revisions.
MODELS = {
    "qwen3vl": {"model": "Qwen/Qwen3-VL-8B-Instruct",
                "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                "enable_thinking": None},
    "qwen35": {"model": "Qwen/Qwen3.5-9B",
               "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
               "enable_thinking": False},
}


def checked_inputs(processor, image_path, response, *, enable_thinking=None):
    """Keep the actual processor's image grid and response-only native inputs."""
    inputs, summary = response_inputs(processor, image_path, response,
                                      enable_thinking=enable_thinking)
    grid = inputs.get("image_grid_thw")
    if grid is None or grid.numel() == 0 or grid.shape != (1, 3):
        raise ValueError("Expected the native grid of exactly one image")
    if summary["image_tokens"] <= 0:
        raise ValueError("Image placeholders were not retained in the prompt")
    decoded = processor.tokenizer.decode(
        inputs["input_ids"][0, summary["prompt_tokens"]:],
        skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != response + processor.tokenizer.eos_token + "\n":
        raise ValueError("Supervised continuation is not exactly the response and native turn end")
    summary["supervised_response_exact"] = True
    summary["image_grid_thw"] = grid.tolist()
    summary["pixel_values_shape"] = list(inputs["pixel_values"].shape)
    summary["renderer"] = "native_complete_assistant"
    summary["native_complete_assistant_template_used"] = True
    return inputs, summary


def main():
    import torch
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    args = parser.parse_args()
    settings = MODELS[args.model]
    torch.set_num_threads(2)
    config = study.dataset_config()
    rows = curated.selected_rows(config)
    selected = probe_blocks(rows)
    asset_root = study.REPO / config["asset_root"]
    report = {
        "checked_at": study.now(), "dataset_id": config["dataset_id"],
        "manifest_sha256": config["manifest_sha256"], "model": settings["model"],
        "processor_revision": settings["revision"],
        "rendering_mode": "native_complete_assistant", "enable_thinking": settings["enable_thinking"],
        "thinking_control": "not_present_in_instruct_template" if settings["enable_thinking"] is None
                            else "explicitly_disabled_in_template",
        "versions": {name: version(name) for name in ("transformers", "torch", "torchvision", "Pillow")},
        "scope": "Complete frozen training matching blocks; processor-only grammar fixtures",
        "fixture_responses_are_not_training_labels": True,
        "fixture_responses": list(PROBE_RESPONSES_NOT_LABELS),
        "source_expression_metadata_not_used_in_prompt_or_responses": True,
        "prompt_sha256": hashlib.sha256(study.IAT_PROMPT.encode()).hexdigest(),
        "model_weights_loaded": False, "optimizer_updates": 0,
        "remote_backend_verified": False, "images_uploaded": False,
        "billable_resources_created": False, "examples": [], "errors": [],
    }
    processor = AutoProcessor.from_pretrained(
        settings["model"], revision=settings["revision"], token=False)
    report["processor_class"] = type(processor).__name__
    template = processor.chat_template
    if not isinstance(template, str):
        raise ValueError("Expected a single pinned native chat template")
    report["chat_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
    report["native_eos_token"] = processor.tokenizer.eos_token
    for row in selected:
        path = asset_root / row["image_path"]
        if not path.is_file() or study.sha(path) != row["image_sha256"]:
            raise ValueError("Selected diagnostic image absent or hash drifted")
        example = {"image_id": row["image_id"], "image_sha256": row["image_sha256"],
                   "dataset": row["dataset"], "split": row["split"],
                   "match_set_id": row["match_set_id"], "branches": []}
        try:
            for response in PROBE_RESPONSES_NOT_LABELS:
                _, summary = checked_inputs(processor, path, response,
                                             enable_thinking=settings["enable_thinking"])
                example["branches"].append(summary)
        except ValueError as exc:
            report["errors"].append({"image_id": row["image_id"], "error": str(exc)})
        report["examples"].append(example)
        if len(report["examples"]) % 12 == 0:
            print(f"{args.model}: checked {len(report['examples'])}/{len(selected)} training images", flush=True)
    report["images_checked"] = len(selected)
    report["branches_checked"] = sum(len(item["branches"]) for item in report["examples"])
    report["by_dataset"] = dict(Counter(row["dataset"] for row in selected))
    report["training_strata_checked"] = dict(Counter(
        row["dataset"] + "/" + row["expression_class"] for row in selected))
    report["local_processor_check_passed"] = not report["errors"]
    study.save_json(study.ROOT / "results" / f"curated_{args.model}_vision_check.json", report)
    summary = {key: value for key, value in report.items() if key not in ("examples", "errors")}
    summary["error_count"] = len(report["errors"])
    summary["first_errors"] = report["errors"][:3]
    print(json.dumps(summary, indent=2))
    return 0 if report["local_processor_check_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
