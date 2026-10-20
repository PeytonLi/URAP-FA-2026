"""Processor-only Gemma 3 checks on the frozen training pool; no weights or labels."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from importlib.metadata import version

import curated
import study
from curated_vision_check import PROBE_RESPONSES_NOT_LABELS, probe_blocks, response_inputs

MODEL = {"model": "google/gemma-3-12b-it",
         "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80"}


def checked_inputs(processor, image_path, response):
    """Keep the Gemma 3 processor's native fields and response-only mask."""
    inputs, summary = response_inputs(processor, image_path, response,
                                      enable_thinking=None)
    pixels = inputs.get("pixel_values")
    if pixels is None or pixels.ndim < 4 or pixels.shape[0] != 1 or not pixels.numel():
        raise ValueError("Expected pixel_values for exactly one image")
    types = inputs.get("token_type_ids")
    if types is None or not int((types[0, :summary["prompt_tokens"]] == 1).sum()):
        raise ValueError("Image token type markers were not retained in the prompt")
    decoded = processor.tokenizer.decode(
        inputs["input_ids"][0, summary["prompt_tokens"]:],
        skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != response + "<end_of_turn>" + "\n":
        raise ValueError(
            "Supervised continuation is not exactly the response and the native turn end")
    summary["supervised_response_exact"] = True
    summary["pixel_values_shape"] = list(pixels.shape)
    summary["renderer"] = "native_complete_assistant"
    summary["native_complete_assistant_template_used"] = True
    return inputs, summary


def main():
    import argparse
    import torch
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(MODEL["model"], revision=MODEL["revision"])
    config = study.dataset_config()
    rows = curated.selected_rows(config)
    selected = probe_blocks(rows)
    asset_root = study.REPO / config["asset_root"]
    report = {
        "checked_at": study.now(), "dataset_id": config["dataset_id"],
        "manifest_sha256": config["manifest_sha256"], "model": MODEL["model"],
        "processor_revision": MODEL["revision"],
        "rendering_mode": "native_complete_assistant",
        "versions": {name: version(name) for name in ("transformers", "torch", "torchvision", "Pillow")},
        "scope": "Complete frozen training matching blocks; processor-only grammar fixtures",
        "fixture_responses_are_not_training_labels": True,
        "fixture_responses": list(PROBE_RESPONSES_NOT_LABELS),
        "prompt_sha256": hashlib.sha256(study.IAT_PROMPT.encode()).hexdigest(),
        "model_weights_loaded": False, "optimizer_updates": 0,
        "images_uploaded": False,
    }
    checks = Counter()
    for row in selected:
        image = asset_root / row["image_path"]
        for response in PROBE_RESPONSES_NOT_LABELS:
            inputs, summary = checked_inputs(processor, image, response)
            checks["blocks_checked"] += 1
            checks["image_tokens"] += summary["image_tokens"]
    report["checks"] = dict(checks)
    report["rows_checked"] = len(selected)
    out = study.ROOT / "results" / "native" / "gemma3_vision_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    study.save_json(out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
