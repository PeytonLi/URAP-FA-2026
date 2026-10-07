"""Local Gemma 4 processor checks on frozen train images; no weights or labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from importlib.metadata import version

import curated
import study
from vision import GEMMA_HF, GEMMA_REVISION

# Grammar/coordinate fixtures only, never image labels, training targets or
# approved preferences. No optimizer is created and no model weights are loaded.
PROBE_RESPONSES_NOT_LABELS = ("Happy, Joyful, Smiling", "Pain, Grief, Sadness")


def probe_blocks(rows):
    blocks = defaultdict(list)
    for row in rows:
        if row["split"] == "train":
            blocks[row["match_set_id"]].append(row)
    by_stratum = {}
    for key, block in sorted(blocks.items()):
        stratum = (block[0]["dataset"], block[0]["expression_class"])
        by_stratum.setdefault(stratum, block)
    selected = [row for stratum, block in sorted(by_stratum.items())
                for row in sorted(block, key=lambda item: item["image_id"])]
    if not selected:
        raise ValueError("No complete frozen training blocks are available")
    return selected


def response_inputs(processor, image_path, response, *, enable_thinking=False):
    """Retain every native processor field; derive the mask from exact prefixes."""
    if not study.exact_target(response):
        raise ValueError("Expected a canonical three-word response")
    user = {"role": "user", "content": [
        {"type": "image", "path": str(image_path)},
        {"type": "text", "text": study.IAT_PROMPT},
    ]}
    options = {"tokenize": True, "return_dict": True, "return_tensors": "pt"}
    if enable_thinking is not None:
        options["enable_thinking"] = enable_thinking
    prompt = processor.apply_chat_template([user], add_generation_prompt=True, **options)
    complete = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": response}]}],
        add_generation_prompt=False, **options,
    )
    return mask_complete(processor, prompt, complete)


def mask_complete(processor, prompt, complete):
    """Fail closed on token/field drift before assigning response-only labels."""
    import torch

    prefix_length = prompt["input_ids"].shape[1]
    ids = complete["input_ids"]
    if ids.shape[0] != 1 or ids.shape[1] <= prefix_length:
        raise ValueError("Missing supervised continuation")
    if not torch.equal(ids[:, :prefix_length], prompt["input_ids"]):
        raise ValueError("Native assistant template does not extend the inference prompt exactly")
    if "pixel_values" not in complete or complete["pixel_values"].numel() == 0:
        raise ValueError("Native image tensors are missing")
    if set(prompt) - set(complete):
        raise ValueError("Full conversation lost native processor fields")
    # Both template calls must process the same single image, with no altered
    # resize policy or hand-constructed image-token span.
    if not torch.equal(complete["pixel_values"], prompt["pixel_values"]):
        raise ValueError("Full conversation changed the image preprocessing")
    for name in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
        if name in prompt:
            if name not in complete or not torch.equal(complete[name][:, :prefix_length], prompt[name]):
                raise ValueError(f"Native prefix field changed: {name}")
    sequence_fields = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
    for name, value in prompt.items():
        if name not in sequence_fields and isinstance(value, torch.Tensor):
            if not torch.equal(value, complete[name]):
                raise ValueError(f"Native image field changed: {name}")
    labels = ids.clone()
    labels[:, :prefix_length] = -100
    complete["labels"] = labels
    image_token = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    if (ids == image_token).any() and not (labels[ids == image_token] == -100).all():
        raise ValueError("Image tokens received target loss")
    return complete, {"prompt_tokens": prefix_length, "total_tokens": ids.shape[1],
                      "response_tokens_including_native_terminator": int((labels != -100).sum()),
                      "image_tokens": int((ids == image_token).sum()),
                      "prefix_equal": True, "image_tensors_retained": True,
                      "prompt_image_labels_all_ignored": True,
                      "processor_fields": sorted(key for key in complete if key != "labels")}


def continuation_inputs(processor, image_path, response):
    """Gemma 4 continuation of the official non-thinking generation prefix.

    This deliberately differs from the complete assistant-message template,
    which omits the empty thought-channel marker at the pinned revision.
    Images are processed natively for the full text; no soft-token lengths or
    multimodal token-type fields are synthesized or truncated.
    """
    from PIL import Image

    if not study.exact_target(response):
        raise ValueError("Expected a canonical three-word response")
    user = {"role": "user", "content": [
        {"type": "image", "path": str(image_path)},
        {"type": "text", "text": study.IAT_PROMPT},
    ]}
    options = {"enable_thinking": False}
    rendered_prefix = processor.apply_chat_template(
        [user], tokenize=False, add_generation_prompt=True, **options)
    native_full = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": response}]}],
        tokenize=False, add_generation_prompt=False, **options)
    terminator = "<turn|>\n"
    if not native_full.endswith(response + terminator):
        raise ValueError("Pinned Gemma 4 native response terminator changed")
    prompt = processor.apply_chat_template(
        [user], tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, **options)
    with Image.open(image_path) as image:
        complete = processor(text=rendered_prefix + response + terminator,
                             images=image.convert("RGB"), return_tensors="pt",
                             add_special_tokens=False)
    inputs, summary = mask_complete(processor, prompt, complete)
    summary["renderer"] = "pinned_gemma4_non_thinking_generation_continuation"
    summary["native_complete_assistant_template_used"] = False
    return inputs, summary


def main():
    from transformers import AutoProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", action="store_true",
                        help="Explicitly test continuation of the pinned non-thinking inference prefix")
    args = parser.parse_args()
    render = continuation_inputs if args.continuation else response_inputs
    config = study.dataset_config()
    rows = curated.selected_rows(config)
    selected = probe_blocks(rows)
    asset_root = study.REPO / config["asset_root"]
    processor = AutoProcessor.from_pretrained(GEMMA_HF, revision=GEMMA_REVISION, token=False)
    report = {"checked_at": study.now(), "dataset_id": config["dataset_id"],
              "manifest_sha256": config["manifest_sha256"], "model": GEMMA_HF,
              "processor_revision": GEMMA_REVISION,
              "rendering_mode": "generation_continuation" if args.continuation else "native_complete_assistant",
              "versions": {name: version(name) for name in ("transformers", "torch", "Pillow")},
              "scope": "Complete frozen training matching blocks; processor-only grammar fixtures",
              "fixture_responses_are_not_training_labels": True,
              "source_expression_metadata_not_used_in_prompt_or_responses": True,
              "model_weights_loaded": False, "optimizer_updates": 0,
              "remote_backend_verified": False, "images_uploaded": False,
              "examples": [], "errors": []}
    for row in selected:
        path = asset_root / row["image_path"]
        if not path.is_file() or study.sha(path) != row["image_sha256"]:
            raise ValueError("Selected diagnostic image absent or hash drifted")
        example = {"image_id": row["image_id"], "image_sha256": row["image_sha256"],
                   "dataset": row["dataset"], "split": row["split"],
                   "match_set_id": row["match_set_id"], "branches": []}
        try:
            for response in PROBE_RESPONSES_NOT_LABELS:
                _, summary = render(processor, path, response)
                example["branches"].append(summary)
        except ValueError as exc:
            report["errors"].append({"image_id": row["image_id"], "error": str(exc)})
        report["examples"].append(example)
        if len(report["examples"]) % 12 == 0:
            print(f"Checked {len(report['examples'])}/{len(selected)} frozen training images", flush=True)
    report["images_checked"] = len(selected)
    report["by_dataset"] = dict(Counter(row["dataset"] for row in selected))
    report["training_strata_checked"] = dict(Counter(
        row["dataset"] + "/" + row["expression_class"] for row in selected))
    report["local_processor_check_passed"] = not report["errors"]
    filename = "curated_gemma4_continuation_check.json" if args.continuation else "curated_gemma4_vision_check.json"
    study.save_json(study.ROOT / "results" / filename, report)
    summary = {key: value for key, value in report.items() if key not in ("examples", "errors")}
    summary["error_count"] = len(report["errors"])
    summary["first_errors"] = report["errors"][:3]
    print(json.dumps(summary, indent=2))
    return 0 if report["local_processor_check_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
