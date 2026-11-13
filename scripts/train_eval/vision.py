"""Inspect image-bearing Gemma 4 training datums before provisioning compute.

This is a local processor/chunk check, not evidence of Fireworks backend support.
It does not export a reduced study, upload data, create a trainer, or update weights.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import study

GEMMA_HF = "google/gemma-4-26B-A4B-it"
GEMMA_REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def probe_rows(asset_root):
    """Choose complete *training* blocks for a modality check, never benchmark/dev rows."""
    groups = defaultdict(list)
    for row in study.read_csv(study.ROOT / "data/pairs.csv"):
        if row["split"] == "train":
            groups[row["match_block_id"]].append(row)
    selected = []
    for expression in ("positive", "neutral", "negative"):
        blocks = [block for _, block in sorted(groups.items())
                  if len(block) == 2 and all(row["expression_class"] == expression for row in block)
                  and {row["fairface_race"] for row in block} == {"Black", "White"}
                  and all((asset_root / row["image_path"]).is_file()
                          and study.sha(asset_root / row["image_path"]) == row["image_sha256"]
                          for row in block)]
        if not blocks:
            raise ValueError(f"No complete hash-verified training block for {expression}")
        selected.extend(sorted(blocks[0], key=lambda row: row["fairface_race"]))
    forbidden = {row["sha256"] for row in study.read_csv(study.IAT12_MANIFEST)}
    if any(study.label_failures(row, forbidden) for row in selected):
        raise ValueError("A modality-probe row failed its historical label audit")
    return selected


def prompt_chunks(processor, image_path, *, include_image_boundaries=False):
    import tinker

    message = {"role": "user", "content": [
        {"type": "image", "path": str(image_path)},
        {"type": "text", "text": study.IAT_PROMPT},
    ]}
    processed = processor.apply_chat_template(
        [message], tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=True, enable_thinking=False,
    )
    tokens = processed["input_ids"][0].tolist()
    image_token = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    indices = [i for i, token in enumerate(tokens) if token == image_token]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Expected one contiguous, nonempty image-token span")
    start, end = indices[0], indices[-1] + 1
    soft_start, soft_end = start, end
    if tokens[start - 1] != processor.tokenizer.convert_tokens_to_ids(processor.boi_token):
        raise ValueError("Missing BOI before the image-token span")
    if tokens[end] != processor.tokenizer.convert_tokens_to_ids(processor.eoi_token):
        raise ValueError("Missing EOI after the image-token span")
    if include_image_boundaries:
        # Retained diagnostic counterexample: attempt 5 proved Fireworks rejects
        # this declared length (258 expected vs 256 actual image soft tokens).
        # Replace the full official span instead of supplying these boundaries
        # again as explicit text. This preserves total HF prompt coordinates,
        # but is NOT a valid training variant and is never the default.
        start -= 1
        end += 1
    chunks = [tinker.types.EncodedTextChunk(tokens=tokens[:start]),
              tinker.types.ImageChunk(data=image_path.read_bytes(), format="png", expected_tokens=end - start),
              tinker.types.EncodedTextChunk(tokens=tokens[end:])]
    if sum(chunk.length for chunk in chunks) != len(tokens):
        raise ValueError("Image chunk length does not match official processor token positions")
    # Image bytes remain a typed chunk, never a string in a text chunk.
    return chunks, tokens, {"image_start": start, "image_tokens": soft_end - soft_start,
                            "image_chunk_expected_tokens": end - start,
                            "image_boundaries_in_chunk": include_image_boundaries,
                            "prompt_tokens": len(tokens),
                            "processor_fields": sorted(processed.keys())}


def response_datum(processor, chunks, prompt_tokens, target):
    import tinker
    import torch

    sys.path.insert(0, str(study.ROOT / "runtime/cookbook"))
    from training.renderer.supervised import datum_from_model_input_weights

    if not study.exact_target(target):
        raise ValueError("Only canonical three-word targets are allowed")
    # Train the continuation of the actual non-thinking inference prefix, including
    # the normal turn terminator. Do not train the prompt, image, or role/channel tags.
    suffix = target + "<turn|>\n"
    target_ids = processor.tokenizer.encode(suffix, add_special_tokens=False)
    decoded_prompt = processor.tokenizer.decode(prompt_tokens, skip_special_tokens=False)
    retokenized = processor.tokenizer.encode(decoded_prompt + suffix, add_special_tokens=False)
    if retokenized != prompt_tokens + target_ids:
        raise ValueError("Prompt/target boundary changed tokenization")
    model_input = tinker.ModelInput(chunks=[*chunks, tinker.types.EncodedTextChunk(tokens=target_ids)])
    weights = torch.tensor([0.0] * len(prompt_tokens) + [1.0] * len(target_ids))
    datum = datum_from_model_input_weights(model_input, weights, reduction="mean")
    shifted = datum.loss_fn_inputs["weights"].data
    if any(shifted[:len(prompt_tokens) - 1]) or not all(w > 0 for w in shifted[len(prompt_tokens) - 1:]):
        raise ValueError("Loss mask is not response-only after the next-token shift")
    if abs(sum(shifted) - 1.0) > 1e-5:
        raise ValueError("SFT response-token mean weights do not sum to one")
    if not any(isinstance(chunk, tinker.types.ImageChunk) for chunk in datum.model_input.chunks):
        raise ValueError("Image was lost during datum construction")
    return datum, target_ids


def main():
    from importlib.metadata import version
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor

    asset_root = study.ROOT / "data/private/assets"
    rows = probe_rows(asset_root)
    processor = AutoProcessor.from_pretrained(GEMMA_HF, revision=GEMMA_REVISION, token=False)
    report = {"checked_at": study.now(), "model": GEMMA_HF, "processor_revision": GEMMA_REVISION,
              "provider_weight_revision_equivalence_verified": False,
              "versions": {name: version(name) for name in ("transformers", "torch", "torchvision", "fireworks-ai", "tinker")},
              "scope": "Six available training images, three complete expression-stratified blocks; local modality validation only",
              "full_dataset_ready": json.loads((study.ROOT / "data/manifest.json").read_text(encoding="utf-8"))["training_ready"],
              "backend_image_support_verified": False,
              "paid_resources_created": False, "optimizer_updates": 0, "examples": []}
    sheet = Image.new("RGB", (800, 960), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        image_path = asset_root / row["image_path"]
        chunks, prompt_tokens, spans = prompt_chunks(processor, image_path)
        branch_info = {}
        for branch in ("chosen", "rejected"):
            datum, target_ids = response_datum(processor, chunks, prompt_tokens, row[branch])
            branch_info[branch] = {"target": row[branch], "target_tokens_including_terminator": len(target_ids),
                                  "datum_tokens": datum.model_input.length,
                                  "prompt_image_loss_weight": 0.0, "response_weight_sum": sum(datum.loss_fn_inputs["weights"].data)}
        report["examples"].append({"pair_id": row["pair_id"], "image_path": row["image_path"],
                                   "image_sha256": row["image_sha256"], "split": row["split"],
                                   "expression": row["expression_class"], **spans, "branches": branch_info})
        left, top = (index % 2) * 400, (index // 2) * 320
        with Image.open(image_path) as original:
            thumb = original.convert("RGB")
            thumb.thumbnail((224, 224))
            sheet.paste(thumb, (left + 10, top + 10))
        draw.text((left + 10, top + 240), f"{row['expression_class']} / {row['expression_fine']} / {row['fairface_race']}", fill="black")
        draw.text((left + 10, top + 258), "Chosen: " + row["chosen"], fill="black")
        draw.text((left + 10, top + 276), "Rejected: " + row["rejected"], fill="black")
        print(f"Rendered {row['expression_class']} training example: {spans['image_tokens']} image tokens; response-only masks", flush=True)
    sheet_path = study.ROOT / "data/private/gemma4_modality_probe_images.png"
    sheet.save(sheet_path)
    report["contact_sheet_private_path"] = str(sheet_path)
    report["local_rendering_passed"] = True
    study.save_json(study.ROOT / "results/gemma4_local_vision_probe.json", report)
    print(json.dumps({"local_rendering_passed": True, "optimizer_updates": 0,
                      "backend_image_support_verified": False, "examples": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
