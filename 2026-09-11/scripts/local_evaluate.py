"""Frozen-weight inference: paired Gemma base/LoRA and a small Qwen VL probe."""
import argparse
import gc
import importlib.metadata
import json
import os
import random
import sys
import time
from pathlib import Path

from common import ROOT, json_read, local_seed, now, read_jsonl, sha, write_csv, write_json

sys.path.insert(0, str(ROOT / "src"))
from multimodel_bias.constants import IAT_PROMPT, NEGATIVE_IAT_SET, POSITIVE_IAT_SET
from multimodel_bias.iat import is_exact_three_word_format, load_stimuli, parse_three_words


def seed_all(seed, torch):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def text_of(result):
    value = result[0]["generated_text"]
    if not isinstance(value, str):
        raise ValueError("Unexpected pipeline return format")
    return value.strip()


def tasks():
    cfg = json_read(ROOT / "config/study.json")
    out = []
    for panel, p in cfg["panels"].items():
        images = load_stimuli(ROOT / p["manifest"], expected_count=12 if panel == "neutral12" else 400)
        for iteration in range(1, p["repeats"] + 1):
            for s in images:
                out.append(dict(panel=panel, image_id=s.image_id, image_sha256=s.sha256,
                                image_path=str(s.path.relative_to(ROOT)), race=s.race, gender=s.gender,
                                iteration=iteration, seed=local_seed(panel, s.image_id, iteration)))
    return out


def annotate(task, key, raw, fallback=""):
    words = parse_three_words(fallback or raw)
    return dict({k: v for k, v in task.items() if k != "image_path"}, model_key=key,
                sampled_raw=raw, fallback_raw=fallback, words="|".join(words),
                response=", ".join(words), valid_three=len(words) == 3,
                word_count=len(words),
                negative_count=sum(w in NEGATIVE_IAT_SET for w in words),
                positive_count=sum(w in POSITIVE_IAT_SET for w in words),
                exact_format=is_exact_three_word_format(fallback or raw, words),
                response_source="fallback" if fallback else "sampled", generation_calls=1+bool(fallback),
                created_at=now(), provenance="new_local", provider="RunPod H100 / Transformers 4.50.0")


def append(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--deadline-epoch", type=float, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()
    cfg = json_read(ROOT / "config/study.json")
    work = tasks()
    if sha(ROOT / "data/private/adapter/adapter_model.safetensors") != cfg["adapter_sha256"]:
        raise ValueError("Adapter hash drift")
    if args.dry_run:
        print(json.dumps(dict(paired_image_tasks=len(work), generations=2*len(work), training=False)))
        return
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, pipeline

    if transformers.__version__ != "4.50.0" or "H100" not in torch.cuda.get_device_name(0):
        raise ValueError("Pinned Transformers 4.50.0 and H100 required")
    out = ROOT / "results/local"
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = args.deadline_epoch or started + cfg["runpod_max_seconds"] - 600
    environment = {n: importlib.metadata.version(n) for n in ("torch", "transformers", "peft", "accelerate", "numpy", "pillow")}
    write_json(out / "environment.json", environment)
    # This small probe is independent of the Gemma comparison, and uses its own model cache.
    if not (out / "qwen_probe.json").exists():
        qrows = []
        try:
            cache = "/tmp/qwen-cache"
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", revision=cfg["qwen_revision"], cache_dir=cache)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct", revision=cfg["qwen_revision"], cache_dir=cache,
                torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
            model.eval().requires_grad_(False)
            for task in [t for t in work if t["panel"] == "neutral12" and t["iteration"] == 1]:
                seed_all(task["seed"], torch)
                with Image.open(ROOT / task["image_path"]) as im:
                    im = im.convert("RGB")
                    prompt = processor.apply_chat_template([{"role": "user", "content": [
                        {"type": "image"}, {"type": "text", "text": IAT_PROMPT}]}], tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[prompt], images=[im], return_tensors="pt").to(model.device)
                    with torch.inference_mode():
                        output = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=1., top_p=1.)
                    raw = processor.decode(output[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                qrows.append(annotate(task, "qwen25_vl_3b_probe", raw))
            write_csv(out / "qwen_probe.csv", qrows)
            write_json(out / "qwen_probe.json", dict(status="complete", model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                       revision=cfg["qwen_revision"], rows=len(qrows), valid=sum(r["valid_three"] for r in qrows),
                       training_performed=False, attention="sdpa", prompt=IAT_PROMPT))
            del model, processor, inputs, output
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            write_json(out / "qwen_probe.json", dict(status="failed", error_type=type(exc).__name__, detail=str(exc)[:500]))
            raise
    if args.probe_only:
        return
    if not args.model_dir or args.model_dir.name != cfg["gemma_revision"]:
        raise ValueError("Pinned Gemma snapshot required")
    files = sorted(p for p in args.model_dir.iterdir() if p.suffix in {".json", ".safetensors", ".model"})
    hashes = {p.name: sha(p) for p in files}
    expected = json_read(ROOT / "reference/gemma_base_manifest.json")["base_snapshot_hashes"]
    if hashes != expected:
        raise ValueError("Gemma snapshot differs from preserved September 5 hashes")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
             torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager")
    model.eval().requires_grad_(False)
    pipe = pipeline("image-text-to-text", model=model, processor=processor, torch_dtype=torch.bfloat16)
    def generate(task):
        seed_all(task["seed"], torch)
        with Image.open(ROOT / task["image_path"]) as im, torch.inference_mode():
            im = im.convert("RGB")
            raw = text_of(pipe(im, text="<start_of_image> " + IAT_PROMPT, return_full_text=False,
                         generate_kwargs=dict(max_new_tokens=30, do_sample=True, temperature=1., top_p=1.)))
            fallback = ""
            if len(parse_three_words(raw)) != 3:
                fallback = text_of(pipe(im, text="<start_of_image> " + IAT_PROMPT, return_full_text=False,
                              generate_kwargs=dict(max_new_tokens=20, do_sample=False, temperature=1., top_p=1., top_k=50)))
        return raw, fallback
    pure_base = generate(work[0])
    generation_config = model.generation_config.to_dict()
    model.load_adapter(str(ROOT / "data/private/adapter"), adapter_name="research")
    model.eval().requires_grad_(False)
    model.disable_adapters()
    disabled_base = generate(work[0])
    if pure_base != disabled_base or generation_config != model.generation_config.to_dict():
        raise ValueError("Disabled adapter does not reproduce pure base probe")
    record_path = out / "responses.jsonl"
    records = read_jsonl(record_path)
    seen = {(r["panel"], r["image_id"], r["iteration"], r["model_key"]) for r in records}
    if len(seen) != len(records):
        raise ValueError("Duplicate local task")
    manifest = dict(started_at=now(), model_revision=cfg["gemma_revision"], adapter_sha256=cfg["adapter_sha256"],
                    status="running", base_weights_updated=False, seed_contract=cfg["local_seed_contract"],
                    pure_base_matches_disabled_adapter=True, base_snapshot_hashes=hashes,
                    planned_responses=5200, generation_config=generation_config, environment=environment)
    write_json(out / "manifest.json", manifest)
    try:
        for task in work:
            if time.time() > deadline or (ROOT / "STOP_REQUESTED").exists():
                manifest["status"] = "time_or_user_stop"
                break
            for key in ("gemma_base", "gemma_tuned"):
                identifier = (task["panel"], task["image_id"], task["iteration"], key)
                if identifier in seen:
                    continue
                tick = time.time()
                if key == "gemma_base":
                    model.disable_adapters()
                else:
                    model.enable_adapters()
                raw, fallback = generate(task)
                row = annotate(task, key, raw, fallback)
                row.update(latency_seconds=time.time()-tick, model_revision=cfg["gemma_revision"],
                           adapter_sha256=cfg["adapter_sha256"] if key == "gemma_tuned" else "")
                append(record_path, row)
                records.append(row)
                seen.add(identifier)
            if len(records) % 50 == 0:
                manifest.update(completed_responses=len(records), elapsed_seconds=time.time()-started,
                                peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
                write_json(out / "manifest.json", manifest)
                print(json.dumps({k: manifest[k] for k in ("completed_responses", "elapsed_seconds", "peak_gpu_gib")}), flush=True)
        else:
            manifest["status"] = "complete"
    finally:
        write_csv(out / "responses.csv", records)
        manifest.update(completed_responses=len(records), finished_at=now(), elapsed_seconds=time.time()-started,
                        base_hashes_unchanged=hashes == {p.name: sha(p) for p in files})
        write_json(out / "manifest.json", manifest)
        if not manifest["base_hashes_unchanged"]:
            raise ValueError("Base snapshot mutated")


if __name__ == "__main__":
    main()
