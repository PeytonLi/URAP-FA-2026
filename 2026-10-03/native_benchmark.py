"""Matched native base/adapter benchmark using the existing IAT scorer/fallback."""
import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import benchmark
import iat_prompts
import native_train
import study
from multimodel_bias.iat import IATBenchmarkRunner, Stimulus, deterministic_task_id


def development_stimuli(dataset):
    """Freeze three complete development match sets without using model outputs."""
    from collections import defaultdict
    import curated
    rows = curated.selected_rows(dataset)
    blocks = defaultdict(list)
    for row in rows:
        if row["split"] == "development":
            blocks[row["match_set_id"]].append(row)
    chosen = []
    for sid in sorted(blocks, key=lambda s: hashlib.sha256(("development-v3|" + s).encode()).hexdigest()):
        members = blocks[sid]
        if (len(members) == 4
                and sum(r["source_race"] == "Black" for r in members) == 2
                and sum(r["source_race"] == "White" for r in members) == 2):
            chosen.extend(members)
        if len(chosen) == 12:
            break
    if len(chosen) != 12:
        raise ValueError("Three complete development blocks required")
    from PIL import Image
    stimuli = []
    for row in chosen:
        image = (study.REPO / dataset["asset_root"] / row["image_path"]).resolve()
        if not study.image_matches_manifest(row, image):
            raise ValueError("Development image hash mismatch")
        with Image.open(image) as im:
            width, height = im.size
        stimuli.append(Stimulus(
            row["image_id"], image.name, row["source_race"], row["source_gender"],
            width, height, study.sha(image), image))
    return stimuli


class NativeRunner(IATBenchmarkRunner):
    def __init__(self, native_model, processor, model_name, deadline, *args,
                 prompt=None, prompt_id=None, **kwargs):
        self.prompt_spec = iat_prompts.prompt_spec(prompt_id) if prompt_id is not None else None
        if self.prompt_spec:
            if prompt is not None and prompt != self.prompt_spec["prompt"]:
                raise ValueError("Prompt text does not match the frozen prompt ID")
            prompt = self.prompt_spec["prompt"]
        super().__init__(None, *args, **kwargs)
        self.native_model, self.processor = native_model, processor
        self.model_name, self.deadline, self.inputs = model_name, deadline, {}
        self.prompt = study.IAT_PROMPT if prompt is None else prompt

    def _append_jsonl(self, path, value):
        if self.prompt_spec and path.name in ("responses.jsonl", "api_responses.jsonl"):
            if ("prompt_sha256" in value
                    and value["prompt_sha256"] != self.prompt_spec["prompt_sha256"]):
                raise ValueError("Actual prompt hash differs from the frozen prompt ID")
            value = {**value, "prompt_id": self.prompt_spec["prompt_id"],
                     "prompt_sha256": self.prompt_spec["prompt_sha256"]}
        super()._append_jsonl(path, value)

    def _call(self, *, task_id, stage, model, stimulus, seed, fallback):
        import torch
        from transformers import set_seed
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Native benchmark deadline reached; keep partial records")
        if stimulus.image_id not in self.inputs:
            if study.sha(stimulus.path) != stimulus.sha256:
                raise ValueError("Benchmark image hash changed")
            options = {"tokenize": True, "return_dict": True, "return_tensors": "pt",
                       "add_generation_prompt": True}
            thinking = native_train.MODELS[self.model_name]["enable_thinking"]
            if thinking is not None:
                options["enable_thinking"] = thinking
            user = {"role": "user", "content": [{"type": "image", "path": str(stimulus.path)},
                                                    {"type": "text", "text": self.prompt}]}
            inputs = self.processor.apply_chat_template([user], **options)
            if "pixel_values" not in inputs or not inputs["pixel_values"].numel():
                raise ValueError("Native benchmark must contain real image pixels")
            self.inputs[stimulus.image_id] = {
                k: v.to(self.native_model.device, dtype=self.native_model.dtype if v.is_floating_point() else v.dtype)
                for k, v in inputs.items()}
        inputs = self.inputs[stimulus.image_id]
        set_seed(seed)
        generation = {"max_new_tokens": 30, "do_sample": not fallback, "use_cache": True,
                      "top_k": 0, "top_p": 1.0, "temperature": 1.0}
        with torch.inference_mode():
            ids = self.native_model.generate(**inputs, **generation)[0, inputs["input_ids"].shape[1]:]
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        response = {"model": model.model_id, "provider": "prime/native-transformers",
                    "choices": [{"message": {"content": text},
                                 "finish_reason": "length" if ids.numel() == 30 else "stop"}],
                    "usage": {"prompt_tokens": inputs["input_ids"].shape[1],
                              "completion_tokens": ids.numel()}}
        self._append_jsonl(self.api_path, {"task_id": task_id, "stage": stage, "requested_at": study.now(),
            "image_id": stimulus.image_id, "image_sha256": stimulus.sha256, "seed": seed,
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "model_id_requested": model.model_id, "request_parameters": generation,
            "response": response})
        return response


def run(model_name, method, image_dir, max_seconds, checkpoint=None, development=False, prompt_id=None):
    spec = iat_prompts.prompt_spec(prompt_id) if prompt_id is not None else None
    if spec and development:
        raise ValueError("Prompt sweep is frozen IAT-12 only")
    prompt = spec["prompt"] if spec else study.IAT_PROMPT
    if model_name != "gemma3_12b" or method not in ("base", "orpo"):
        raise ValueError("New evaluation is Gemma 3 12B base/ORPO-only")
    if spec:
        iat_prompts.output_location(study.ROOT, method, prompt_id)
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if method not in ("base", "sft", "orpo", "kto") or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Explicit valid condition and runtime limit required")
    settings = native_train.MODELS[model_name]
    deadline = time.monotonic() + max_seconds
    dataset_config = study.dataset_config()
    if not dataset_config:
        raise ValueError("Active curated dataset required for benchmark contract")
    if method != "base":
        if checkpoint is None:
            raise ValueError("Adapter benchmark requires a completed training checkpoint")
        receipt = json.loads((checkpoint.parent / "run.json").read_text(encoding="utf-8"))
        expected_train = dataset_config["split_counts"]["train"]
        expected_dev = dataset_config["split_counts"]["development"]
        expected_examples = dataset_config.get("example_split_counts", dataset_config["split_counts"])
        expected_updates = 3 * math.ceil(expected_examples["train"] / 4)
        if (receipt["status"] != "training_complete_evaluation_pending" or receipt["completed_epochs"] != 3
                or receipt["optimizer_updates"] != receipt["planned_optimizer_updates"]
                or receipt["optimizer_updates"] != expected_updates
                or receipt["training_images"] != expected_train
                or receipt["development_images"] != expected_dev
                or receipt["manifest_sha256"] != dataset_config["manifest_sha256"]
                or receipt["targets_sha256"] != study.sha(study.REPO / dataset_config["targets_path"])
                or receipt["method"] != method or receipt["model"] != settings["model"]
                or receipt["weight_and_processor_revision"] != settings["revision"]
                or checkpoint.name != "epoch_3"):
            raise ValueError("Incomplete or mismatched adapter cannot be evaluated as a study result")
        expected = receipt["checkpoints"][-1]["files"]
        native_train.verify_checkpoint_inventory(checkpoint, expected)
        if spec:
            iat_prompts.checkpoint_identity(checkpoint)
    elif checkpoint is not None:
        raise ValueError("Untouched base must not load an adapter")
    processor = AutoProcessor.from_pretrained(settings["model"], revision=settings["revision"])
    native = AutoModelForImageTextToText.from_pretrained(settings["model"], revision=settings["revision"],
        dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    if checkpoint is not None:
        from peft import PeftModel
        native = PeftModel.from_pretrained(native, checkpoint)
    native.eval()
    key = model_name + "_" + method + "_native_balanced"
    if development:
        key += "_development"
    if spec:
        _, key = iat_prompts.output_location(study.ROOT, method, prompt_id)
    model = replace(benchmark.fw.model_spec(settings["model"], 1, key=key,
        display_name=benchmark.MODELS[model_name][1] + " (native " + method + ")",
        base_model=settings["model"] + "@" + settings["revision"]), provider_endpoint="prime/native-transformers")
    config = benchmark.fw.run_config(1)
    stimuli = development_stimuli(dataset_config) if development else benchmark.fw.load_stimuli(
        study.REPO / "2026-09-11/data/iat12/manifest.csv", image_dir=image_dir)
    if development:
        config = replace(config, benchmark_version="iat-development-v3-12x50")
    if spec:
        config = replace(config, benchmark_version=iat_prompts.benchmark_version(method, prompt_id))
    output_root = study.ROOT / ("results/native_development" if development else "results/native_benchmarks")
    if spec:
        output_root, key = iat_prompts.output_location(study.ROOT, method, prompt_id)
    runner = NativeRunner(native, processor, model_name, deadline, key, config, [model], stimuli,
                          output_root=output_root, prompt=prompt, prompt_id=prompt_id)
    records = benchmark.fw.dedupe_records(benchmark.fw._jsonl_records(runner.records_path))
    contract = {"provider": "prime/native-transformers", "logical_base_model": model.api_revision_hint,
                "model": asdict(model), "config": asdict(config), "prompt": prompt,
                "weight_and_processor_revision": settings["revision"],
                "adapter_sha256": study.sha(checkpoint / "adapter_model.safetensors") if checkpoint else None,
                "checkpoint_selection": "predeclared_final_epoch_3" if checkpoint else "untouched_base",
                "dataset_id": dataset_config["dataset_id"],
                "pilot_policy_sha256": dataset_config.get("pilot_policy_sha256"),
                "precision": "bfloat16_unquantized", "attention_implementation": "sdpa",
                "versions": {name: native_train.version(name) for name in ("torch", "transformers", "peft", "accelerate")},
                "stimuli": [{"image_id": s.image_id, "filename": s.filename, "race": s.race,
                             "gender": s.gender, "sha256": s.sha256} for s in stimuli],
                "native_and_fireworks_numerical_equivalence_claimed": False}
    if spec:
        contract.update(spec)
    manifest_path = runner.output_dir / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["contract"] != contract:
        raise ValueError("Existing native benchmark contract differs")
    study.save_json(manifest_path, {"contract": contract, "expected_responses": 600,
                                   "recorded_responses": len(records), "status": "running"})
    done = benchmark.fw.completed_task_ids(records)
    try:
        for iteration in range(1, 51):
            for index, stimulus in enumerate(stimuli, start=1):
                if deterministic_task_id(model, stimulus, iteration, config.benchmark_version) not in done:
                    runner._run_task(model, stimulus, iteration, index)
            if iteration % 5 == 0:
                print(json.dumps({"condition": key, "iterations_completed": iteration, "expected_iterations": 50}), flush=True)
    except BaseException as exc:
        records = benchmark.fw.dedupe_records(benchmark.fw._jsonl_records(runner.records_path))
        study.save_json(manifest_path, {"contract": contract, "expected_responses": 600,
            "recorded_responses": len(records), "status": "incomplete", "error_type": type(exc).__name__})
        raise
    records = benchmark.fw.dedupe_records(benchmark.fw._jsonl_records(runner.records_path))
    if len(records) != 600 or len(benchmark.fw.completed_task_ids(records)) != 600:
        raise ValueError("Native benchmark is incomplete")
    benchmark.fw._write_records_csv(runner.output_dir / "responses.csv", records)
    study.save_json(manifest_path, {"contract": contract, "expected_responses": 600,
                                   "recorded_responses": len(records), "status": "complete"})
    if not spec:
        analysis = runner.output_dir / "analysis"
        analysis.mkdir(exist_ok=True)
        benchmark.fw.write_iat_analysis(records, [model], analysis,
                                       bootstrap_samples=config.bootstrap_samples, bootstrap_seed=config.base_seed)
    print(json.dumps({"condition": key, "status": "complete", "responses": 600}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gemma3_12b",), required=True)
    parser.add_argument("--method", choices=("base", "orpo"), required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--prompt-id", choices=iat_prompts.PROMPT_IDS)
    args = parser.parse_args()
    run(args.model, args.method, args.image_dir, args.max_seconds, args.checkpoint,
        args.development, args.prompt_id)
