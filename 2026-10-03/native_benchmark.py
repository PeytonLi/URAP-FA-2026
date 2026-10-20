"""Matched native base/adapter benchmark using the existing IAT scorer/fallback."""
import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import benchmark
import native_train
import pilot
import study
from multimodel_bias.iat import IATBenchmarkRunner, deterministic_task_id


class NativeRunner(IATBenchmarkRunner):
    def __init__(self, native_model, processor, model_name, deadline, *args, **kwargs):
        super().__init__(None, *args, **kwargs)
        self.native_model, self.processor = native_model, processor
        self.model_name, self.deadline, self.inputs = model_name, deadline, {}

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
                                                    {"type": "text", "text": study.IAT_PROMPT}]}
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
            "prompt_sha256": hashlib.sha256(study.IAT_PROMPT.encode()).hexdigest(),
            "model_id_requested": model.model_id, "request_parameters": generation,
            "response": response})
        return response


def run(model_name, method, image_dir, max_seconds, checkpoint=None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if method not in ("base", "sft", "orpo") or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Explicit valid condition and runtime limit required")
    settings = native_train.MODELS[model_name]
    deadline = time.monotonic() + max_seconds
    if method != "base":
        if checkpoint is None:
            raise ValueError("Adapter benchmark requires a completed training checkpoint")
        receipt = json.loads((checkpoint.parent / "run.json").read_text(encoding="utf-8"))
        if (receipt["status"] != "training_complete_evaluation_pending" or receipt["completed_epochs"] != 3
                or receipt["optimizer_updates"] != receipt["planned_optimizer_updates"]
                or receipt["optimizer_updates"] != 507 or receipt.get("cohort") != "resolved-balanced"
                or receipt["training_images"] != 676 or receipt["development_images"] != 180
                or receipt["targets_sha256"] != study.sha(pilot.FROZEN)
                or receipt["pilot_policy_sha256"] != study.sha(pilot.POLICY)
                or receipt["method"] != method or receipt["model"] != settings["model"]
                or receipt["weight_and_processor_revision"] != settings["revision"]
                or checkpoint.name != "epoch_3"):
            raise ValueError("Incomplete or mismatched adapter cannot be evaluated as a study result")
        expected = receipt["checkpoints"][-1]["files"]
        if any(study.sha(checkpoint / name) != digest for name, digest in expected.items()):
            raise ValueError("Adapter/processor checkpoint checksum mismatch")
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
    model = replace(benchmark.fw.model_spec(settings["model"], 1, key=key,
        display_name=benchmark.MODELS[model_name][1] + " (native " + method + ")",
        base_model=settings["model"] + "@" + settings["revision"]), provider_endpoint="prime/native-transformers")
    config = benchmark.fw.run_config(1)
    stimuli = benchmark.fw.load_stimuli(study.REPO / "2026-09-11/data/iat12/manifest.csv", image_dir=image_dir)
    runner = NativeRunner(native, processor, model_name, deadline, key, config, [model], stimuli,
                          output_root=study.ROOT / "results/native_benchmarks")
    records = benchmark.fw.dedupe_records(benchmark.fw._jsonl_records(runner.records_path))
    contract = {"provider": "prime/native-transformers", "logical_base_model": model.api_revision_hint,
                "model": asdict(model), "config": asdict(config), "prompt": study.IAT_PROMPT,
                "weight_and_processor_revision": settings["revision"],
                "adapter_sha256": study.sha(checkpoint / "adapter_model.safetensors") if checkpoint else None,
                "checkpoint_selection": "predeclared_final_epoch_3" if checkpoint else "untouched_base",
                "dataset_id": json.loads(pilot.POLICY.read_text(encoding="utf-8"))["dataset_id"],
                "pilot_policy_sha256": study.sha(pilot.POLICY),
                "precision": "bfloat16_unquantized", "attention_implementation": "sdpa",
                "versions": {name: native_train.version(name) for name in ("torch", "transformers", "peft", "accelerate")},
                "stimuli": [{"image_id": s.image_id, "filename": s.filename, "race": s.race,
                             "gender": s.gender, "sha256": s.sha256} for s in stimuli],
                "native_and_fireworks_numerical_equivalence_claimed": False}
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
    # Reuse scorer, all-61-word reference ordering and blue/gold graph style.
    analysis = runner.output_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    benchmark.fw.write_iat_analysis(records, [model], analysis,
                                   bootstrap_samples=config.bootstrap_samples, bootstrap_seed=config.base_seed)
    print(json.dumps({"condition": key, "status": "complete", "responses": 600}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=native_train.MODELS, required=True)
    parser.add_argument("--method", choices=("base", "sft", "orpo"), required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-seconds", type=float, required=True)
    args = parser.parse_args()
    run(args.model, args.method, args.image_dir, args.max_seconds, args.checkpoint)
