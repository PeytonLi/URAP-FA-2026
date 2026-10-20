import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "2026-09-11" / "src"))
sys.path.insert(0, str(ROOT))

import native_train
import study
from multimodel_bias import bold, decodingtrust, visbias
from multimodel_bias.iat import ModelSpec
from multimodel_bias.text_runner import TextBenchmarkRunner, TextRunConfig
from multimodel_bias.visbias import VisBiasRunConfig, VisBiasRunner

VISBIAS_INPUT_DIR = REPO / "2026-09-05" / "data" / "frozen" / "visbias"
OUT_ROOT = ROOT / "results" / "extra_benchmarks"


class LocalGemmaClient:
    """Duck-typed stand-in for OpenRouterClient: same response shape, local weights."""

    timeout_seconds = 0
    max_retries = 0

    def __init__(self, native_model, processor, model_id, revision):
        self.model = native_model
        self.processor = processor
        self.model_id = model_id
        self.revision = revision

    def credits(self):
        return {"remaining": 10 ** 9, "total": 10 ** 9}

    def key_metadata(self):
        return {"source": "prime/native-transformers"}

    def model_catalog(self):
        return [{"id": self.model_id, "name": self.model_id,
                 "canonical_slug": self.revision, "created": 0,
                 "context_length": 131072, "architecture": {},
                 "pricing": {}, "supported_parameters": []}]

    def chat_completion(self, payload):
        import torch
        from transformers import set_seed
        content = payload["messages"][-1]["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        images = []
        parts = []
        for part in content:
            if part.get("type") == "image_url":
                from PIL import Image
                data_url = part["image_url"]["url"]
                encoded = data_url.split(",", 1)[1]
                image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                images.append(image)
                parts.append({"type": "image"})
            else:
                parts.append({"type": "text", "text": part.get("text", "")})
        messages = payload["messages"][:-1] + [{"role": "user", "content": parts}]
        options = {"tokenize": True, "return_dict": True, "return_tensors": "pt",
                   "add_generation_prompt": True}
        if images:
            options["images"] = images
        inputs = self.processor.apply_chat_template(messages, **options)
        inputs = {k: v.to(self.model.device, dtype=self.model.dtype if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}
        seed = int(payload.get("seed") or 0)
        set_seed(seed)
        temperature = payload.get("temperature")
        generation = {"max_new_tokens": int(payload["max_tokens"]), "use_cache": True}
        if temperature:
            generation.update(do_sample=True, temperature=float(temperature),
                              top_p=float(payload.get("top_p", 1.0)), top_k=0)
        else:
            generation["do_sample"] = False
        with torch.inference_mode():
            ids = self.model.generate(**inputs, **generation)[0, inputs["input_ids"].shape[1]:]
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        return {"model": self.model_id, "provider": "prime/native-transformers",
                "choices": [{"message": {"content": text},
                             "finish_reason": "length" if ids.numel() == int(payload["max_tokens"]) else "stop"}],
                "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]),
                          "completion_tokens": int(ids.numel()), "cost": 0.0}}


def load_native(model_name, checkpoint):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    settings = native_train.MODELS[model_name]
    processor = AutoProcessor.from_pretrained(settings["model"], revision=settings["revision"])
    native = AutoModelForImageTextToText.from_pretrained(
        settings["model"], revision=settings["revision"], dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa")
    if checkpoint is not None:
        from peft import PeftModel
        native = PeftModel.from_pretrained(native, checkpoint)
    native.eval()
    return native, processor, settings


def model_spec(settings, method):
    return ModelSpec(
        key="gemma3_12b_" + method + "_native",
        display_name="Gemma 3 12B IT (native " + method + ")",
        model_id=settings["model"], api_revision_hint=settings["revision"],
        release_date="", weight_access="local-weights",
        provider_endpoint="prime/native-transformers", max_concurrency=1,
        supports_temperature=True, supports_top_p=True,
        reasoning_effort=None, max_tokens=512)


def text_config(version, repeats, temperature, seed):
    return TextRunConfig(
        benchmark_version=version, repeats=repeats, temperature=temperature, top_p=1.0,
        base_seed=seed, max_workers=1, minimum_request_interval_seconds=0.0,
        max_run_cost_usd=10 ** 9, minimum_account_balance_usd=0.0,
        provider={"require_parameters": True}, retry_empty_responses=False)


def records_of(output_dir):
    path = output_dir / "responses.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_decodingtrust(client, spec, out_dir):
    tasks = decodingtrust.load_race_tasks(16, 260903)
    runner = TextBenchmarkRunner(client, "decodingtrust_native_" + spec.key,
                                 text_config("decodingtrust-race-v1", 1, 1.0, 260903),
                                 [spec], tasks, out_dir,
                                 source_manifest=decodingtrust.source_manifest())
    runner.run()
    records = records_of(out_dir)
    decodingtrust.analyze(records, out_dir, [(spec.key, spec.display_name)])
    return {"tasks": len(tasks), "records": len(records)}


def run_bold(client, spec, out_dir):
    tasks = bold.load_tasks(300, 260903)
    runner = TextBenchmarkRunner(client, "bold_native_" + spec.key,
                                 text_config("bold-v1", 1, 1.0, 260903),
                                 [spec], tasks, out_dir,
                                 source_manifest=bold.source_manifest())
    runner.run()
    records = records_of(out_dir)
    bold.score_records(records, out_dir, [(spec.key, spec.display_name)])
    return {"tasks": len(tasks), "records": len(records)}


def run_visbias(client, spec, out_dir):
    input_dir = VISBIAS_INPUT_DIR.resolve()
    tasks = visbias.load_tasks(input_dir)
    config = VisBiasRunConfig(
        benchmark_version="visbias-race-v1", temperature=1.0, top_p=1.0,
        base_seed=260903, max_description_tokens=512, max_comparison_tokens=16,
        max_workers=1, minimum_request_interval_seconds=0.0,
        max_run_cost_usd=10 ** 9, minimum_account_balance_usd=0.0,
        bootstrap_samples=0, provider={"require_parameters": True})
    runner = VisBiasRunner(client, "visbias_native_" + spec.key, config, [spec], tasks,
                           out_dir.parent)
    runner.run()
    return {"tasks": len(tasks), "records": len(records_of(runner.output_dir))}


def run(model_name, method, checkpoint, benchmarks, deadline_seconds):
    native, processor, settings = load_native(model_name, checkpoint)
    client = LocalGemmaClient(native, processor, settings["model"], settings["revision"])
    spec = model_spec(settings, method)
    results = {}
    deadline = time.monotonic() + deadline_seconds
    for name in benchmarks:
        if time.monotonic() >= deadline:
            results["deadline_exceeded_before"] = name
            break
        out_dir = OUT_ROOT / name / spec.key
        out_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        if name == "decodingtrust":
            results[name] = run_decodingtrust(client, spec, out_dir)
        elif name == "bold":
            results[name] = run_bold(client, spec, out_dir)
        elif name == "visbias":
            results[name] = run_visbias(client, spec, out_dir)
        results[name]["seconds"] = round(time.monotonic() - started, 1)
    study.save_json(OUT_ROOT / f"{spec.key}_summary.json",
                    {"model": settings["model"], "revision": settings["revision"],
                     "method": method, "checkpoint": str(checkpoint) if checkpoint else None,
                     "adapter_sha256": study.sha(checkpoint / "adapter_model.safetensors") if checkpoint else None,
                     "benchmarks": results, "finished_at": study.now()})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(native_train.MODELS))
    parser.add_argument("--method", required=True, choices=("base", "orpo"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--benchmarks", default="decodingtrust,bold,visbias")
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.method != "base" and args.checkpoint is None:
        raise ValueError("Adapter benchmark requires the epoch_3 checkpoint")
    benchmarks = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    unknown = set(benchmarks) - {"decodingtrust", "bold", "visbias"}
    if unknown:
        raise ValueError(f"Unknown benchmarks: {sorted(unknown)}")
    print(json.dumps(run(args.model, args.method, args.checkpoint, benchmarks,
                         args.max_seconds), indent=2))


if __name__ == "__main__":
    main()
