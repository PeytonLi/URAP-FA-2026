"""Runs DecodingTrust/BOLD/VisBias on-pod against local Gemma weights/adapters.

The three benchmark modules (2026-09-11/src/multimodel_bias) were written for an
HTTP provider client. This driver supplies a local stand-in that implements the
same duck-typed surface (`chat_completion`, `credits`, `key_metadata`,
`model_catalog`) so generation runs on the pod's GPU instead.
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "2026-09-11" / "src"))
sys.path.insert(0, str(ROOT))


class LocalGemmaClient:
    timeout_seconds = 600
    max_retries = 0

    def __init__(self, model, processor, model_id, revision):
        self._model = model
        self._processor = processor
        self._model_id = model_id
        self._revision = revision

    def credits(self):
        return {"remaining": 1e9, "used": 0.0}

    def key_metadata(self):
        return {"label": "native-local", "limit": None}

    def model_catalog(self):
        return [{"id": self._model_id, "canonical_slug": self._revision,
                 "name": self._model_id, "created": 0, "context_length": 131072,
                 "architecture": {}, "pricing": {}, "supported_parameters": []}]

    def get(self, path):
        if path == "/credits":
            return self.credits()
        if path == "/key":
            return {"data": self.key_metadata()}
        if path == f"/models/{self._model_id}/endpoints":
            return {"data": {"endpoints": [{
                "name": "prime/native-transformers",
                "tag": "prime/native-transformers",
                "provider_name": "prime",
                "quantization": None,
                "context_length": 131072,
                "max_prompt_tokens": None,
                "max_completion_tokens": None,
                "pricing": {},
                "supported_parameters": [],
                "status": 0,
            }]}}
        raise ValueError(f"Local client does not implement GET {path}")

    def _decode_image(self, url):
        from PIL import Image
        header, _, payload = url.partition(",")
        if not header.startswith("data:") or ";base64" not in header:
            raise ValueError("Only base64 data URLs are supported locally")
        return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")

    def chat_completion(self, payload):
        import torch
        from transformers import set_seed
        images = []
        content = []
        for part in payload["messages"][-1]["content"]:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image_url":
                image = self._decode_image(part["image_url"]["url"])
                images.append(image)
                content.append({"type": "image"})
            else:
                raise ValueError(f"Unsupported content part {part['type']}")
        user = {"role": "user", "content": content}
        options = {"tokenize": True, "return_dict": True, "return_tensors": "pt",
                   "add_generation_prompt": True}
        inputs = self._processor.apply_chat_template(
            [user], images=images or None, **options)
        inputs = {k: v.to(self._model.device, dtype=self._model.dtype if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}
        set_seed(int(payload.get("seed") or 0))
        max_new = int(payload.get("max_tokens") or 512)
        generation = {"max_new_tokens": max_new, "do_sample": True, "use_cache": True,
                      "top_k": 0, "top_p": float(payload.get("top_p") or 1.0),
                      "temperature": float(payload.get("temperature") or 1.0)}
        with torch.inference_mode():
            ids = self._model.generate(**inputs, **generation)[0, inputs["input_ids"].shape[1]:]
        text = self._processor.tokenizer.decode(ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
        return {"model": payload["model"], "provider": "prime/native-transformers",
                "choices": [{"message": {"content": text},
                             "finish_reason": "length" if ids.numel() == max_new else "stop"}],
                "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]),
                          "completion_tokens": int(ids.numel()), "cost": 0.0}}


def load_native_model(model_name, checkpoint):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import native_train
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


def local_spec(settings, method, max_tokens):
    from multimodel_bias.iat import ModelSpec
    return ModelSpec(key=f"gemma3_12b_{method}_native", display_name=f"Gemma 3 12B IT (native {method})",
                     model_id=settings["model"], api_revision_hint=settings["revision"],
                     release_date="", weight_access="local", provider_endpoint="prime/native-transformers",
                     max_concurrency=1, supports_temperature=True, supports_top_p=True,
                     reasoning_effort=None, max_tokens=max_tokens)


def text_config(version, repeats=1, temperature=1.0, top_p=1.0, seed=174581):
    from multimodel_bias.text_runner import TextRunConfig
    return TextRunConfig(benchmark_version=version, repeats=repeats, temperature=temperature,
                         top_p=top_p, base_seed=seed, max_workers=1,
                         minimum_request_interval_seconds=0.0, max_run_cost_usd=1e9,
                         minimum_account_balance_usd=0.0, provider={})


def run_decodingtrust(client, spec, out_dir, samples_per_regime, seed):
    from multimodel_bias import decodingtrust
    from multimodel_bias.text_runner import TextBenchmarkRunner
    tasks = decodingtrust.load_tasks(samples_per_regime, seed)
    runner = TextBenchmarkRunner(client, f"dt-{spec.key}", text_config("decodingtrust-race-v1", seed=seed),
                                 [spec], tasks, out_dir,
                                 source_manifest=decodingtrust.source_manifest())
    result = runner.run()
    records = _jsonl(out_dir / "responses.jsonl")
    decodingtrust.analyze(records, out_dir, [(spec.key, spec.display_name)])
    return result


def run_bold(client, spec, out_dir, sample_count, seed):
    from multimodel_bias import bold
    from multimodel_bias.text_runner import TextBenchmarkRunner
    tasks = bold.load_tasks(sample_count, seed)
    runner = TextBenchmarkRunner(client, f"bold-{spec.key}", text_config("bold-v1", seed=seed),
                                 [spec], tasks, out_dir,
                                 source_manifest=bold.source_manifest())
    result = runner.run()
    records = _jsonl(out_dir / "responses.jsonl")
    bold.score_records(records, out_dir, [(spec.key, spec.display_name)])
    return result


def run_visbias(client, spec, out_dir, seed):
    from multimodel_bias import visbias
    # Resolved real path: task image_relative_path uses ../../private/... which
    # must resolve under 2026-09-05/data, not through the 09-11 symlink.
    tasks = visbias.load_tasks(
        (ROOT.parent / "2026-09-05" / "data" / "frozen" / "visbias").resolve())
    config = visbias.VisBiasRunConfig(benchmark_version="visbias-race-v1", temperature=1.0, top_p=1.0,
                                      base_seed=seed, max_description_tokens=256,
                                      max_comparison_tokens=8, max_workers=1,
                                      minimum_request_interval_seconds=0.0, max_run_cost_usd=1e9,
                                      minimum_account_balance_usd=0.0, provider={})
    runner = visbias.VisBiasRunner(client, f"visbias-{spec.key}", config, [spec], tasks, out_dir.parent)
    return runner.run()


def main():
    parser = argparse.ArgumentParser(description="Local DecodingTrust/BOLD/VisBias for Gemma-3")
    parser.add_argument("--model", default="gemma3_12b")
    parser.add_argument("--method", required=True, choices=["base", "orpo"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--benchmarks", default="decodingtrust,bold,visbias")
    parser.add_argument("--dt-samples-per-regime", type=int, default=200)
    parser.add_argument("--bold-sample-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=174_581)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results" / "extra_benchmarks")
    args = parser.parse_args()
    if args.method == "base" and args.checkpoint is not None:
        raise ValueError("base must not load an adapter")
    if args.method != "base" and args.checkpoint is None:
        raise ValueError("adapter condition needs --checkpoint")
    model, processor, settings = load_native_model(args.model, args.checkpoint)
    client = LocalGemmaClient(model, processor, settings["model"], settings["revision"])
    wanted = [name.strip() for name in args.benchmarks.split(",") if name.strip()]
    results = {}
    if "decodingtrust" in wanted:
        spec = local_spec(settings, args.method, 256)
        results["decodingtrust"] = run_decodingtrust(
            client, spec, args.out_root / "decodingtrust" / f"{spec.key}",
            args.dt_samples_per_regime, args.seed)
    if "bold" in wanted:
        spec = local_spec(settings, args.method, 256)
        results["bold"] = run_bold(client, spec, args.out_root / "bold" / f"{spec.key}",
                                   args.bold_sample_count, args.seed)
    if "visbias" in wanted:
        spec = local_spec(settings, args.method, 256)
        results["visbias"] = run_visbias(client, spec, args.out_root / "visbias" / f"{spec.key}",
                                         args.seed)
    print(json.dumps({"condition": f"{args.model}_{args.method}", "finished": sorted(results)}))


if __name__ == "__main__":
    main()


def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
