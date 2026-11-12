from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodel_bias.analysis import write_iat_analysis
from multimodel_bias.constants import BENCHMARK_VERSION, IAT_PROMPT
from multimodel_bias.iat import (
    IATBenchmarkRunner,
    IATRunConfig,
    ModelSpec,
    Stimulus,
    _jsonl_records,
    _safe_response_snapshot,
    _usage,
    _write_records_csv,
    dedupe_records,
    deterministic_task_id,
    load_stimuli,
    utc_now,
)
from multimodel_bias.openrouter import OpenRouterClient

BASE_MODEL = "accounts/fireworks/models/gemma-4-26b-a4b-it"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def fireworks_payload(
    model_id: str,
    image_data_url: str,
    config: IATRunConfig,
    seed: int,
    *,
    fallback: bool,
) -> dict[str, Any]:
    return {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": IAT_PROMPT},
                ],
            }
        ],
        "max_tokens": config.fallback_max_tokens if fallback else 30,
        "seed": seed,
        "temperature": 0.0 if fallback else config.temperature,
        "top_p": config.top_p,
        "reasoning_effort": "none",
    }


class FireworksRunner(IATBenchmarkRunner):
    def _call(
        self,
        *,
        task_id: str,
        stage: str,
        model: ModelSpec,
        stimulus: Stimulus,
        seed: int,
        fallback: bool,
    ) -> dict[str, Any]:
        self._wait_for_model_slot(model)
        payload = fireworks_payload(
            model.model_id,
            self._data_urls[stimulus.image_id],
            self.config,
            seed,
            fallback=fallback,
        )
        response = self.client.chat_completion(payload)
        response.setdefault("provider", "fireworks")
        self._append_jsonl(
            self.api_path,
            {
                "task_id": task_id,
                "stage": stage,
                "requested_at": utc_now(),
                "model_id_requested": model.model_id,
                "image_id": stimulus.image_id,
                "image_sha256": stimulus.sha256,
                "prompt_sha256": hashlib.sha256(IAT_PROMPT.encode()).hexdigest(),
                "seed": seed,
                "request_parameters": {
                    key: value for key, value in payload.items() if key != "messages"
                },
                "response": _safe_response_snapshot(response),
            },
        )
        self._reserve_response_cost(_usage(response)["cost_usd"])
        return response


def model_spec(model_id: str, workers: int, *, key: str = "gemma4_26b_a4b_fireworks",
               display_name: str = "Gemma 4 26B A4B IT (Fireworks base)",
               base_model: str = BASE_MODEL) -> ModelSpec:
    return ModelSpec(
        key=key,
        display_name=display_name,
        model_id=model_id,
        api_revision_hint=base_model,
        release_date="",
        weight_access="open-weight",
        provider_endpoint="fireworks/on-demand" if "/deployments/" in model_id else "fireworks/serverless",
        max_concurrency=workers,
        supports_temperature=True,
        supports_top_p=True,
        reasoning_effort="none",
        max_tokens=30,
    )


def run_config(workers: int) -> IATRunConfig:
    return IATRunConfig(
        benchmark_version=BENCHMARK_VERSION,
        repeats=50,
        temperature=1.0,
        top_p=1.0,
        base_seed=17_082_026,
        fallback_enabled=True,
        fallback_max_tokens=30,
        max_workers=workers,
        minimum_request_interval_seconds=0.0,
        max_run_cost_usd=1_000_000.0,
        minimum_account_balance_usd=0.0,
        bootstrap_samples=100_000,
        provider={},
    )


def write_manifest(
    path: Path,
    model: ModelSpec,
    config: IATRunConfig,
    stimuli: list[Stimulus],
    records: list[dict[str, Any]],
) -> None:
    contract = {
        "provider": "fireworks",
        "logical_base_model": model.api_revision_hint,
        "deployment_model": model.model_id,
        "model": asdict(model),
        "prompt": IAT_PROMPT,
        "prompt_sha256": hashlib.sha256(IAT_PROMPT.encode()).hexdigest(),
        "config": asdict(config),
        "stimuli": [
            {
                "image_id": item.image_id,
                "filename": item.filename,
                "race": item.race,
                "gender": item.gender,
                "sha256": item.sha256,
            }
            for item in stimuli
        ],
    }
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if existing and existing.get("contract") != contract:
        raise ValueError(f"Run contract differs from existing {path}; choose a new --run-id")
    payload = {
        "created_at": existing.get("created_at", utc_now()),
        "updated_at": utc_now(),
        "contract": contract,
        "expected_responses": config.repeats * len(stimuli),
        "recorded_responses": len(records),
        "billing_note": "API usage costs may be absent; dedicated GPU charges require separate billing records.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def completed_task_ids(records: list[dict[str, Any]]) -> set[str]:
    return {
        str(record["task_id"])
        for record in records
        if record.get("status") == "completed"
        or (
            record.get("status") == "invalid"
            and record.get("response_source") not in {"api_error", "budget_stopped"}
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a Fireworks model on the unchanged 61-word/12-image benchmark."
    )
    parser.add_argument("action", choices=("dry-run", "smoke", "run"))
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("FIREWORKS_MODEL", ""))
    parser.add_argument("--run-id", default="gemma4_26b_a4b_fireworks_no_reasoning")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model-key", default="gemma4_26b_a4b_fireworks")
    parser.add_argument("--display-name", default="Gemma 4 26B A4B IT (Fireworks base)")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/fireworks")
    args = parser.parse_args()

    if not args.model.startswith("accounts/") or not any(part in args.model for part in ("/models/", "/deployments/")):
        parser.error(
            "set --model to accounts/<account>/models/<model> or accounts/<account>/deployments/<deployment>"
        )
    if args.base_model != BASE_MODEL and args.model_key == "gemma4_26b_a4b_fireworks":
        parser.error("provide --model-key and --display-name when benchmarking another base model")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")

    manifest_path = ROOT / "data" / "iat12" / "manifest.csv"
    stimuli = load_stimuli(manifest_path, image_dir=args.image_dir)
    expected_names = {item.filename for item in stimuli}
    actual_names = {
        path.name
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    }
    if actual_names != expected_names:
        raise ValueError(
            f"Image directory must contain exactly the 12 manifest files; "
            f"missing={sorted(expected_names - actual_names)}, extra={sorted(actual_names - expected_names)}"
        )

    model = model_spec(args.model, args.workers, key=args.model_key,
                       display_name=args.display_name, base_model=args.base_model)
    config = run_config(args.workers)
    output_root = args.output_root
    output_dir = output_root / args.run_id
    records_path = output_dir / "responses.jsonl"
    records = dedupe_records(_jsonl_records(records_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_output = output_dir / "run_manifest.json"
    write_manifest(manifest_output, model, config, stimuli, records)

    iterations = range(1, 2 if args.action == "smoke" else config.repeats + 1)
    done = completed_task_ids(records)
    tasks = [
        (stimulus, iteration, image_index)
        for iteration in iterations
        for image_index, stimulus in enumerate(stimuli, start=1)
        if deterministic_task_id(model, stimulus, iteration, config.benchmark_version) not in done
    ]
    print(
        f"Validated {len(stimuli)} images; {len(tasks)} calls pending "
        f"({config.repeats * len(stimuli)} in the full benchmark)."
    )
    if args.action == "dry-run":
        return

    api_key = os.environ.get("FIREWORKS_API_KEY", "")
    if not api_key:
        parser.error("set FIREWORKS_API_KEY in the runtime secret manager")
    client = OpenRouterClient(
        api_key,
        app_name="urap-gemma4-base-benchmark",
        base_url=FIREWORKS_BASE_URL,
    )
    runner = FireworksRunner(
        client,
        args.run_id,
        config,
        [model],
        stimuli,
        output_root=output_root,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(runner._run_task_limited, model, stimulus, iteration, image_index)
            for stimulus, iteration, image_index in tasks
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            if completed % 12 == 0 or completed == len(futures):
                print(f"Completed {completed}/{len(futures)} pending calls.")

    records = dedupe_records(_jsonl_records(runner.records_path))
    _write_records_csv(output_dir / "responses.csv", records)
    write_manifest(manifest_output, model, config, stimuli, records)
    if args.action == "run" and len(completed_task_ids(records)) == 600:
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        write_iat_analysis(
            records,
            [model],
            analysis_dir,
            bootstrap_samples=config.bootstrap_samples,
            bootstrap_seed=config.base_seed,
        )
        print(f"Benchmark and evaluation artifacts are complete in {output_dir}")
    elif args.action == "run":
        raise RuntimeError("Benchmark has unresolved technical failures; raw responses were retained")


if __name__ == "__main__":
    main()
