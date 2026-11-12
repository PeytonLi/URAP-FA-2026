#!/usr/bin/env python3
"""Bounded, resumable VisBias12 generation with pinned routes and no fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, sha, utcnow, write_csv, write_json
from preflight_access import secret

sys.path.insert(0, str(ROOT / "src"))
from multimodel_bias.artifacts import sanitize_for_artifact
from multimodel_bias.iat import image_data_url, load_model_specs, validated_model_snapshot
from multimodel_bias.openrouter import OpenRouterClient
from multimodel_bias.visbias import _message_text, _parse_comparison, _seed, load_tasks


class Budget:
    def __init__(self, client, initial_usage, ceiling=5.0, reserve=10.0):
        self.client, self.initial_usage = client, initial_usage
        self.ceiling, self.reserve = ceiling, reserve
        self.inflight = 0.0
        self.lock = threading.Lock()

    def acquire(self, upper_bound):
        with self.lock:
            credits = self.client.credits()
            available = min(
                self.ceiling - (credits["total_usage"] - self.initial_usage),
                credits["remaining"] - self.reserve,
            )
            if available < self.inflight + upper_bound:
                raise RuntimeError("Budget or required credit reserve would be crossed")
            self.inflight += upper_bound

    def release(self, upper_bound):
        with self.lock:
            self.inflight -= upper_bound


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="One description and comparison/model")
    parser.add_argument("--models", default="all")
    args = parser.parse_args()
    client = OpenRouterClient(
        secret("openrouter-api-key"),
        timeout_seconds=90,
        max_retries=0,
        app_name="research-evidence-september5",
    )
    tasks = load_tasks(ROOT / "data/frozen/visbias12")
    _, models = load_model_specs(ROOT / "config/models_aug31.json")
    if args.models != "all":
        models = [m for m in models if m.key in args.models.split(",")]
    destination = ROOT / "results/hosted_visbias12"
    destination.mkdir(parents=True, exist_ok=True)
    starting = json.loads((ROOT / "results/preflight/access.json").read_text())["openrouter"]
    budget = Budget(client, starting["total_usage"])
    image_urls = {p: image_data_url(p) for t in tasks for p in t["image_paths"]}
    probe_indices = {
        next(i for i, t in enumerate(tasks) if t["task_kind"] == kind)
        for kind in ("description", "comparison")
    }

    def run_model(model):
        folder = destination / model.key
        folder.mkdir(parents=True, exist_ok=True)
        manifest = {
            "model": model.__dict__,
            "started_at": utcnow(),
            "task_contract_sha256": sha(ROOT / "data/frozen/visbias12/protocol.json"),
            "status": "checking",
            "probe": args.probe,
            "planned_tasks": len(tasks),
            "provider_fallback": False,
        }
        try:
            snapshot = validated_model_snapshot(client, [model])[0]
        except Exception as exc:
            manifest.update(status="route_unavailable", error=str(exc))
            write_json(folder / "manifest.json", manifest)
            print(f"{model.key}: route unavailable; no paid calls", flush=True)
            return
        manifest["catalog_snapshot"] = snapshot
        pricing = snapshot["selected_endpoint"]["pricing"]
        # A conservative reservation covers two small images, prompt, and output ceiling.
        ceiling = max(220, 4096 if model.key == "step37_flash" else model.max_tokens)
        bound = (
            20000 * float(pricing.get("prompt") or 0)
            + ceiling * float(pricing.get("completion") or 0)
            + 0.02
        )
        record_path = folder / "responses.jsonl"
        existing = (
            [json.loads(line) for line in record_path.read_text().splitlines()]
            if record_path.exists()
            else []
        )
        attempted = {r["task_key"] for r in existing}
        errors = 0
        write_json(folder / "manifest.json", manifest)
        for index, task in enumerate(tasks):
            if task["task_key"] in attempted or (args.probe and index not in probe_indices):
                continue
            tokens = 220 if task["task_kind"] == "description" else 32
            if model.reasoning_effort not in {None, "none"}:
                tokens = max(tokens, 4096 if model.key == "step37_flash" else model.max_tokens)
            content = [{"type": "text", "text": task["prompt"]}]
            for position, path in enumerate(task["image_paths"]):
                if task["task_kind"] == "comparison":
                    content.append({"type": "text", "text": f"Image {'AB'[position]}:"})
                content.append({"type": "image_url", "image_url": {"url": image_urls[path]}})
            seed = _seed(31082026, model.key, index + 1)
            payload = {
                "model": model.model_id,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": tokens,
                "seed": seed,
                "provider": {
                    "order": [model.provider_endpoint],
                    "only": [model.provider_endpoint],
                    "allow_fallbacks": False,
                    "data_collection": "deny",
                    "zdr": True,
                },
            }
            if model.supports_temperature:
                payload["temperature"] = 0.0
            if model.supports_top_p:
                payload["top_p"] = 1.0
            if model.reasoning_effort is not None:
                payload["reasoning"] = {"effort": model.reasoning_effort, "exclude": True}
            try:
                budget.acquire(bound)
            except Exception as exc:
                manifest.update(status="budget_stop", error=str(exc))
                break
            start = time.monotonic()
            row = {k: v for k, v in task.items() if k not in {"image_paths", "image_hashes"}}
            row.update(
                model_key=model.key,
                model_display_name=model.display_name,
                model_id_requested=model.model_id,
                provider_requested=model.provider_endpoint,
                api_revision_hint=model.api_revision_hint,
                dataset="neutral12",
                image_hashes="|".join(task["image_hashes"]),
                seed=seed,
                temperature_requested=0.0,
                temperature_effective=payload.get("temperature"),
                top_p_requested=1.0,
                max_tokens_requested=tokens,
                created_at=utcnow(),
                task_id=hashlib.sha256(f"{model.key}|{task['task_key']}".encode()).hexdigest(),
                status="api_error",
                response="",
                cost_usd=0.0,
            )
            try:
                response = client.chat_completion(payload)
                usage = response.get("usage") or {}
                text = _message_text(response)
                row.update(
                    response=text,
                    api_model_returned=response.get("model"),
                    provider=response.get("provider"),
                    cost_usd=usage.get("cost", 0.0),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    generation_id=response.get("id"),
                    finish_reason=(response.get("choices") or [{}])[0].get("finish_reason"),
                    request_parameters={k: v for k, v in payload.items() if k != "messages"},
                )
                if response.get("model") not in {model.model_id, model.api_revision_hint}:
                    raise ValueError("Returned model differs from frozen model identity")
                expected_provider = snapshot["selected_endpoint"]["provider_name"]
                if response.get("provider") != expected_provider:
                    raise ValueError(f"Provider drift: expected {expected_provider}")
                answer = _parse_comparison(text) if task["task_kind"] == "comparison" else ""
                row.update(
                    parsed_answer=answer,
                    selected_race=(
                        task.get("race_a")
                        if answer == "A"
                        else task.get("race_b")
                        if answer == "B"
                        else ""
                    ),
                )
                valid = bool(text) and answer != "UNCLASSIFIED" and row["finish_reason"] != "length"
                row["status"] = "completed" if valid else "invalid"
                errors = 0 if valid else errors + 1
            except Exception as exc:
                row["error_type"] = type(exc).__name__
                row["error_message"] = str(exc)[:400]
                errors += 1
                if isinstance(exc, ValueError):
                    errors = 3
            finally:
                budget.release(bound)
            row["latency_seconds"] = time.monotonic() - start
            row = sanitize_for_artifact(row)
            with record_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
            existing.append(row)
            write_csv(folder / "responses.csv", existing)
            print(f"{model.key}: {len(existing)}/36 {row['status']}", flush=True)
            if errors >= 3:
                manifest.update(status="systematic_error_stop")
                break
            time.sleep(1.0)
        if manifest["status"] == "checking":
            manifest["status"] = "complete" if len(existing) == len(tasks) else "probe_complete"
        manifest.update(
            finished_at=utcnow(),
            recorded_calls=len(existing),
            valid_responses=sum(r["status"] == "completed" for r in existing),
            response_cost_usd=sum(float(r.get("cost_usd") or 0) for r in existing),
        )
        write_json(folder / "manifest.json", manifest)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_model, models))
    rows = []
    for path in sorted(destination.glob("*/responses.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    write_csv(destination / "responses.csv", rows)
    final = client.credits()
    write_json(
        destination / "cost_summary.json",
        {
            "initial": starting,
            "final": final,
            "account_usage_delta_usd": final["total_usage"] - starting["total_usage"],
            "model_calls_recorded": len(rows),
            "finished_at": utcnow(),
        },
    )


if __name__ == "__main__":
    main()
