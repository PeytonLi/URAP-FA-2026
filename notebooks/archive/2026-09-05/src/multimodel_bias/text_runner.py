from __future__ import annotations

import csv
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import sanitize_for_artifact
from .iat import ModelSpec, utc_now, validated_model_snapshot
from .openrouter import OpenRouterClient, OpenRouterError


@dataclass(frozen=True)
class TextTask:
    key: str
    prompt: str
    system_prompt: str
    metadata: dict[str, Any]
    max_tokens: int


@dataclass(frozen=True)
class TextRunConfig:
    benchmark_version: str
    repeats: int
    temperature: float
    top_p: float
    base_seed: int
    max_workers: int
    minimum_request_interval_seconds: float
    max_run_cost_usd: float
    minimum_account_balance_usd: float
    provider: dict[str, Any]
    retry_empty_responses: bool = False


class TextRunBudgetExceeded(RuntimeError):
    pass


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return records


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fields = sorted({field for record in records for field in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def freeze_selected_prompts(path: Path, records: list[dict[str, Any]]) -> None:
    """Write selected prompts without changing the task set of an existing run."""
    response_artifacts = (path.parent / "responses.jsonl", path.parent / "responses.csv")
    if path.exists() and any(candidate.exists() for candidate in response_artifacts):
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
        existing_keys = [row.get("task_key", "") for row in existing]
        requested_keys = [str(row.get("task_key", "")) for row in records]
        if existing_keys != requested_keys:
            raise ValueError(
                f"Refusing to replace frozen prompts for existing run {path.parent.name}: "
                f"stored={len(existing_keys)}, requested={len(requested_keys)}. "
                "Use a new run ID for a different task set."
            )
    write_csv(path, records)


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first success; otherwise keep the latest attempt for each task."""
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = record["task_id"]
        current = selected.get(task_id)
        if current is None or current.get("status") != "completed":
            selected[task_id] = record
    return list(selected.values())


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        ).strip()
    return str(content).strip()


def _usage(response: dict[str, Any]) -> dict[str, int | float]:
    usage = response.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
        "cost_usd": float(usage.get("cost") or 0.0),
    }


def _task_id(benchmark_version: str, model: ModelSpec, task: TextTask, iteration: int) -> str:
    identity = f"{benchmark_version}|{model.model_id}|{task.key}|{iteration}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _task_seed(base_seed: int, task: TextTask, iteration: int) -> int:
    offset = int(hashlib.sha256(task.key.encode("utf-8")).hexdigest()[:8], 16)
    return (base_seed + iteration * 1_000_003 + offset) % 2_147_483_647


class TextBenchmarkRunner:
    def __init__(
        self,
        client: OpenRouterClient,
        run_id: str,
        config: TextRunConfig,
        models: list[ModelSpec],
        tasks: list[TextTask],
        output_dir: Path,
        *,
        source_manifest: dict[str, Any],
        model_universe: list[ModelSpec] | None = None,
        retry_existing_api_errors: bool = True,
    ) -> None:
        if config.repeats < 1 or config.max_workers < 1:
            raise ValueError("repeats and max_workers must be positive")
        if config.minimum_request_interval_seconds < 0:
            raise ValueError("minimum_request_interval_seconds must not be negative")
        if config.max_run_cost_usd <= 0:
            raise ValueError("max_run_cost_usd must be positive")
        if len({task.key for task in tasks}) != len(tasks):
            raise ValueError("Text task keys must be unique")
        reserved = {
            "task_id",
            "run_id",
            "model_key",
            "response",
            "status",
            "cost_usd",
        }
        for task in tasks:
            overlap = reserved.intersection(task.metadata)
            if overlap:
                raise ValueError(f"Task metadata uses reserved fields: {sorted(overlap)}")

        self.client = client
        self.run_id = run_id
        self.config = config
        self.models = models
        self.tasks = tasks
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_manifest = source_manifest
        self.model_universe = model_universe or models
        self.retry_existing_api_errors = retry_existing_api_errors
        self.records_path = output_dir / "responses.jsonl"
        self.api_path = output_dir / "api_responses.jsonl"
        self.sessions_path = output_dir / "run_sessions.jsonl"
        self._write_lock = threading.Lock()
        self._cost_lock = threading.Lock()
        self._rate_limit_lock = threading.Lock()
        self._invocation_cost = 0.0
        self._stop_requested = False
        self._model_next_request_at = {model.key: 0.0 for model in models}
        self._model_semaphores = {
            model.key: threading.BoundedSemaphore(model.max_concurrency) for model in models
        }

    def _append_jsonl(self, path: Path, value: dict[str, Any]) -> None:
        with self._write_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()

    def _reserve_cost(self, cost: float) -> None:
        with self._cost_lock:
            self._invocation_cost += cost
            if self._invocation_cost >= self.config.max_run_cost_usd:
                self._stop_requested = True

    def _check_budget(self) -> None:
        with self._cost_lock:
            if self._stop_requested:
                raise TextRunBudgetExceeded(
                    f"Run cost ceiling reached at ${self._invocation_cost:.6f}"
                )

    def _payload(
        self,
        model: ModelSpec,
        task: TextTask,
        seed: int,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if task.system_prompt:
            messages.append({"role": "system", "content": task.system_prompt})
        messages.append({"role": "user", "content": task.prompt})
        provider = dict(self.config.provider)
        provider.update({"order": [model.provider_endpoint], "allow_fallbacks": False})
        payload: dict[str, Any] = {
            "model": model.model_id,
            "messages": messages,
            "max_tokens": max(task.max_tokens, model.max_tokens),
            "seed": seed,
            "provider": provider,
        }
        if model.supports_temperature:
            payload["temperature"] = self.config.temperature
        if model.supports_top_p:
            payload["top_p"] = self.config.top_p
        if model.reasoning_effort is not None:
            payload["reasoning"] = {
                "effort": model.reasoning_effort,
                "exclude": True,
            }
        return payload

    def _run_task(
        self,
        model: ModelSpec,
        task: TextTask,
        iteration: int,
    ) -> dict[str, Any]:
        task_id = _task_id(self.config.benchmark_version, model, task, iteration)
        seed = _task_seed(self.config.base_seed, task, iteration)
        started = time.monotonic()
        response: dict[str, Any] = {}
        response_text = ""
        response_source = "api"
        error_type = ""
        error_status = 0
        error_message = ""
        try:
            self._check_budget()
            payload = self._payload(model, task, seed)
            response = self.client.chat_completion(payload)
            response_text = _response_text(response)
            usage = _usage(response)
            self._reserve_cost(float(usage["cost_usd"]))
            self._append_jsonl(
                self.api_path,
                {
                    "task_id": task_id,
                    "requested_at": utc_now(),
                    "model_id_requested": model.model_id,
                    "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
                    "seed": seed,
                    "request_parameters": {
                        key: value for key, value in payload.items() if key != "messages"
                    },
                    "response": sanitize_for_artifact(
                        {key: value for key, value in response.items() if key != "_client"}
                    ),
                },
            )
        except TextRunBudgetExceeded as exc:
            response_source = "budget_stopped"
            error_type = type(exc).__name__
            error_message = str(exc)
        except OpenRouterError as exc:
            response_source = "api_error"
            error_type = type(exc).__name__
            error_status = exc.failure.status_code
            error_message = exc.failure.message
            self._append_jsonl(
                self.api_path,
                {
                    "task_id": task_id,
                    "requested_at": utc_now(),
                    "model_id_requested": model.model_id,
                    "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
                    "seed": seed,
                    "error": sanitize_for_artifact(exc.failure.payload),
                },
            )

        usage = _usage(response)
        choices = response.get("choices") or []
        finish_reason = str(choices[0].get("finish_reason") or "") if choices else ""
        client_metadata = response.get("_client") or {}
        record: dict[str, Any] = {
            "task_id": task_id,
            "benchmark_version": self.config.benchmark_version,
            "run_id": self.run_id,
            "created_at": utc_now(),
            "status": "completed" if response_text else "invalid",
            "response_source": response_source,
            "model_key": model.key,
            "model_display_name": model.display_name,
            "model_id_requested": model.model_id,
            "api_model_returned": str(response.get("model") or ""),
            "provider": str(response.get("provider") or ""),
            "iteration": iteration,
            "seed": seed,
            "system_prompt": task.system_prompt,
            "prompt": task.prompt,
            "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
            "response": response_text,
            "finish_reason": finish_reason,
            "temperature_requested": self.config.temperature,
            "temperature_effective": (
                self.config.temperature if model.supports_temperature else "provider_default"
            ),
            "top_p_requested": self.config.top_p,
            "top_p_effective": (self.config.top_p if model.supports_top_p else "provider_default"),
            "max_tokens": max(task.max_tokens, model.max_tokens),
            "reasoning_effort": model.reasoning_effort or "not_applicable",
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "reasoning_tokens": usage["reasoning_tokens"],
            "cost_usd": usage["cost_usd"],
            "latency_seconds": time.monotonic() - started,
            "retry_count": int(client_metadata.get("retry_count") or 0),
            "error_type": error_type,
            "error_status": error_status,
            "error_message": error_message,
            **task.metadata,
        }
        self._append_jsonl(self.records_path, record)
        return record

    def _run_task_limited(
        self,
        model: ModelSpec,
        task: TextTask,
        iteration: int,
    ) -> dict[str, Any]:
        with self._model_semaphores[model.key]:
            with self._rate_limit_lock:
                now = time.monotonic()
                request_at = max(now, self._model_next_request_at[model.key])
                self._model_next_request_at[model.key] = (
                    request_at + self.config.minimum_request_interval_seconds
                )
            delay = request_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            return self._run_task(model, task, iteration)

    def _model_snapshot(self) -> list[dict[str, Any]]:
        return validated_model_snapshot(self.client, self.models)

    def run(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        manifest_path = self.output_dir / "run_manifest.json"
        previous_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )
        invocation_started_at = utc_now()
        credits_before = self.client.credits()
        if credits_before["remaining"] < self.config.minimum_account_balance_usd:
            raise TextRunBudgetExceeded(
                f"Account balance ${credits_before['remaining']:.2f} is below reserve "
                f"${self.config.minimum_account_balance_usd:.2f}"
            )

        existing = _jsonl_records(self.records_path)
        if not existing:
            # Shared repositories omit raw logs. A paid rerun in a fresh clone must
            # not inherit the archived account baseline or session history.
            previous_manifest = {}
            self.sessions_path.unlink(missing_ok=True)
            self.api_path.unlink(missing_ok=True)
        recorded_model_keys = {str(record.get("model_key")) for record in existing}
        run_model_keys = recorded_model_keys | {model.key for model in self.models}
        run_model_keys.update(
            str(item["key"])
            for item in previous_manifest.get("models", [])
            if isinstance(item, dict) and item.get("key")
        )
        run_models_by_key = {
            str(item["key"]): ModelSpec(**item)
            for item in previous_manifest.get("models", [])
            if isinstance(item, dict) and item.get("key")
        }
        run_models_by_key.update(
            {model.key: model for model in self.model_universe if model.key in run_model_keys}
        )
        run_models_by_key.update({model.key: model for model in self.models})
        unresolved_model_keys = run_model_keys - set(run_models_by_key)
        if unresolved_model_keys:
            raise ValueError(
                "Cannot reconstruct model contracts for resumed records: "
                f"{sorted(unresolved_model_keys)}"
            )
        universe_order = [model.key for model in self.model_universe]
        ordered_keys = [key for key in universe_order if key in run_models_by_key]
        ordered_keys.extend(sorted(set(run_models_by_key) - set(ordered_keys)))
        run_models = [run_models_by_key[key] for key in ordered_keys]
        completed = {
            record["task_id"]
            for record in existing
            if record.get("status") == "completed"
            or (
                record.get("status") == "invalid"
                and (
                    not self.retry_existing_api_errors
                    or record.get("response_source") not in {"api_error", "budget_stopped"}
                )
                and (
                    not self.config.retry_empty_responses
                    or record.get("response_source") in {"api_error", "budget_stopped"}
                )
            )
        }
        pending = [
            (model, task, iteration)
            for iteration in range(1, self.config.repeats + 1)
            for task in self.tasks
            for model in self.models
            if _task_id(self.config.benchmark_version, model, task, iteration) not in completed
        ]
        manifest = {
            "run_id": self.run_id,
            "benchmark_version": self.config.benchmark_version,
            "initial_started_at": previous_manifest.get(
                "initial_started_at",
                previous_manifest.get("started_at", invocation_started_at),
            ),
            "latest_invocation_started_at": invocation_started_at,
            "config": self.config.__dict__,
            "client": {
                "timeout_seconds": self.client.timeout_seconds,
                "max_retries": self.client.max_retries,
            },
            "models": [model.__dict__ for model in run_models],
            "model_catalog_snapshot": validated_model_snapshot(self.client, run_models),
            "source_manifest": self.source_manifest,
            "task_count": len(self.tasks),
            "tasks": [
                {
                    "task_key": task.key,
                    "prompt_sha256": hashlib.sha256(task.prompt.encode("utf-8")).hexdigest(),
                    "system_prompt_sha256": hashlib.sha256(
                        task.system_prompt.encode("utf-8")
                    ).hexdigest(),
                    "max_tokens": task.max_tokens,
                    "metadata": task.metadata,
                }
                for task in self.tasks
            ],
            "planned_records": len(run_models) * self.config.repeats * len(self.tasks),
            "pending_records": len(pending),
            "resume_policy": {
                "retry_existing_api_errors": self.retry_existing_api_errors,
                "retry_empty_responses": self.config.retry_empty_responses,
            },
            "credits_initial": previous_manifest.get(
                "credits_initial",
                previous_manifest.get("credits_before", credits_before),
            ),
            "credits_before_latest_invocation": credits_before,
            "key_metadata": self.client.key_metadata(),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        completed_now = 0
        if pending:
            pending_by_model = {
                model.key: [
                    (task, iteration)
                    for pending_model, task, iteration in pending
                    if pending_model.key == model.key
                ]
                for model in self.models
            }
            progress_lock = threading.Lock()

            def run_model_lane(model: ModelSpec) -> None:
                nonlocal completed_now
                for task, iteration in pending_by_model[model.key]:
                    try:
                        self._run_task_limited(model, task, iteration)
                    except Exception as exc:  # Preserve and continue other tasks in this lane.
                        self._append_jsonl(
                            self.records_path,
                            {
                                "task_id": _task_id(
                                    self.config.benchmark_version, model, task, iteration
                                ),
                                "benchmark_version": self.config.benchmark_version,
                                "run_id": self.run_id,
                                "created_at": utc_now(),
                                "status": "runner_error",
                                "response_source": "runner_error",
                                "model_key": model.key,
                                "model_display_name": model.display_name,
                                "model_id_requested": model.model_id,
                                "iteration": iteration,
                                "seed": _task_seed(self.config.base_seed, task, iteration),
                                "system_prompt": task.system_prompt,
                                "prompt": task.prompt,
                                "response": "",
                                "cost_usd": 0.0,
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                                **task.metadata,
                            },
                        )
                    with progress_lock:
                        completed_now += 1
                        if (
                            completed_now == 1
                            or completed_now % 10 == 0
                            or completed_now == len(pending)
                        ):
                            print(
                                f"completed {completed_now}/{len(pending)} pending records; "
                                f"reported invocation cost ${self._invocation_cost:.6f}",
                                flush=True,
                            )

            active_models = [model for model in self.models if pending_by_model[model.key]]
            executor = ThreadPoolExecutor(
                max_workers=min(self.config.max_workers, len(active_models))
            )
            futures = {}
            try:
                futures = {
                    executor.submit(run_model_lane, model): model for model in active_models
                }
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                with self._cost_lock:
                    self._stop_requested = True
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

        all_records = _jsonl_records(self.records_path)
        records = sorted(
            dedupe_records(all_records),
            key=lambda record: (
                record.get("model_key", ""),
                record.get("iteration", 0),
                record.get("task_key", ""),
            ),
        )
        write_csv(self.output_dir / "responses.csv", records)
        credits_after = self.client.credits()
        credits_initial = manifest["credits_initial"]
        manifest.update(
            {
                "finished_at": utc_now(),
                "credits_after": credits_after,
                "latest_invocation_usage_delta_usd": (
                    credits_after["total_usage"] - credits_before["total_usage"]
                ),
                "account_usage_delta_usd": (
                    credits_after["total_usage"] - credits_initial["total_usage"]
                ),
                "reported_response_cost_usd": sum(
                    float(record.get("cost_usd") or 0.0) for record in records
                ),
                "record_count": len(records),
                "valid_record_count": sum(
                    record.get("status") == "completed" for record in records
                ),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._append_jsonl(
            self.sessions_path,
            {
                "invocation_started_at": invocation_started_at,
                "invocation_finished_at": manifest["finished_at"],
                "credits_before": credits_before,
                "credits_after": credits_after,
                "usage_delta_usd": manifest["latest_invocation_usage_delta_usd"],
                "pending_records_at_start": len(pending),
                "records_after": len(records),
                "valid_records_after": manifest["valid_record_count"],
            },
        )
        return records, manifest
