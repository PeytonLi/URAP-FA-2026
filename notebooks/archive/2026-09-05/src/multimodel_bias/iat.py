from __future__ import annotations

import base64
import csv
import hashlib
import json
import mimetypes
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from .analysis import write_iat_analysis
from .artifacts import sanitize_for_artifact
from .constants import (
    BENCHMARK_VERSION,
    IAT_PROMPT,
    IAT_WORD_LOOKUP,
    NEGATIVE_IAT_SET,
    POSITIVE_IAT_SET,
)
from .openrouter import OpenRouterClient, OpenRouterError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORD_RE = re.compile(r"[A-Za-z]+")
SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Stimulus:
    image_id: str
    filename: str
    race: str
    gender: str
    width: int
    height: int
    sha256: str
    path: Path


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    model_id: str
    api_revision_hint: str
    release_date: str
    weight_access: str
    provider_endpoint: str
    max_concurrency: int
    supports_temperature: bool
    supports_top_p: bool
    reasoning_effort: str | None
    max_tokens: int


@dataclass(frozen=True)
class IATRunConfig:
    benchmark_version: str
    repeats: int
    temperature: float
    top_p: float
    base_seed: int
    fallback_enabled: bool
    fallback_max_tokens: int
    max_workers: int
    minimum_request_interval_seconds: float
    max_run_cost_usd: float
    minimum_account_balance_usd: float
    bootstrap_samples: int
    provider: dict[str, Any]


class BudgetExceededError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stimuli(
    manifest_path: Path | None = None, *, expected_count: int | None = 12
) -> list[Stimulus]:
    manifest_path = manifest_path or PROJECT_ROOT / "data" / "iat12" / "manifest.csv"
    stimuli: list[Stimulus] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = (manifest_path.parent / row["source_relative_path"]).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Missing IAT stimulus: {path}")
            actual_hash = sha256_file(path)
            if actual_hash != row["sha256"]:
                raise ValueError(
                    f"Stimulus hash mismatch for {path.name}: {actual_hash} != {row['sha256']}"
                )
            with Image.open(path) as image:
                actual_size = image.size
            expected_size = (int(row["width"]), int(row["height"]))
            if actual_size != expected_size:
                raise ValueError(
                    f"Stimulus dimensions changed for {path.name}: {actual_size} != {expected_size}"
                )
            stimuli.append(
                Stimulus(
                    image_id=row["image_id"],
                    filename=row["filename"],
                    race=row["race"],
                    gender=row["gender"],
                    width=expected_size[0],
                    height=expected_size[1],
                    sha256=row["sha256"],
                    path=path,
                )
            )
    if expected_count is not None and len(stimuli) != expected_count:
        raise ValueError(f"Expected {expected_count} IAT stimuli, found {len(stimuli)}")
    if not stimuli:
        raise ValueError("The stimulus manifest contains no images")
    if len({stimulus.image_id for stimulus in stimuli}) != len(stimuli):
        raise ValueError("Stimulus image_id values must be unique")
    if {stimulus.race for stimulus in stimuli} - {"Black", "White"}:
        raise ValueError("This analysis supports only Black and White race labels")
    if {stimulus.gender for stimulus in stimuli} - {"Female", "Male"}:
        raise ValueError("This analysis supports only Female and Male gender labels")
    return stimuli


def load_model_specs(path: Path | None = None) -> tuple[dict[str, Any], list[ModelSpec]]:
    path = path or PROJECT_ROOT / "config" / "models_aug31.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    models = [ModelSpec(**entry) for entry in document["models"]]
    if len({model.key for model in models}) != len(models):
        raise ValueError("Model keys must be unique")
    if any(model.max_concurrency < 1 for model in models):
        raise ValueError("Each model max_concurrency must be at least 1")
    return document, models


def validated_model_snapshot(
    client: OpenRouterClient,
    models: list[ModelSpec],
) -> list[dict[str, Any]]:
    catalog = {entry.get("id"): entry for entry in client.model_catalog()}
    fields = {
        "id",
        "name",
        "canonical_slug",
        "created",
        "context_length",
        "architecture",
        "pricing",
        "supported_parameters",
    }
    snapshots = []
    for model in models:
        entry = catalog.get(model.model_id)
        if entry is None:
            raise ValueError(f"OpenRouter catalog no longer contains {model.model_id}")
        canonical_slug = entry.get("canonical_slug")
        if canonical_slug != model.api_revision_hint:
            raise ValueError(
                f"OpenRouter alias drift for {model.model_id}: expected "
                f"{model.api_revision_hint}, found {canonical_slug}"
            )
        endpoint_document = client.get(f"/models/{model.model_id}/endpoints").get("data", {})
        endpoints = endpoint_document.get("endpoints") or []
        matches = [
            endpoint
            for endpoint in endpoints
            if endpoint.get("tag") == model.provider_endpoint
        ]
        if len(matches) != 1:
            available = sorted(
                str(endpoint.get("tag")) for endpoint in endpoints if endpoint.get("tag")
            )
            raise ValueError(
                f"OpenRouter endpoint drift for {model.model_id}: expected exactly one "
                f"{model.provider_endpoint!r} route, found {len(matches)}; available={available}"
            )
        selected_endpoint = matches[0]
        if selected_endpoint.get("status") != 0:
            raise ValueError(
                f"OpenRouter endpoint {model.provider_endpoint!r} for {model.model_id} "
                f"is unavailable (status={selected_endpoint.get('status')!r})"
            )
        endpoint_fields = {
            "name",
            "tag",
            "provider_name",
            "quantization",
            "context_length",
            "max_prompt_tokens",
            "max_completion_tokens",
            "pricing",
            "supported_parameters",
            "status",
        }
        snapshots.append(
            {
                **{key: entry.get(key) for key in fields},
                "selected_endpoint": {
                    key: selected_endpoint.get(key) for key in endpoint_fields
                },
            }
        )
    return snapshots


def load_run_config(path: Path | None = None, **overrides: Any) -> IATRunConfig:
    path = path or PROJECT_ROOT / "config" / "iat_utkface_aug31.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    for key, value in overrides.items():
        if value is not None:
            document[key] = value
    config = IATRunConfig(**document)
    if config.repeats < 1:
        raise ValueError("repeats must be at least 1")
    if config.max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if config.minimum_request_interval_seconds < 0:
        raise ValueError("minimum_request_interval_seconds must not be negative")
    if config.max_run_cost_usd <= 0:
        raise ValueError("max_run_cost_usd must be positive")
    return config


def parse_three_words(text: str) -> list[str]:
    picks: list[str] = []
    for match in WORD_RE.finditer(str(text)):
        canonical = IAT_WORD_LOOKUP.get(match.group(0).lower())
        if canonical is None:
            continue
        if not picks or picks[-1] != canonical:
            picks.append(canonical)
        if len(picks) == 3:
            break
    return picks


def is_exact_three_word_format(text: str, parsed_words: list[str]) -> bool:
    if len(parsed_words) != 3:
        return False
    expected = ", ".join(parsed_words)
    normalized = re.sub(r"\s*,\s*", ", ", str(text).strip())
    return normalized.casefold() == expected.casefold()


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def deterministic_seed(base_seed: int, iteration: int, image_index: int) -> int:
    return int(base_seed + iteration * 1000 + image_index)


def deterministic_task_id(
    model: ModelSpec,
    stimulus: Stimulus,
    iteration: int,
    benchmark_version: str = BENCHMARK_VERSION,
) -> str:
    value = f"{benchmark_version}|{model.model_id}|{stimulus.sha256}|{iteration}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _message_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
        return "".join(parts).strip()
    return str(content).strip()


def _request_payload(
    model: ModelSpec,
    stimulus_data_url: str,
    config: IATRunConfig,
    seed: int,
    *,
    fallback: bool,
) -> dict[str, Any]:
    provider = dict(config.provider)
    provider.update(
        {
            "order": [model.provider_endpoint],
            "allow_fallbacks": False,
        }
    )
    payload: dict[str, Any] = {
        "model": model.model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": stimulus_data_url}},
                    {"type": "text", "text": IAT_PROMPT},
                ],
            }
        ],
        "max_tokens": (
            max(config.fallback_max_tokens, model.max_tokens)
            if fallback and model.reasoning_effort not in {None, "none"}
            else config.fallback_max_tokens
            if fallback
            else model.max_tokens
        ),
        "seed": seed,
        "provider": provider,
    }
    if model.supports_temperature:
        payload["temperature"] = 0.0 if fallback else config.temperature
    if model.supports_top_p and not fallback:
        payload["top_p"] = config.top_p
    if model.reasoning_effort is not None:
        payload["reasoning"] = {
            "effort": model.reasoning_effort,
            "exclude": True,
        }
    return payload


def _usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        "image_tokens": int(completion_details.get("image_tokens") or 0),
        "cost_usd": float(usage.get("cost") or 0.0),
    }


def _safe_response_snapshot(response: dict[str, Any]) -> dict[str, Any]:
    return sanitize_for_artifact(
        {key: value for key, value in response.items() if key != "_request_payload"}
    )


def _jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return records


def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first success; otherwise keep the latest attempt for each task."""
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = record["task_id"]
        current = selected.get(task_id)
        if current is None or current.get("status") != "completed":
            selected[task_id] = record
    return list(selected.values())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    columns = sorted({key for record in records for key in record})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda item: (
                item.get("model_key", ""),
                item.get("iteration", 0),
                item.get("image_id", ""),
            ),
        ):
            writer.writerow(record)


class IATBenchmarkRunner:
    def __init__(
        self,
        client: OpenRouterClient,
        run_id: str,
        config: IATRunConfig,
        models: list[ModelSpec],
        stimuli: list[Stimulus],
        *,
        output_root: Path | None = None,
        retry_existing_api_errors: bool = True,
    ) -> None:
        if not SAFE_RUN_ID_RE.fullmatch(run_id):
            raise ValueError("run_id may contain only letters, numbers, '.', '_' and '-'")
        self.client = client
        self.run_id = run_id
        self.config = config
        self.models = models
        self.stimuli = stimuli
        self.output_dir = (output_root or PROJECT_ROOT / "results" / "iat12") / run_id
        self.retry_existing_api_errors = retry_existing_api_errors
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.output_dir / "responses.jsonl"
        self.api_path = self.output_dir / "api_responses.jsonl"
        self.sessions_path = self.output_dir / "run_sessions.jsonl"
        self._write_lock = threading.Lock()
        self._cost_lock = threading.Lock()
        self._rate_limit_lock = threading.Lock()
        self._run_cost = 0.0
        self._stop_requested = False
        self._model_next_request_at = {model.key: 0.0 for model in models}
        self._data_urls = {stimulus.image_id: image_data_url(stimulus.path) for stimulus in stimuli}
        self._model_semaphores = {
            model.key: threading.BoundedSemaphore(model.max_concurrency) for model in models
        }

    def _append_jsonl(self, path: Path, value: dict[str, Any]) -> None:
        with self._write_lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
            handle.flush()

    def _reserve_response_cost(self, cost: float) -> None:
        with self._cost_lock:
            self._run_cost += cost
            if self._run_cost >= self.config.max_run_cost_usd:
                self._stop_requested = True

    def _check_budget(self) -> None:
        with self._cost_lock:
            if self._stop_requested or self._run_cost >= self.config.max_run_cost_usd:
                raise BudgetExceededError(
                    f"Run cost ceiling reached: ${self._run_cost:.6f} / "
                    f"${self.config.max_run_cost_usd:.2f}"
                )

    def _wait_for_model_slot(self, model: ModelSpec) -> None:
        with self._rate_limit_lock:
            now = time.monotonic()
            request_at = max(now, self._model_next_request_at[model.key])
            self._model_next_request_at[model.key] = (
                request_at + self.config.minimum_request_interval_seconds
            )
        delay = request_at - time.monotonic()
        if delay > 0:
            time.sleep(delay)

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
        self._check_budget()
        self._wait_for_model_slot(model)
        self._check_budget()
        payload = _request_payload(
            model,
            self._data_urls[stimulus.image_id],
            self.config,
            seed,
            fallback=fallback,
        )
        response = self.client.chat_completion(payload)
        usage = _usage(response)
        self._reserve_response_cost(usage["cost_usd"])
        self._append_jsonl(
            self.api_path,
            {
                "task_id": task_id,
                "stage": stage,
                "requested_at": utc_now(),
                "model_id_requested": model.model_id,
                "image_id": stimulus.image_id,
                "image_sha256": stimulus.sha256,
                "prompt_sha256": hashlib.sha256(IAT_PROMPT.encode("utf-8")).hexdigest(),
                "seed": seed,
                "request_parameters": {
                    key: value for key, value in payload.items() if key not in {"messages"}
                },
                "response": _safe_response_snapshot(response),
            },
        )
        return response

    def _run_task(
        self,
        model: ModelSpec,
        stimulus: Stimulus,
        iteration: int,
        image_index: int,
    ) -> dict[str, Any]:
        task_id = deterministic_task_id(
            model, stimulus, iteration, self.config.benchmark_version
        )
        seed = deterministic_seed(self.config.base_seed, iteration, image_index)
        started = time.monotonic()
        sampled_response: dict[str, Any] | None = None
        fallback_response: dict[str, Any] | None = None
        sampled_text = ""
        fallback_text = ""
        error_type = ""
        error_message = ""
        error_status = 0

        try:
            sampled_response = self._call(
                task_id=task_id,
                stage="sampled",
                model=model,
                stimulus=stimulus,
                seed=seed,
                fallback=False,
            )
            sampled_text = _message_text(sampled_response)
            words = parse_three_words(sampled_text)
            response_source = "sampled"

            if len(words) != 3 and self.config.fallback_enabled:
                fallback_response = self._call(
                    task_id=task_id,
                    stage="fallback",
                    model=model,
                    stimulus=stimulus,
                    seed=seed + 1_000_000,
                    fallback=True,
                )
                fallback_text = _message_text(fallback_response)
                words = parse_three_words(fallback_text)
                response_source = "fallback" if len(words) == 3 else "fallback_invalid"
        except BudgetExceededError as exc:
            words = []
            response_source = "budget_stopped"
            error_type = type(exc).__name__
            error_message = str(exc)
        except OpenRouterError as exc:
            words = []
            response_source = "api_error"
            error_type = type(exc).__name__
            error_message = exc.failure.message
            error_status = exc.failure.status_code
            self._append_jsonl(
                self.api_path,
                {
                    "task_id": task_id,
                    "stage": "error",
                    "requested_at": utc_now(),
                    "model_id_requested": model.model_id,
                    "image_id": stimulus.image_id,
                    "image_sha256": stimulus.sha256,
                    "seed": seed,
                    "error": sanitize_for_artifact(exc.failure.payload),
                },
            )

        final_text = fallback_text if response_source.startswith("fallback") else sampled_text
        valid_three = len(words) == 3
        sampled_usage = _usage(sampled_response or {})
        fallback_usage = _usage(fallback_response or {})
        response_cost = sampled_usage["cost_usd"] + fallback_usage["cost_usd"]
        api_model = ""
        provider = ""
        finish_reason = ""
        final_api_response = fallback_response or sampled_response or {}
        choices = final_api_response.get("choices") or []
        if choices:
            finish_reason = str(choices[0].get("finish_reason") or "")
        api_model = str(final_api_response.get("model") or "")
        provider = str(final_api_response.get("provider") or "")
        client_meta = final_api_response.get("_client") or {}

        record = {
            "task_id": task_id,
            "benchmark_version": self.config.benchmark_version,
            "run_id": self.run_id,
            "created_at": utc_now(),
            "status": "completed" if valid_three else "invalid",
            "model_key": model.key,
            "model_display_name": model.display_name,
            "model_id_requested": model.model_id,
            "api_model_returned": api_model,
            "provider": provider,
            "image_id": stimulus.image_id,
            "image_filename": stimulus.filename,
            "image_sha256": stimulus.sha256,
            "race": stimulus.race,
            "gender": stimulus.gender,
            "iteration": iteration,
            "seed": seed,
            "temperature_requested": self.config.temperature,
            "temperature_effective": (
                self.config.temperature if model.supports_temperature else "provider_default"
            ),
            "top_p_requested": self.config.top_p,
            "top_p_effective": self.config.top_p if model.supports_top_p else "provider_default",
            "max_tokens": model.max_tokens,
            "reasoning_effort": model.reasoning_effort or "not_applicable",
            "response": ", ".join(words) if words else "N/A",
            "words": "|".join(words),
            "sampled_raw": sampled_text,
            "fallback_raw": fallback_text,
            "response_source": response_source,
            "valid_three": valid_three,
            "exact_format": is_exact_three_word_format(final_text, words),
            "word_count": len(words),
            "positive_count": sum(word in POSITIVE_IAT_SET for word in words),
            "negative_count": sum(word in NEGATIVE_IAT_SET for word in words),
            "finish_reason": finish_reason,
            "prompt_tokens": sampled_usage["prompt_tokens"] + fallback_usage["prompt_tokens"],
            "completion_tokens": (
                sampled_usage["completion_tokens"] + fallback_usage["completion_tokens"]
            ),
            "reasoning_tokens": (
                sampled_usage["reasoning_tokens"] + fallback_usage["reasoning_tokens"]
            ),
            "image_tokens": sampled_usage["image_tokens"] + fallback_usage["image_tokens"],
            "cost_usd": response_cost,
            "latency_seconds": time.monotonic() - started,
            "final_call_retry_count": int(client_meta.get("retry_count") or 0),
            "error_type": error_type,
            "error_status": error_status,
            "error_message": error_message,
        }
        self._append_jsonl(self.records_path, record)
        return record

    def _run_task_limited(
        self,
        model: ModelSpec,
        stimulus: Stimulus,
        iteration: int,
        image_index: int,
    ) -> dict[str, Any]:
        with self._model_semaphores[model.key]:
            return self._run_task(model, stimulus, iteration, image_index)

    def _model_snapshot(self) -> list[dict[str, Any]]:
        return validated_model_snapshot(self.client, self.models)

    def run(self) -> dict[str, Any]:
        manifest_path = self.output_dir / "run_manifest.json"
        previous_manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )
        invocation_started_at = utc_now()
        credits_before = self.client.credits()
        if credits_before["remaining"] < self.config.minimum_account_balance_usd:
            raise BudgetExceededError(
                f"Account balance ${credits_before['remaining']:.2f} is below required reserve "
                f"${self.config.minimum_account_balance_usd:.2f}"
            )

        existing_records = _jsonl_records(self.records_path)
        if not existing_records:
            # Shared repositories omit raw logs. A paid rerun in a fresh clone must
            # not inherit the archived account baseline or session history.
            previous_manifest = {}
            self.sessions_path.unlink(missing_ok=True)
            self.api_path.unlink(missing_ok=True)
        recorded_model_keys = {str(record.get("model_key")) for record in existing_records}
        run_models_by_key = {
            str(item["key"]): ModelSpec(**item)
            for item in previous_manifest.get("models", [])
            if isinstance(item, dict) and item.get("key")
        }
        run_models_by_key.update({model.key: model for model in self.models})
        missing_model_keys = recorded_model_keys - set(run_models_by_key)
        if missing_model_keys:
            _, default_models = load_model_specs()
            run_models_by_key.update(
                {model.key: model for model in default_models if model.key in missing_model_keys}
            )
        unresolved_model_keys = recorded_model_keys - set(run_models_by_key)
        if unresolved_model_keys:
            raise ValueError(
                "Cannot reconstruct model contracts for resumed records: "
                f"{sorted(unresolved_model_keys)}"
            )
        _, default_models = load_model_specs()
        default_order = [model.key for model in default_models]
        ordered_keys = [key for key in default_order if key in run_models_by_key]
        ordered_keys.extend(sorted(set(run_models_by_key) - set(ordered_keys)))
        run_models = [run_models_by_key[key] for key in ordered_keys]
        completed = {
            record["task_id"]
            for record in existing_records
            if record.get("status") == "completed"
            or (
                record.get("status") == "invalid"
                and (
                    not self.retry_existing_api_errors
                    or record.get("response_source") not in {"api_error", "budget_stopped"}
                )
            )
        }
        tasks = []
        for iteration in range(1, self.config.repeats + 1):
            for image_index, stimulus in enumerate(self.stimuli, start=1):
                for model in self.models:
                    task_id = deterministic_task_id(
                        model, stimulus, iteration, self.config.benchmark_version
                    )
                    if task_id not in completed:
                        tasks.append((model, stimulus, iteration, image_index))

        run_manifest = {
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
            "stimuli": [
                {
                    **stimulus.__dict__,
                    "path": str(stimulus.path.relative_to(PROJECT_ROOT)),
                }
                for stimulus in self.stimuli
            ],
            "prompt": IAT_PROMPT,
            "prompt_sha256": hashlib.sha256(IAT_PROMPT.encode("utf-8")).hexdigest(),
            "credits_initial": previous_manifest.get(
                "credits_initial",
                previous_manifest.get("credits_before", credits_before),
            ),
            "credits_before_latest_invocation": credits_before,
            "key_metadata": self.client.key_metadata(),
            "planned_tasks": len(run_models) * self.config.repeats * len(self.stimuli),
            "resumed_records": len(existing_records),
            "pending_tasks": len(tasks),
            "resume_policy": {
                "retry_existing_api_errors": self.retry_existing_api_errors,
            },
        }
        _write_json(manifest_path, run_manifest)

        new_records: list[dict[str, Any]] = []
        if tasks:
            tasks_by_model = {
                model.key: [
                    (stimulus, iteration, image_index)
                    for task_model, stimulus, iteration, image_index in tasks
                    if task_model.key == model.key
                ]
                for model in self.models
            }
            progress_lock = threading.Lock()

            def run_model_lane(model: ModelSpec) -> None:
                for stimulus, iteration, image_index in tasks_by_model[model.key]:
                    try:
                        record = self._run_task_limited(
                            model, stimulus, iteration, image_index
                        )
                    except Exception as exc:  # Keep later tasks in this model lane recoverable.
                        record = {
                            "task_id": deterministic_task_id(
                                model,
                                stimulus,
                                iteration,
                                self.config.benchmark_version,
                            ),
                            "benchmark_version": self.config.benchmark_version,
                            "run_id": self.run_id,
                            "created_at": utc_now(),
                            "status": "runner_error",
                            "model_key": model.key,
                            "model_display_name": model.display_name,
                            "model_id_requested": model.model_id,
                            "image_id": stimulus.image_id,
                            "image_filename": stimulus.filename,
                            "image_sha256": stimulus.sha256,
                            "race": stimulus.race,
                            "gender": stimulus.gender,
                            "iteration": iteration,
                            "valid_three": False,
                            "word_count": 0,
                            "positive_count": 0,
                            "negative_count": 0,
                            "cost_usd": 0.0,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                        self._append_jsonl(self.records_path, record)
                    with progress_lock:
                        new_records.append(record)
                        done = len(new_records)
                        if done == 1 or done % 10 == 0 or done == len(tasks):
                            print(
                                f"completed {done}/{len(tasks)} pending tasks; "
                                f"reported run cost ${self._run_cost:.6f}",
                                flush=True,
                            )

            active_models = [model for model in self.models if tasks_by_model[model.key]]
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
        records = dedupe_records(all_records)
        _write_records_csv(self.output_dir / "responses.csv", records)
        analysis_manifest = write_iat_analysis(
            records,
            run_models,
            self.output_dir,
            bootstrap_samples=self.config.bootstrap_samples,
            bootstrap_seed=self.config.base_seed + 77,
        )
        credits_after = self.client.credits()
        credits_initial = run_manifest["credits_initial"]
        invocation_usage_delta = credits_after["total_usage"] - credits_before["total_usage"]
        run_manifest.update(
            {
                "finished_at": utc_now(),
                "credits_after": credits_after,
                "latest_invocation_usage_delta_usd": invocation_usage_delta,
                "account_usage_delta_usd": credits_after["total_usage"]
                - credits_initial["total_usage"],
                "reported_response_cost_usd": sum(
                    float(record.get("cost_usd") or 0.0) for record in records
                ),
                "record_count": len(records),
                "valid_record_count": sum(bool(record.get("valid_three")) for record in records),
                "analysis": analysis_manifest,
            }
        )
        _write_json(manifest_path, run_manifest)
        self._append_jsonl(
            self.sessions_path,
            {
                "invocation_started_at": invocation_started_at,
                "invocation_finished_at": run_manifest["finished_at"],
                "credits_before": credits_before,
                "credits_after": credits_after,
                "usage_delta_usd": invocation_usage_delta,
                "pending_tasks_at_start": len(tasks),
                "records_after": len(records),
                "valid_records_after": run_manifest["valid_record_count"],
            },
        )
        return run_manifest
