from __future__ import annotations

import json
import random
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import certifi

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class OpenRouterFailure:
    status_code: int
    message: str
    payload: dict[str, Any]


class OpenRouterError(RuntimeError):
    def __init__(self, failure: OpenRouterFailure):
        super().__init__(f"OpenRouter HTTP {failure.status_code}: {failure.message}")
        self.failure = failure


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 120.0,
        max_retries: int = 7,
        app_name: str = "multimodel-bias-benchmark-august31",
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.app_name = app_name
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{OPENROUTER_BASE_URL}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.app_name,
                "X-Title": self.app_name,
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
                context=self._ssl_context,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                error_payload = json.loads(raw)
            except json.JSONDecodeError:
                error_payload = {"error": {"message": raw or str(exc)}}
            message = str(error_payload.get("error", {}).get("message", str(exc)))
            raise OpenRouterError(OpenRouterFailure(exc.code, message, error_payload)) from exc
        except urllib.error.URLError as exc:
            payload = {"error": {"message": str(exc.reason)}}
            raise OpenRouterError(OpenRouterFailure(0, str(exc.reason), payload)) from exc

    def get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def credits(self) -> dict[str, float]:
        data = self.get("/credits").get("data", {})
        total_credits = float(data.get("total_credits") or 0.0)
        total_usage = float(data.get("total_usage") or 0.0)
        return {
            "total_credits": total_credits,
            "total_usage": total_usage,
            "remaining": total_credits - total_usage,
        }

    def key_metadata(self) -> dict[str, Any]:
        data = dict(self.get("/auth/key").get("data", {}))
        for private_field in ("label", "creator_user_id", "hash", "key", "name"):
            data.pop(private_field, None)
        return data

    def model_catalog(self) -> list[dict[str, Any]]:
        return list(self.get("/models").get("data", []))

    def chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        last_error: OpenRouterError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request("POST", "/chat/completions", payload)
                embedded_error = response.get("error")
                if embedded_error and not response.get("choices"):
                    raw_code = embedded_error.get("code", 500)
                    try:
                        status_code = int(raw_code)
                    except (TypeError, ValueError):
                        status_code = 500
                    raise OpenRouterError(
                        OpenRouterFailure(
                            status_code=status_code,
                            message=str(embedded_error.get("message") or "Provider returned error"),
                            payload=response,
                        )
                    )
                response["_client"] = {
                    "retry_count": attempt,
                    "elapsed_seconds": time.monotonic() - started,
                }
                return response
            except OpenRouterError as exc:
                last_error = exc
                status = exc.failure.status_code
                if attempt >= self.max_retries or (
                    status not in RETRYABLE_STATUS_CODES and status != 0
                ):
                    break
                delay = min(30.0, 0.8 * (2**attempt)) + random.uniform(0.0, 0.35)
                time.sleep(delay)

        assert last_error is not None
        raise last_error
