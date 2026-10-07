from __future__ import annotations

import base64
import json
import mimetypes
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
RETRYABLE_HTTP_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class GeminiTextResponse:
    text: str
    response_id: str
    model_version: str
    finish_reason: str
    prompt_tokens: int
    output_tokens: int
    thought_tokens: int
    total_tokens: int


class GoogleAuthAdcTokenProvider:
    def __init__(self, credentials: Any | None = None, request_factory: Any | None = None):
        self._credentials = credentials
        self._request_factory = request_factory
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            if self._credentials is None:
                try:
                    import google.auth
                    from google.auth.transport.requests import Request
                except ImportError as exc:
                    raise RuntimeError(
                        "google-auth[requests] is required for Vertex ADC"
                    ) from exc
                self._credentials, _ = google.auth.default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                self._request_factory = Request
            if not self._credentials.valid or not self._credentials.token:
                request = self._request_factory() if self._request_factory else None
                self._credentials.refresh(request)
            token = str(self._credentials.token or "").strip()
            if not token:
                raise RuntimeError("Vertex ADC did not produce an access token")
            return token

    def invalidate(self) -> None:
        with self._lock:
            if self._credentials is not None:
                self._credentials.token = None


def vertex_endpoint(project: str, location: str, model: str) -> str:
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/{project}/locations/{location}/"
        f"publishers/google/models/{model}:generateContent"
    )


def image_part(path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
    }


def image_parts(paths: Path | Sequence[Path]) -> list[dict[str, Any]]:
    resolved = [paths] if isinstance(paths, Path) else list(paths)
    if not resolved or not all(isinstance(path, Path) for path in resolved):
        raise ValueError("at least one Path image is required")
    return [image_part(path) for path in resolved]


def parse_text_response(payload: dict[str, Any]) -> GeminiTextResponse:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError(f"Vertex returned no candidates: {payload.get('promptFeedback')}")
    candidate = candidates[0]
    parts = ((candidate.get("content") or {}).get("parts") or [])
    text = "".join(
        str(part["text"])
        for part in parts
        if "text" in part and not bool(part.get("thought"))
    ).strip()
    if not text:
        raise ValueError(
            "Vertex returned no non-thought text "
            f"(finish_reason={candidate.get('finishReason')!r}, "
            f"finish_message={candidate.get('finishMessage')!r}, "
            f"parts={len(parts)}, thought_parts="
            f"{sum(bool(part.get('thought')) for part in parts)}, "
            f"usage={payload.get('usageMetadata') or {}})"
        )
    usage = payload.get("usageMetadata") or {}
    return GeminiTextResponse(
        text=text,
        response_id=str(payload.get("responseId") or ""),
        model_version=str(payload.get("modelVersion") or ""),
        finish_reason=str(candidate.get("finishReason") or ""),
        prompt_tokens=int(usage.get("promptTokenCount") or 0),
        output_tokens=int(usage.get("candidatesTokenCount") or 0),
        thought_tokens=int(usage.get("thoughtsTokenCount") or 0),
        total_tokens=int(usage.get("totalTokenCount") or 0),
    )


def build_generation_config(
    *,
    temperature: float | None,
    top_p: float | None,
    max_output_tokens: int,
    response_mime_type: str | None,
    response_schema: dict[str, Any] | None,
    thinking_level: str | None,
    seed: int | None,
) -> dict[str, Any]:
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    generation_config: dict[str, Any] = {"maxOutputTokens": max_output_tokens}
    if temperature is not None:
        generation_config["temperature"] = float(temperature)
    if top_p is not None:
        generation_config["topP"] = float(top_p)
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type
    if response_schema is not None:
        if response_mime_type != "application/json":
            raise ValueError(
                "response_schema requires response_mime_type='application/json'"
            )
        generation_config["responseSchema"] = response_schema
    if thinking_level is not None:
        normalized_level = thinking_level.strip().upper()
        if normalized_level not in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}:
            raise ValueError(f"unsupported Gemini thinking level: {thinking_level}")
        generation_config["thinkingConfig"] = {
            "thinkingLevel": normalized_level,
            "includeThoughts": False,
        }
    if seed is not None:
        generation_config["seed"] = int(seed)
    return generation_config


class VertexGeminiProvider:
    def __init__(
        self,
        *,
        project: str,
        location: str = "global",
        model: str = DEFAULT_GEMINI_MODEL,
        max_attempts: int = 4,
        timeout_seconds: int = 180,
        token_provider: GoogleAuthAdcTokenProvider | None = None,
        opener: Callable[..., Any] | None = None,
    ):
        if not project:
            raise ValueError("Vertex project is required")
        self.endpoint = vertex_endpoint(project, location, model)
        self.model = model
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.token_provider = token_provider or GoogleAuthAdcTokenProvider()
        self.opener = opener or urllib.request.urlopen

    def generate_text(
        self,
        *,
        system: str,
        prompt: str,
        image_path: Path | Sequence[Path],
        temperature: float | None,
        top_p: float | None,
        max_output_tokens: int,
        response_mime_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        thinking_level: str | None = None,
        seed: int | None = None,
    ) -> GeminiTextResponse:
        generation_config = build_generation_config(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
            thinking_level=thinking_level,
            seed=seed,
        )
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [*image_parts(image_path), {"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
        }
        encoded = json.dumps(body, ensure_ascii=True).encode("utf-8")
        errors = []
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=encoded,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.token_provider.get()}",
                    "Content-Type": "application/json; charset=utf-8",
                },
            )
            try:
                with self.opener(request, timeout=self.timeout_seconds) as response:
                    return parse_text_response(json.loads(response.read().decode("utf-8")))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                errors.append(f"HTTP {exc.code}: {detail[:1000]}")
                if exc.code == 401:
                    self.token_provider.invalidate()
                retryable = exc.code in RETRYABLE_HTTP_CODES or exc.code == 401
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                retryable = True
            if attempt >= self.max_attempts or not retryable:
                break
            time.sleep(min(8.0, 1.5 * 2 ** (attempt - 1)))
        raise RuntimeError("Vertex Gemini generation failed: " + " | ".join(errors))
