from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CLAUDE_MODEL = "claude-opus-5"
CLAUDE_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class ClaudeJSONResponse:
    value: dict[str, Any]
    model: str
    response_id: str
    stop_reason: str
    input_tokens: int
    output_tokens: int


def image_content_block(path: Path) -> dict[str, Any]:
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        raise ValueError(f"Unsupported Claude image type for {path}: {media_type}")
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        },
    }


def _usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, Mapping):
        return int(usage.get(name, 0) or 0)
    return int(getattr(usage, name, 0) or 0)


def _response_text(response: Any) -> str:
    blocks = response.get("content", []) if isinstance(response, Mapping) else response.content
    texts: list[str] = []
    for block in blocks:
        block_type = block.get("type") if isinstance(block, Mapping) else getattr(block, "type", None)
        if block_type == "text":
            text = block.get("text", "") if isinstance(block, Mapping) else getattr(block, "text", "")
            texts.append(str(text))
    if not texts:
        raise RuntimeError("Claude response did not contain a text block")
    return "".join(texts)


class ClaudeProvider:
    """Small injectable wrapper around Anthropic's Messages API."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_CLAUDE_MODEL,
        backend: str = "direct",
        api_key: str | None = None,
        vertex_project_id: str | None = None,
        vertex_region: str = "global",
        client: Any | None = None,
        max_attempts: int = 4,
        retry_base_seconds: float = 2.0,
        effort: str = "high",
    ) -> None:
        if effort not in CLAUDE_EFFORT_LEVELS:
            raise ValueError(f"Unsupported Claude effort level: {effort}")
        self.model = model
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.effort = effort
        if client is not None:
            self.client = client
            return
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Install the project dependencies before calling Claude"
            ) from exc
        if backend == "direct":
            resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not resolved_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self.client = anthropic.Anthropic(api_key=resolved_key)
        elif backend == "vertex":
            project_id = vertex_project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if not project_id:
                raise RuntimeError(
                    "vertex_project_id or GOOGLE_CLOUD_PROJECT is required"
                )
            self.client = anthropic.AnthropicVertex(
                project_id=project_id,
                region=vertex_region,
            )
        else:
            raise ValueError(f"Unsupported Claude backend: {backend}")

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        image_path: Path | None,
        max_tokens: int = 2048,
    ) -> ClaudeJSONResponse:
        content: list[dict[str, Any]] = []
        if image_path is not None:
            content.append(image_content_block(image_path))
        content.append({"type": "text", "text": prompt})
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": content}],
                    output_config={
                        "effort": self.effort,
                        "format": {
                            "type": "json_schema",
                            "schema": schema,
                        }
                    },
                )
                value = json.loads(_response_text(response))
                if not isinstance(value, dict):
                    raise ValueError("Claude structured response must be a JSON object")
                usage = response.get("usage") if isinstance(response, Mapping) else response.usage
                return ClaudeJSONResponse(
                    value=value,
                    model=str(
                        response.get("model", self.model)
                        if isinstance(response, Mapping)
                        else getattr(response, "model", self.model)
                    ),
                    response_id=str(
                        response.get("id", "")
                        if isinstance(response, Mapping)
                        else getattr(response, "id", "")
                    ),
                    stop_reason=str(
                        response.get("stop_reason", "")
                        if isinstance(response, Mapping)
                        else getattr(response, "stop_reason", "")
                    ),
                    input_tokens=_usage_value(usage, "input_tokens"),
                    output_tokens=_usage_value(usage, "output_tokens"),
                )
            except Exception as exc:  # Provider errors vary by SDK release.
                last_error = exc
                if attempt == self.max_attempts:
                    break
                time.sleep(self.retry_base_seconds * (2 ** (attempt - 1)))
        raise RuntimeError(
            f"Claude request failed after {self.max_attempts} attempts: "
            f"{type(last_error).__name__}: {str(last_error)[:1500]}"
        ) from last_error
