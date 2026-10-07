from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CODEX_MODEL = "gpt-5.6-sol"
PROFILE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "response": {
            "type": "string",
            "description": "The image-grounded portrait description only.",
        }
    },
    "required": ["response"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class CodexTextResponse:
    text: str
    thread_id: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_output_tokens: int
    total_tokens: int


def parse_codex_jsonl(text: str) -> tuple[str, dict[str, int]]:
    thread_id = ""
    usage: Mapping[str, Any] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("type") == "thread.started":
            thread_id = str(event.get("thread_id") or "")
        if event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
    normalized = {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "cached_input_tokens": int(usage.get("cached_input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_output_tokens": int(
            usage.get("reasoning_output_tokens") or 0
        ),
    }
    normalized["total_tokens"] = (
        normalized["input_tokens"] + normalized["output_tokens"]
    )
    return thread_id, normalized


def parse_profile_output(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"response"} or not isinstance(payload["response"], str):
        raise ValueError("Codex profile output does not match the frozen schema")
    text = payload["response"].strip()
    if not text:
        raise ValueError("Codex returned an empty profile")
    return text


class CodexCliProvider:
    def __init__(
        self,
        *,
        model: str = DEFAULT_CODEX_MODEL,
        reasoning_effort: str = "ultra",
        service_tier: str = "fast",
        timeout_seconds: int = 900,
        executable: str = "codex",
        auth_json: Path | None = None,
        temp_root: Path = Path("/private/tmp"),
    ):
        if not model.strip():
            raise ValueError("Codex model is required")
        if reasoning_effort not in {
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
            "ultra",
        }:
            raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
        if service_tier not in {"standard", "fast"}:
            raise ValueError(f"unsupported service tier: {service_tier}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.service_tier = service_tier
        self.timeout_seconds = timeout_seconds
        self.executable = executable
        self.auth_json = (
            auth_json.expanduser().resolve()
            if auth_json is not None
            else (Path.home() / ".codex" / "auth.json").resolve()
        )
        self.temp_root = temp_root.resolve()

    def cli_version(self) -> str:
        completed = subprocess.run(
            [self.executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.stdout.strip()

    def generate_profile(
        self,
        *,
        system: str,
        prompt: str,
        image_path: Path,
    ) -> CodexTextResponse:
        if not self.auth_json.is_file():
            raise RuntimeError(
                "ChatGPT-managed Codex authentication is unavailable; run "
                "`codex login` before generation"
            )
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="august3-codex-profile-", dir=self.temp_root
        ) as raw_temp:
            run_root = Path(raw_temp)
            codex_home = run_root / "codex-home"
            codex_home.mkdir(mode=0o700)
            copied_auth = codex_home / "auth.json"
            shutil.copy2(self.auth_json, copied_auth)
            copied_auth.chmod(0o600)
            schema_path = run_root / "profile_schema.json"
            schema_path.write_text(
                json.dumps(PROFILE_OUTPUT_SCHEMA, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            output_path = run_root / "profile.json"
            full_prompt = (
                f"{system.strip()}\n\n{prompt.strip()}\n\n"
                "Return exactly one JSON object matching the supplied schema. "
                "Do not call tools, inspect files, or discuss the task."
            )
            command = [
                self.executable,
                "exec",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "-C",
                str(run_root),
                "--model",
                self.model,
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "-c",
                f'service_tier="{self.service_tier}"',
            ]
            if self.service_tier == "fast":
                command.extend(("-c", "features.fast_mode=true"))
            command.extend(
                (
                    "--image",
                    str(image_path),
                    "--output-schema",
                    str(schema_path),
                    "--json",
                    "--output-last-message",
                    str(output_path),
                    full_prompt,
                )
            )
            environment = dict(os.environ)
            environment["CODEX_HOME"] = str(codex_home)
            environment.pop("OPENAI_API_KEY", None)
            environment.pop("CODEX_API_KEY", None)
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=environment,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"Codex profile generation timed out after {self.timeout_seconds}s"
                ) from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-4000:]
                raise RuntimeError(
                    f"Codex profile generation exited {completed.returncode}: {detail}"
                )
            if not output_path.is_file():
                raise RuntimeError("Codex did not write the structured profile output")
            text = parse_profile_output(output_path)
            thread_id, usage = parse_codex_jsonl(completed.stdout)
            return CodexTextResponse(
                text=text,
                thread_id=thread_id,
                model=self.model,
                **usage,
            )
