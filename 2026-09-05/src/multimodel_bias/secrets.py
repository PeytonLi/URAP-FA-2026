from __future__ import annotations

import getpass
import os
import platform
import subprocess


class MissingSecretError(RuntimeError):
    pass


def get_openrouter_api_key() -> str:
    """Load the OpenRouter key without writing it to logs or project files."""
    value = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if value:
        return value

    if platform.system() == "Darwin":
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                getpass.getuser(),
                "-s",
                "openrouter-api-key",
                "-w",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        if result.returncode == 0 and value:
            return value

    raise MissingSecretError(
        "OPENROUTER_API_KEY is not set and the macOS Keychain item "
        "'openrouter-api-key' was not found."
    )
