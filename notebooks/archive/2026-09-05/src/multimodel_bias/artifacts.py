from __future__ import annotations

from typing import Any

PRIVATE_FIELD_NAMES = {
    "api_key",
    "creator_user_id",
    "hash",
    "key_hash",
    "key_metadata",
    "user_id",
}


def sanitize_for_artifact(value: Any) -> Any:
    """Remove account identifiers and embedded image bytes from persisted API metadata."""
    if isinstance(value, dict):
        return {
            key: sanitize_for_artifact(item)
            for key, item in value.items()
            if key.casefold() not in PRIVATE_FIELD_NAMES
        }
    if isinstance(value, list):
        return [sanitize_for_artifact(item) for item in value]
    if isinstance(value, str) and "data:image/" in value.casefold():
        return "[redacted embedded image payload]"
    return value
