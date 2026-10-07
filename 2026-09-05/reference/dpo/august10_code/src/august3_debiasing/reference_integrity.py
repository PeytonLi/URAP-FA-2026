from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sha256sums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid SHA256SUMS line {line_number}")
        digest, filename = parts
        filename = filename.lstrip("*")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"invalid SHA-256 digest at line {line_number}")
        if not filename or filename in checksums:
            raise ValueError(f"invalid or duplicate SHA256SUMS filename at line {line_number}")
        checksums[filename] = digest
    if not checksums:
        raise ValueError("SHA256SUMS is empty")
    return checksums


def verify_named_reference_files(
    checksums_path: Path,
    files: Mapping[str, Path],
) -> dict[str, str]:
    """Verify immutable reference bytes against the repository checksum ledger."""

    checksums = read_sha256sums(checksums_path)
    actual: dict[str, str] = {}
    failures: list[str] = []
    for canonical_name, path in files.items():
        expected = checksums.get(canonical_name)
        if expected is None:
            failures.append(f"{canonical_name}:missing_from_checksum_ledger")
            continue
        if not path.is_file():
            failures.append(f"{canonical_name}:file_missing")
            continue
        digest = sha256_file(path)
        actual[canonical_name] = digest
        if digest != expected:
            failures.append(f"{canonical_name}:sha256_mismatch")
    if failures:
        raise ValueError("frozen reference integrity failure: " + ", ".join(failures))
    return actual


def verify_manifest_internal_hashes(
    manifest: Mapping[str, object],
    actual: Mapping[str, str],
) -> None:
    source = manifest.get("source_sha256")
    portable = manifest.get("portable_sha256")
    if not isinstance(source, Mapping) or not isinstance(portable, Mapping):
        raise ValueError("frozen input manifest is missing its internal hash maps")
    expected = {
        "base_gemma3_fairface_iat.csv": source.get("base_eval"),
        "base_gemma3_neutral_iat.csv": source.get("base_neutral"),
        "eval_metadata.csv": portable.get("eval_metadata"),
        "neutral_benchmark_metadata.csv": portable.get("neutral_metadata"),
    }
    mismatches = [
        name
        for name, digest in expected.items()
        if not isinstance(digest, str) or actual.get(name) != digest
    ]
    if mismatches:
        raise ValueError(
            "frozen manifest internal hashes disagree with reference bytes: "
            + ", ".join(mismatches)
        )
