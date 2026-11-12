#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
records = []
for directory in (root / "results", root / "runtime"):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".log"}:
            records.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            )
print(json.dumps(records))
