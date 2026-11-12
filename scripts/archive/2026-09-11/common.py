import csv
import getpass
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runtime/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "runtime/cache"))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def json_read(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError(f"No records: {path}")
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def secret(service):
    env = service.upper().replace("-", "_")
    if os.environ.get(env):
        return os.environ[env].strip()
    result = subprocess.run(["security", "find-generic-password", "-a", getpass.getuser(),
                             "-s", service, "-w"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Credential unavailable: {service}")
    return result.stdout.strip()


def local_seed(panel, image_id, iteration):
    message = f"9112026|{panel}|{image_id}|{iteration}"
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:4], "big") % (2**31)


def merge_slots(archived, new):
    """Allow replacing technical failures, never completed or invalid model answers."""
    merged={}
    for row in archived+new:
        key=(row["model_key"],row["image_id"],int(row["iteration"]))
        if key in merged and merged[key].get("response_source") not in {"api_error","budget_stopped"}:
            raise ValueError("Attempted replacement of a completed model answer")
        merged[key]=row
    return list(merged.values())
