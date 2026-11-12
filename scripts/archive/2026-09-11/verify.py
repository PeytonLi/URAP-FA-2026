"""Verify artifacts, statistical totals, privacy boundaries and publishable inventory."""
import argparse
import json
import re
from collections import Counter

from common import ROOT, json_read, read_jsonl, rows, sha, write_json

EXCLUDED_PARTS={".venv","runtime","__pycache__",".pytest_cache",".ruff_cache",".git","private"}
ROOT_FILES={"README.md","METHODS.md","RESULTS.md","COVERAGE.md","DATA_DICTIONARY.md","QWEN_COMPATIBILITY.md","SESSION_NOTES.md",
            "LICENSE","requirements.lock","pyproject.toml",".gitignore"}


def public_files():
    selected=[]
    for name in ROOT_FILES:
        p=ROOT / name
        if p.exists():
            selected.append(p)
    for dirname in ("scripts","src","tests","config","reference","results","plots","data"):
        for p in (ROOT / dirname).rglob("*"):
            relative=p.relative_to(ROOT)
            if not p.is_file() or set(relative.parts)&EXCLUDED_PARTS or p.name in {"worker.lock","integrity.json","artifact_checksums.json"}:
                continue
            if dirname=="data" and p.suffix!=".csv":
                continue
            if p.suffix in {".pyc",".safetensors",".bin",".tar",".gz"}:
                continue
            selected.append(p)
    return sorted(set(selected))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--freeze",action="store_true")
    args=parser.parse_args()
    files=public_files()
    secrets=re.compile(rb"(?:sk-or-v1-|sk-ant-api\d+-|sk-proj-|rpa_|hf_|AIza)[A-Za-z0-9_-]{15,}|data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/]{20,}|-----BEGIN (?:(?:OPENSSH|RSA|EC|ENCRYPTED) )?PRIVATE KEY-----")
    for p in files:
        if p.stat().st_size>=95*1024**2:
            raise ValueError(f"File too large for safe GitHub release: {p}")
        if p.suffix in {".json",".jsonl",".csv",".md",".py",".toml",".lock"} and secrets.search(p.read_bytes()):
            raise ValueError(f"Secret or image payload in publishable file: {p}")
    summary=rows(ROOT / "results/analysis/summary.csv")
    words=rows(ROOT / "results/analysis/word_counts.csv")
    for panel in ("neutral12","utkface"):
        rr=rows(ROOT / f"results/analysis/{panel}_responses.csv")
        keys={(r["model_key"],r["image_id"],int(r["iteration"])) for r in rr}
        assert len(keys)==len(rr), "Duplicate analysis slots"
        for s in (r for r in summary if r["panel"]==panel):
            selected=[r for r in rr if r["model_key"]==s["model_key"]]
            valid=[r for r in selected if r["valid_three"]=="True"]
            assert len(valid)==int(s["valid_responses"])
            assert len(selected)==int(s["attempted_slots"])
            wc=[r for r in words if r["panel"]==panel and r["model_key"]==s["model_key"]]
            assert len(wc)==122 and sum(int(r["count"]) for r in wc)==3*len(valid)
            for race in ("Black","White"):
                observed=Counter(w for r in valid if r["race"]==race for w in r["words"].split("|"))
                assert all(int(r["count"])==observed[r["word"]] for r in wc if r["race"]==race)
    calls=read_jsonl(ROOT / "results/hosted/calls.jsonl")
    session=json_read(ROOT / "results/hosted/session.json")
    assert len(calls)==session["api_calls"]
    assert abs(sum(float(r.get("cost_usd",0)) for r in calls)-session["reported_cost_usd"])<1e-7
    assert max(session["reported_cost_usd"],session["account_usage_delta"])<=json_read(ROOT / "config/study.json")["openrouter_max_usd"]
    inventory={str(p.relative_to(ROOT)):sha(p) for p in files}
    checksums=ROOT / "artifact_checksums.json"
    if args.freeze:
        write_json(checksums,inventory)
    else:
        frozen=json_read(checksums)
        for name,digest in frozen.items():
            if not (ROOT / name).exists() or sha(ROOT / name)!=digest:
                raise ValueError(f"Artifact checksum mismatch: {name}")
    report=dict(files_checked=len(files),duplicate_slots=False,word_counts_reconciled=True,
                ledger_reconciled=True,privacy_scan_passed=True,
                note="Private images/weights intentionally excluded; hashes are not proof of person-disjointness.")
    write_json(ROOT / "results/integrity.json",report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
