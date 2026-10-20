"""Teacher-API draft pass for the gated 61-word IAT labeling.

Provider-agnostic over the OpenAI chat-completions shape: OpenAI, Prime
Inference gateway, or any compatible endpoint. Auth and model come from env:
  URAP_TEACHER_API_KEY  (required)
  URAP_TEACHER_BASE_URL (default https://api.openai.com/v1)
  URAP_TEACHER_MODEL    (e.g. gpt-5-mini, anthropic/claude-haiku-4.5 on a
                         gateway that maps model names)

Writes append-only JSONL drafts so an interrupted run resumes without
repeating or selectively resampling images.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import os
import time
import urllib.error
import urllib.request
from decimal import Decimal
from pathlib import Path

import common
import policy

MAX_OUTPUT_TOKENS = 700
REQUEST_RESERVATION_USD = Decimal("0.02")


def teacher_payload(image_path: Path, model: str) -> dict:
    data_url = ("data:image/png;base64,"
                + base64.b64encode(image_path.read_bytes()).decode("ascii"))
    body = {
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "stream": False,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "iat_preference_draft", "strict": True,
            "schema": policy.RESPONSE_SCHEMA}},
        "messages": [
            {"role": "system", "content": policy.POLICY},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",
                 "text": "Draft one image-specific preference pair as JSON."}]}],
    }
    # Reasoning models (gpt-5*, o*) reject sampling temperature; gpt-4o
    # accepts an explicit 0 for determinism.
    if not model.startswith(("gpt-5", "o1", "o3", "o4")):
        body["temperature"] = 0
    else:
        body["reasoning_effort"] = "low"
        body["max_completion_tokens"] = 2000
    return body


def call_teacher(payload: dict, model: str, base_url: str, key: str,
                 timeout: int = 120, retries: int = 3) -> tuple[dict, dict]:
    body = dict(payload)
    body["model"] = model
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            content = data["choices"][0]["message"]["content"]
            value = json.loads(content)
            receipt = {"model": data.get("model"), "id": data.get("id"),
                       "usage": data.get("usage"),
                       "created": data.get("created")}
            return value, receipt
        except urllib.error.HTTPError as exc:  # pragma: no cover
            last = f"http_{exc.code}"
            if exc.code in (400, 401, 403, 404, 422):
                raise RuntimeError(f"teacher_http_{exc.code}") from exc
        except Exception as exc:  # pragma: no cover
            last = repr(exc)[:200]
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"teacher_failed:{last}")


def run_pass(pass_tag: str, limit: int | None = None,
             workers: int = 8) -> dict:
    key = os.environ.get("URAP_TEACHER_API_KEY", "")
    model = os.environ.get("URAP_TEACHER_MODEL", "gpt-5-mini")
    base_url = os.environ.get("URAP_TEACHER_BASE_URL",
                              "https://api.openai.com/v1")
    if not key:
        raise SystemExit("URAP_TEACHER_API_KEY not set")
    rows = common.labelable_rows()
    store = common.JsonlStore(
        common.PRIVATE / f"drafts_{pass_tag}.jsonl", key="image_id")
    todo = [r for r in rows if r["image_id"] not in store]
    if limit:
        todo = todo[:limit]
    results = {"attempted": 0, "drafted": 0, "failed": []}
    errors_path = common.PRIVATE / f"errors_{pass_tag}.jsonl"

    def work(row):
        value, receipt = call_teacher(
            teacher_payload(common.image_file(row), model),
            model, base_url, key)
        return row, value, receipt

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, r): r for r in todo}
        for fut in cf.as_completed(futures):
            row = futures[fut]
            results["attempted"] += 1
            try:
                row, value, receipt = fut.result()
                rec = policy.draft_record(
                    row, value, pass_tag, f"{base_url}|{model}", receipt)
                store.append(rec)
                results["drafted"] += 1
            except Exception as exc:
                err = {"image_id": row["image_id"], "error": str(exc)[:300],
                       "pass_tag": pass_tag}
                with errors_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(err) + "\n")
                results["failed"].append(err)
            if results["attempted"] % 100 == 0:
                print(f"{pass_tag}: {results['attempted']}/{len(todo)} "
                      f"drafted={results['drafted']} "
                      f"failed={len(results['failed'])}", flush=True)
    common.write_json(common.PRIVATE / f"run_{pass_tag}_summary.json",
                      {"pass_tag": pass_tag, "model": model,
                       "base_url": base_url,
                       "todo": len(todo), **{k: (len(v) if isinstance(v, list)
                                                else v)
                                            for k, v in results.items()}})
    print("pass done:", results["attempted"], "attempted,",
          results["drafted"], "drafted,", len(results["failed"]), "failed")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass-tag", required=True, choices=["A", "B"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run_pass(args.pass_tag, args.limit, args.workers)


if __name__ == "__main__":
    main()
