"""Resumable two-teacher candidate elicitation with a shared spend ledger.

No held-out image or benchmark result is used to choose curriculum words.
Requests and receipts remain private, including invalid/refused responses.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import hashlib
import json
import os
import random
import urllib.error
import urllib.request
from collections import Counter, defaultdict

import common
import policy_v3
import teacher_api

TEACHERS = {"A": "gpt-5-mini-2025-08-07", "B": "gpt-4.1-mini-2025-04-14"}
# USD per million tokens, including image inputs in the provider usage count.
PRICES = {"A": (0.25, 2.0), "B": (0.40, 1.60)}
PROFILE = "openai"
API_URL = "https://api.openai.com/v1/chat/completions"
KEY_ENV = "URAP_TEACHER_API_KEY"
MAX_TOKENS = 12000
INPUT_TOKEN_CEILING = 8000
STORE = common.PRIVATE / "candidates_v3"


def configure(provider):
    global PROFILE, API_URL, KEY_ENV, TEACHERS, PRICES
    if provider == "openai":
        PROFILE = "openai"
        API_URL = "https://api.openai.com/v1/chat/completions"
        KEY_ENV = "URAP_TEACHER_API_KEY"
        TEACHERS = {"A": "gpt-5-mini-2025-08-07", "B": "gpt-4.1-mini-2025-04-14"}
        PRICES = {"A": (0.25, 2.0), "B": (0.40, 1.60)}
    elif provider == "bitdeer":
        PROFILE = "bitdeer"
        API_URL = "https://api-inference.bitdeer.ai/v1/chat/completions"
        KEY_ENV = "URAP_BITDEER_API_KEY"
        TEACHERS = {"A": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                    "B": "zai-org/GLM-5.3-Flash"}
        # Official pricing page list prices, not temporary sale/cache prices:
        # https://www.bitdeer.ai/en/pricing/ai-models
        PRICES = {"A": (0.30, 1.20), "B": (0.15, 0.50)}
    else:
        raise ValueError("Unknown teacher provider")


def target_rows():
    return [r for r in common.labelable_rows() if r["source_race"] in ("Black", "White")]


def stable_key(*parts):
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True).encode()).hexdigest()


def pilot_rows(rows, limit):
    """Round-robin complete blocks without selecting on teacher acceptance."""
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["match_set_id"]].append(row)
    strata = defaultdict(list)
    for sid in sorted(blocks, key=lambda s: stable_key("pilot", s)):
        row = blocks[sid][0]
        strata[(row["dataset"], row["split"],
                row["expression_class"])].append(blocks[sid])
    selected: list[dict] = []
    while strata and len(selected) < limit:
        for key in sorted(list(strata)):
            block = strata[key].pop()
            if len(selected) + len(block) <= limit:
                selected.extend(sorted(block, key=lambda r: r["image_id"]))
            if not strata[key]:
                del strata[key]
            if len(selected) == limit:
                break
    return selected


def request_payload(row, tag, words):
    body = teacher_api.teacher_payload(
        common.image_file(row), TEACHERS[tag], policy_v3)
    body["model"] = TEACHERS[tag]
    body["max_completion_tokens"] = MAX_TOKENS
    order = list(words)
    random.Random(stable_key(row["image_id"], tag, words)).shuffle(order)
    body["messages"][1]["content"][0]["image_url"]["detail"] = "low"
    body["messages"][1]["content"][1]["text"] = (
        "Evaluate only these candidate words; return an empty candidates list "
        "if none fit. No minimum count: " + ", ".join(order)
        + "\nThe rejected answer must contain exactly THREE DIFFERENT words "
        "from the FULL negative-example vocabulary in the system message, "
        "not just this candidate subset. Never repeat a rejected word, even "
        "when the candidates list is empty.")
    schema = copy.deepcopy(policy_v3.RESPONSE_SCHEMA)
    schema["properties"]["candidates"]["items"]["properties"]["word"]["enum"] = order
    body["response_format"]["json_schema"]["schema"] = schema
    if PROFILE == "bitdeer":
        body["max_tokens"] = body.pop("max_completion_tokens")
        body["messages"][1]["content"][0]["image_url"].pop("detail", None)
    return body


def send(payload):
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise RuntimeError(KEY_ENV + " required")
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + key,
                 "Content-Type": "application/json", "User-Agent": "curl/8.5.0"})
    # No hidden automatic retries: an ambiguous network failure consumes its
    # reservation in the ledger and requires an explicit later attempt.
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.load(response)


def load_records():
    path = STORE / "responses.jsonl"
    if not path.exists():
        return []
    return active_policy_records(
        [json.loads(line) for line in path.read_text().splitlines() if line])


def active_policy_records(records):
    """Keep old raw responses private, but never reinterpret a prior protocol."""
    return [r for r in records
            if r.get("policy_id", "visible_expression_candidates_v3") == policy_v3.POLICY_ID]


def approved_candidates(rows, records):
    """Latest explicit assessment wins; missing words in a follow-up are vetoes."""
    by_image = {r["image_id"]: r for r in rows}
    latest = {}
    for record in records:
        if record.get("teacher_profile", "openai") != PROFILE:
            continue
        row = by_image.get(record.get("image_id"))
        if row is None:
            continue
        if (record.get("policy_sha256") != policy_v3.POLICY_SHA
                or record.get("schema_sha256") != policy_v3.SCHEMA_SHA
                or record.get("image_sha256") != row["image_sha256"]
                or record.get("pixel_sha256") != row.get("pixel_sha256")
                or record.get("split") != row["split"]
                or row["split"] not in common.LABEL_SPLITS):
            raise ValueError("candidate_provenance_mismatch")
        tag = record["tag"]
        if tag not in TEACHERS or record["model"] != TEACHERS[tag]:
            raise ValueError("unexpected_teacher")
        if record.get("status") != "valid":
            continue
        value = policy_v3.validate(record["value"], record["evaluated_words"])
        candidates = {c["word"]: c for c in value["candidates"]}
        for word in record["evaluated_words"]:
            latest[row["image_id"], tag, word] = (
                candidates.get(word) if not value["review_flags"] else None,
                record)
    approved = {}
    for row in rows:
        words: list[str] = []
        provenance: dict[str, list[dict]] = {}
        for word in policy_v3.ELIGIBLE:
            entries = [latest.get((row["image_id"], tag, word)) for tag in TEACHERS]
            supported = [e for e in entries if e and e[0] and e[0]["confidence"] >= 0.8]
            if len(supported) == len(TEACHERS):
                words.append(word)
                provenance[word] = []
                for candidate, record in supported:
                    assert candidate is not None
                    provenance[word].append({
                        "request_id": record["request_id"], "teacher": record["model"],
                        "api_id": record["raw_response"]["id"], **candidate})
        if words:
            # Use a teacher-judged negative example, never a synthetic default.
            alternatives = [r for r in records if r.get("image_id") == row["image_id"]
                            and r.get("teacher_profile", "openai") == PROFILE
                            and r.get("status") == "valid"
                            and not r["value"]["review_flags"]
                            and not set(r["value"]["rejected"]).intersection(words)]
            if alternatives:
                negative = alternatives[-1]
                approved[row["image_id"]] = {
                    "words": words, "provenance": provenance,
                    "rejected": negative["value"]["rejected"],
                    "rejected_reason": negative["value"]["rejected_reason"],
                    "rejected_request_id": negative["request_id"],
                }
    return approved


def coverage_plan(rows, approved, per_word=24, round_id=1):
    """Probe missing/rare TRAIN words; each word gets a fixed image budget."""
    train = [r for r in rows if r["split"] == "train"]
    counts = Counter((r["source_race"], w) for r in train
                     for w in approved.get(r["image_id"], {}).get("words", []))
    focus = [w for w in policy_v3.ELIGIBLE
             if min(counts[race, w] for race in ("Black", "White")) < 8]
    plan = defaultdict(set)
    for word in focus:
        # Demographics affect scheduling parity only, never teacher input.
        for race in ("Black", "White"):
            pool = [r for r in train if r["source_race"] == race
                    and word not in approved.get(r["image_id"], {}).get("words", [])]
            pool.sort(key=lambda r: stable_key("curriculum", round_id, word, r["image_id"]))
            for row in pool[:per_word // 2]:
                plan[row["image_id"]].add(word)
    return {iid: sorted(words) for iid, words in sorted(plan.items())}


def run(stage, limit=None, workers=8, cap_usd=20.0, round_id=1, retry_failed=False):
    if workers < 1 or workers > 32 or not 0 < cap_usd <= 20:
        raise ValueError("workers 1..32 and total v3 cap in (0,20] required")
    permission = json.loads((common.REPO / "2026-10-03/config/prime_budget.json").read_text())
    authorization = permission.get("ai_target_approval_candidates_v3", {})
    if (authorization.get("authorized") is not True
            or authorization.get("external_teacher_generation_authorized") is not True
            or cap_usd > authorization.get("labeling_spend_cap_usd", 0)):
        raise ValueError("Candidate teacher calls or requested budget not authorized")
    if stage == "curriculum" and round_id not in (1, 2):
        raise ValueError("At most two predeclared curriculum rounds")
    STORE.mkdir(parents=True, exist_ok=True)
    # flock prevents two concurrent pilot/full processes from overspending.
    import fcntl
    with (STORE / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(stage, limit, workers, cap_usd, round_id, retry_failed)


def _run(stage, limit, workers, cap_usd, round_id, retry_failed=False):
    rows = target_rows()
    forbidden = {r["sha256"] for r in common.read_csv(
        common.REPO / "2026-09-11/data/iat12/manifest.csv")}
    if any(r["image_sha256"] in forbidden for r in rows):
        raise ValueError("benchmark_overlap")
    previous = load_records()
    approved = approved_candidates(rows, previous)
    if stage == "pilot":
        rows = pilot_rows(rows, limit or 48)
    plan = coverage_plan(rows, approved, round_id=round_id) if stage == "curriculum" else {
        r["image_id"]: list(policy_v3.ELIGIBLE) for r in rows}
    ledger = common.JsonlStore(STORE / "ledger.jsonl", "request_id")
    store = common.JsonlStore(STORE / "responses.jsonl", "request_id")
    prior_attempts = defaultdict(list)
    for entry in ledger.values():
        prior_attempts[entry.get("logical_request_id", entry["request_id"])].append(entry)
    used = sum(r.get("charged_usd", r["reservation_usd"]) for r in ledger.values())
    pending = []
    skipped_unresolved = 0
    for row in rows:
        words = plan.get(row["image_id"], [])
        if not words:
            continue
        for tag in TEACHERS:
            logical_id = stable_key(row["image_id"], row["image_sha256"], tag,
                                    policy_v3.POLICY_SHA, policy_v3.SCHEMA_SHA,
                                    "base" if stage != "curriculum" else f"curriculum{round_id}",
                                    sorted(words), PROFILE, TEACHERS[tag])
            attempts = prior_attempts[logical_id]
            if attempts:
                latest = store.get(attempts[-1]["request_id"])
                if not retry_failed or (latest and latest.get("status") == "valid"):
                    if not latest or latest.get("status") != "valid":
                        skipped_unresolved += 1
                    continue
            request_id = logical_id if not attempts else stable_key(logical_id, len(attempts))
            pending.append((row, tag, words, request_id, logical_id))
    if limit and stage != "pilot":
        ids = {r["image_id"] for r, *_ in pending[:2 * limit]}
        pending = [task for task in pending if task[0]["image_id"] in ids]
    report = {"stage": stage, "round": round_id, "policy": policy_v3.POLICY_ID,
              "teacher_profile": PROFILE, "teachers": dict(TEACHERS),
              "accounting": "conservative_uncached_list_prices_usd_per_million_tokens",
              "accounting_prices": dict(PRICES),
              "cap_usd": cap_usd, "completed": 0, "invalid": 0, "failed": 0,
              "budget_exhausted": False, "scheduled": len(pending)}
    report["previous_unresolved_skipped"] = skipped_unresolved

    def work(task):
        row, tag, words, _, _ = task
        if not common.verify_image_file(row):
            raise ValueError("image_hash_mismatch")
        payload = request_payload(row, tag, words)
        return payload, send(payload)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        active: dict[cf.Future, tuple] = {}
        iterator = iter(pending)
        exhausted = False
        while active or not exhausted:
            while not exhausted and len(active) < workers:
                task = next(iterator, None)
                if task is None:
                    exhausted = True
                    break
                row, tag, words, request_id, logical_id = task
                inp, out = PRICES[tag]
                reservation = (INPUT_TOKEN_CEILING * inp + MAX_TOKENS * out) / 1e6
                if used + reservation > cap_usd:
                    report["budget_exhausted"] = True
                    exhausted = True
                    break
                entry = {"request_id": request_id, "logical_request_id": logical_id,
                         "reservation_usd": reservation}
                ledger.append(entry)
                used += reservation
                active[pool.submit(work, task)] = (task, entry)
            if not active:
                break
            done, _ = cf.wait(active, return_when=cf.FIRST_COMPLETED)
            for future in done:
                (row, tag, words, request_id, logical_id), entry = active.pop(future)
                record = {"request_id": request_id, "logical_request_id": logical_id,
                          "image_id": row["image_id"],
                          "teacher_profile": PROFILE, "api_url": API_URL,
                          "image_sha256": row["image_sha256"],
                          "pixel_sha256": row.get("pixel_sha256"),
                          "split": row["split"], "tag": tag, "model": TEACHERS[tag],
                          "evaluated_words": words, "policy_id": policy_v3.POLICY_ID,
                          "policy_sha256": policy_v3.POLICY_SHA,
                          "schema_sha256": policy_v3.SCHEMA_SHA}
                try:
                    payload, raw = future.result()
                    record["request_sha256"] = stable_key(payload)
                    record["raw_response"] = raw
                    usage = raw.get("usage", {})
                    if {"prompt_tokens", "completion_tokens"} <= usage.keys():
                        inp, out = PRICES[tag]
                        cost = (usage["prompt_tokens"] * inp
                                + usage["completion_tokens"] * out) / 1e6
                        used += cost - entry["reservation_usd"]
                        ledger.append({**entry, "charged_usd": cost, "usage": usage})
                    try:
                        choice = raw["choices"][0]
                        if raw.get("model") != TEACHERS[tag] or not isinstance(raw.get("id"), str):
                            raise ValueError("teacher_response_identity_mismatch")
                        if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
                            raise ValueError("refused_or_truncated")
                        record["value"] = policy_v3.validate(
                            json.loads(choice["message"]["content"]), words)
                        record["status"] = "valid"
                        report["completed"] += 1
                    except (ValueError, KeyError, TypeError) as error:
                        record.update(status="invalid", error=str(error))
                        report["invalid"] += 1
                except urllib.error.HTTPError as error:
                    record.update(status="failed", error_type="HTTPError", http_status=error.code)
                    # Only error codes, never provider messages or auth headers.
                    try:
                        code = json.load(error).get("error", {}).get("code")
                        record["provider_error_code"] = code if isinstance(code, str) else None
                    except (ValueError, AttributeError):
                        pass
                    # Rejected requests do not incur token usage.
                    if 400 <= error.code < 500:
                        used -= entry["reservation_usd"]
                        ledger.append({**entry, "charged_usd": 0, "http_status": error.code})
                    report["provider_blocked"] = {
                        "http_status": error.code, "code": record.get("provider_error_code")}
                    exhausted = True
                    report["failed"] += 1
                except Exception as error:
                    # Never persist exception bodies that might contain auth.
                    record.update(status="failed", error_type=type(error).__name__)
                    report["failed"] += 1
                    report["provider_blocked"] = {"error_type": type(error).__name__}
                    exhausted = True
                store.append(record)
                if sum(report[k] for k in ("completed", "invalid", "failed")) % 20 == 0:
                    print(json.dumps({**report, "accounted_usd": round(used, 4)}), flush=True)
    allrows = target_rows()
    approved = approved_candidates(allrows, active_policy_records(store.values()))
    counts = Counter(w for r in allrows if r["split"] == "train"
                     for w in approved.get(r["image_id"], {}).get("words", []))
    report.update(accounted_usd=used,
                  images_with_supported_words=len(approved),
                  approved_images=sum(len(v["words"]) >= 3 for v in approved.values()),
                  train_candidate_words=dict(counts),
                  uncovered_train_words=sorted(set(policy_v3.ELIGIBLE) - counts.keys()))
    common.write_json(
        STORE / f"{PROFILE}_{policy_v3.POLICY_ID}_{stage}_{round_id}_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilot", "full", "curriculum"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cap-usd", type=float, default=20)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--provider", choices=("openai", "bitdeer"), default="openai")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Explicitly retry failed attempts; raw history and reservations remain")
    args = parser.parse_args()
    configure(args.provider)
    result = run(args.stage, args.limit, args.workers, args.cap_usd, args.round, args.retry_failed)
    if any(result.get(key) for key in ("failed", "invalid", "budget_exhausted",
                                      "provider_blocked", "previous_unresolved_skipped")):
        raise SystemExit(2)
