"""Resumable IAT expansion. API keys/images never enter saved API artifacts."""
import argparse
import dataclasses
import fcntl
import hashlib
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, json_read, merge_slots, now, read_jsonl, rows, secret, sha, write_csv, write_json

sys.path.insert(0, str(ROOT / "src"))
from multimodel_bias.artifacts import sanitize_for_artifact
from multimodel_bias.constants import IAT_PROMPT
from multimodel_bias.iat import (
    BudgetExceededError, EndpointUnavailableError, IATBenchmarkRunner, IATRunConfig, ModelSpec,
    _request_payload, _safe_response_snapshot, _usage,
    load_stimuli, parse_three_words, validated_model_snapshot,
)
from multimodel_bias.openrouter import OpenRouterClient, OpenRouterError, OpenRouterFailure

OUT = ROOT / "results/hosted"


def plan_panel(panel):
    study = json_read(ROOT / "config/study.json")
    cfg = study["panels"][panel]
    archive = [r for r in rows(ROOT / f"reference/iat_{panel}_responses.csv")
               if r["model_key"] != "gemma_local"]
    stimuli = load_stimuli(ROOT / cfg["manifest"], expected_count=12 if panel == "neutral12" else 400)
    image_map = {s.image_id: s for s in stimuli}
    indexes = {}
    seen = set()
    retained = []
    for r in archive:
        image = image_map[r["image_id"]]
        key = (r["model_key"], r["image_id"], int(r["iteration"]))
        if key in seen or r["image_sha256"] != image.sha256:
            raise ValueError("Duplicate archive task or image drift")
        seen.add(key)
        offset = int(r["seed"]) - cfg["base_seed"] - 1000 * int(r["iteration"])
        if indexes.get(image.image_id, offset) != offset:
            raise ValueError("Archived image ordering differs between models")
        indexes[image.image_id] = offset
        parsed = parse_three_words(r.get("fallback_raw") or r.get("sampled_raw") or r["response"])
        if r["valid_three"].lower() == "true" and "|".join(parsed) != r["words"]:
            raise ValueError("Archived parser contract differs")
        if panel == "neutral12" and r["model_key"] == "mimo_v25":
            continue
        retained.append(dict(r, panel=panel, provenance="archived", new_cost_usd=0.0))
    if sorted(indexes.values()) != list(range(1, len(stimuli) + 1)):
        raise ValueError("Invalid archived image indexes")
    stimuli.sort(key=lambda s: indexes[s.image_id])
    models = [ModelSpec(**m) for m in cfg["models"]]
    concurrency = json_read(ROOT / "config/execution.json").get("operational_concurrency", {})
    models = [dataclasses.replace(m, max_concurrency=concurrency.get(f"{panel}/{m.key}", m.max_concurrency)) for m in models]
    config = IATRunConfig(
        benchmark_version="iat12-pipeline-reference-v1" if panel == "neutral12" else "utkface-legacy61-iat-v1",
        repeats=cfg["repeats"], temperature=1.0, top_p=1.0, base_seed=cfg["base_seed"],
        fallback_enabled=True, fallback_max_tokens=cfg["fallback_max_tokens"], max_workers=16,
        minimum_request_interval_seconds=0.1, max_run_cost_usd=study["openrouter_max_usd"],
        minimum_account_balance_usd=study["openrouter_min_balance_usd"], bootstrap_samples=100000,
        provider={"allow_fallbacks": False, "data_collection": "deny", "zdr": True},
    )
    return config, models, stimuli, retained


class SessionBudget:
    def __init__(self, client, snapshots):
        self.client, self.lock = client, threading.RLock()
        self.inflight = 0.0
        self.stopped = False
        self.blocked_routes = set()
        self.snapshots = snapshots
        self.route_slots = {key: threading.Semaphore(8 if key == ("minimax/minimax-m3", "venice/fp8") else
                                                    6 if key[0]=="x-ai/grok-4.6" else
                                                    4 if key[0]=="moonshotai/kimi-k3" else 3)
                            for key in snapshots}
        self.path = OUT / "session.json"
        self.study = json_read(ROOT / "config/study.json")
        if self.path.exists():
            self.state = json_read(self.path)
        else:
            self.state = dict(started_at=now(), credits_initial=client.credits(),
                              reported_cost_usd=0.0, api_calls=0, errors=0)
        ledger = read_jsonl(OUT / "calls.jsonl")
        self.recovered = {(r["panel"],r["task_id"],r["stage"]):r["response"]
                          for r in ledger if not r.get("error") and r.get("response")}
        self.state["reported_cost_usd"] = sum(float(r.get("cost_usd", 0)) for r in ledger)
        self.state["api_calls"] = len(ledger)
        self.state["unknown_cost_reserve_usd"] = sum(float(r.get("unknown_cost_reserve_usd", .10)) for r in ledger if r.get("error")=="unknown_call_outcome")
        self.refresh()

    def refresh(self):
        credit = self.client.credits()
        self.state.update(credits_latest=credit, checked_at=now(),
                          account_usage_delta=credit["total_usage"] - self.state["credits_initial"]["total_usage"])
        write_json(self.path, self.state)

    def reserve(self, model, payload):
        price = self.snapshots[model.model_id, model.provider_endpoint]["selected_endpoint"]["pricing"]
        reservation = 16384 * float(price.get("prompt") or 0) + payload["max_tokens"] * float(price.get("completion") or 0) + 0.01
        with self.lock:
            spent = max(self.state["reported_cost_usd"], self.state["account_usage_delta"]) + self.state["unknown_cost_reserve_usd"]
            remaining = self.state["credits_initial"]["remaining"] - spent
            if (self.stopped or (ROOT / "STOP_REQUESTED").exists()
                    or (model.model_id,model.provider_endpoint) in self.blocked_routes
                    or spent + self.inflight + reservation > self.study["openrouter_max_usd"]
                    or remaining - self.inflight - reservation < self.study["openrouter_min_balance_usd"]):
                raise BudgetExceededError("Session stop or pre-call budget reservation failed")
            self.inflight += reservation
        return reservation

    def record(self, reservation, record):
        with self.lock:
            self.inflight -= reservation
            with (OUT / "calls.jsonl").open("a") as f:
                f.write(json.dumps(sanitize_for_artifact(record), sort_keys=True) + "\n")
                f.flush()
            self.state["reported_cost_usd"] += float(record.get("cost_usd", 0))
            self.state["api_calls"] += 1
            if record.get("error"):
                self.state["errors"] += 1
                if record["error"] == "unknown_call_outcome":
                    self.state["unknown_cost_reserve_usd"] += record.get("unknown_cost_reserve_usd", .10)
            if self.state["api_calls"] % 64 == 0:
                self.refresh()
            else:
                write_json(self.path, self.state)


class Runner(IATBenchmarkRunner):
    def __init__(self, panel, client, config, models, stimuli, budget):
        super().__init__(client, panel, config, models, stimuli, output_root=OUT,
                         retry_existing_api_errors=False)
        self.panel, self.budget = panel, budget
        self.failures = Counter()

    def _call(self, *, task_id, stage, model, stimulus, seed, fallback):
        # Reuse a returned API stage if interruption happened before its final row
        # was committed. This avoids paying again or resampling an invalid sample.
        cached = self.budget.recovered.get((self.panel,task_id,stage))
        if cached is not None:
            expected=self.budget.snapshots[model.model_id,model.provider_endpoint]["selected_endpoint"]["provider_name"]
            if cached.get("model") not in {model.model_id,model.api_revision_hint} or cached.get("provider")!=expected:
                raise RuntimeError("Recovered API stage has incompatible model/provider")
            with self.budget.lock:
                with (OUT / "recovery_events.jsonl").open("a") as f:
                    f.write(json.dumps(dict(panel=self.panel,task_id=task_id,stage=stage,
                                           response_id=cached.get("id"),reused_at=now()))+"\n")
            return cached
        self._wait_for_model_slot(model)
        payload = _request_payload(model, self._data_urls[stimulus.image_id], self.config, seed, fallback=fallback)
        payload["provider"]["only"] = [model.provider_endpoint]
        reservation = self.budget.reserve(model, payload)
        record = dict(panel=self.panel, task_id=task_id, stage=stage, requested_at=now(),
                      model_key=model.key, provider_endpoint=model.provider_endpoint,
                      image_sha256=stimulus.sha256, seed=seed,
                      prompt_sha256=hashlib.sha256(IAT_PROMPT.encode()).hexdigest(),
                      request_parameters={k:v for k,v in payload.items() if k != "messages"})
        try:
            with self.budget.route_slots[model.model_id, model.provider_endpoint]:
                response = self.client.chat_completion(payload)
        except OpenRouterError as exc:
            record.update(error=f"HTTP {exc.failure.status_code}", cost_usd=0.0)
            self.budget.record(reservation, record)
            self.failures[model.key] += 1
            if self.failures[model.key] >= 3:
                self.budget.blocked_routes.add((model.model_id,model.provider_endpoint))
            raise OpenRouterError(type(exc.failure)(exc.failure.status_code,
                                                   f"HTTP {exc.failure.status_code}", {})) from None
        except (TimeoutError, OSError) as exc:
            record.update(error="unknown_call_outcome", error_type=type(exc).__name__, cost_usd=0.0,
                          unknown_cost_reserve_usd=reservation)
            self.budget.record(reservation, record)
            self.failures[model.key] += 1
            if self.failures[model.key] >= 3:
                self.budget.blocked_routes.add((model.model_id,model.provider_endpoint))
            raise OpenRouterError(OpenRouterFailure(0, "Transport timeout; billed outcome unknown", {})) from None
        except Exception:
            record.update(error="unknown_call_outcome", cost_usd=0.0)
            self.budget.record(reservation, record)
            self.budget.stopped = True
            raise
        usage = _usage(response)
        snapshot = _safe_response_snapshot(response)
        for choice in snapshot.get("choices", []):
            choice.get("message", {}).pop("reasoning_details", None)
        record.update(usage, response=snapshot)
        self.budget.record(reservation, record)
        self.failures[model.key] = 0
        if response.get("model") not in {model.model_id, model.api_revision_hint}:
            self.budget.stopped = True
            raise RuntimeError("Returned model identifier drift")
        if (response.get("usage") or {}).get("cost") is None:
            self.budget.stopped = True
            raise RuntimeError("Missing per-call cost; account reconciliation required")
        expected = self.budget.snapshots[model.model_id, model.provider_endpoint]["selected_endpoint"]["provider_name"]
        if response.get("provider") != expected:
            self.budget.stopped = True
            raise RuntimeError("Returned provider drift")
        return response

    def _append_jsonl(self, path, record):
        if path == self.records_path:
            record.update(panel=self.panel, provenance="new", new_cost_usd=record.get("cost_usd", 0))
        super()._append_jsonl(path, record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("dry-run", "preflight", "probe", "run"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plans = {p: plan_panel(p) for p in ("neutral12", "utkface")}
    exclusions = json_read(ROOT / "config/execution.json")["excluded_cells"]
    report = {p: dict(archived=len(v[3]), total=len(v[1])*len(v[2])*v[0].repeats,
                     new=len(v[1])*len(v[2])*v[0].repeats-len(v[3])) for p,v in plans.items()}
    write_json(OUT / "reuse_audit.json", dict(panels=report, prompt_sha256=hashlib.sha256(IAT_PROMPT.encode()).hexdigest(),
                 input_checksums_sha256=sha(ROOT / "reference/input_checksums.json"),
                 mimo_neutral12="600 fresh FP8 outputs; BF16 archive excluded; authorized new condition",
                 archived_invalids="retained, never silently replaced", checked_at=now()))
    print(json.dumps(report), flush=True)
    if args.action == "dry-run":
        return
    client = OpenRouterClient(secret("openrouter-api-key"), max_retries=0, timeout_seconds=120)
    catalog = client.model_catalog()
    write_json(ROOT / "results/preflight/catalog.json", catalog)
    snapshots, unavailable, availability = {}, {}, []
    for panel, (config, models, stimuli, archived) in plans.items():
        for model in models:
            if f"{panel}/{model.key}" in exclusions:
                continue
            key = (model.model_id, model.provider_endpoint)
            if key not in snapshots and key not in unavailable:
                try:
                    snapshots[key] = validated_model_snapshot(client, [model])[0]
                except EndpointUnavailableError as exc:
                    unavailable[key]=dict(status=exc.status, reason=str(exc))
            if key in unavailable:
                exclusions[f"{panel}/{model.key}"]=unavailable[key]["reason"]
            availability.append(dict(panel=panel,model_key=model.key,endpoint=model.provider_endpoint,
                                     status=unavailable.get(key,{}).get("status",0),checked_at=now()))
    write_json(ROOT / "results/preflight/routes.json", list(snapshots.values()))
    write_json(ROOT / "results/preflight/availability.json",availability)
    with (ROOT / "results/preflight/availability_history.jsonl").open("a") as f:
        f.write(json.dumps(availability)+"\n")
    if args.action == "preflight":
        print(json.dumps(dict(available=len(snapshots),unavailable_cells=exclusions)),flush=True)
        return
    budget = SessionBudget(client, snapshots)
    runners, tasks, probe_slots = {}, {}, {}
    for panel, (config, models, stimuli, archived) in plans.items():
        runner = Runner(panel, client, config, models, stimuli, budget)
        runners[panel] = runner
        write_csv(runner.output_dir / "archived_responses.csv", archived)
        previous = read_jsonl(runner.records_path)
        previous_latest = list({r["task_id"]: r for r in previous}.values())
        completed = {(r["model_key"], r["image_id"], int(r["iteration"])) for r in archived+previous_latest
                     if r.get("response_source") not in {"api_error", "budget_stopped"}}
        write_json(runner.output_dir / "contract.json", dict(config=dataclasses.asdict(config), models=[dataclasses.asdict(m) for m in models],
                   prompt=IAT_PROMPT, stimuli=[dict(image_id=s.image_id, sha256=s.sha256) for s in stimuli]))
        for model in models:
            if f"{panel}/{model.key}" in exclusions:
                tasks[panel, model.key] = []
                continue
            archive_keys = {(r["model_key"], r["image_id"], int(r["iteration"])) for r in archived
                            if r.get("response_source") not in {"api_error", "budget_stopped"}}
            pending = [(model,s,i,j) for i in range(1,config.repeats+1)
                       for j,s in enumerate(stimuli,1) if (model.key,s.image_id,i) not in archive_keys]
            if args.action == "probe":
                selected, strata = [], set()
                for task in pending:
                    stratum = (task[1].race, task[1].gender)
                    if stratum not in strata:
                        selected.append(task)
                        strata.add(stratum)
                pending = selected
                probe_slots[panel,model.key] = {(t[1].image_id,t[2]) for t in selected}
            tasks[panel,model.key] = [t for t in pending if (model.key,t[1].image_id,t[2]) not in completed]
    if args.action == "run":
        probe = json_read(OUT / "probe.json")
        if not probe["technically_valid"]:
            raise ValueError("Compatibility probe failed")
    tick = time.time()
    completed_new = []
    progress_lock = threading.Lock()
    # Two lanes per model: disjoint tasks; provider cap is additionally enforced by each runner.
    concurrency = json_read(ROOT / "config/execution.json").get("operational_concurrency", {})
    lanes = [(panel, key, tasks[panel,key][offset::concurrency.get(f"{panel}/{key}",2)])
             for panel,key in tasks for offset in range(concurrency.get(f"{panel}/{key}",2))]
    def lane(item):
        panel, key, work = item
        for model, stimulus, iteration, index in work:
            if (budget.stopped or (model.model_id,model.provider_endpoint) in budget.blocked_routes
                    or time.time()-tick > 6*3600 or (ROOT / "STOP_REQUESTED").exists()):
                break
            try:
                row = runners[panel]._run_task_limited(model,stimulus,iteration,index)
                if row.get("response_source") == "budget_stopped":
                    if (model.model_id,model.provider_endpoint) not in budget.blocked_routes:
                        budget.stopped = True
                with progress_lock:
                    completed_new.append(row)
                    if len(completed_new) % 100 == 0:
                        print(f"new={len(completed_new)} cost=${budget.state['reported_cost_usd']:.4f} elapsed={time.time()-tick:.1f}s",flush=True)
                        write_json(OUT / "progress.json", dict(new_this_invocation=len(completed_new), elapsed_seconds=time.time()-tick,
                                   cost_usd=budget.state["reported_cost_usd"], last_update=now()))
            except Exception as exc:
                budget.stopped = True
                write_json(OUT / "stop_reason.json", dict(type=type(exc).__name__, detail=str(exc)[:200], at=now()))
                break
    with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
        list(executor.map(lane, lanes))
    for panel, runner in runners.items():
        latest = list({r["task_id"]: r for r in read_jsonl(runner.records_path)}.values())
        write_csv(runner.output_dir / "responses.csv", merge_slots(plans[panel][3],latest))
    budget.refresh()
    if args.action == "probe":
        groups = {}
        for panel,key in tasks:
            if f"{panel}/{key}" in exclusions:
                continue
            rr=list({r["task_id"]:r for r in read_jsonl(runners[panel].records_path) if r["model_key"]==key}.values())
            rr=[r for r in rr if (r["image_id"],int(r["iteration"])) in probe_slots[panel,key]]
            groups[f"{panel}/{key}"]=dict(rows=len(rr),valid=sum(bool(r["valid_three"]) for r in rr))
        valid = bool(groups) and not budget.stopped and not budget.blocked_routes and all(g["rows"]==4 and g["valid"]>=3 for g in groups.values())
        write_json(OUT / "probe.json",dict(technically_valid=valid, groups=groups, excluded_cells=exclusions, completed_at=now()))
        if not valid:
            raise RuntimeError("Probe requires review before expansion")
    write_json(OUT / "invocation_exit.json",dict(action=args.action, stopped=budget.stopped, finished_at=now(),
               new_rows=len(completed_new), elapsed_seconds=time.time()-tick, unavailable_cells=exclusions,
               blocked_routes=sorted(budget.blocked_routes)))
    print(json.dumps(budget.state),flush=True)


if __name__ == "__main__":
    main()
