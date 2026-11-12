"""Audit and analyze prompt batches (v1: prompts 2-4; A: 5-8; B: 9-13) without training, inference or provider calls."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
sys.path.insert(0, str(REPO / "scripts" / "train_eval"))
sys.path.insert(0, str(REPO / "scripts" / "archive" / "multimodel_bias_src"))

import iat_significance_v4 as stats
import variety
from finetune_significance import holm, sign_flip_pvalue
import benchmark
import iat_prompts
import native_train
import study
from multimodel_bias.iat import deterministic_seed

try:
    from repo_paths import PROMPT_SWEEP_PLANS, RESULTS_V4_STATS
    PLANS = dict(PROMPT_SWEEP_PLANS)
    PROMPT1_PAIRED = RESULTS_V4_STATS / "paired_responses.csv"
except Exception:  # pragma: no cover - fallback if repo_paths missing
    PLANS = {
        batch: REPO / "configs" / "prompt_sweep" / Path(name).name
        for batch, name in iat_prompts.PLAN_FILES.items()
    }
    PROMPT1_PAIRED = REPO / "results" / "v4" / "iat" / "iat_stats_v4" / "paired_responses.csv"
CANNOT_TELL = re.compile(r"^[\W_]*(i\s+)?cannot\s+tell\b", re.IGNORECASE)
SEPARATE_FAMILIES = {"p8": "mitigation_", "p9": "negative_control_", "p15": "behavioral_probe_"}
PRIMARY_TESTS = ("negative_gap_change_did", "word_entropy_change", "unordered_triples_change")
VS_PROMPT1_TESTS = (
    ("negative_rate", "negative_gap_vs_prompt_1", "gap", "proportion",
     "exact race-label permutation of per-image prompt-minus-prompt-1 negative rates"),
    ("entropy", "word_entropy_vs_prompt_1", None, "normalized entropy",
     "exact sign-flip of per-image prompt-minus-prompt-1 entropy"),
    ("triples", "unordered_triples_vs_prompt_1", None, "distinct triples per image",
     "exact sign-flip of per-image prompt-minus-prompt-1 distinct triples"),
)
ADAPTER_SHA256 = iat_prompts.ADAPTER_SHA256
SWEEP_SOURCES = stats.FINAL_SOURCES | {"fallback_invalid"}


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def normalized_contract(contract):
    contract = json.loads(json.dumps(contract))
    for key in ("adapter_sha256", "checkpoint_selection"):
        contract.pop(key)
    for key in ("key", "display_name"):
        contract["model"].pop(key)
    contract["config"].pop("benchmark_version")
    return contract


def audit_prompt(recovered, prompt_id):
    spec = iat_prompts.prompt_spec(prompt_id)
    settings = native_train.MODELS["gemma3_12b"]
    frozen = study.read_csv(REPO / "2026-09-11/data/iat12/manifest.csv")
    identities = [{key: row[key] for key in
                   ("image_id", "filename", "race", "gender", "sha256")} for row in frozen]
    data, contracts, audits, sources = {}, {}, {}, []
    for method, label in (("base", "base"), ("orpo", "tuned")):
        folder = recovered / "results/native_prompt_sweep" / ("prompt_" + prompt_id) / (
            "gemma3_12b_" + method + "_native_balanced_prompt_" + prompt_id)
        manifest = json.loads((folder / "run_manifest.json").read_text())
        stats.require(manifest["status"] == "complete"
                      and manifest["recorded_responses"] == 600, "Incomplete condition")
        contract = manifest["contract"]
        stats.require(all(contract[key] == value for key, value in spec.items()),
                      "Prompt identity/text/hash mismatch")
        stats.require(contract["model"]["model_id"] == settings["model"]
                      and contract["weight_and_processor_revision"] == settings["revision"],
                      "Wrong model or revision")
        expected_config = asdict(replace(benchmark.fw.run_config(1),
            benchmark_version=iat_prompts.benchmark_version(method, prompt_id)))
        stats.require(contract["config"] == expected_config, "Generation/scoring contract drift")
        stats.require(contract["stimuli"] == identities, "Frozen image/order/metadata drift")
        stats.require(contract["adapter_sha256"] == (None if method == "base" else ADAPTER_SHA256),
                      "Wrong ORPO adapter")
        rows = stats.load_csv(folder / "responses.csv")
        variety.validate_slots(rows)
        records = jsonl(folder / "responses.jsonl")
        calls = jsonl(folder / "api_responses.jsonl")
        stats.require(len(records) == 600 and len({r["task_id"] for r in records}) == 600,
                      "Missing/duplicate JSONL responses")
        races = {r["image_id"]: r["race"] for r in identities}
        for record in records + calls:
            seed = int(record["seed"]) - (1_000_000 if record.get("stage") == "fallback" else 0)
            text = iat_prompts.render(prompt_id, record["image_id"], races[record["image_id"]], seed)
            stats.require(record["prompt_id"] == prompt_id
                          and record["prompt_sha256"] == hashlib.sha256(text.encode()).hexdigest(),
                          "Per-record prompt identity mismatch")
        keyed = {}
        for call in calls:
            stats.require(call["stage"] in ("sampled", "fallback"), "Unknown call stage")
            key = call["task_id"], call["stage"]
            stats.require(key not in keyed, "Duplicate API call stage")
            keyed[key] = call
            expected = {"max_new_tokens": (spec.get("options") or {}).get("max_new_tokens", 30),
                        "do_sample": call["stage"] == "sampled",
                        "use_cache": True, "top_k": 0, "top_p": 1.0, "temperature": 1.0}
            stats.require(call["request_parameters"] == expected, "Decoding drift")
            options = spec.get("options") or {}
            if options:
                stats.require(call["image_input"] == options["image"]
                              and ("gray_image_sha256" in call) == (options["image"] != "face"),
                              "Image substitution not provable")
                stats.require(call["response"]["choices"][0]["message"]["content"]
                              == iat_prompts.answer_text(prompt_id, call["generated_text"]),
                              "Scored text is not the frozen extraction of the generation")
        sampled = {task for task, stage in keyed if stage == "sampled"}
        fallback = {task for task, stage in keyed if stage == "fallback"}
        stats.require(sampled == {r["task_id"] for r in rows}
                      == {r["task_id"] for r in records}, "Sampled task set mismatch")
        stats.require(fallback == {r["task_id"] for r in rows
                                   if r["response_source"].startswith("fallback")},
                      "Fallback/scorer mismatch")
        image_indices = {r["image_id"]: i for i, r in enumerate(identities, start=1)}
        image_map = {r["image_id"]: r for r in identities}
        for row in rows:
            image = image_map[row["image_id"]]
            stats.require((row["image_filename"], row["image_sha256"], row["race"], row["gender"])
                          == (image["filename"], image["sha256"], image["race"], image["gender"]),
                          "Response image metadata mismatch")
            seed = deterministic_seed(expected_config["base_seed"], int(row["iteration"]),
                                      image_indices[row["image_id"]])
            value = (expected_config["benchmark_version"] + "|" + settings["model"] + "|"
                     + image["sha256"] + "|" + str(row["iteration"]))
            expected_id = hashlib.sha256(value.encode()).hexdigest()[:24]
            stats.require(row["task_id"] == expected_id and int(row["seed"]) == seed,
                          "Task namespace or deterministic seed mismatch")
            for stage in ("sampled", "fallback"):
                if stage == "fallback" and row["task_id"] not in fallback:
                    continue
                call = keyed[row["task_id"], stage]
                stats.require(call["image_id"] == row["image_id"]
                              and call["image_sha256"] == row["image_sha256"]
                              and call["seed"] == seed + (1_000_000 if stage == "fallback" else 0),
                              "API image/seed mismatch")
            text = keyed[row["task_id"], "sampled"]["response"]["choices"][0]["message"]["content"]
            stats.require(text.strip() == row["sampled_raw"].strip(), "API sampled text mismatch")
            if row["task_id"] in fallback:
                text = keyed[row["task_id"], "fallback"]["response"]["choices"][0]["message"]["content"]
                stats.require(text.strip() == row["fallback_raw"].strip(),
                              "API fallback text mismatch")
        contracts[label], data[label] = contract, rows
        sources.append((label, folder / "responses.csv"))
        audits[label] = {"files_sha256": {name: stats.digest(folder / name) for name in (
            "run_manifest.json", "responses.csv", "responses.jsonl", "api_responses.jsonl")},
            "sampled_calls": len(sampled), "fallback_calls": len(fallback),
            "length_finish_share": sum(c["response"]["choices"][0]["finish_reason"] == "length"
                                       for c in calls) / len(calls),
            "adapter_sha256": contract["adapter_sha256"]}
    stats.require(normalized_contract(contracts["base"]) == normalized_contract(contracts["tuned"]),
                  "Base/ORPO contracts differ beyond the intended condition namespace")
    stats.validate_pair(data["base"], data["tuned"], SWEEP_SOURCES)
    return data, {"prompt": spec, "conditions": audits, "original_sources_verified": True,
                  "historical_baseline_used_for_tuning_inference": False}, sources


def adjust_across_prompts(results, batch="v1"):
    ids = iat_prompts.batch_ids(batch)
    families = defaultdict(list)
    for row in results:
        families[row["family"]].append(row)
    primary = [p for p in ids if batch == "v1" or p not in SEPARATE_FAMILIES]
    stats.require(len(families["primary"]) == 3 * len(primary), "Need every primary hypothesis")
    label = "_across_prompts_" + "_".join(iat_prompts.REPORT_NUMBERS[p] for p in ids)
    for family, members in families.items():
        for row, p in zip(members, holm([r["raw_p"] for r in members]), strict=True):
            row["holm_p"] = p
            row["multiplicity_family"] = family + label


def unavailable_rows(reason):
    nan = float("nan")
    return [{**stats.result_row(test, "primary", "sampled_exact", None, None, None, (nan, nan), 1.0, 0,
                                "unavailable: " + reason, "unavailable"), "status": "unavailable"}
            for test in PRIMARY_TESTS]


def condition_rows(paired_csv, condition):
    rows = [r for r in stats.load_csv(paired_csv) if r["condition"] == condition]
    stats.require(len(rows) == 600, f"{paired_csv} needs 600 {condition} rows")
    return rows


def align_images(prompt_images, prompt1_images):
    by_id = {row["image_id"]: row for row in prompt1_images}
    aligned = []
    for row in prompt_images:
        other = by_id.get(row["image_id"])
        stats.require(other is not None, "Prompt-1 missing image " + row["image_id"])
        stats.require((row["race"], row["gender"], row["image_sha256"])
                      == (other["race"], other["gender"], other["image_sha256"]),
                      "Image identity drift versus prompt 1")
        aligned.append(other)
    return aligned


def compare_prompt_to_prompt1(prompt_rows, prompt1_rows):
    """Primary-metric contrasts of one prompt versus frozen prompt 1 within one model."""
    prompt = stats.per_image(prompt_rows, "sampled_exact")
    baseline = align_images(prompt, stats.per_image(prompt1_rows, "sampled_exact"))
    races = [row["race"] for row in prompt]
    resamples = stats.bootstrap_indices(races)
    rows = []
    for key, name, metric, unit, method in VS_PROMPT1_TESTS:
        left = np.array([row[key] for row in prompt])
        right = np.array([row[key] for row in baseline])
        delta = left - right
        if metric == "gap":
            effect, p, count = stats.race_test(delta, races)
            ci = stats.interval(delta, resamples, races, "gap")
            perm = count
            used_method = method
        else:
            effect = float(delta.mean())
            p, used_method = sign_flip_pvalue(delta, stats.SEED)
            ci = stats.interval(delta, resamples, races)
            perm = 4096
        rows.append(stats.result_row(
            name, "vs_prompt_1", "sampled_exact",
            float(right.mean()) if key != "negative_rate" else stats.race_test(right, races)[0],
            float(left.mean()) if key != "negative_rate" else stats.race_test(left, races)[0],
            effect, ci, p, perm, used_method, unit))
    return rows


def adjust_vs_prompt1(results, batch):
    ids = iat_prompts.batch_ids(batch)
    families = defaultdict(list)
    for row in results:
        families[row["family"]].append(row)
    primary_ids = [p for p in ids if p not in SEPARATE_FAMILIES]
    for condition in ("base", "tuned"):
        family = "vs_prompt_1_" + condition
        stats.require(len(families[family]) == 3 * len(primary_ids),
                      "Need every vs-prompt-1 primary hypothesis for " + condition)
    label = "_across_prompts_" + "_".join(iat_prompts.REPORT_NUMBERS[p] for p in ids)
    for family, members in families.items():
        for row, p in zip(members, holm([r["raw_p"] for r in members]), strict=True):
            row["holm_p"] = p
            row["multiplicity_family"] = family + label


def run_vs_prompt1(batch_dir, out, batch):
    """Required C/D follow-up: each prompt versus frozen prompt-1 within base and within ORPO."""
    stats.require(batch in ("C", "D"), "vs-prompt-1 is required only for batches C and D")
    stats.require(not out.exists(), "vs-prompt-1 output must start fresh")
    stats.require(PROMPT1_PAIRED.is_file(), "Frozen prompt-1 paired responses missing")
    plan = json.loads(PLANS[batch].read_text())
    ids = iat_prompts.batch_ids(batch)
    prompt1 = {condition: condition_rows(PROMPT1_PAIRED, condition)
               for condition in ("base", "tuned")}
    results, prepared = [], {}
    for prompt_id in ids:
        number = iat_prompts.REPORT_NUMBERS[prompt_id]
        paired = batch_dir / ("prompt_" + number) / "paired_responses.csv"
        stats.require(paired.is_file(), "Missing batch paired responses for prompt " + number)
        prompt_rows = {}
        rows = []
        for condition in ("base", "tuned"):
            try:
                prompt_rows[condition] = condition_rows(paired, condition)
                part = compare_prompt_to_prompt1(prompt_rows[condition], prompt1[condition])
            except ValueError as exc:
                if "Zero valid answers" not in str(exc):
                    raise
                part = [{**stats.result_row(name, "vs_prompt_1", "sampled_exact", None, None, None,
                                            (float("nan"), float("nan")), 1.0, 0,
                                            "unavailable: " + str(exc), "unavailable"),
                         "status": "unavailable"}
                        for _, name, *_ in VS_PROMPT1_TESTS]
            family_prefix = SEPARATE_FAMILIES.get(prompt_id, "")
            for row in part:
                row["family"] = family_prefix + "vs_prompt_1_" + condition
                row.update(prompt=number, run_prompt_id=prompt_id,
                           interpretation=iat_prompts.KINDS[prompt_id],
                           model_condition=condition, reference_prompt="1")
            rows.extend(part)
        results.extend(rows)
        prepared[prompt_id] = prompt_rows, rows
    adjust_vs_prompt1(results, batch)
    out.mkdir(parents=True)
    stats.save_json(out / "analysis_plan.json", {
        **{k: plan[k] for k in ("status", "cross_prompt_vs_prompt_1", "primary_tests",
                                "batches", "report_numbers", "interpretations") if k in plan},
        "reference": {
            "prompt": "1",
            "paired_responses": str(PROMPT1_PAIRED.relative_to(REPO)),
            "paired_inputs_sha256": stats.digest(PROMPT1_PAIRED),
            "source": "2026-10-27/analysis/iat_stats_v4",
        },
        "batch": batch,
        "batch_dir": str(batch_dir),
    })
    variety.write_csv(out / "tests.csv", results)
    flat = [{"prompt": row["prompt"], "run_prompt_id": row["run_prompt_id"],
             "interpretation": row["interpretation"],
             "model_condition": row["model_condition"],
             "reference_prompt": row["reference_prompt"],
             **stats.flat_rows([row])[0]} for row in results]
    variety.write_csv(out / "statistics_table.csv", flat)
    for prompt_id, (_, rows) in prepared.items():
        number = iat_prompts.REPORT_NUMBERS[prompt_id]
        folder = out / ("prompt_" + number)
        folder.mkdir()
        variety.write_csv(folder / "tests.csv", rows)
    primary = [row for row in flat if row["Family"].startswith("vs_prompt_1_")]
    numbers = ", ".join(iat_prompts.REPORT_NUMBERS[p] for p in ids)
    lines = [
        f"# Prompts {numbers} versus prompt 1",
        "",
        "Required Batch " + batch + " follow-up: within base and within saved ORPO separately,",
        "each prompt is compared to the frozen prompt-1 native paired responses on the same",
        "12 images for the three primary metric families (negative Black-White gap via exact",
        "924 race permutations of per-image prompt−prompt-1 differences; entropy and unordered",
        "triples via exact 2^12 sign-flips). Prompt 1 values are listed under Base; the batch",
        "prompt under Tuned; Change is prompt minus prompt 1.",
        "",
        "Prompt 15 stays in behavioral_probe_vs_prompt_1_* families and is excluded from the",
        "primary vs-prompt-1 Holm families. Holm adjusts within each model-condition family",
        "across prompts in this batch. Plots are not a substitute for these tests.",
        "",
        "Reference: `2026-10-27/analysis/iat_stats_v4/paired_responses.csv`.",
        f"Source batch analysis: `{batch_dir}`.",
        "",
        "See statistics_table.csv for effects, 95% image-bootstrap CIs, raw and Holm p-values.",
        "",
        f"Primary vs-prompt-1 rows: {len(primary)}.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats.save_json(out / "provenance.json", {
        "analysis_script_sha256": stats.digest(Path(__file__)),
        "statistical_engine_sha256": stats.digest(Path(stats.__file__)),
        "prompt1_paired_sha256": stats.digest(PROMPT1_PAIRED),
        "batch_dir": str(batch_dir),
        "files_sha256": {str(p.relative_to(out)): stats.digest(p) for p in out.rglob("*")
                         if p.is_file()},
    })
    print(json.dumps(primary, indent=2))


def abstention(rows):
    """Per-image sampled 'Cannot tell' rate; Black minus White with exact 924-assignment p."""
    by_image = defaultdict(list)
    for row in rows:
        by_image[row["image_id"]].append(row)
    rates = {i: sum(bool(CANNOT_TELL.match(r["sampled_raw"])) for r in v) / len(v)
             for i, v in by_image.items()}
    race = {i: v[0]["race"] for i, v in by_image.items()}
    images = sorted(rates)
    black = [i for i in images if race[i] == "Black"]

    def gap(group):
        inside = [rates[i] for i in group]
        outside = [rates[i] for i in images if i not in group]
        return sum(inside) / len(inside) - sum(outside) / len(outside)
    observed = gap(set(black))
    null = [gap(set(c)) for c in itertools.combinations(images, len(black))]
    p = sum(abs(v) >= abs(observed) - 1e-12 for v in null) / len(null)
    other = {i: sum(not CANNOT_TELL.match(r["sampled_raw"]) and not variety.exact_sample(r["sampled_raw"])
                    for r in v) / len(v) for i, v in by_image.items()}
    return {"black_other_invalid_rate": sum(other[i] for i in black) / len(black),
            "white_other_invalid_rate": sum(other[i] for i in images if i not in black) / (len(images) - len(black)),
            "adjustment": "none; descriptive","black_cannot_tell_rate": sum(rates[i] for i in black) / len(black),
            "white_cannot_tell_rate": sum(rates[i] for i in images if i not in black) / (len(images) - len(black)),
            "black_minus_white": observed, "exact_permutation_p": p, "assignments": len(null)}


def verify(recovered, batch="v1"):
    """Launcher gate: exactly the expected base/ORPO conditions, each fully audited."""
    ids = iat_prompts.batch_ids(batch)
    root = recovered / "results/native_prompt_sweep"
    expected = {("prompt_" + p, "gemma3_12b_" + m + "_native_balanced_prompt_" + p)
                for p in ids for m in ("base", "orpo")}
    found = {(d.parent.name, d.name) for d in root.glob("*/*") if d.is_dir()}
    stats.require(found == expected and {d.name for d in root.iterdir()}
                  == {"prompt_" + p for p in ids},
                  "Prompt sweep must contain exactly the " + ("six" if batch == "v1" else str(2 * len(ids))) + " expected conditions")
    stats.require(not any((recovered / "results" / name).exists() for name in
                          ("native_benchmarks", "native_development", "extra_benchmarks")),
                  "Prompt sweep recovered outputs outside its scope")
    return {p: audit_prompt(recovered, p)[1] for p in ids}


def run(recovered, out, batch="v1"):
    ids = iat_prompts.batch_ids(batch)
    stats.require(not out.exists(), "Analysis output must start fresh")
    plan = json.loads(PLANS[batch].read_text())
    stats.require(batch == "v1" or all(plan["options"][p] == iat_prompts.OPTIONS[p] for p in ids),
                  "Frozen prompt options drift")
    stats.require((plan["prompt_ids"] if batch == "v1" else plan["batches"][batch]) == list(ids)
                  and plan["bootstrap"] == {
                      "draws": stats.BOOTSTRAPS, "seed": stats.SEED,
                      "unit": "paired images resampled within race",
                      "interval": "95% percentile; marginal, not simultaneous"}, "Analysis plan drift")
    audited = {p: audit_prompt(recovered, p) for p in ids}
    results, per_image, prepared = [], [], {}
    for prompt_id, (data, audit, sources) in audited.items():
        try:
            rows, images, _ = stats.analyze(data["base"], data["tuned"], SWEEP_SOURCES)
        except ValueError as exc:
            if batch == "v1" or "Zero valid answers" not in str(exc):
                raise
            rows, images = unavailable_rows(str(exc)), []
        if batch != "v1" and prompt_id in SEPARATE_FAMILIES:
            for row in rows:
                row["family"] = SEPARATE_FAMILIES[prompt_id] + row["family"]
        for row in rows:
            row.update(prompt=iat_prompts.REPORT_NUMBERS[prompt_id], run_prompt_id=prompt_id,
                       interpretation=iat_prompts.KINDS[prompt_id])
        results.extend(rows)
        per_image.extend({"prompt": iat_prompts.REPORT_NUMBERS[prompt_id], "run_prompt_id": prompt_id,
                          **image} for image in images)
        prepared[prompt_id] = data, audit, sources, rows
    adjust_across_prompts(results, batch)
    out.mkdir(parents=True)
    stats.save_json(out / "analysis_plan.json", plan)
    stats.save_json(out / "protocol_audit.json", {"prompt_" + iat_prompts.REPORT_NUMBERS[p]: values[1]
                                                    for p, values in prepared.items()})
    variety.write_csv(out / "tests.csv", results)
    variety.write_csv(out / "per_image.csv", per_image)
    flat = [{"prompt": row["prompt"], "run_prompt_id": row["run_prompt_id"],
             "interpretation": row["interpretation"],
              **stats.flat_rows([row])[0]} for row in results]
    variety.write_csv(out / "statistics_table.csv", flat)
    for prompt_id, (data, audit, sources, rows) in prepared.items():
        number = iat_prompts.REPORT_NUMBERS[prompt_id]
        folder = out / ("prompt_" + number)
        folder.mkdir()
        inputs = [{"condition": label, **{key: r[key] for key in stats.INPUT_FIELDS}}
                  for label in ("base", "tuned") for r in data[label]]
        variety.write_csv(folder / "paired_responses.csv", inputs)
        variety.write_csv(folder / "tests.csv", rows)
        stats.save_json(folder / "protocol_audit.json", {
            **audit, "paired_inputs_sha256": stats.digest(folder / "paired_responses.csv")})
        variety.main(sources, folder / "graphs", title_prefix="Prompt " + number)
    primary = [row for row in flat if row["Family"] == "primary"]
    abstentions = {}
    for prompt_id, (data, _, _, _) in prepared.items():
        if prompt_id == "p12":
            abstentions[iat_prompts.REPORT_NUMBERS[prompt_id]] = {
                label: abstention(rows) for label, rows in data.items()}
    if abstentions:
        stats.save_json(out / "cannot_tell.json", abstentions)
    numbers = ", ".join(iat_prompts.REPORT_NUMBERS[p] for p in ids)
    attempts = len(ids) * 1200
    if batch == "v1":
        lines = [
            "# Prompts 2, 3 and 4: base versus saved ORPO",
            "",
            "Evaluation only: 3,600 sampled attempts, no retraining or extra benchmarks.",
            "Prompt 1 is the original baseline prompt (study.IAT_PROMPT), evaluated earlier in",
            "iat_stats_v4. Run IDs 3, 4 and 5 in raw outputs are reported as prompts 2, 3 and 4.",
            "Every prompt is reported separately. Prompt 2 changes wording; prompts 3 and 4",
            "change task interpretation. They are not pooled as one task.",
        ]
    else:
        lines = [
            f"# Prompts {numbers}: base versus saved ORPO",
            "",
            f"Evaluation only: {attempts:,} sampled attempts, no retraining or extra benchmarks.",
            "Prompt 1 is the original baseline prompt; prompts 2-4 are in 2026-10-27/analysis/prompt_sweep_v1.",
            "Raw run IDs carry a p prefix (p10a = prompt 10a). Every prompt is reported",
            "separately and none are pooled. Interpretations:",
            "",
            *[f"- Prompt {iat_prompts.REPORT_NUMBERS[p]}: {plan['interpretations'][p]}" for p in ids],
            "",
            "Prompt 8 (a fix), prompt 9 (gray-image null check) and prompt 15",
            "(behavioral-evidence probe) have their own Holm families",
            "(mitigation_*, negative_control_*, behavioral_probe_*) and are not in the primary family.",
            "Any prompt with an image that had zero valid answers is marked unavailable.",
            "After recovery, prompt-X versus prompt-1 contrasts are required for batches C/D",
            "and must include statistical tests; plots alone are not sufficient.",
        ]
    lines += [
        "",
        "See statistics_table.csv for effects, 95% image-bootstrap confidence intervals,",
        f"raw p-values and Holm p-values. Holm spans all {'nine' if len(primary) == 9 else len(primary)} primary hypotheses.",
        "Diagnostic and sensitivity adjustments span prompts within their labeled families.",
        "The photo is the independent unit, not each answer or selected word.",
        "",
        "Strict sampled output is primary; fallback output and repeated-word exclusion",
        "are sensitivities. Entropy is normalized to log2(61); triples are unordered.",
        "Graphs/summary.csv within each prompt reports per-group rates and vocabulary",
        "coverage. Word-count graphs are separate and share the fixed reference order.",
        *(["cannot_tell.json reports prompt 12 abstention rates per race and model."] if abstentions else []),
        "",
        "These are exploratory results on 12 already-inspected photos. Nonsignificance",
        "does not establish equivalence; variety does not establish grounded accuracy.",
        "The historical screenshot baseline is unchanged and excluded from tuning tests.",
        "",
        "Reproduce with the recovered checksummed run:",
        "`python 2026-10-20/analysis/prompt_sweep.py --recovered <recovered> --out <new-output>"
        + ("" if batch == "v1" else " --batch " + batch) + "`",
        "",
        f"Primary rows: {len(primary)}. Source and script checksums are in provenance.json.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    stats.save_json(out / "provenance.json", {
        "analysis_script_sha256": stats.digest(Path(__file__)),
        "statistical_engine_sha256": stats.digest(Path(stats.__file__)),
        "graph_engine_sha256": stats.digest(Path(variety.__file__)),
        "files_sha256": {str(p.relative_to(out)): stats.digest(p) for p in out.rglob("*")
                         if p.is_file()},
    })
    print(json.dumps(primary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", type=Path,
                        help="Recovered prompt-sweep tree (required unless --vs-prompt-1)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--batch", choices=tuple(iat_prompts.BATCHES), default="v1")
    parser.add_argument("--vs-prompt-1", type=Path,
                        help="Existing batch analysis dir; run required prompt-X vs prompt-1 tests")
    args = parser.parse_args()
    if args.vs_prompt_1 is not None:
        stats.require(args.out is not None, "--out is required for vs-prompt-1 analysis")
        stats.require(args.batch in ("C", "D"), "--vs-prompt-1 requires --batch C or D")
        run_vs_prompt1(args.vs_prompt_1, args.out, args.batch)
    elif args.verify_only:
        stats.require(args.recovered is not None, "--recovered is required")
        print(json.dumps(verify(args.recovered, args.batch), indent=2, default=str))
    else:
        stats.require(args.recovered is not None, "--recovered is required")
        stats.require(args.out is not None, "--out is required for the analysis")
        run(args.recovered, args.out, args.batch)
