"""Produce the full native comparison only from six finished runs and nine benchmarks."""
import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import benchmark
import native_train
import report
import study


def percent(value):
    return report.pct(value) if value is not None and math.isfinite(value) else "Unavailable"


def verified_results(recovered, require_all=True):
    dataset = study.dataset_config()
    if not dataset:
        raise ValueError("Active curated dataset required for report validation")
    expected_images = [{k: row[k] for k in ("image_id", "filename", "race", "gender", "sha256")}
                       for row in study.read_csv(study.REPO / "2026-09-11/data/iat12/manifest.csv")]
    slots = {(row["image_id"], iteration) for row in expected_images for iteration in range(1, 51)}
    hashes = {row["image_id"]: row["sha256"] for row in expected_images}
    training, conditions = {}, {}
    methods = dataset.get("training_methods", ("sft", "orpo", "kto"))
    model_settings = ({"gemma3_12b": native_train.MODELS["gemma3_12b"]}
                      if dataset.get("dataset_id") == "image-curation-2026-09-29-v3"
                      else native_train.MODELS)
    for name, settings in model_settings.items():
        for method in methods:
            runs = list((recovered / "data/private/native_runs").glob(name + "_" + method + "_*/run.json"))
            if not runs and not require_all:
                continue
            if len(runs) != 1:
                raise ValueError(f"Exactly one completed {name}/{method} run required")
            run = json.loads(runs[0].read_text(encoding="utf-8"))
            if not require_all and run["status"] != "training_complete_evaluation_pending":
                continue
            dataset = study.dataset_config()
            if not dataset:
                raise ValueError("Active curated dataset required for report validation")
            expected_train = dataset["split_counts"]["train"]
            expected_dev = dataset["split_counts"]["development"]
            expected_examples = dataset.get("example_split_counts", dataset["split_counts"])
            if (run["status"] != "training_complete_evaluation_pending"
                    or run["optimizer_updates"] != run["planned_optimizer_updates"]
                    or run["planned_optimizer_updates"] != 3 * -(-expected_examples["train"] // 4)
                    or run["completed_epochs"] != 3 or run.get("cohort") != "full"
                    or run["model"] != settings["model"] or run["method"] != method
                    or run["weight_and_processor_revision"] != settings["revision"]
                    or run["targets_sha256"] != study.sha(study.REPO / dataset["targets_path"])
                    or run["manifest_sha256"] != dataset["manifest_sha256"]
                    or run["training_images"] != expected_train or run["development_images"] != expected_dev
                    or run["test_images_used"] != 0 or run["profile_targets_used"] != 0):
                raise ValueError(f"Incomplete or mismatched training evidence: {name}/{method}")
            recipe = {"seed": 0, "logical_batch_size": 4, "requested_epochs": 3,
                      "optimizer": "AdamW", "learning_rate": 1e-5, "weight_decay": 0,
                      "orpo_coefficient": 0.1 if method == "orpo" else None,
                      "kto_beta": 0.1 if method == "kto" else None,
                      "loss_weighting": "equal_examples"}
            if any(run.get(field) != value for field, value in recipe.items()):
                raise ValueError("Actual training recipe differs from the reported protocol")
            checkpoint = runs[0].parent / "epoch_3"
            for filename, digest in run["checkpoints"][-1]["files"].items():
                if Path(filename).name != filename or study.sha(checkpoint / filename) != digest:
                    raise ValueError("Final adapter/processor/optimizer checksum mismatch")
            training[name, method] = (run, checkpoint)
        contracts = []
        for method in ("base", *methods):
            key = name + "_" + method + "_native_balanced"
            folder = recovered / "results/native_benchmarks" / key
            if not require_all and not (folder / "run_manifest.json").exists():
                continue
            manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
            if not require_all and manifest["status"] != "complete":
                continue
            contract = manifest["contract"]
            if (manifest["status"] != "complete" or manifest["recorded_responses"] != 600
                    or contract["provider"] != "prime/native-transformers"
                    or contract["weight_and_processor_revision"] != settings["revision"]
                    or contract["precision"] != "bfloat16_unquantized"
                    or contract["prompt"] != study.IAT_PROMPT or contract["stimuli"] != expected_images
                    or contract["pilot_policy_sha256"] != dataset.get("pilot_policy_sha256")
                    or contract["config"] != asdict(benchmark.fw.run_config(1))):
                raise ValueError(f"Unmatched or incomplete benchmark contract: {key}")
            if method != "base" and (name, method) not in training:
                raise ValueError("A completed adapter benchmark requires its verified training checkpoint")
            adapter = None if method == "base" else study.sha(training[name, method][1] / "adapter_model.safetensors")
            if contract["adapter_sha256"] != adapter:
                raise ValueError("Evaluated adapter does not match the recovered final checkpoint")
            model = benchmark.fw.ModelSpec(**contract["model"])
            records = benchmark.fw.dedupe_records(benchmark.fw._jsonl_records(folder / "responses.jsonl"))
            if (model.key != key or len(records) != 600
                    or {(r["image_id"], r["iteration"]) for r in records} != slots
                    or len(benchmark.fw.completed_task_ids(records)) != 600
                    or any(r["model_key"] != key or r["image_sha256"] != hashes[r["image_id"]] for r in records)):
                raise ValueError(f"Missing, duplicate or mismatched benchmark slots: {key}")
            report.per_image_metrics(records)  # Reconcile valid/invalid counts and actual denominators.
            contracts.append({field: contract[field] for field in
                              ("versions", "precision", "attention_implementation", "config", "stimuli", "prompt")})
            conditions[name, method] = (model, records, folder, contract)
        if any(contract != contracts[0] for contract in contracts[1:]):
            raise ValueError(f"Base and adapter runtimes/benchmarks differ: {name}")
        if contracts and any(training[name, method][0]["versions"] != contracts[0]["versions"]
                             for method in ("sft", "orpo", "kto") if (name, method) in training):
            raise ValueError(f"Training and benchmark runtime versions differ: {name}")
    return training, conditions


def main(recovered):
    if study.dataset_config().get("dataset_id") == "image-curation-2026-09-29-v3":
        return coverage_report(recovered)
    training, conditions = verified_results(recovered)
    dataset = study.dataset_config()
    cohort = json.loads((study.REPO / dataset["targets_path"]).read_text(encoding="utf-8"))
    cohort_sha = study.sha(study.REPO / dataset["targets_path"])
    example = next(row for row in cohort if row["split"] == "train")
    source_counts = Counter((row["dataset"], row["split"]) for row in cohort)
    target_counts = Counter(row["chosen"] for row in cohort if row["split"] == "train")
    rejected_counts = Counter(row["rejected"] for row in cohort if row["split"] == "train")
    target_words = {word.strip() for target in target_counts for word in target.split(",")}
    top_targets = "; ".join(f"`{target}`: {count} images" for target, count in target_counts.most_common(2))
    top_rejected, top_rejected_count = rejected_counts.most_common(1)[0]
    output = study.ROOT / "reports/native_balanced"
    analysis = output / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    # Task IDs are deduplicated within each condition, never across adapters.
    models = [item[0] for item in conditions.values()]
    records = [row for item in conditions.values() for row in item[1]]
    artifacts = benchmark.fw.write_iat_analysis(records, models, analysis,
                                               bootstrap_samples=100_000, bootstrap_seed=17_082_026)
    metrics = json.loads((analysis / "metrics.json").read_text(encoding="utf-8"))["models"]
    summary = ["| Model / condition | Valid / 600 | Black negative rate | White negative rate | Gap (pp) | Unique triples |",
               "| --- | ---: | ---: | ---: | ---: | ---: |"]
    provenance = {"created_at": study.now(), "benchmark_slots": len(records), "training_conditions": len(training),
                  "optimizer_updates": sum(r[0]["optimizer_updates"] for r in training.values()),
                  "policy_sha256": dataset.get("pilot_policy_sha256"), "cohort_sha256": cohort_sha,
                  "source_manifest_sha256": dataset["manifest_sha256"], "source_commit": dataset["source_commit"],
                  "training_implementation_sha256": study.sha(Path(native_train.__file__)),
                  "chosen_training_triple_counts": dict(target_counts),
                  "rejected_training_triple_counts": dict(rejected_counts),
                  "identity_disjointness_established": False,
                  "response_sha256": {}, "analysis": artifacts, "training_performed": True}
    for name, settings in native_train.MODELS.items():
        available_methods = [method for method in ("sft", "orpo", "kto")
                             if (name, method) in training]
        lines = [f"# {benchmark.MODELS[name][1]}: native base, SFT and ORPO", "",
            f"Pinned weights and processor: `{settings['model']}@{settings['revision']}`. BF16, native Transformers.", "",
            f"[Official model card at the tested revision](https://huggingface.co/{settings['model']}/blob/{settings['revision']}/README.md).", "",
            "## Training data and procedure", "",
            f"Source: [manifest](../../../{dataset['manifest_path']}) gated from the September 29 "
            f"FairFace/UTKFace pool; repository commit `{dataset['source_commit']}`, "
            f"manifest SHA-256 `{dataset['manifest_sha256']}`. "
            f"Targets file SHA-256 `{cohort_sha}`.", "",
            f"The confidence-gated cohort supplied {dataset['split_counts']['train']} training images and "
            f"{dataset['split_counts']['development']} development images. Unresolved images under the two-pass "
            "0.8-confidence gate were excluded per the recorded authorization. Original splits were retained; "
            "the 480 reserved test photos and 12 benchmark photos supplied no training targets. "
            "There are no profile targets.", "",
            "Each input contains the image and unchanged 61-word prompt. SFT learns the chosen three-word "
            "answer; ORPO additionally compares its rejected answer. Response tokens and native turn endings "
            "receive loss. All examples carry equal loss weight in this cohort.", "",
            "ORPO adds an odds-ratio preference term to chosen-response likelihood and uses no separate "
            "frozen reference model. [Original ORPO paper](https://arxiv.org/abs/2403.07691).", "",
            "### Exact prompt and an actual training record", "",
            "The model receives the image pixels and this user prompt:", "",
            "```text", study.IAT_PROMPT, "```", "",
            f"Retained training image `{example['image_id']}` has chosen answer `{example['chosen']}` "
            f"and rejected answer `{example['rejected']}`. SFT supervises the chosen answer; ORPO uses "
            "both answers with the same image and prompt.", "",
            f"The chosen training answers contain {len(target_counts)} distinct ordered triples and "
            f"{len(target_words)} of the 61 permitted words. Most frequent targets: {top_targets}. "
            "These are unweighted image counts; the category loss weights above apply during training.", "",
            f"The rejected answers contain {len(rejected_counts)} distinct ordered triples. "
            f"`{top_rejected}` is rejected for {top_rejected_count}/{sum(target_counts.values())} "
            f"training images ({percent(top_rejected_count/sum(target_counts.values()))}). "
            "These AI-proposed counterexamples are not observed human preferences or necessarily subtle "
            "expression distinctions. Their repetition limits what high development preference accuracy establishes.", "",
            "Targets are gated two-pass AI proposals with adjudication and user-authorized AI approval, "
            "with zero independent human approvals. They express proposed associations, not objective emotion "
            "or personality truth. A per-triple variety cap limits answer concentration; gating and capping "
            "break the original demographic matching balance. The labeling assistant's exact model revision "
            "is unavailable.", "",
            "Both methods start independently from the untouched base. Language-attention LoRA rank 8, "
            "alpha 16, dropout zero; AdamW learning rate 1e-5, logical batch four, three epochs, seed zero; "
            "ORPO coefficient 0.1; gradient-norm clipping at 1.0 before each optimizer step. "
            "Equal category loss weight does not guarantee equal gradient or optimizer influence. "
            "Vision, experts, routers, embeddings and base parameters are frozen. "
            "The final epoch-three checkpoint was predeclared. Development never receives optimizer updates.", "",
            "| Source | Retained training images | Retained development images |",
            "| --- | ---: | ---: |",
            *[f"| {source} | {source_counts[source, 'train']} | {source_counts[source, 'development']} |"
              for source in sorted({row['dataset'] for row in cohort})], "",
            "| Method | Updates | Dev chosen loss | Dev macro preference accuracy |",
            "| --- | ---: | ---: | ---: |"]
        for method in available_methods:
            run, checkpoint = training[name, method]
            dev = run["development"][-1]
            lines.append(f"| {method.upper()} | {run['optimizer_updates']} | {dev['response_loss']:.4f} | "
                         f"{percent(dev['macro_category_preference_accuracy'])} |")
        lines += ["", "Preference accuracy asks whether the chosen response receives higher mean conditional "
                  "log probability than the rejected response. It is not free-generation accuracy.", "",
                  "### Actual execution evidence", ""]
        for method in available_methods:
            run, checkpoint = training[name, method]
            lines += [f"{method.upper()}: `{run['gpu']}`; {run['adapter']['trainable_parameters']:,} trainable "
                f"adapter parameters. Elapsed {run['elapsed_seconds']/60:.1f} minutes including weight loading "
                f"and development checks; peak Torch-allocated GPU memory {run['peak_gpu_memory_gib']:.2f} GiB. "
                f"Runtime versions: `{json.dumps(run['versions'], sort_keys=True)}`. Pre-training pixel-only control "
                f"maximum response-log-probability change: {run['image_control']['maximum_response_logprob_change']:.6f} "
                "(evidence that image inputs affect scoring before training, not a post-training grounding or accuracy measurement).", "",
                f"[Training receipt](<{(checkpoint.parent / 'run.json').as_posix()}>) · "
                f"[Final LoRA adapter](<{(checkpoint / 'adapter_model.safetensors').as_posix()}>) · "
                f"[Adapter configuration](<{(checkpoint / 'adapter_config.json').as_posix()}>).", ""]
        lines += ["## Benchmark measurements", "",
                  "Each condition uses 12 photos × 50 repeats = 600 slots, temperature 1, top-p 1, top-k disabled, "
                  "30 new tokens, reasoning disabled and the original seed schedule. Invalid samples receive "
                  "the existing deterministic fallback. Invalid final responses remain in validity counts; "
                  "word rates use the actual valid-response denominators. Confidence intervals resample image "
                  "clusters, not individual repetitions. There are only six photos per race."]
        for method in ("base", *available_methods):
            model, rows, folder, contract = conditions[name, method]
            values = metrics[model.key]
            words = Counter(word for row in rows if row["valid_three"] for word in row["words"].split("|"))
            total = sum(words.values())
            top = words.most_common(3)
            fallback = sum(r["stage"] == "fallback" for r in benchmark.fw._jsonl_records(folder / "api_responses.jsonl"))
            race = values["race"]
            summary.append(f"| {benchmark.MODELS[name][1]} / {method.upper()} | {values['valid_responses']}/600 | "
                f"{percent(race['Black']['negative_rate'])} | {percent(race['White']['negative_rate'])} | "
                f"{100 * values['black_minus_white_negative_rate_gap']:+.2f} | {values['unique_response_tuples']} |")
            lines += ["", f"### {method.upper()}", "",
                f"Valid: {values['valid_responses']}/600; invalid: {600-values['valid_responses']}. "
                f"Exact final format: {sum(r['exact_format'] for r in rows)}/600. Fallback calls: {fallback}.", "",
                f"Counted words: {total}; unique triples: {values['unique_response_tuples']}; distinct words: "
                f"{values['distinct_iat_words']}/61; normalized word entropy: {values['normalized_word_entropy']:.4f}. "
                f"Top three words: {top}; combined share: {percent(sum(n for _, n in top)/total if total else None)}.", "",
                "| Photo group | Positive / negative words | Counted words | Negative rate | 95% image-cluster CI |",
                "| --- | ---: | ---: | ---: | --- |"]
            for group in ("Black", "White"):
                r = race[group]
                lines.append(f"| {group} | {r['total_terms']-r['negative_terms']} / {r['negative_terms']} | "
                    f"{r['total_terms']} | {percent(r['negative_rate'])} | {percent(r['ci_low'])} to {percent(r['ci_high'])} |")
            invalid_by_photo = Counter(row["image_id"] for row in rows if not row["valid_three"])
            if invalid_by_photo:
                lines += ["", f"Invalid final slots by photo: `{dict(invalid_by_photo)}`. "
                          "These slots contribute no positive or negative words; refusals are not scored as neutral."]
            lines += ["", f"Black-minus-White negative-rate gap: {100*values['black_minus_white_negative_rate_gap']:+.2f} pp; "
                f"95% CI {100*values['gap_ci_low']:+.2f} to {100*values['gap_ci_high']:+.2f} pp.", "",
                f"![{method.upper()} word counts](analysis/plots/{model.key}_01_word_counts.png)", "",
                f"[Per-image results]({model.key}_per_image.csv). Adapter SHA-256: `{contract['adapter_sha256']}`.", "",
                f"[Final response records](<{(folder / 'responses.jsonl').as_posix()}>) · "
                f"[All sampled/fallback attempts](<{(folder / 'api_responses.jsonl').as_posix()}>) · "
                f"[Benchmark contract](<{(folder / 'run_manifest.json').as_posix()}>)."]
            image_rows = report.per_image_metrics(rows)
            with (output / f"{model.key}_per_image.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(image_rows[0]))
                writer.writeheader()
                writer.writerows(image_rows)
            provenance["response_sha256"][model.key] = study.sha(folder / "responses.jsonl")
        lines += ["", "## Changes from the native base", ""]
        base_values = metrics[conditions[name, "base"][0].key]
        for method in ("sft", "orpo"):
            values = metrics[conditions[name, method][0].key]
            gap_change = 100 * (values["black_minus_white_negative_rate_gap"] - base_values["black_minus_white_negative_rate_gap"])
            absolute_change = 100 * (abs(values["black_minus_white_negative_rate_gap"]) - abs(base_values["black_minus_white_negative_rate_gap"]))
            validity_change = 100 * (values["validity_rate"] - base_values["validity_rate"])
            lines += [f"{method.upper()}: signed gap change {gap_change:+.2f} pp; absolute gap change "
                      f"{absolute_change:+.2f} pp (negative means a smaller absolute gap). Validity changed "
                      f"{validity_change:+.2f} pp, and unique triples changed from "
                      f"{base_values['unique_response_tuples']} to {values['unique_response_tuples']}.", ""]
            for group in ("Black", "White"):
                before, after = base_values["race"][group]["negative_rate"], values["race"][group]["negative_rate"]
                change = f"{100*(after-before):+.2f} pp" if all(v is not None and math.isfinite(v) for v in (before, after)) else "Unavailable"
                lines += [f"{group}-photo negative-word rate: {percent(before)} to {percent(after)} "
                          f"(change {change}).", ""]
        lines += ["## Interpretation limits", "", "Compare each adapter to the native base above. "
            "The prior Fireworks results use a different serving stack. A smaller gap alone does not establish "
            "better image grounding: validity, diversity and the very small negative development subset also "
            "matter. This filtered pilot cannot establish population fairness or accuracy on neutral expressions. "
            "The benchmark gap is descriptive; its individual condition intervals are not a paired confidence "
            "interval for the training effect. When validity changes, conditional word rates can cover "
            "different outputs; changed refusal rates can affect the gap independently of word choice. "
            "File and decoded-pixel exclusion checks do not establish person-disjointness: differently "
            "cropped/resized/recompressed copies and multiple photos of the same person are not "
            "comprehensively excluded. Identity generalization remains unverified. "
            "Zero legacy API cost fields do not report rented-GPU charges."]
        (output / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    comparison = ["# Native SFT / ORPO comparison", "", *summary, "",
        "Six completed training conditions and nine complete 600-slot benchmarks are included. All 61 words "
        "retain the original order, including zero counts. Blue bars represent Black-photo counts and gold "
        "bars White-photo counts. Labels use the original dataset metadata.", "",
        "[Gemma 4 report](gemma4.md) · [Qwen3-VL report](qwen3vl.md) · [Qwen3.5 report](qwen35.md)", "",
        "![Negative rates](analysis/plots/comparison_01_negative_rates.png)", "",
        "![Race gaps](analysis/plots/comparison_02_race_gaps.png)", "",
        "![Validity and diversity](analysis/plots/comparison_03_validity_diversity.png)", "",
        "Read rates alongside invalid outputs and response concentration. The original 12-photo benchmark "
        "provides limited evidence about unseen faces or population-level fairness. [Machine-readable metrics](analysis/metrics.json)."]
    (output / "COMPARISON.md").write_text("\n".join(comparison) + "\n", encoding="utf-8")
    study.save_json(output / "report_manifest.json", provenance)
    print(json.dumps({"output": str(output), "training_conditions": len(training), "benchmark_slots": len(records)}))


def coverage_report(recovered):
    training, conditions = verified_results(recovered)
    sys.path.insert(0, str(study.REPO / "2026-10-20/analysis"))
    import variety
    output = study.ROOT / "reports/native_coverage_v3"
    dataset = study.dataset_config()
    labels = [(method, conditions["gemma3_12b", method][2] / "responses.csv")
              for method in ("base", "orpo")]
    summaries = variety.main(
        labels, output, [("orpo", study.REPO / dataset["targets_path"])])
    run = training["gemma3_12b", "orpo"][0]
    lines = [
        "# ORPO coverage experiment v3", "",
        "Only Gemma 3 12B-IT ORPO was trained. The checkpoint is the predeclared final third epoch.",
        f"Training: {run['training_images']} images / {run['training_examples']} preference examples; "
        f"{run['optimizer_updates']} optimizer updates. Development received no optimizer updates.",
        "The frozen test protocol remains 12 photos, 50 repetitions, temperature 1.0, identical prompts and seeds.",
        "Training uses independently supported candidate words, not human ground-truth expression labels.", "",
        "| Condition / output view | Valid | Vocabulary union | Black / White words | Median unordered triples per image | Word TV |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        tv = summary.get("word_tv_image_weighted")
        tv_text = f"{tv:.4f}" if tv is not None else "Not estimable"
        lines.append(
            f"| {summary['condition']} / {summary['view']} | {summary['valid_responses']}/600 | "
            f"{summary['distinct_words_union']}/61 | {summary['distinct_words_black']} / "
            f"{summary['distinct_words_white']} | {summary['per_image_unordered_triples_median']} | "
            f"{tv_text} |")
    tuned = next(s for s in summaries if s["condition"] == "orpo" and s["view"] == "sampled_exact")
    lines.extend([
        "", f"The >=50-word union target was {'met' if tuned['target_50_met'] else 'not met'} on strict sampled outputs.",
        "Report per-group coverage and overlap alongside union coverage. Non-significance does not establish distribution equality.",
        "A low race gap alone is not success: inspect coverage, entropy, dominant-triple concentration and validity together.",
        "These 12 stimuli do not establish population fairness or visible-expression accuracy.", "",
        "![Word counts](word_counts.png)", "![Per-image variety](per_image_variety.png)",
        "![Valence rates](per_image_rates.png)", "",
        "See summary.csv, per_image.csv, word_counts.csv and provenance.json for denominators, exact 924-assignment "
        "image-level permutation tests, 10,000 image-cluster bootstrap draws and input checksums.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "training_conditions": len(training)}))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovered", type=Path, required=True)
    main(parser.parse_args().recovered.resolve())
