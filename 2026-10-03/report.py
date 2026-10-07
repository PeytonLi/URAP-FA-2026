"""Report verified base-model runs without implying that training has occurred."""
import csv
import json
from collections import Counter, defaultdict

import benchmark
import study
from multimodel_bias.analysis import write_iat_analysis


def pct(value):
    return f"{100 * value:.2f}%"


def per_image_metrics(records):
    """Keep invalid outputs visible beside each image's counted-word denominator."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["image_id"]].append(record)
    output = []
    for image_id, rows in sorted(grouped.items()):
        for field in ("race", "gender", "image_sha256"):
            if len({row[field] for row in rows}) != 1:
                raise ValueError(f"Inconsistent image metadata: {image_id}/{field}")
        valid = [row for row in rows if row["valid_three"]]
        total = sum(int(row["word_count"]) for row in valid)
        negative = sum(int(row["negative_count"]) for row in valid)
        if total != 3 * len(valid) or not 0 <= negative <= total:
            raise ValueError("Per-image word totals do not reconcile")
        output.append({"image_id": image_id, "race": rows[0]["race"],
                       "gender": rows[0]["gender"], "image_sha256": rows[0]["image_sha256"],
                       "attempts": len(rows), "valid": len(valid), "invalid": len(rows) - len(valid),
                       "exact_format": sum(row["exact_format"] for row in rows),
                       "fallback_final_outputs": sum(row["response_source"] != "sampled" for row in rows),
                       "negative_words": negative, "positive_words": total - negative,
                       "counted_words": total, "negative_rate": negative / total if total else None})
    return output


def main():
    output = study.ROOT / "reports"
    output.mkdir(exist_ok=True)
    models, records, runs = [], [], {}
    for name in benchmark.MODELS:
        run = study.ROOT / f"results/benchmarks/{name}_base_20261003"
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        model = benchmark.fw.ModelSpec(**manifest["contract"]["model"])
        data = benchmark.fw.dedupe_records(benchmark.fw._jsonl_records(run / "responses.jsonl"))
        expected = {(item["image_id"], iteration) for item in manifest["contract"]["stimuli"]
                    for iteration in range(1, 51)}
        if len(data) != 600 or {(row["image_id"], row["iteration"]) for row in data} != expected:
            raise ValueError(f"Incomplete benchmark slots for {name}")
        hashes = {item["image_id"]: item["sha256"] for item in manifest["contract"]["stimuli"]}
        if any(row["image_sha256"] != hashes[row["image_id"]] for row in data):
            raise ValueError(f"Image provenance mismatch for {name}")
        if len(benchmark.fw.completed_task_ids(data)) != 600:
            raise ValueError(f"Unresolved technical failures for {name}")
        models.append(model)
        records.extend(data)
        runs[name] = (run, data, manifest)
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    artifacts = write_iat_analysis(records, models, analysis,
                                   bootstrap_samples=100_000, bootstrap_seed=17_082_026)
    metrics = json.loads((analysis / "metrics.json").read_text(encoding="utf-8"))
    table = ["| Base model | Valid / attempts | Black negative rate | White negative rate | Gap (pp) | Unique triples |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, model in zip(benchmark.MODELS, models):
        run, data, manifest = runs[name]
        values = metrics["models"][model.key]
        race = values["race"]
        source_counts = Counter(row["response_source"] for row in data)
        fallback_attempts = sum(row["stage"] == "fallback" for row in benchmark.fw._jsonl_records(run / "api_responses.jsonl"))
        valid = [row for row in data if row["valid_three"]]
        word_counts = Counter(word for row in valid for word in row["words"].split("|"))
        if sum(word_counts.values()) != len(valid) * 3:
            raise ValueError("Valid-response word totals do not reconcile")
        exact = sum(row["exact_format"] for row in data)
        top3 = word_counts.most_common(3)
        top3_share = sum(count for _, count in top3) / sum(word_counts.values())
        image_rows = per_image_metrics(data)
        with (output / f"{name}_base_per_image.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(image_rows[0]))
            writer.writeheader()
            writer.writerows(image_rows)
        image_table = ["| Image | Photo group | Valid / attempts | Negative / counted words | Negative rate |",
                       "| --- | --- | ---: | ---: | ---: |"]
        for item in image_rows:
            rate = pct(item["negative_rate"]) if item["negative_rate"] is not None else "Unavailable"
            image_table.append(f"| {item['image_id']} | {item['race']} | {item['valid']}/{item['attempts']} | "
                               f"{item['negative_words']}/{item['counted_words']} | {rate} |")
        deployment = json.loads((study.ROOT / f"results/deployments/{name}_created.json").read_text(encoding="utf-8"))
        cleanup = json.loads((study.ROOT / f"results/deployments/{name}_cleanup.json").read_text(encoding="utf-8"))
        if not cleanup["delete_acknowledged"]:
            raise ValueError(f"Deployment cleanup is unacknowledged for {name}")
        row = (f"| {benchmark.MODELS[name][1]} | {len(valid)}/600 | {pct(race['Black']['negative_rate'])} | "
               f"{pct(race['White']['negative_rate'])} | {100 * values['black_minus_white_negative_rate_gap']:+.2f} | "
               f"{values['unique_response_tuples']} |")
        table.append(row)
        race_table = ["| Photo group | Valid responses | Negative / all counted words | Negative rate | 95% image-cluster CI |",
                      "| --- | ---: | ---: | ---: | --- |"]
        for group in ("Black", "White"):
            item = race[group]
            race_table.append(f"| {group} | {item['valid_responses']}/300 | {item['negative_terms']}/{item['total_terms']} | "
                              f"{pct(item['negative_rate'])} | {pct(item['ci_low'])}–{pct(item['ci_high'])} |")
        interpretation = {
            "gemma4": "Negative-word selection is high, especially for Black photos, with substantial concentration in a few words. Invalid outputs are excluded from word-rate denominators, so group-specific validity must accompany the observed gap. This alone does not identify a causal source of bias.",
            "qwen3vl": "Almost-all-positive output yields a tiny negative-word gap, but does not establish expression-grounded behavior. The model has a strong positive-word prior on this task; the benchmark alone cannot distinguish desirable behavior from task avoidance or response concentration.",
            "qwen35": "The output is more varied than the other two base models, with an observed Black-minus-White negative-word gap. Image-cluster uncertainty is substantial. Its FP8 serving configuration is an additional cross-model comparison caveat, not a proven explanation for its behavior.",
        }[name]
        text = f"""# {benchmark.MODELS[name][1]} — base benchmark report

Status: **base evaluation complete; SFT and ORPO not run**. No training data was used for this base model. This is not a report of a fine-tuned checkpoint.

## Model, serving and test inputs

Logical model: `{manifest['contract']['logical_base_model']}`. Requested deployment: `{model.model_id}`. Returned API model labels: {', '.join(sorted({row['api_model_returned'] for row in data}))}.

Serving used {deployment['accelerator_count']} × `{deployment['accelerator_type']}` with deployment shape `{deployment['deployment_shape']}`. Precision: `{deployment.get('precision', 'not explicitly reported')}`. A provider model label is not an immutable Hugging Face weight revision; exact historical weight identity is not established.

The test used the 12 hash-verified reference photos, six per race, and the unchanged 61-word prompt. There were 50 attempts per image, temperature 1, top-p 1, 30 output tokens, reasoning disabled and the existing seed schedule. Invalid samples received the existing fallback (temperature 0, 30 tokens); invalid final outputs were retained, not resampled away. Provider seed requests do not guarantee reproducibility across serving engines.

## Measured output quality

- Valid three-word outputs: {len(valid)}/600 ({pct(values['validity_rate'])}); invalid: {600 - len(valid)}.
- Exact final-format compliance: {exact}/600 ({pct(exact / 600)}). Legacy three-word validity and exact formatting are different criteria.
- Fallback API attempts: {fallback_attempts}; final response sources: `{dict(source_counts)}`.
- Counted words: {sum(word_counts.values())}, not an assumed 1,800. Unique ordered triples: {values['unique_response_tuples']}; distinct words: {values['distinct_iat_words']}/61.
- Normalized word entropy: {values['normalized_word_entropy']:.4f}. Top three words: {', '.join(f'{word} ({count})' for word, count in top3)}; together {pct(top3_share)} of counted words.

{chr(10).join(race_table)}

Black-minus-White negative-rate gap: **{100 * values['black_minus_white_negative_rate_gap']:+.2f} percentage points**; 95% image-cluster CI {100 * values['gap_ci_low']:+.2f} to {100 * values['gap_ci_high']:+.2f} pp. The uncertainty unit is the image, not each repeated response. The benchmark contains only six images per race and is not a population-level fairness test.

{interpretation}

## Per-image results

{chr(10).join(image_table)}

[Per-image counts and provenance](./{name}_base_per_image.csv) include invalid responses, exact formatting and fallback reliance. Rates exclude invalid outputs; a missing rate is not zero negativity.

## Training data and remaining work

No SFT or ORPO checkpoint has been created. The proposed shared pilot contains 38 training and 16 development images, with no profiles. SFT will use image + unchanged prompt + chosen triple; ORPO will additionally use the rejected triple. Existing targets are AI-judged templates, not human ground truth. The negative subset consists of sad faces, and neutral targets favor positive terms. See [the full data audit and protocol](../README.md).

Images must all pass the recorded hash checks before training. An image-capable training path must also be verified; Gemma 4's stock cookbook formatter is text-only. Base, SFT and ORPO must be evaluated separately and never presented as interchangeable.

## Evidence and figures

- [Raw responses](../results/benchmarks/{name}_base_20261003/responses.csv), [all API attempts](../results/benchmarks/{name}_base_20261003/api_responses.jsonl), [run contract](../results/benchmarks/{name}_base_20261003/run_manifest.json).
- [Joint comparison metrics](analysis/metrics.json), [word counts](analysis/word_counts.csv), [intersectional counts](analysis/intersectional_metrics.csv).
- [61-word count figure](analysis/plots/{model.key}_01_word_counts.png), [negative-rate figure](analysis/plots/{model.key}_02_negative_rates.png).

Temporary deployment deletion was acknowledged. Dedicated GPU billing is not included in the API response; a zero in the legacy `total_cost_usd` field means cost was unreported, **not free compute**.
"""
        (output / f"{name}_base.md").write_text(text, encoding="utf-8")
    comparison = f"""# Three-model base comparison

These are real new Fireworks base-model evaluations, **not fine-tuning results**. All three have 600 completed benchmark slots with no unresolved technical errors. Profiles were not used, and none of these models was trained in this study.

{chr(10).join(table)}

Positive/negative rates are conditional on valid final outputs, including deterministic fallback outputs. Gemma 4 has invalid final outputs, so its counted-word denominators differ from the other two models. Keep fallback reliance, exact formatting and response diversity alongside any claimed reduction in a race gap.

Gemma 4's negative-word concentration, Qwen3-VL's positive-word concentration and Qwen3.5's broader word use are different behaviors. The small Qwen3-VL gap is not proof it is the best expression-grounded model. The prompt does not explicitly define an emotion-grounding task, these photos are not a representative population, and serving hardware/precision differ. No causal attribution to architecture, training data or quantization follows from this benchmark alone.

## Reports and visualizations

- [Gemma 4 base](gemma4_base.md)
- [Qwen3-VL base](qwen3vl_base.md)
- [Qwen3.5 base](qwen35_base.md)

![Base negative-word rates](analysis/plots/comparison_01_negative_rates.png)

![Base race gaps](analysis/plots/comparison_02_race_gaps.png)

![Base validity and diversity](analysis/plots/comparison_03_validity_diversity.png)

All per-model word-count charts use the original 61-word order, blue Black-photo bars and gold White-photo bars, including words with zero counts. Raw data, image-cluster confidence intervals and exploratory permutation/BH results are in [the joint metrics](analysis/metrics.json).

## Goal remains incomplete

Required next: finish the training-image audit, verify the image-capable training routes, run SFT and ORPO from each untouched base, evaluate the resulting checkpoints using the same benchmark, and extend these reports/figures to compare the actual trained models. The original goal is not reduced to these base evaluations.
"""
    (output / "BASE_COMPARISON.md").write_text(comparison, encoding="utf-8")
    study.save_json(output / "baseline_report_manifest.json", {
        "created_at": study.now(), "models": [model.key for model in models], "benchmark_slots": len(records),
        "training_performed": False, "goal_complete": False, "analysis": artifacts,
        "response_file_sha256": {name: study.sha(run / "responses.jsonl") for name, (run, _, _) in runs.items()}})
    print(f"Verified {len(records)} benchmark slots; baseline reports and graphs: {output}")


if __name__ == "__main__":
    main()
