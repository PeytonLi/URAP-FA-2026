"""Build a factual findings report and requested coverage tables from saved results."""
import json
from collections import Counter

from analyze import NAMES
from common import ROOT, json_read, read_jsonl, rows, write_csv

GITHUB="https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/"
OLD="https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/"


def link(path,label=None,old=False):
    return f"[{label or path}]({(OLD if old else GITHUB)+path})"


def main():
    summary=rows(ROOT / "results/analysis/summary.csv")
    significance=rows(ROOT / "results/analysis/significance.csv")
    sig={(r["panel"],r["model_key"]):r for r in significance}
    lines=["# September 11 Evaluation Results", "", "## Scope", "",
           "Inference-only expansion of IAT61 and a paired evaluation of the existing August 10 Gemma LoRA adapter. "
           "No new DPO training or judge calls were made. The selected adapter is a retrospective directional example, not a validated best model.","",
           "## Response Counts and Race Gaps", "",
           "Each valid answer contains three permitted words. Race rates below are image-averaged percentages. "
           "The gap is Black minus White in percentage points. p values are exploratory two-sided tests; q values are BH-adjusted within each panel.","",
           "| Panel | Model | Answered / target | Valid | Black % | White % | Gap [95% CI], pp | p | q |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        s=sig[r["panel"],r["model_key"]]
        lines.append(f"| {r['panel']} | {r['model_name']} | {r['responses']} / {r['planned_responses']} | {r['valid_responses']} | "
                     f"{float(r['black_rate'])*100:.2f} | {float(r['white_rate'])*100:.2f} | {float(s['effect'])*100:+.2f} "
                     f"[{float(s['ci_low'])*100:+.2f}, {float(s['ci_high'])*100:+.2f}] | {float(s['raw_p_value']):.4g} | {float(s['bh_adjusted_q_value']):.4g} |")
    lines += ["", "Use "+link("results/analysis/summary.csv")+" and "+link("results/analysis/significance.csv")+" for unrounded values and sample units.","",
              "## Cross-Model Findings", ""]
    for panel in ('neutral12','utkface'):
        tests=[r for r in significance if r['panel']==panel]
        detected=[r for r in tests if float(r['bh_adjusted_q_value'])<.05]
        lines += [f"**{panel}:** {len(detected)} of {len(tests)} available conditions have an exploratory "
                  "race-gap test with BH q < 0.05"+
                  (" ("+", ".join(r['model_name'] for r in detected)+")" if detected else "")+". "
                  "Incomplete cells are not full-size estimates, and a nonsignificant test does not establish no bias.", ""]
    lines += ["Image panels, expression distributions, sample units and model serving paths affect these measures. "
              "Do not interpret cross-model bar height as a general fairness ranking or differences from older plots as evidence that base weights were trained.", "",
              "## Direct Gemma Base Versus Adapter Comparison",""]
    path=ROOT / "results/analysis/gemma_paired_effect.csv"
    if path.exists():
        for d in rows(path):
            base=next(r for r in summary if r["panel"]==d["panel"] and r["model_key"]=="gemma_base")
            tuned=next(r for r in summary if r["panel"]==d["panel"] and r["model_key"]=="gemma_tuned")
            a=float(sig[d["panel"],"gemma_base"]["effect"])
            b=float(sig[d["panel"],"gemma_tuned"]["effect"])
            entropy=100*float(tuned["word_entropy_bits"])/float(base["word_entropy_bits"]) if float(base["word_entropy_bits"]) else float("nan")
            unique=100*int(tuned["unique_response_tuples"])/int(base["unique_response_tuples"]) if int(base["unique_response_tuples"]) else float("nan")
            lines += [f"**{d['panel']}:** the absolute observed gap {'fell' if abs(b)<abs(a) else 'rose or stayed unchanged'} "
                      f"from {abs(a)*100:.2f} to {abs(b)*100:.2f} pp. The direct paired signed-gap change was "
                      f"{float(d['effect'])*100:+.2f} pp (95% CI {float(d['ci_low'])*100:+.2f} to {float(d['ci_high'])*100:+.2f}; "
                      f"p={float(d['raw_p_value']):.4g}, q={float(d['bh_adjusted_q_value']):.4g}). "
                      f"Adapter word-entropy retention was {entropy:.1f}% and unique-response retention {unique:.1f}% "
                      f"({base['unique_response_tuples']} to {tuned['unique_response_tuples']}).", ""]
        lines += ["A smaller signed gap is not automatically a smaller absolute gap. Read both race rates and diversity. "
                  "These results do not establish preserved general accuracy or causal racial invariance. "
                  "The direct test uses only tasks valid in both conditions, so missingness can matter.", "",
                  link("plots/gemma_04_paired_tuning_effect.png", "Direct paired tuning-effect graph"), "",
                  link("plots/gemma_05_base_vs_lora.png", "Focused base-versus-adapter bars"), "",
                  link("plots/gemma_06_diversity_retention.png", "Vocabulary-retention graph"), ""]
        tests=rows(path)
        significant=sum(float(r['bh_adjusted_q_value'])<.05 for r in tests)
        failed_diversity=[]
        for panel in ('neutral12','utkface'):
            base=next(r for r in summary if r['panel']==panel and r['model_key']=='gemma_base')
            tuned=next(r for r in summary if r['panel']==panel and r['model_key']=='gemma_tuned')
            if any(float(tuned[k])/float(base[k])<.95 for k in ('word_entropy_bits','unique_response_tuples')):
                failed_diversity.append(panel)
        lines += [f"**Research decision:** {significant} of {len(tests)} direct tuning-effect tests have q < 0.05. "
                  f"The historical 95% diversity-retention screen was not met on: {', '.join(failed_diversity) or 'neither panel'}. "
                  "The observed directional changes therefore do not establish a successful accuracy-preserving debiasing result. "
                  "Low UTKFace baseline negativity also leaves limited room for a reduction; equality alone is insufficient evidence of useful image understanding.", ""]
    else:
        lines += ["The local paired evaluation is incomplete or unavailable; no adapter effect is claimed.", ""]
    lines += ["## Figure Guide", "",
              "All figures place local Gemma conditions first when available. Each figure includes a short interpretation.", ""]
    descriptions={"01_negative_term_rates":"Group rates: percent of selected words that are negative. Compare uncertainty and missingness, not only bar height.",
                  "02_race_gap_significance":"Effect size, 95% interval, and exploratory p/q values. Above zero means more negative associations for Black photos.",
                  "03_validity_diversity":"Output compliance and vocabulary entropy. Broad vocabulary alone does not demonstrate accurate image understanding."}
    for panel in ("neutral12","utkface"):
        for stem,description in descriptions.items():
            lines.append("- "+link(f"plots/{panel}_{stem}.png")+": "+description)
        lines.append(f"- Individual {panel} word-count graphs: `plots/{panel}_<model_key>_word_counts.png`. "
                     "The 61 x-axis categories include unselected words. Compare rates, not raw counts, when valid sample sizes differ.")
    lines += ["", "## Changes, Failures and Interpretation Limits", "",
              "- The same image bytes and IAT61 vocabulary are retained. The reuse plan included 4,040 archived slots, of which eight were technical errors rather than model answers. Final retained archive counts are in summary.csv.",
              "- MiMo Neutral12 is a fresh DeepInfra FP8 condition; its old BF16 data are excluded from pooling.",
              "- MiniMax Neutral12's pinned DeepInfra route returned eleven HTTP 429 errors across three probes and was excluded for the rest of this session. Its UTKFace Venice route briefly became unavailable (catalog status -2), then recovered and resumed on the same endpoint. Completed answers were preserved.",
              "- Kimi Neutral12's DeepInfra route also reported unavailable status during expansion. Availability snapshots, recovery attempts and final completion counts are preserved; no alternate provider was substituted.",
              "- One MiMo transport timeout interrupted the first expansion. The uncertain charge is covered by an extra budget reservation and account reconciliation; completed response slots were retained on resume.",
              "- Runpod's container lacked rsync and apt update timed out; archive transfer with no ownership restoration was used. Historical model environments were not upgraded.",
              "- Retrieval initially failed because local Python 3.11.0 lacks the newer tar extraction filter argument. The transfer now explicitly rejects unsafe members; final remote/local checksums matched before compute shutdown.",
              "- Gemma now resets a common seed per image/iteration. Historical dataset-stream baselines can differ without any base-weight changes. The new paired base is the relevant comparator.",
              "- Fifty repetitions do not turn 12 faces into 600 independent subjects. The 12-face panel was used for retrospective adapter selection.",
              "- UTKFace matching uses fallible automated expression and image-quality labels; negative expressions can justify negative words.",
              "- Hosted backends and quantization differ from local inference. Catalog revision/route checks cannot prove immutable hosted weight bytes.",
              "- Sampling was fixed by target counts and technical/budget stops, not repeated until bias became significant. Null results are retained.",
              "- These are IAT-style implicit-association measures only. Existing text-only DecodingTrust/BOLD and image-based VisBias have different constructs and are linked in the coverage tables, not rerun here.",""]
    incomplete=[r for r in summary if r["complete"]!="True"]
    lines += ["### Incomplete Cells",""]
    lines += [f"- {r['panel']} / {r['model_name']}: {r['responses']} of {r['planned_responses']} answered slots; {r['api_error_slots']} final API-error slots." for r in incomplete] or ["All scheduled cells reached their response target."]
    session=json_read(ROOT / "results/hosted/session.json")
    ledger=read_jsonl(ROOT / "results/hosted/calls.jsonl")
    calls=Counter(r["model_key"] for r in ledger)
    cost=sum(float(r.get("cost_usd",0)) for r in ledger)
    lines += ["", "## Costs and Calls", "", f"OpenRouter: {len(ledger):,} model-call attempts, including technical errors and parsing fallbacks; "
              f"${cost:.6f} reported call cost. Latest account balance: ${session['credits_latest']['remaining']:.6f}. "
              f"Account usage delta: ${session['account_usage_delta']:.6f}. Timestamps and reconciliation are in "+link("results/hosted/session.json")+".","",
              "| Model | New API-call attempts | Reported cost (USD) |","|---|---:|---:|"]
    cost_rows=[]
    for key,n in calls.items():
        c=sum(float(r.get("cost_usd",0)) for r in ledger if r["model_key"]==key)
        cost_rows.append(dict(model_key=key,new_api_call_attempts=n,reported_cost_usd=c))
        lines.append(f"| {NAMES[key]} | {n} | {c:.6f} |")
    write_csv(ROOT / "results/analysis/model_call_costs.csv",cost_rows)
    audit_path=ROOT / "results/postflight/audit.json"
    if audit_path.exists():
        audit=json_read(audit_path)
        lines += ["", f"There were {audit['successful_model_calls']:,} successful returned API calls and "
                  f"{audit['unknown_outcome_calls']} call with an unknown transport outcome. "
                  f"The reconciled account charge minus reported call costs is ${audit['account_minus_ledger_usd']:.8f}; "
                  "billing timing can contribute to this difference. An uncertainty reservation is a spending safeguard, not an additional confirmed charge."]
    local_path=ROOT / "results/local/responses.jsonl"
    if local_path.exists():
        local=read_jsonl(local_path)
        lines += ["", f"Local Gemma: {len(local):,} saved evaluation answers, with "
                  f"{sum(int(r['generation_calls']) for r in local):,} recorded generation calls including parsing fallbacks. "
                  "The separate Qwen probe has 12 answers; Gemma's preflight base/disabled-adapter equivalence probes are not included in those response counts."]
    cleanup=ROOT / "results/runpod/cleanup.json"
    if cleanup.exists():
        c=json_read(cleanup)
        lines += ["",f"Runpod compute wall-time estimate: ${c['estimated_compute_usd']:.4f}. "
                  f"Own pod absent: {c['pod_absent']}; network volume preserved: {c['network_volume_preserved']}. "
                  "Account balance includes storage and billing timing; see "+link("results/runpod/cleanup.json")+"."]
    lines += ["", "## Next Steps", "",
              "1. Review the direct paired effect and anti-collapse metrics before choosing any new training objective.",
              "2. Use a new person-disjoint held-out sample and better expression review for a confirmatory study; do not tune on these reported p values.",
              "3. Treat Qwen2.5-VL-3B as a separate future training experiment, not a proven replacement or an already debiased model.",
              "4. Resolve any still-unavailable pinned endpoints before completing partial cells; any new route must be a separately labeled condition.","",
              "## Reproducibility", "", "See "+link("README.md")+", "+link("METHODS.md")+", "+link("DATA_DICTIONARY.md")+", and "+link("COVERAGE.md")+". "
              "Run `scripts/analyze.py`, `scripts/plot.py`, and `scripts/report.py` offline to regenerate the reported derived artifacts.",""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    coverage()
    qwen=ROOT / "results/local/qwen_probe.json"
    q=json_read(qwen) if qwen.exists() else {"status":"not available"}
    qr=rows(ROOT / 'results/local/qwen_probe.csv') if (ROOT / 'results/local/qwen_probe.csv').exists() else []
    qdetail=f"The saved probe has {q.get('valid',0)} parser-valid answers among {q.get('rows',0)} responses (status: {q['status']}). "
    if qr:
        qdetail += 'By recorded image group: '+', '.join(
            f"{race} {sum(r['valid_three']=='True' for r in qr if r['race']==race)}/{sum(r['race']==race for r in qr)} valid"
            for race in ('Black','White'))+'. '
    (ROOT / "QWEN_COMPATIBILITY.md").write_text("# Qwen2.5-VL-3B Compatibility\n\n"
        "Model: `Qwen/Qwen2.5-VL-3B-Instruct`; frozen revision `66285546d2b821cf421d4f5eb2576359d3770cd3`.\n\n"
        f"Saved probe result: `{json.dumps({k:v for k,v in q.items() if k!='prompt'},sort_keys=True)}`\n\n"
        "The probe uses the same 12 face bytes and 61-word request, but Qwen's own image/chat template, SDPA, BF16, "
        "and a 64-token sampled output budget. It is a format/inference compatibility test, not a full bias estimate. "
        +qdetail+"Parser-invalid visible answers are retained in `results/local/qwen_probe.csv`; the current failures use out-of-list positive synonyms. "
        "No retry or parsing fallback was used for this probe. A small, selectively valid subset cannot establish that the model is unbiased "
        "or that DPO will improve it; format compliance needs work before a full IAT comparison.\n\n"
        "The [official model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) documents image-and-text inference. "
        "[TRL's DPOTrainer documentation](https://huggingface.co/docs/trl/dpo_trainer#training-vision-language-models) "
        "includes this model in its vision-language DPO example and supports PEFT adapters. Thus multimodal DPO/LoRA "
        "is an available training route; no Qwen training or gradient test was performed here. Future training needs "
        "image-grounded chosen/rejected pairs, correctly masked visual/prompt tokens, a frozen reference, and held-out evaluation.\n\n"
        "A 3B model generally needs less weight/optimizer memory than Gemma 12B, but actual dollar savings depend on "
        "GPU rental and throughput. An H100 still costs the same per hour when running the smaller model. The probe "
        "download used container-local storage, not replacement of the existing network-volume models.\n")


def coverage():
    iat=link("results/analysis/summary.csv","new 12-face and UTKFace results")
    oldvis=link("results/analysis/visbias_neutral12_responses.csv","prior image-based results",old=True)
    olddt=link("results/analysis/decodingtrust_responses.csv","prior text-only results",old=True)
    oldbold=link("results/analysis/bold_scored_responses.csv","prior text-only results",old=True)
    incomplete=[r['model_name']+' ('+r['panel']+')' for r in rows(ROOT / 'results/analysis/summary.csv') if r['complete']!='True']
    # Paths are verified against the published historical package before release.
    lines=["# Benchmark Coverage", "", "## Model Groups", "",
           "The requested labels '8 other open source' and '2 closed source' do not match the models actually tested. "
           "Below, the comparison cohort is five open-weight and three closed-source models. Qwen2.5-VL-3B has only a compatibility probe.","",
           "Open-weight cohort: Qwen 3.5 9B, Kimi K3, MiMo-V2.5, MiniMax M3 and Step 3.7 Flash. "
           "Closed-source cohort: GPT-5.6 Luna, Gemini 3.6 Flash and Grok 4.6.", "",
           "| Benchmark | Base Gemma | Other open-weight models (5) | Closed-source models (3) | Tuned Gemma |",
           "|---|---|---|---|---|",
           f"| IAT61 | {iat} | {iat} | {iat} | {iat}; direct paired effect in results |",
           f"| VisBias | {oldvis} | {oldvis} | {oldvis} | Not done |",
           f"| DecodingTrust | {olddt}; local subset has severe truncation limitation | {olddt} | {olddt} | Not done |",
           f"| BOLD | {oldbold} | {oldbold} | {oldbold} | Not done |", "",
           "Incomplete new IAT cells: "+('; '.join(incomplete) if incomplete else 'none')+". Counts and reasons are in RESULTS.md.", "",
           "## Input Components", "",
           "Text means the prompt/wrapper accompanying the task, not a claim that image-based tasks were also tested without pictures.", "",
           "| Benchmark | Text | Image1: standard 12 | Image2: UTKFace |",
           "|---|---|---|---|",
           f"| IAT61 | {link('src/multimodel_bias/constants.py','61-word prompt')} | {link('results/analysis/neutral12_responses.csv','responses')} | {link('results/analysis/utkface_responses.csv','responses')} |",
           f"| VisBias | {link('src/multimodel_bias/visbias.py','description/comparison wrappers',old=True)} | {oldvis} | Not done |",
           f"| DecodingTrust | {olddt} | N/A: text-only test | N/A: text-only test |",
           f"| BOLD | {oldbold} | N/A: text-only test | N/A: text-only test |", "",
           "## Prior DPO Evidence", "",
           link("report/FORMAL_RESEARCH_REPORT.md","Audited prior methods, failed attempts, datasets, and outcomes",old=True),"",
           link("reference/dpo/august10/august10_negative_expression_pilot_v4/dataset/mixed_dpo_pairs_frozen.csv","Frozen 100-row FairFace preference dataset (80 train/20 development)",old=True),"",
           link("reference/dpo/august10_code/scripts/train_mixed_dpo_lora.py","Historical training implementation",old=True),"",
           "The new iteration does not rerun explicit bias benchmarks or add UTKFace training preferences. 'Not done' is not a null result.",""]
    (ROOT / "COVERAGE.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
