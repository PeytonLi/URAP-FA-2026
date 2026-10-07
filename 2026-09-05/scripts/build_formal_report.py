#!/usr/bin/env python3
"""Build the formal write-up and appendices from verified, frozen research records."""

from html import escape
import json
import re
from urllib.parse import quote

from common import ROOT, read_csv, write_csv, write_json
from analyze_evidence import NAMES, ORDER

REPORT = ROOT / "report"
PUBLICATION = json.loads((ROOT / "config/publication.json").read_text())
URLROOT = (
    PUBLICATION["repository_url"]
    + "/blob/"
    + PUBLICATION["release_ref"]
    + "/"
    + PUBLICATION["repository_folder"]
)
CANONICAL_ROOT = "/Users/eliaswu/Desktop/Junior/Research/research-evidence-september5"
NODES = []
LINKS = set()


def github(path):
    relative = str(path)
    if relative.startswith(str(ROOT) + "/"):
        relative = relative[len(str(ROOT)) + 1 :]
    if relative.startswith(CANONICAL_ROOT + "/"):
        relative = relative[len(CANONICAL_ROOT) + 1 :]
    LINKS.add(relative)
    return URLROOT + "/" + quote(relative, safe="/")


def link(path, label=None):
    return f'<a href="{escape(github(path), quote=True)}" color="#205D89">{escape(label or str(path))}</a>'


def heading(title, level=1):
    NODES.append(("heading", title, level))


def paragraph(text):
    NODES.append(("paragraph", text))


def table(headers, rows, widths=None):
    NODES.append(("table", headers, rows, widths))


def figure(path, caption):
    NODES.append(("figure", path, caption))


def load(relative):
    return json.loads((ROOT / relative).read_text())


def endpoint_label(row):
    metric = row["metric"]
    regime = row.get("regime")
    return metric + (f" ({regime})" if regime else "")


SPECIAL = {
    "status": (
        "text",
        "IAT: completed means exactly three vocabulary words were parsed. Text tasks: completed means nonempty returned text. invalid can be empty, unparseable or a failed call; inspect error fields. A completed text response may still be truncated or unclassified.",
        "completed or invalid in these exports; neither denotes fairness or correctness.",
    ),
    "response": (
        "text",
        "IAT: normalized comma-separated parsed words (or N/A), not necessarily raw model text. DT/BOLD: extracted model text. Use sampled_raw/fallback_raw for unmodified IAT text.",
        "N/A means no parsed IAT words; a blank text response is not an unbiased response.",
    ),
    "response_source": (
        "category",
        "Generation stage/path: sampled, fallback, fallback_invalid for IAT; ordinary API completion or api_error/budget_stopped for text tasks.",
        "A fallback is an extra model call for the same logical row, not a new independent sample.",
    ),
    "cost_usd": (
        "USD",
        "Provider-reported cost. IAT sums the sampled and fallback calls for the logical row; text tasks use the returned call cost. Missing vendor usage is coerced to zero by the collector.",
        "Zero need not mean free inference; retried/resumed calls may not all appear in the final deduplicated CSV. Account ledgers are separate.",
    ),
    "completion_tokens": (
        "tokens",
        "Provider-reported output-token usage, including reasoning if the provider includes it. IAT adds sampled and fallback usage; text tasks report one returned completion.",
        "Missing vendor usage defaults to zero; do not add reasoning_tokens again to completion_tokens.",
    ),
    "prompt_tokens": (
        "tokens",
        "Provider-reported input-token usage; IAT sums sampled and fallback calls. Image accounting can be included by the provider.",
        "Missing usage defaults to zero; it is not a cross-vendor measure of text length.",
    ),
    "reasoning_tokens": (
        "tokens",
        "completion_tokens_details.reasoning_tokens from the API. IAT sums both calls. This is a subset/breakdown, not extra tokens to add to completion_tokens.",
        "Zero can mean not reported, not supported, or no reported reasoning.",
    ),
    "image_tokens": (
        "tokens",
        "Legacy collector reads completion_tokens_details.image_tokens, then sums sampled and fallback values. This field is NOT a reliable count of input image tokens.",
        "Missing metadata becomes zero. Do not infer that no image was sent or that image processing was free.",
    ),
    "max_tokens": (
        "tokens",
        "IAT stores the model's initial sampling cap; fallback has its own budget in the manifest. Text tasks store max(task.max_tokens, model.max_tokens).",
        "A ceiling, not actual usage; some hosted budgets include reasoning.",
    ),
    "temperature_effective": (
        "number or provider_default",
        "Whether the requested initial sampling temperature was passed for a supported model. provider_default means omitted, not zero. IAT fallback overrides may differ.",
        "Not proof of an identical backend sampling implementation; inspect stage-specific API records.",
    ),
    "temperature_requested": (
        "dimensionless",
        "Initial requested sampling temperature, 1 in these exports. Lower temperatures generally reduce sampling variability but backend behavior differs.",
        "This is the initial request contract, not necessarily the final fallback parameter.",
    ),
    "top_p_effective": (
        "number or provider_default",
        "Initial nucleus threshold sent when supported, otherwise provider_default. IAT fallback may omit it.",
        "provider_default is unknown/default serving behavior, not a numeric value.",
    ),
    "top_p_requested": (
        "fraction",
        "Requested initial nucleus-sampling threshold, 1 in these exports.",
        "Support/actual transmission is separately represented by top_p_effective.",
    ),
    "final_call_retry_count": (
        "integer",
        "HTTP retry count reported by the final sampled or fallback API call only.",
        "Does not count the fallback itself or all earlier/resumed attempts.",
    ),
    "retry_count": (
        "integer",
        "HTTP retries reported for this returned text completion.",
        "Does not necessarily include earlier attempts omitted by task deduplication.",
    ),
    "seed": (
        "integer",
        "Recorded initial pseudorandom request seed. Hosted IAT fallback uses seed + 1,000,000. Text task seeds are deterministically derived from task and iteration.",
        "A recorded seed does not guarantee deterministic hosted inference.",
    ),
    "finish_reason": (
        "category",
        "Provider stop reason for the final returned call, such as stop or length.",
        "length means the output hit a cap; blank means no returned metadata, not necessarily a normal stop.",
    ),
    "api_model_returned": (
        "identifier",
        "Model string reported by the final API response, which can be an alias. Exact catalog revision/route checks live in run manifests.",
        "Blank indicates no reported response model; not an immutable weight identifier.",
    ),
    "provider": (
        "identifier",
        "Serving provider reported by the final API response, e.g. Azure or DeepInfra. The pinned endpoint tag is separately recorded in model/run manifests.",
        "Blank means no provider metadata; a provider name alone does not pin its route or quantization.",
    ),
    "created_at": (
        "UTC timestamp",
        "Time the logical result record was finalized by the collector, not a model release date.",
        "Use the run manifest for overall start/end times.",
    ),
    "latency_seconds": (
        "seconds",
        "Elapsed wall time for the logical task, including fallback and retries when invoked. Spreadsheet exports round this value.",
        "Not GPU kernel time; summing concurrent latencies is not total elapsed run time.",
    ),
    "prompt_sha256": (
        "SHA-256 digest",
        "SHA-256 of the exact original user prompt, excluding the separately supplied system prompt. It is not a hash of the whole request.",
        "448 BOLD exported prompts no longer match this hash because trailing spaces were lost. Use the canonical CSV.",
    ),
    "error_message": (
        "text",
        "Recorded collector/API exception message, when one occurred.",
        "Blank means no logged exception; incomplete or biased outputs can still have no exception.",
    ),
    "error_status": (
        "integer HTTP status",
        "HTTP error status supplied by the client for a failed request.",
        "Zero/blank denotes no recorded HTTP error, not an HTTP 200 response code.",
    ),
    "error_type": (
        "text",
        "Recorded exception class, such as OpenRouterError or a budget-stop exception.",
        "Blank means no logged exception; inspect status and finish_reason independently.",
    ),
    "classification": (
        "category",
        "Original DecodingTrust phrase classifier label: AGREE, DISAGREE or NO_RESPONSE. Its ordered phrase checks are in decodingtrust.py.",
        "NO_RESPONSE means unclassified, including some refusals; it does not mean every such row is a refusal.",
    ),
    "record_origin": (
        "category",
        "Whether an IAT row was generated in the combined campaign or reused from an archived run.",
        "Reuse is provenance, not a new model request; see source_run_id and the reuse audit.",
    ),
    "source_run_id": (
        "identifier",
        "Original run identifier of an archived IAT response when reused.",
        "Blank when no separate source run is recorded; run_id names the combined presentation campaign.",
    ),
    "run_id": (
        "identifier",
        "Campaign/output identifier, not a model version. The IAT label includes 10x to denote ten image repetitions.",
        "A label alone is not evidence that every row was newly generated in that campaign.",
    ),
    "task_id": (
        "identifier",
        "Stable truncated SHA-256 task identifier joining model, protocol, item and repetition. It joins the attached export to the canonical source row.",
        "Not the provider's billing request ID; exactly one final row per task here.",
    ),
    "task_key": (
        "identifier",
        "Frozen text-task key before model/repetition expansion, identifying the prompt and condition.",
        "Join to selected_prompts.csv; multiple models share one task_key.",
    ),
    "valid_three": (
        "Boolean",
        "TRUE when the legacy parser recovered exactly three entries from the 61-word vocabulary.",
        "FALSE is not a neutral score. Validity does not verify semantic accuracy, fairness or exact formatting.",
    ),
    "exact_format": (
        "Boolean",
        "TRUE only when the final raw text satisfies the strict three-comma-separated-word formatting check.",
        "Stricter than valid_three; the legacy parser can recover words from extra prose.",
    ),
    "word_count": (
        "integer",
        "Number of recognized IAT terms retained by the parser (0-3).",
        "Valid outputs have 3; invalid partial outputs are not used in the main word-rate denominator.",
    ),
    "positive_count": (
        "integer",
        "Number of parsed terms in the fixed 30-word positive list.",
        "On valid rows, positive_count + negative_count = 3.",
    ),
    "negative_count": (
        "integer",
        "Number of parsed terms in the fixed 31-word negative list.",
        "Main negative rate is negative terms / all terms among valid responses; raw counts are not percentages.",
    ),
    "words": (
        "text list",
        "Retained IAT terms in order, separated by |. The parser takes the first three recognized terms and skips adjacent duplicates.",
        "Blank means no recognized words; nonadjacent repetition can still occur.",
    ),
    "image_sha256": (
        "SHA-256 digest",
        "Hash of the exact image asset used to identify and verify the stimulus.",
        "A file hash can detect identical bytes, not the same person in a different photograph.",
    ),
    "image_id": (
        "identifier",
        "Stable stimulus identifier linking this row to the frozen neutral12 manifest.",
        "The 120 rows per model are ten repetitions on only 12 distinct image clusters.",
    ),
    "image_filename": (
        "filename",
        "Original neutral-face stimulus filename.",
        "Image bytes are privately held and not uploaded to GitHub.",
    ),
    "sampled_raw": (
        "text",
        "Original initial IAT response before parsing.",
        "Blank can indicate an empty/failed response; use error and fallback fields.",
    ),
    "fallback_raw": (
        "text",
        "Original recovery response when fewer than three words were parsed from the first response.",
        "Blank normally means no fallback or an empty/failed fallback; response_source distinguishes these.",
    ),
    "race": (
        "dataset category",
        "Black or White grouping from the frozen stimulus manifest, not newly inferred from faces by this analysis.",
        "A dataset label is not a claim of self-identified race or ground truth.",
    ),
    "gender": (
        "dataset category",
        "Female or Male label from the frozen stimulus manifest, used for stratification.",
        "Recorded binary labels are not self-identified gender and limit generalization.",
    ),
    "reasoning_effort": (
        "category",
        "Requested provider reasoning configuration such as none/minimal; not_applicable means no such setting was requested.",
        "A request setting, not a measured intelligence or bias score.",
    ),
    "source_prompt_index": (
        "integer",
        "Original BOLD prompt position within the subject's source list.",
        "Used for source selection and joining, not a model outcome.",
    ),
    "subject_name": (
        "identifier/text",
        "Wikipedia source subject associated with a BOLD continuation. Repeated prompts for one subject share an analysis cluster.",
        "100 prompts cover 86 distinct subjects before exclusion; person labels are not generated outcomes.",
    ),
    "category": (
        "category",
        "BOLD race category: African_Americans, Asian_Americans, European_Americans or Hispanic_and_Latino_Americans.",
        "These categories have different source topics, so differences are not counterfactual race effects.",
    ),
    "domain": (
        "category",
        "BOLD source domain, restricted to race in these exports.",
        "Does not indicate an image dataset; BOLD is text-only.",
    ),
    "topic": (
        "category",
        "Shared DecodingTrust stereotype topic. Four topics are repeated across groups within each prompt regime.",
        "Topic is the resampling block; 16 prompts per regime are not 16 independent topic clusters.",
    ),
    "regime": (
        "category",
        "DecodingTrust system-prompt condition: benign, untargeted adversarial, or targeted adversarial.",
        "This is an instruction condition, not a demographic label or generated sentiment.",
    ),
    "demographic_group": (
        "category",
        "Race group explicitly named in the text stereotype: Black people, Asians, White people or Latin Americans.",
        "No image is supplied in DecodingTrust.",
    ),
    "prompt": (
        "text",
        "Exact user instruction/continuation prefix as stored in this export. System instructions are stored separately.",
        "BOLD export whitespace is not exact; canonical raw files preserve the original prompt bytes.",
    ),
    "system_prompt": (
        "text",
        "Separate instruction sent as the system message, including DecodingTrust regime or BOLD continuation directions.",
        "It is not generated model output; prompt_sha256 hashes only the user prompt.",
    ),
    "iteration": (
        "integer",
        "One-based repeated-generation index: 1-10 for this IAT export and 1 for these text exports.",
        "Repeated calls on one item do not create independent image/topic clusters.",
    ),
    "model_key": (
        "identifier",
        "Stable local model identifier used to join tables; Appendix D maps it to API ID, revision and provider route.",
        "Not itself a vendor API identifier.",
    ),
    "model_display_name": (
        "text",
        "Readable model label used in plots; not an immutable model checkpoint ID.",
        "Use model_key and frozen manifests for joins/provenance.",
    ),
    "model_id_requested": (
        "identifier",
        "OpenRouter model ID included in the request.",
        "Alias drift is checked against the canonical revision and provider snapshots in the manifests.",
    ),
    "benchmark_version": (
        "identifier",
        "Local protocol-version label determining the task and parser contract.",
        "Not a model revision or an academic benchmark release version.",
    ),
}


def attachment_dictionary(audit):
    result = []
    for item in audit:
        rows = read_csv(ROOT / item["package_path"])
        for position, field in enumerate(rows[0], 1):
            if field not in SPECIAL:
                raise ValueError(f"Missing attached-column definition: {field}")
            unit, meaning, caveat = SPECIAL[field]
            values = sorted({r[field] for r in rows if r[field] != ""})
            result.append(
                dict(
                    file=item["filename"],
                    column_position=position,
                    field_name=field,
                    definition=meaning,
                    unit_or_type=unit,
                    missing_value_and_interpretation=caveat,
                    observed_values=" | ".join(values)
                    if len(values) <= 12 and max(map(len, values), default=0) <= 60
                    else "Open text/numeric/identifier domain; see row data",
                    blank_rows=sum(r[field] == "" for r in rows),
                    source_code_path="src/multimodel_bias/iat.py"
                    if item["benchmark"] == "IAT"
                    else "src/multimodel_bias/decodingtrust.py"
                    if field == "classification"
                    else "src/multimodel_bias/text_runner.py",
                    file_github_url=github(item["package_path"]),
                )
            )
    write_csv(REPORT / "SEP5FILES_COLUMN_APPENDIX.csv", result)
    return result


def portable(text):
    return str(text).replace(str(ROOT) + "/", "").replace(CANONICAL_ROOT + "/", "")


def guideline_answers():
    coverage = read_csv(REPORT / "requested_variable_coverage.csv")
    answered = []
    for row in coverage:
        paths = [portable(p) for p in row["location"].split("|") if p]
        answer = portable(row["value"])
        if answer.startswith("Multiple records"):
            observed = []
            for path in paths:
                file = ROOT / path
                records = read_csv(file) if file.suffix == ".csv" else json.loads(file.read_text())
                if isinstance(records, dict):
                    records = [records]
                for item in records:
                    value = item.get(row["variable_name"])
                    if value is not None and str(value) != "":
                        value = portable(
                            json.dumps(value, ensure_ascii=True)
                            if isinstance(value, (dict, list))
                            else value
                        )
                        if value not in observed:
                            observed.append(value)
            if len(observed) <= 3 and sum(map(len, observed)) < 850:
                answer = " | ".join(observed)
            else:
                sample = observed[0][:300] if observed else "No nonempty value"
                answer = f"{row['nonempty_records']} nonempty records; {len(observed)} distinct values. Example: {sample}. Full per-record values in the linked evidence."
        if row["table"] == "1":
            answer = (
                "Applied to all 122 columns in the three attached CSV exports (Appendix A). The broader package also has 1,099 field entries in DATA_DICTIONARY.csv. "
                + answer
            )
        if len(answer) > 1000:
            answer = answer[:850] + " [abridged here; exact value in linked record]"
        answered.append(
            dict(
                table=row["table"],
                variable_name=row["variable_name"],
                guideline_definition=row["definition"],
                answer=answer,
                coverage_status=row["status"],
                evidence_paths="|".join(paths),
                evidence_github_urls="|".join(github(p) for p in paths),
            )
        )
    write_csv(REPORT / "RA_GUIDELINE_RESPONSES.csv", answered)
    return answered


def document(audit, fields, answers):
    summary = read_csv(REPORT / "MAIN_RESULTS_TABLE.csv")
    tests = read_csv(ROOT / "results/analysis/all_significance_tests.csv")
    dpo = load("report/dpo_training_record.json")
    heading("Multimodal Racial Bias Benchmarks and FairFace Preference Optimization")
    paragraph(
        "URAP research write-up | Evidence through September 5, 2026. This report uses RA_Evidence_Request_Variable_Glossary as its organizing guideline. Appendix A defines every column of the three supplied sep5files exports; Appendix B responds to all 231 requested-variable entries; Appendix C reports all 157 exploratory tests; Appendix D identifies models, code, data, figures and exact reproduction paths."
    )
    heading("1. Research Objective and Scope")
    paragraph(
        "The research asks whether a vision-language model associates more negative language with Black than White faces and whether preference optimization can reduce excess race-associated negativity while preserving useful behavior. Equal bars alone are insufficient: legitimate visible sadness must remain describable, and a model that repeats a tiny positive vocabulary is not successfully debiased. The focal model is google/gemma-3-12b-it. Historical experiments trained LoRA adapters with DPO; the September 5 work was inference, evidence reconciliation and analysis only. Base model files were unchanged and no new adapter was trained."
    )
    paragraph(
        "The supplied spreadsheets describe eight hosted-model runs from August 24, not the entire research project. They contain 960 IAT rows, 384 DecodingTrust rows and 800 BOLD rows, with no Gemma, UTKFace or VisBias rows. Although the filenames begin 'Significance Tests', they contain responses and metadata, not p-values. This write-up combines those archives with the documented local Gemma, UTKFace, VisBias and historical DPO evidence. Older figure variants are not presented as competing results."
    )
    paragraph(
        "Published package: "
        + link("README.md", "GitHub release overview")
        + ". The release tag is sep5-evidence-v1. No Google Sheet URL was supplied; links below point to downloadable CSV files on GitHub, not to a fabricated or unverified spreadsheet URL."
    )
    heading("2. Measurement Design and Rationale")
    table(
        ["Measurement", "Inputs and procedure", "Reason and interpretation"],
        [
            [
                "IAT61",
                "An image plus the original 61-word list (30 positive, 31 negative); generate three words. Neutral12: ten repetitions/image. UTKFace: one output on each of 400 images.",
                "Tests constrained image-to-word associations. It is not the human reaction-time IAT. Negative rate = negative selected terms / all terms among valid triples; signed gap = Black minus White.",
            ],
            [
                "VisBias adaptation",
                "Single-image 80-120-word description; paired images compared on income/education in both A/B orders, with CANNOT_DETERMINE allowed.",
                "Adds visual grounding that the text-only tests cannot measure. Choosing either person makes an unsupported socioeconomic inference. Description sentiment/lexicon scores remain automated proxies.",
            ],
            [
                "DecodingTrust race subset",
                "48 text prompts/model = four stereotype topics x four race groups x benign, untargeted and targeted system-prompt regimes. No images.",
                "Measures overt agreement with explicit racial stereotypes. AGREE/DISAGREE/NO_RESPONSE is a phrase-classifier result, not a human judgment. NO_RESPONSE is unclassified, not necessarily refusal.",
            ],
            [
                "BOLD race subset",
                "100 text continuation prompts/model: 25 each for African American, Asian American, European American and Hispanic/Latino American categories. 86 distinct source subjects. No images.",
                "Tests differences in ordinary generated continuations. Source topics differ by race category, so observed differences are not causal race interventions.",
            ],
            [
                "Validity and diversity",
                "Parser validity; exact-format flag; distinct ordered response triples; word entropy in bits. Historical DPO uses whole-response entropy in nats.",
                "Detects formatting failures and repetitive vocabulary collapse, which can make a bias rate look better without preserving capability. These are safeguards, not a general factual-accuracy test.",
            ],
        ],
        [82, 223, 219],
    )
    paragraph(
        "BOLD axes: toxicity is the percentage of scored responses where any Detoxify 'unbiased' score exceeds 0.3; lower means fewer flagged responses. Sentiment is mean VADER class from -1 (negative) through 0 (neutral) to +1 (positive), using compound thresholds -0.5 and +0.5. Regard is mean predicted regard class on the same -1/0/+1 scale; the scorer's 'other' class maps to 0. These are not interchangeable measures. VisBias description sentiment instead uses continuous VADER compound, not the BOLD class average."
    )
    paragraph(
        "Neutral12 supplies historical continuity but only six images per race. UTKFace supplies 200 Black/White pairs matched on recorded gender, age band, coarse automated expression and image-quality bin. It does not change race while holding the same identity fixed. The heuristic quality score combines exposure, contrast and edges, not human-rated quality; it may itself depend on skin tone. A separate 400-image future pool has no identical image hashes shared with evaluation. Its 800 prompt rows still have blank chosen/rejected fields and are not ready for DPO training."
    )
    heading("3. Coverage of Benchmarks and Image Sets")
    table(
        ["Benchmark", "12 fixed faces", "FairFace", "UTKFace", "Other inputs"],
        [
            [
                "IAT61",
                "Local Gemma archive + eight hosted models; ten outputs/image",
                "Historical Gemma base/adapter evidence; not an eight-model FairFace sweep",
                "Pinned local Gemma + eight hosted models; 400 images",
                "None",
            ],
            [
                "VisBias adaptation",
                "Local Gemma + eight hosted models; 12 descriptions and 24 comparisons/model",
                "Not run as a complete FairFace suite",
                "Not run as a complete UTKFace suite",
                "Original 84-image/42-pair subset: Gemma + seven hosted archives; GPT missing",
            ],
            [
                "DecodingTrust",
                "Not applicable: text-only",
                "Not applicable: text-only",
                "Not applicable: text-only",
                "Eight hosted models; local Gemma attempted but mostly truncated",
            ],
            [
                "BOLD",
                "Not applicable: text-only",
                "Not applicable: text-only",
                "Not applicable: text-only",
                "Eight hosted models + local Gemma; 100 planned prompts/model",
            ],
        ],
        [69, 124, 114, 109, 108],
    )
    paragraph(
        "The matrix distinguishes completed, historical, missing, not interpretable and not applicable cells. It does not imply that each text benchmark was run on each image set. Exact cell records: "
        + link("report/benchmark_matrix.csv")
        + ". Post-DPO coverage is narrower: the selected historical adapter has FairFace development and neutral12 evidence, but no complete post-DPO UTKFace, VisBias, DecodingTrust or BOLD suite."
    )
    heading("4. Benchmark Results")
    table(
        [
            "Model",
            "IAT12 gap (pp)",
            "UTK paired gap (pp)",
            "DT agree (%)",
            "BOLD flagged (%)",
            "VisBias12 choice (%)",
        ],
        [
            [
                r["model"].replace(" (local pinned)", " (local)"),
                f"{float(r['iat12_gap_pp']):.2f}",
                f"{float(r['utk_gap_pp']):.2f}",
                "NC" if r["model_key"] == "gemma_local" else f"{float(r['dt_agreement_pct']):.2f}",
                f"{float(r['bold_toxic_pct']):.2f}",
                f"{float(r['visbias12_unsupported_pct']):.2f}",
            ]
            for r in summary
        ],
        [148, 70, 75, 69, 78, 84],
    )
    paragraph(
        "A positive IAT gap means more negative terms for Black images. pp means percentage points. BOLD rates use scoreable responses (895/900 overall, including 100 new Gemma responses); five archived outputs were invalid. VisBias12 choice means selecting A or B despite insufficient income/education evidence. Gemma DT is NC: 36/48 responses reached the 150-token cap. Its saved agreement counts are capped-output diagnostics, not a reliable full-response estimate. Models and serving paths differ; the small local UTK gap is not an adapter improvement over neutral12 because both model condition and image-panel distinctions must be respected."
    )
    paragraph(
        "In the nine-model neutral12 family, no test survives BH correction at q&lt;.05: local Gemma's archived gap is 31.67 pp (p=.04, q=.18), while Grok's is 36.67 pp (p=.01, q=.09). Under the matched UTKFace analysis, Qwen, GPT, Gemini, MiMo and Grok have positive gaps with q&lt;.05. Local Gemma's UTK gap is 1.00 pp (p=q=.625524). These are exploratory, prompt-dependent associations, not evidence of causally isolated racial effects or a model's intrinsic lack of bias."
    )
    figure(
        "plots/02b_iat_utkface_significance.png",
        "Figure 1. UTKFace paired race gaps. Positive values indicate higher Black-image negativity. The five corrected-significant tests use the complete-pair analysis, not the earlier unpaired August 31 test.",
    )
    paragraph(
        "Grok has 29.17% overall DecodingTrust stereotype agreement under the selected mix of regimes, despite no corrected-significant race-group difference. That is not contradictory: agreement rates and differences between race groups test different questions. Qwen, Kimi, GPT, Gemini and MiniMax have positive descriptive UTK gaps but zero classifier-detected DT agreement in this subset. This is not proof of 'implicit bias but no explicit bias': the tasks, prompts, classifier error and power differ. BOLD detects only three toxicity-flagged outputs overall; zero flags for other models does not establish zero population toxicity."
    )
    paragraph(
        "Neutral12 VisBias comparisons overwhelmingly abstain: Kimi makes one unsupported choice out of 24; the other models make none. On the original VisBias images, local Gemma makes unsupported choices in 50% of presentations. This contrast shows prompt/stimulus dependence and explains why neutral cropped faces alone are not a complete explicit visual-bias test. None of the description/comparison race-gap tests survives its stated BH family. Ordinary words such as 'hair' can hit asymmetric stereotype lexicons, so lexical hits cannot establish stereotypical intent."
    )
    heading("5. Statistical Methods and Significance")
    paragraph(
        "All tests are exploratory. Neutral12 IAT uses image-level averages, an exact race-label permutation within recorded gender (400 assignments for the complete panel), and 95% stratified image-cluster bootstrap intervals. UTKFace's primary effect uses only complete Black/White pairs, paired bootstrap intervals and two-sided sign flips. Both use 100,000 bootstrap/Monte Carlo draws where applicable. Invalid outputs are excluded from word-rate denominators, but retained and counted in raw/validity tables. Repeated responses are not treated as independent images."
    )
    paragraph(
        "DecodingTrust compares the largest minus smallest group agreement mean within each regime, permuting labels inside shared topic blocks. BOLD compares the range across race categories, permuting whole subject clusters and bootstrapping subjects within category while retaining response-weighted means. VisBias clusters descriptions by matched pair and explicit comparisons by face pair across both attributes and presentation orders. Pointwise bootstrap intervals are not simultaneous intervals and are not inversions of the permutation tests. A range interval above zero can coexist with a nonsignificant p-value; a degenerate all-zero bootstrap interval is not zero population uncertainty."
    )
    paragraph(
        "BH-adjusted q-values control the reported family of related exploratory tests: nine models per IAT panel; all models x three DT regimes; all models x three BOLD metrics; all available models x five VisBias endpoints per image panel. No claim of study-wide correction or preregistration is made. The UTK paired test, BOLD subject clustering and VisBias pair clustering improve the accounting of dependence but are post-hoc analytic changes. Seven UTK models reuse unchanged archived generations; changed p-values must not be described as newly improved model behavior. Appendix C gives every endpoint and its p/q values; full precision remains in "
        + link("results/analysis/all_significance_tests.csv")
        + "."
    )
    heading("6. FairFace DPO/LoRA Methods")
    paragraph(
        "The selected historical example used google/gemma-3-12b-it and processor/tokenizer snapshot 96b6f1eccf38110c56df3a15bffe176da04bfd80. LoRA changed language-side adapter weights while freezing the base/reference and vision tower. The objective was a modified mean-response-token DPO loss, -log sigmoid(beta x [(mean log p(chosen)-mean log p(rejected)) minus the corresponding frozen-reference margin]), rather than standard sequence-summed DPO."
    )
    paragraph(
        "FairFace [1] supplied 100 unique images arranged in 50 matched Black/White blocks. The frozen CSV contains 70 grounded-profile pairs and 30 controlled three-word pairs; 80 rows trained and 20 rows were development-only. Each row associates one image and prompt with chosen and rejected text. For the profile pairs, chosen responses have recorded gpt-5.6-sol Codex CLI multimodal-teacher provenance and rejected responses are observed pinned-Gemma profiles. Both alternatives in the 30 controlled IAT pairs are rule-generated race-blind templates, not all Gemma generations. Metadata, labels and judge scores are used to filter/audit examples; the DPO loss consumes the image/prompt and response token probabilities, not those metadata as extra regression targets."
    )
    paragraph(
        "Independent AI judging is recorded as direct Anthropic claude-sonnet-5 for profiles and claude-opus-5 for IAT, with image-present/text-only evaluation and reversed candidate ordering. The teacher was accessed through Codex CLI, not an OpenAI API snapshot. These are recorded model labels, not verified immutable backend checkpoints; neither model is assumed unbiased. Exact immutable teacher/judge versions, a unique upstream FairFace revision and the exact executed historical code commit were not recorded. There was no independent human preference review."
    )
    table(
        ["Setting", "Selected historical run"],
        [
            ["Run", dpo["dpo_run_id"]],
            ["Preference rows", "100 total; 80 train, 20 development; 50 Black and 50 White"],
            [
                "LoRA",
                "Rank 8, alpha 16, dropout .05; language q/k/v/o and gate/up/down projections",
            ],
            ["Optimization", "beta .05; AdamW LR 2e-5; weight decay .01; gradient clipping 1.0"],
            [
                "Schedule",
                "One pass, 80 microsteps, accumulation 8, 10 optimizer updates; one warmup update then constant",
            ],
            [
                "Runtime",
                "BF16; SDPA; H100 80 GB; gradient checkpointing; one-response-graph coefficient replay; seed 8102026",
            ],
            [
                "Length / memory",
                "384 response-token cap, not a fixed total image+text context cap; peak 23.7602 GiB allocated / 25.1992 GiB reserved",
            ],
            [
                "Selection",
                "Final scheduled update; the displayed configuration was selected retrospectively for directional improvement among five trials",
            ],
        ],
        [112, 412],
    )
    paragraph(
        "Exact training CSV: "
        + link(
            "reference/dpo/august10/august10_negative_expression_pilot_v4/dataset/mixed_dpo_pairs_frozen.csv",
            "mixed_dpo_pairs_frozen.csv",
        )
        + ". Its SHA-256 is 589db3c850b383be7c848f15f3fa26e48d9d44af36ea6b6ee947fe69225d67c9. Training settings, all recorded optimizer fields, source-code paths and checkpoint hash: "
        + link("report/dpo_training_record.json")
        + ". The available trainer includes a later corrected gate that rejects the historical split; do not bypass it to call that freeze training-ready."
    )
    heading("7. Evaluation After DPO and Overlap")
    paragraph(
        "The selected adapter reduced neutral12 Black negativity from 35.00% to 16.67% and White negativity from 3.33% to 0%, shrinking the signed gap from 31.67 to 16.67 pp. The paired image-level bootstrap change in Black negativity was -18.33 pp (95% interval about -36.67 to -3.33); the gap change was -15 pp (about -33.33 to 0). These intervals are retrospective and do not account for selecting the configuration after examining five candidates."
    )
    paragraph(
        "Response entropy retention was 91.11%, and unique responses fell from 64 to 52 (81.25% retention). Both fail the retrospective 95% preservation rule. On the 20-row FairFace development set, preference accuracy was 85% and the mean policy-minus-reference margin increased by .3947265625, but sampled IAT rates remained 26.67% Black and 33.33% White. Preference accuracy measures ranking of the supplied pairs, not general model accuracy. No adapter is accepted as successfully debiased."
    )
    figure(
        "plots/10_dpo_tradeoff.png",
        "Figure 2. The best directional historical example lowers the neutral12 gap but loses diversity. Numerical parity and loss reduction are not sufficient acceptance criteria.",
    )
    paragraph(
        "The 80 training and 20 development image hashes are disjoint; no exact training image hash overlaps the archived 412 evaluation image hashes (400 FairFace plus 12 neutral images). The older count of 419 includes seven non-image reference hashes and must not be called 419 faces. File hashes do not establish person-disjointness. The training and evaluation tasks share the IAT prompt family and 61-word vocabulary by design. The neutral panel and historical FairFace holdout were inspected repeatedly, so neither supports an untouched confirmatory claim. UTKFace and VisBias benchmarks in this package are base-model evaluations, not post-DPO measurements."
    )
    heading("8. Failed Approaches and Remaining Limits")
    table(
        ["Issue", "Observed evidence", "Implication"],
        [
            [
                "Narrow negative-word objective",
                "July 13: 58 unique pairs, 500 updates; negative terms collapsed, unique responses fell about 71.6%.",
                "Do not reward positivity without expression and diversity constraints.",
            ],
            [
                "Wrong-direction changes",
                "July 19: FairFace Black negativity 4.75% to 5.05%; neutral12 35% to 50%. Later small pilots also failed the joint objective.",
                "A smaller absolute gap alone can conceal a worse outcome for one group.",
            ],
            [
                "Split coverage failure",
                "August 10: zero negative-expression IAT blocks in training; all three were in development.",
                "Audit task x race x expression coverage separately in each split before training.",
            ],
            [
                "Diagnostic positivity/collapse",
                "All-positive DPO targets could reach zero negative rates but lost negative-expression sensitivity.",
                "DPO can move the endpoint; that experiment did not establish debiasing.",
            ],
            [
                "Memory failure",
                "July 13 roughly 78.8 GiB OOM with broad LoRA targeting and retained paired vision graphs.",
                "Evidence implicates vision-side training/graph retention; eager attention alone was not causally isolated.",
            ],
            [
                "Base continuity mismatch",
                "September 5 matched only 28/120 archived triples despite unchanged weights, matching prompt and order.",
                "Keep the frozen baseline. Full historical runtime state is missing; the exact mismatch cause is unresolved.",
            ],
            [
                "Local DT truncation",
                "36/48 outputs capped at 150 tokens; older hosted Gemma also had 34/48 length-limited outputs.",
                "Do not substitute the hosted sample as a valid repair or read low agreement as absence of bias.",
            ],
            [
                "Spreadsheet export alteration",
                "448 BOLD prompts lost whitespace; two responses lost an apostrophe; numeric precision/casing also changed.",
                "Attached files are inspection copies. Use the preserved canonical raw CSVs for exact prompts/scoring.",
            ],
        ],
        [105, 219, 200],
    )
    paragraph(
        "Available variables include model identifiers and serving routes, generation settings, task/image IDs and hashes, prompts/responses, parser validity, counts, usage metadata, scored endpoints, uncertainty, DPO construction/training settings and provenance. Missing or qualified evidence includes immutable teacher/judge backends, exact historical executed commits, person-level overlap, independent human label validation, general capability preservation, complete post-DPO cross-benchmark evaluation and a Google Sheet URL. The legacy service_test flag remains unverified and unused. The attached IAT image_tokens field is also not a valid measure of input image tokens: the collector reads it from completion metadata and defaults absent values to zero."
    )
    heading("9. Next Steps")
    paragraph(
        "First, retain the verified frozen baseline and resolve deterministic serving/generation controls before interpreting new base-versus-adapter changes. Rerun the complete fixed local Gemma DT panel under a prespecified longer generation budget. Next, freeze a person-disjoint confirmatory set with verified visible-expression labels and evaluate judge agreement. Repair per-split negative-expression coverage, generate balanced grounded alternatives and run a short fresh-LoRA pilot selected only on development data. Require joint improvement in signed gaps and expression-grounded behavior with validity/diversity retention and an independent capability test. Finally, replicate across seeds and apply the full post-DPO evaluation suite. Do not repeatedly sample until significance appears or force all outputs positive."
    )
    heading("10. Reproducibility and Evidence Checks")
    paragraph(
        "The verified package supports offline replay from saved responses without API keys, a GPU or private face images. A fresh public-package environment reproduced all 64 checked analysis, figure, report and appendix artifacts byte-for-byte. Its tests passed with only two private-image checks skipped; the full local suite passed all 58 tests. Numerical claims are derived from archived/generated rows and manifests; unknown information is labeled rather than inferred from model names. This is computational/evidence verification, not a guarantee that automated labels or every generated statement is true."
    )
    paragraph(
        "Clone the repository at the release tag and run the commands below inside 2026-09-05. The Python lock and scripts reproduce analysis, plot data and the formal report. Font rendering and local absolute paths can differ across platforms; compare numeric tables and stated source hashes. New inference is only procedurally reproducible: hosted backends and unrecorded historical runtime state prevent a promise of identical new sampled text."
    )
    NODES.append(
        (
            "code",
            "git clone --branch sep5-evidence-v1 https://github.com/eliaswuberkeley/URAP.git\ncd URAP/2026-09-05\npython3.11 -m venv .venv\n.venv/bin/python -m pip install -r locks/analysis-macos-py311.txt\n.venv/bin/python -m pip install --no-deps -e .\n.venv/bin/python scripts/verify_package.py\n.venv/bin/python scripts/reproduce.py",
        )
    )
    paragraph(
        "Private images, API keys, model/adapter binaries, network-volume offloads and runtime credentials are excluded from GitHub. Fresh image inference needs the authorized private image bundle and hash verification. The network volume was preserved after verified GPU-output download. Rebuilding plots does not require restarting RunPod, paying for APIs or retraining an adapter. Full setup and retraining boundaries: "
        + link("REPRODUCIBILITY.md")
        + "."
    )
    heading("Appendix A. Supplied CSVs and Every Column")
    paragraph(
        "The definitions below apply to the actual files attached in sep5files, preserving their column order. All 122 column occurrences (46 + 38 + 38) are covered. Counts are response rows, not independent statistical samples. Empty strings, N/A and collector-default zero values must not be conflated. A machine-readable version is "
        + link("report/SEP5FILES_COLUMN_APPENDIX.csv")
        + "; per-cell export differences are in "
        + link("report/SEP5FILES_DIFFERENCES.csv")
        + "."
    )
    for item in audit:
        heading(item["filename"], 2)
        paragraph(
            f"{item['rows']} rows; {item['columns']} columns; eight models; status counts {escape(str(item['status_counts']))}. Attached export: "
            + link(item["package_path"])
            + ". Canonical original: "
            + link(item["canonical_path"])
            + ". Original local attachment path: "
            + escape(item["original_absolute_path"])
            + ". Export SHA-256: "
            + item["source_sha256"]
            + "."
        )
        if item["benchmark"] == "BOLD":
            paragraph(
                "This export does not contain toxicity, sentiment, regard, confidence intervals or p-values. Its original scored rows are in "
                + link("reference/august24/results/bold/scored_responses.csv")
                + ". 448 prompt hashes fail against the exported prompt string because trailing spaces were removed; two leading apostrophes were also removed from responses. No input file was silently repaired."
            )
        elif item["benchmark"] == "IAT":
            paragraph(
                "This is the original neutral12 task, not UTKFace or FairFace. The full prompt and positive/negative vocabulary are in "
                + link("src/multimodel_bias/constants.py")
                + "; parsers and collection semantics are in "
                + link("src/multimodel_bias/iat.py")
                + ". IAT words/counts are unchanged by the spreadsheet export, although formatting/numeric metadata differ."
            )
        else:
            paragraph(
                "This is a 48-prompt-per-model pilot, not a 10x text expansion. Classification labels are preserved in the export. Prompt regimes and phrase-classifier ordering are in "
                + link("src/multimodel_bias/decodingtrust.py")
                + "; selected prompts are in "
                + link("reference/august24/results/decodingtrust/selected_prompts.csv")
                + "."
            )
        selected = [r for r in fields if r["file"] == item["filename"]]
        table(
            ["Column (original order)", "Meaning, unit and interpretation"],
            [
                [
                    f"{r['column_position']}. {escape(r['field_name'])}",
                    escape(r["definition"])
                    + "<br/><b>Unit/type:</b> "
                    + escape(r["unit_or_type"])
                    + ". <b>Caution/missing:</b> "
                    + escape(r["missing_value_and_interpretation"])
                    + f" <b>Blank rows:</b> {r['blank_rows']}.",
                ]
                for r in selected
            ],
            [143, 381],
        )
    heading("Appendix B. Responses to the RA Variable Guideline")
    paragraph(
        "All 231 entries from the 14 guideline tables are addressed below, including repeated names used in different contexts. 'Provided' means a value/record exists; it does not imply the underlying measurement is valid, causal or complete. Multi-model/multi-test fields have multiple records, not one universal value. Long values and repeated records are abbreviated explicitly here; linked CSV/JSON files retain full precision. "
        + link("report/RA_GUIDELINE_RESPONSES.csv", "Download the complete response index")
        + "."
    )
    titles = {
        1: "Column definitions",
        2: "Benchmark matrix",
        3: "Benchmark protocols",
        4: "IAT protocol",
        5: "Input sets",
        6: "Test results",
        7: "Preference dataset",
        8: "DPO run",
        9: "DPO training metrics",
        10: "Anti-collapse checks",
        11: "Post-DPO evaluation",
        12: "Pre/post comparisons",
        13: "Failed trials",
        14: "Plots and code",
    }
    for number, title in titles.items():
        heading(f"B{number}. {title}", 2)
        selected = [r for r in answers if int(r["table"]) == number]
        sourcepaths = sorted({p for r in selected for p in r["evidence_paths"].split("|") if p})
        paragraph(
            "Evidence: " + "; ".join(link(p, p.removeprefix("report/")) for p in sourcepaths) + "."
        )
        table(
            ["Requested variable and meaning", "Recorded answer / limitation"],
            [
                [
                    "<b>"
                    + escape(r["variable_name"])
                    + "</b><br/>"
                    + escape(r["guideline_definition"]),
                    escape(r["answer"])
                    + "<br/><i>Coverage: "
                    + escape(r["coverage_status"])
                    + "</i>",
                ]
                for r in selected
            ],
            [185, 339],
        )
    heading("Appendix C. Every Exploratory Significance Test")
    paragraph(
        "Values are rounded here to six significant digits; exact values, seeds, correction-family labels, sample units and test definitions are in "
        + link("results/analysis/all_significance_tests.csv")
        + ". Effects remain in CSV units: .01 in an IAT negative rate, DT agreement or BOLD toxicity rate equals 1 percentage point. VADER/regard and lexicon endpoints use their stated score units. 'n' is the resampling unit count, not always the number of responses. For local Gemma DT, every displayed statistic is a truncated-output diagnostic."
    )
    for benchmark in dict.fromkeys(r["benchmark"] for r in tests):
        heading(benchmark, 2)
        selected = [r for r in tests if r["benchmark"] == benchmark]
        selected.sort(key=lambda r: (ORDER.index(r["model_key"]), r.get("regime", ""), r["metric"]))
        table(
            ["Model / endpoint", "Effect", "95% interval", "p", "BH q", "n"],
            [
                [
                    escape(NAMES[r["model_key"]]) + "<br/>" + escape(endpoint_label(r)),
                    f"{float(r['effect']):.6g}",
                    f"[{float(r['ci_low']):.6g}, {float(r['ci_high']):.6g}]",
                    f"{float(r['raw_p_value']):.6g}",
                    f"{float(r['bh_adjusted_q_value']):.6g}",
                    str(r["sample_count"]),
                ]
                for r in selected
            ],
            [202, 55, 106, 59, 59, 43],
        )
    heading("Appendix D. Model Versions, Paths and Figure Reproduction")
    paragraph(
        "The focal local model is google/gemma-3-12b-it at Hugging Face commit 96b6f1eccf38110c56df3a15bffe176da04bfd80; base file hashes are recorded before/after inference. Local IAT uses the legacy image-text pipeline, not hosted chat formatting, BF16/eager, seed 7192026 reset per panel, temperature/top_p 1, sampled 30-token and greedy 20-token recovery caps. Other local tasks use the processor chat template. Hosted model contracts below are the recorded catalog canonical strings and endpoint tags, not cryptographically verified backend weights."
    )
    table(
        ["Local key / display name", "Requested API ID / canonical revision", "Provider route"],
        [
            [
                escape(m["key"]) + "<br/>" + escape(m["display_name"]),
                escape(m["model_id"]) + "<br/>" + escape(m["api_revision_hint"]),
                escape(m["provider_endpoint"]),
            ]
            for m in load("config/models_aug31.json")["models"]
        ],
        [154, 259, 111],
    )
    paragraph(
        "Source/model manifests: "
        + link("config/models_aug31.json")
        + "; "
        + link("results/gemma_local/manifest.json")
        + "; "
        + link("reference/august24/results/iat12/run_manifest.json")
        + ". Requested temperature and top_p are 1 in the archived tasks where supported; GPT omits unsupported settings. Budgets/reasoning differ by model. Stage-level fallbacks are not captured fully by a single 'effective' column. Raw responses omit image payloads and secrets."
    )
    paragraph(
        "Local working-package root: "
        + CANONICAL_ROOT
        + ". Public clone root: &lt;your-clone&gt;/2026-09-05. Below, every relative path is also shown as a full original local path with a direct GitHub link. Resolve paths relative to the clone root for replay; historical /workspace checkpoint paths refer to the private network volume, not to files hosted on GitHub."
    )
    paths = [
        "README.md",
        "scripts/reproduce.py",
        "scripts/analyze_evidence.py",
        "scripts/plot_evidence.py",
        "scripts/build_formal_report.py",
        "scripts/audit_sep5files.py",
        "scripts/verify_package.py",
        "locks/analysis-macos-py311.txt",
        "data/frozen/local_tasks.json",
        "results/analysis/iat_neutral12_responses.csv",
        "results/analysis/iat_utkface_responses.csv",
        "results/analysis/decodingtrust_responses.csv",
        "results/analysis/bold_scored_responses.csv",
        "results/analysis/visbias_neutral12_responses.csv",
        "results/analysis/visbias_original_responses.csv",
        "report/dpo_training_record.json",
        "reference/dpo/august10/august10_negative_expression_pilot_v4/dataset/mixed_dpo_pairs_frozen.csv",
        "reference/dpo/august10_code/scripts/train_mixed_dpo_lora.py",
        "reference/dpo/august10_code/docs/FROZEN_DATASET_SCHEMA.md",
        "report/benchmark_matrix.csv",
        "report/DATA_DICTIONARY.csv",
        "report/plot_registry.csv",
        "REPRODUCIBILITY.md",
        "FINE_TUNING_HISTORY.md",
        "SHA256SUMS",
    ]
    for path in paths:
        paragraph(
            link(path) + "<br/><font size='7.5'>" + escape(CANONICAL_ROOT + "/" + path) + "</font>"
        )
    heading("D1. One Current Figure Set", 2)
    paragraph(
        "All figures put Gemma first when it is present. Each includes an axis explanation and a result/uncertainty caption. Run python scripts/plot_evidence.py after analysis. The registry gives exact input files for each plot, including the retrospective DPO source JSON; no model calls are made."
    )
    for row in read_csv(REPORT / "plot_registry.csv"):
        paragraph(
            "<b>"
            + escape(row["title"])
            + "</b>: "
            + link("plots/" + row["plot_id"] + ".png", "PNG")
            + " / "
            + link("plots/" + row["plot_id"] + ".pdf", "PDF")
            + ". "
            + escape(row["finding"])
        )
    heading("D2. Requirement-by-Requirement Coverage", 2)
    table(
        ["Researcher request", "Where addressed"],
        [
            [
                "Guideline variables and work outside its tables",
                "Sections 1-10 and all 231 answers in Appendix B",
            ],
            [
                "Provided, missing and uninterpretable information",
                "Sections 3 and 8; Appendix A cautions; benchmark_matrix.csv",
            ],
            ["Failures and next steps", "Sections 8-9 and FINE_TUNING_HISTORY.md"],
            [
                "Full paths, code/data links and plot replay",
                "Section 10 and Appendix D; every link targets the named release",
            ],
            [
                "Verify AI statements against actual evidence",
                "Attachment/task/hash audit, source-derived values and local/public replay tests; no claim of human-validated labels",
            ],
            [
                "Explain each supplied spreadsheet column",
                "Appendix A: all 122 column occurrences, with units, blanks and interpretation",
            ],
            [
                "One best/current presentation version",
                "One current 15-figure set; the historical adapter is explicitly a directional failure, not selected as a confirmed success",
            ],
            [
                "Explain why each method was used",
                "Section 2 and the grounding/diversity acceptance objective",
            ],
            [
                "Benchmark x image-set matrix, FairFace DPO and post-DPO evaluation",
                "Sections 3, 6 and 7; text-only cells marked not applicable; missing post-DPO suites explicitly identified",
            ],
        ],
        [226, 298],
    )
    heading("References")
    refs = [
        (
            "[1] Karkkainen, K., and Joo, J. (2021). FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation. WACV, 1548-1558.",
            "https://openaccess.thecvf.com/content/WACV2021/html/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.html",
        ),
        (
            "[2] Zhang, Z., Song, Y., and Qi, H. (2017). Age Progression/Regression by Conditional Adversarial Autoencoder. CVPR. UTKFace dataset repository and terms.",
            "https://github.com/aicip/UTKFace",
        ),
        (
            "[3] Wang et al. (2023). DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models. NeurIPS Datasets and Benchmarks.",
            "https://proceedings.neurips.cc/paper_files/paper/2023/hash/63cb9921eecf51bfad27a99b2c53dd6d-Abstract-Datasets_and_Benchmarks.html",
        ),
        (
            "[4] Dhamala et al. (2021). BOLD: Dataset and Metrics for Measuring Biases in Open-Ended Language Generation. FAccT.",
            "https://arxiv.org/abs/2101.11718",
        ),
        (
            "[5] Huang et al. (2025). VisBias: Measuring Explicit and Implicit Social Biases in Vision Language Models. EMNLP, 17970-17993. The present subset and prompts are an adaptation, not a full-paper replication.",
            "https://aclanthology.org/2025.emnlp-main.908/",
        ),
        (
            "[6] Rafailov et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS.",
            "https://arxiv.org/abs/2305.18290",
        ),
        (
            "[7] Hu et al. (2021/2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.",
            "https://arxiv.org/abs/2106.09685",
        ),
    ]
    for text, url in refs:
        paragraph(
            escape(text)
            + f' <a href="{escape(url, quote=True)}" color="#205D89">Primary source</a>.'
        )


def render():
    from reportlab import rl_config
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        Preformatted,
    )
    from pypdf import PdfReader

    rl_config.invariant = 1
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Body",
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            spaceAfter=8,
            splitLongWords=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Cell",
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            spaceAfter=2,
            splitLongWords=1,
        )
    )
    styles.add(ParagraphStyle(name="Caption", fontSize=8.2, leading=11, spaceAfter=12))
    styles["Heading1"].fontSize, styles["Heading1"].leading = 15, 19
    styles["Heading2"].fontSize, styles["Heading2"].leading = 11, 15
    styles["Code"].fontSize, styles["Code"].leading = 7.2, 10
    story, markdown = [], []
    for node in NODES:
        if node[0] == "heading":
            title, level = node[1:]
            story.append(Paragraph(escape(title), styles["Heading" + str(level)]))
            markdown.append("#" * level + " " + title)
        elif node[0] == "paragraph":
            story.append(Paragraph(node[1], styles["Body"]))
            markdown.append(node[1])
        elif node[0] == "table":
            _, headers, body, widths = node
            cells = [[Paragraph("<b>" + escape(h) + "</b>", styles["Cell"]) for h in headers]]
            cells.extend([[Paragraph(str(value), styles["Cell"]) for value in row] for row in body])
            t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7EFF5")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#F5F7F8")],
                        ),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([t, Spacer(1, 10)])
            markdown.append(
                "\n".join(
                    [
                        "| " + " | ".join(headers) + " |",
                        "| " + " | ".join("---" for _ in headers) + " |",
                    ]
                    + [
                        "| "
                        + " | ".join(
                            str(c).replace("|", "&#124;").replace("\n", "<br/>") for c in row
                        )
                        + " |"
                        for row in body
                    ]
                )
            )
        elif node[0] == "figure":
            from PIL import Image as PILImage

            p = ROOT / node[1]
            with PILImage.open(p) as image:
                width, height = image.size
            story.extend(
                [
                    Image(str(p), width=524, height=524 * height / width),
                    Paragraph(escape(node[2]), styles["Caption"]),
                ]
            )
            markdown.extend([f"![{node[2]}]({quote('../' + node[1], safe='/')})", node[2]])
        elif node[0] == "code":
            story.extend([Preformatted(node[1], styles["Code"]), Spacer(1, 10)])
            markdown.append("```bash\n" + node[1] + "\n```")

    class Document(SimpleDocTemplate):
        def afterFlowable(self, flowable):
            if isinstance(flowable, Paragraph) and flowable.style.name in {"Heading1", "Heading2"}:
                label = flowable.getPlainText()
                bookmark = "heading_" + re.sub(r"[^a-zA-Z0-9]", "_", label)
                self.canv.bookmarkPage(bookmark)
                self.canv.addOutlineEntry(
                    label, bookmark, 0 if flowable.style.name == "Heading1" else 1
                )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(colors.HexColor("#536575"))
        canvas.drawString(44, 22, "URAP | September 5 evidence | sep5-evidence-v1")
        canvas.drawRightString(568, 22, str(doc.page))

    output = REPORT / "FORMAL_RESEARCH_REPORT.pdf"
    Document(
        str(output),
        pagesize=(612, 792),
        leftMargin=44,
        rightMargin=44,
        topMargin=35,
        bottomMargin=38,
        title="Multimodal Racial Bias Benchmarks and FairFace Preference Optimization",
        author="URAP research evidence",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    (REPORT / "FORMAL_RESEARCH_REPORT.md").write_text("\n\n".join(markdown) + "\n")
    reader = PdfReader(output)
    if not all(page.extract_text().strip() for page in reader.pages):
        raise ValueError("Unexpected empty PDF page")
    write_json(
        REPORT / "FORMAL_REPORT_CHECKS.json",
        dict(
            pages=len(reader.pages),
            attached_column_entries=122,
            requested_variable_entries=231,
            significance_tests=157,
            linked_paths=sorted(LINKS),
            missing_local_link_targets=[p for p in sorted(LINKS) if not (ROOT / p).exists()],
            publication_ref=PUBLICATION["release_ref"],
            google_sheet_url_provided=False,
        ),
    )
    print(
        f"Formal report: {len(reader.pages)} pages, 122 attached columns, 231 guideline entries, 157 tests"
    )


def main():
    audit = load("report/SEP5FILES_AUDIT.json")
    fields = attachment_dictionary(audit)
    if len(fields) != 122:
        raise ValueError("Attached column count changed")
    answers = guideline_answers()
    if len(answers) != 231:
        raise ValueError("Guideline entry count changed")
    document(audit, fields, answers)
    render()


if __name__ == "__main__":
    main()
