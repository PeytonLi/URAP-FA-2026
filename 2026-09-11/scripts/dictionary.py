"""Column appendix for the presented CSVs; fail on an undocumented new column."""
from common import ROOT, rows, write_csv

DEFINITIONS = {
    "panel":"Image panel: neutral12 or utkface; analyses are separate.",
    "model_key":"Stable condition identifier; gemma_base and gemma_tuned use the same pinned local base.",
    "model_name":"Human-readable condition label.", "model_display_name":"Human-readable model label saved by the generation runner.",
    "model_id_requested":"Model identifier sent to the API, or local base model ID.",
    "api_model_returned":"Model identifier returned by OpenRouter; checked against the pinned catalog ID/revision.",
    "model_revision":"Pinned Hugging Face commit for local Gemma or the Qwen probe.",
    "adapter_sha256":"SHA-256 of the adapter weights; blank for the base condition.",
    "provider":"Actual serving provider reported by OpenRouter, or recorded local runtime.",
    "run_id":"Saved inference run identifier.", "source_run_id":"Earlier source run identifier retained during consolidation.",
    "benchmark_version":"Protocol identifier for the IAT runner.", "task_id":"Deterministic hosted task identifier.",
    "created_at":"UTC response-record timestamp.",
    "image_id":"Stable manifest image identifier; join image metadata by this key.",
    "image_filename":"Original image filename from the stimulus manifest.", "filename":"Source image filename.",
    "image_sha256":"SHA-256 of actual image bytes used for the response.", "sha256":"SHA-256 of the manifest image file.",
    "source_relative_path":"Private image path relative to its manifest directory; not an embedded image.",
    "race":"Recorded dataset race group, Black or White; not an inference by this agent.",
    "gender":"Recorded dataset gender category, Female or Male; labels may be imperfect.",
    "iteration":"One-based repetition index for a given image.", "seed":"Requested generation RNG seed; local seeds reset Python/NumPy/Torch/CUDA per image and repetition.",
    "temperature_requested":"Sampling temperature requested by the protocol.",
    "temperature_effective":"Temperature actually sent when supported; omission/blank means unsupported or unrecorded, not zero.",
    "top_p_requested":"Requested nucleus sampling probability threshold.",
    "top_p_effective":"Nucleus threshold actually sent when supported; blank does not mean zero.",
    "reasoning_effort":"Provider-specific requested reasoning setting; not comparable as a fixed token count.",
    "max_tokens":"Configured hosted sampled-output budget; some providers account for reasoning separately.",
    "sampled_raw":"Unmodified visible text of the sampled answer, excluding image payloads.",
    "fallback_raw":"Visible text from the deterministic parsing fallback, if used.",
    "response":"Comma-separated parsed words from the final selected answer; inspect raw text for extra prose.",
    "words":"Pipe-separated legacy parsed word matches in order; non-consecutive duplicate words can remain.",
    "word_count":"Number of permitted word matches retained by the legacy parser (at most three).",
    "valid_three":"True when the legacy parser retained three matches; not proof of exact format, uniqueness, accuracy, or absence of refusal prose.",
    "exact_format":"True when the visible final answer equals the parsed comma-separated sequence, ignoring case/spacing around commas.",
    "negative_count":"Number of parsed words in the fixed negative lexicon; primary rates use valid answers only.",
    "positive_count":"Number of parsed words in the fixed positive lexicon.",
    "response_source":"sampled/fallback for model outputs, api_error for failed requests, budget_stopped for halted tasks; historical variants are retained.",
    "status":"Recorded completion/invalid/runtime status from the originating runner; use valid_three for common parsing analysis.",
    "error_status":"HTTP status or originating error code; blank means no recorded error.",
    "error_type":"Recorded technical exception category, when available.",
    "error_message":"Sanitized technical error explanation; no API credential or image payload.",
    "finish_reason":"Provider stop/length/error termination label; backend meanings can differ.",
    "final_call_retry_count":"Client retry count reported for the last generation call, not all historical attempts.",
    "latency_seconds":"Elapsed generation time in seconds, as recorded by the runner.",
    "prompt_tokens":"Reported input tokens across this response slot's calls; image representation depends on provider.",
    "completion_tokens":"Reported generated token count, including provider-specific reasoning accounting.",
    "reasoning_tokens":"Provider-reported reasoning token count when available.",
    "image_tokens":"Provider-reported image-token count when available; zero/blank need not mean no image was sent.",
    "cost_usd":"Original reported per-slot inference cost in USD; local GPU inference does not have a per-response API price.",
    "new_cost_usd":"API cost attributed to the new session for this retained slot; zero for reused archives. Use the call ledger for retries and full billed costs.",
    "record_origin":"Earlier consolidation provenance, preserved from the archive.",
    "provenance":"archived, new, or new_local generation source for this study.",
    "generation_calls":"Number of local generate calls for the answer, including a parsing fallback.",
    "planned_responses":"Target response slots: 600 for neutral12 and 2000 for utkface per full condition.",
    "attempted_slots":"Unique retained image/repetition slots including technical failures, not the number of API requests.",
    "responses":"Slots with a model answer rather than api_error/budget_stopped; invalid model answers can be included.",
    "valid_responses":"Number of slots with three parsed permitted word matches.",
    "validity_rate":"Valid slots divided by attempted_slots, as a fraction from 0 to 1.",
    "exact_format_rate":"Exact-format slots divided by attempted_slots, as a fraction from 0 to 1.",
    "repeated_word_response_rate":"Fraction of valid responses containing fewer than three globally unique words.",
    "distinct_images":"Distinct images with at least one valid response in this summary/test.",
    "complete":"Whether answered slots reached the target; a false value must not be presented as full sampling.",
    "archived_responses":"Retained historical slots (including archived failures/invalidity), not new API calls.",
    "unique_response_tuples":"Distinct ordered parsed three-word sequences among valid answers; sample-size dependent.",
    "distinct_iat_words":"Number of the 61 permitted words occurring in valid responses.",
    "word_entropy_bits":"Shannon entropy of pooled selected-word frequencies, using log base 2.",
    "normalized_word_entropy":"Word entropy divided by log2(61); 0 is maximal concentration and 1 is uniform use of all permitted words.",
    "top_three_word_share":"Share of all selected word occurrences occupied by the three most frequent words, from 0 to 1.",
    "api_error_slots":"Final retained slots ending in an API error; full call attempts including retries are in calls.jsonl.",
    "new_inference_cost_usd":"Sum of new_cost_usd for retained slots only; excludes some retried/failed attempts, so not the session bill.",
    "black_rate":"Equal-image mean negative-word fraction for Black images; UTK uses complete matched pairs.",
    "white_rate":"Equal-image mean negative-word fraction for White images; UTK uses complete matched pairs.",
    "black_fallback_rate":"Fraction of Black-image slots whose final source is a parsing fallback, including fallback_invalid.",
    "white_fallback_rate":"Fraction of White-image slots whose final source is a parsing fallback, including fallback_invalid.",
    "black_ci_low":"Lower 95% image-bootstrap bound for black_rate, expressed as a fraction.",
    "black_ci_high":"Upper 95% image-bootstrap bound for black_rate, expressed as a fraction.",
    "white_ci_low":"Lower 95% image-bootstrap bound for white_rate, expressed as a fraction.",
    "white_ci_high":"Upper 95% image-bootstrap bound for white_rate, expressed as a fraction.",
    "metric":"Name of the measured outcome; distinguishes a race gap from a tuned-minus-base change.",
    "effect":"Signed point estimate in fractional rate units. Multiply by 100 for percentage points.",
    "ci_low":"Lower 95% percentile-bootstrap bound in the same units as effect.",
    "ci_high":"Upper 95% percentile-bootstrap bound in the same units as effect.",
    "raw_p_value":"Exploratory two-sided permutation/sign-flip p value under the stated exchangeability null; not a probability the null is true.",
    "bh_adjusted_q_value":"Benjamini-Hochberg adjusted p value within the declared family; q<0.05 is the exploratory false-discovery threshold.",
    "correction_family":"Which collection of tests is included in the multiple-comparison adjustment.",
    "family_tests":"Number of finite p values adjusted together.",
    "exploratory":"True: these tests are not preregistered confirmatory evidence.",
    "sample_count":"Number of statistical units: images for Neutral12 or complete matched image pairs for UTKFace.",
    "sample_unit":"Definition of the independent unit used by the stated test.",
    "test":"Statistical procedure used to compute raw_p_value.",
    "negative_rate":"Per-image mean negative-word fraction; in gemma_paired_image_changes.csv this is tuned minus base, as specified by metric.",
    "valid_repetitions":"Valid answers averaged into this image score; does not increase independent-image count.",
    "paired_valid_repetitions":"Image/repetition tasks valid in both Gemma conditions and sharing the same image hash and seed.",
    "word":"One canonical word from the fixed IAT61 list.", "count":"Number of selections of this word in valid answers for the model/panel/race.",
    "negative":"Whether word belongs to the fixed negative list (otherwise it belongs to the positive list).",
    "new_api_call_attempts":"New model HTTP calls, including recorded errors and parsing fallback calls.",
    "reported_cost_usd":"Sum of provider-reported call costs in USD; compare account reconciliation for lag or unknown outcomes.",
    "width":"Width of the stored cropped image in pixels.", "height":"Height of the stored cropped image in pixels.",
    "age":"UTKFace recorded age in years; not verified by this study.",
    "age_band":"Age-matching bin assigned by the preserved dataset preparation code.",
    "race_code":"UTKFace source code: 0 White, 1 Black in this selected panel; other source groups were not selected here.",
    "gender_code":"UTKFace source coding: 0 Male, 1 Female; not a complete model of gender identity.",
    "expression_class":"Coarse expression-matching class such as negative/neutral/positive from the automated classifier.",
    "expression_label":"Classifier's fine expression label, such as angry or sad; labels are fallible.",
    "expression_confidence":"Model probability assigned to the predicted expression, not human-verified confidence.",
    "expression_model_id":"Hugging Face identifier of the expression classifier used in dataset construction.",
    "expression_model_revision":"Pinned expression-classifier source revision.",
    "quality_score":"Heuristic 0.30*exposure + 0.30*contrast + 0.40*sharpness score on grayscale pixels; components are normalized/clipped in utkface.py. Not a validated fairness or perceptual-quality score.",
    "quality_bin":"High for quality_score >=0.65, medium for >=0.40, otherwise low; used for approximate matching.",
    "match_pair_id":"Identifier connecting one Black and one White UTKFace image in the matched evaluation pool.",
    "pool_role":"Evaluation versus future fine-tuning pool designation; this manifest contains evaluation images.",
    "finetuning_split":"Training/development assignment when relevant; not_applicable for this evaluation pool.",
}


def main():
    files=list((ROOT / "results/analysis").glob("*.csv"))
    files += list((ROOT / "results/local").glob("*.csv"))
    files += [ROOT / "data/iat12/manifest.csv",ROOT / "data/frozen/utkface/utkface_iat_evaluation_manifest.csv"]
    appendix=[]
    for path in sorted(files):
        records=rows(path)
        if not records:
            continue
        for column in records[0]:
            if column not in DEFINITIONS:
                raise ValueError(f"Undocumented column {column} in {path.name}")
            appendix.append(dict(file=str(path.relative_to(ROOT)),column=column,definition=DEFINITIONS[column]))
    write_csv(ROOT / "results/data_dictionary.csv",appendix)
    text=["# Data Dictionary", "", "This appendix defines every column in the analysis CSVs, local Gemma/Qwen CSVs and the two image manifests. "
          "Blank means unavailable/not applicable, not a measured zero. Raw API ledger JSON and untouched historical reference files preserve their original schemas.", "",
          "## File Roles", "", "- `*_responses.csv`: one retained model/image/repetition slot, including invalidity and final technical failures.",
          "- `summary.csv`: one model-condition/panel summary; no claim that entropy is general accuracy.",
          "- `significance.csv`: one exploratory race-gap test per model-condition/panel.",
          "- `gemma_paired_effect.csv`: direct paired tuning-effect tests, not subtraction of separately obtained p values.",
          "- `*_image_scores.csv`: per-image averages; repetitions stay clustered within image.",
          "- `word_counts.csv`: all 61 words for each model/panel/race, including zero counts.",
          "- `model_call_costs.csv`: new session requests and reported cost; not archived costs.",
          "- `results/hosted/calls.jsonl`: append-only attempt ledger with `panel`, `task_id`, `stage`, request time, model/route, image/prompt hashes, seed, generation parameters, sanitized provider response, usage and cost. "
          "`stage` distinguishes sample from fallback; nested `choices[].message.content` is the visible answer. Image payloads and API keys are omitted.",
          "- `results/local/manifest.json`: base/adapter hashes, generation defaults, environment, status and completed responses.",
          "- `reference/`: historical source contracts and data, not new model results. See the prior September 5 appendix for archived-only fields.", "",
          "## Columns", "", "| Column | Meaning |", "|---|---|"]
    used=sorted({r["column"] for r in appendix})
    text += [f"| `{k}` | {DEFINITIONS[k]} |" for k in used]
    text += ["", "`results/data_dictionary.csv` also maps each column to every presented CSV where it occurs. "
             "For UTKFace quality/expression construction, inspect `src/multimodel_bias/utkface.py` and the frozen source manifest. "
             "All raw-response CSV files are UTF-8 and can be imported directly into a spreadsheet; no public Google Sheet was created.", ""]
    (ROOT / "DATA_DICTIONARY.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
