# Data Dictionary

## Frozen UTKFace files

### `data/frozen/utkface_evaluation_manifest.csv`

One row is one evaluation image. `image_id` is a hash-derived public identifier;
`filename` and `source_relative_path` reconstruct the private source file. `race`,
`gender`, and their codes are UTKFace metadata. `age_band`, `expression_class`, and
`quality_bin` are exact matching strata. `expression_label` and
`expression_confidence` come from the pinned FER model. `sha256` freezes image bytes;
`match_pair_id` links a Black/White matched pair. `pool_role=evaluation` distinguishes
this pool from all future training images.

### `data/frozen/utkface_finetuning_manifest.csv`

Columns match the evaluation manifest. `pool_role=future_finetuning`, and
`finetuning_split` is either `train` or `development`. No SHA-256 occurs in the
evaluation file.

### `data/frozen/utkface_future_dpo_tasks.csv`

One image produces two rows, identified by `candidate_task_id`: `iat_legacy61` and
`grounded_profile`. `prompt` is the exact proposed user task and `prompt_sha256` freezes
it. `race_metadata_in_prompt=false` confirms that the model is not told the sampling
label. `chosen` and `rejected` are blank; `candidate_generation_complete=false` and
`dpo_training_ready=false` mean the row must not enter DPO training.

## UTKFace IAT output

### `data/raw/utkface_responses.csv`

One deduplicated row is one model/image generation cell. Important field groups are:

- Contract: `task_id`, `benchmark_version`, `run_id`, `image_id`, `image_sha256`,
  `iteration`, and `seed`.
- Model route: `model_key`, `model_id_requested`, `api_model_returned`, `provider`, and
  requested/effective temperature, top-p, reasoning, and token limits.
- Output: `response`, `words`, `sampled_raw`, `fallback_raw`, `response_source`,
  `valid_three`, `exact_format`, and `word_count`.
- Endpoint counts: `positive_count` and `negative_count` count recognized fixed-lexicon
  terms only.
- Quality/cost: `status`, finish/error fields, retries, latency, tokens, and `cost_usd`.

`valid_three=true` means three words from the fixed 61-word list could be parsed. It does
not mean that the response was factually correct, unbiased, or exactly comma formatted.

### Aggregate files

- `data/derived/utkface_model_summary.csv`: Black/White negative rates, signed gap, image-cluster intervals,
  validity, response diversity, entropy, and cost by model.
- `data/derived/utkface_intersectional.csv`: the same count/rate family by race and recorded gender.
- `data/derived/utkface_word_counts.csv`: every model/race/word frequency, including zeros.
- `data/derived/utkface_significance.csv`: image-level effect, raw permutation p-value,
  permutation count/method, and BH-adjusted q-value.

## Frozen VisBias inputs

### `data/frozen/visbias_description_tasks.csv`

One row is a normalized image-description task. `matched_pair_id` links the Black/White
images with the same occupation, gender, and source index. Source and normalized hashes
and dimensions are separate. `prompt` and `prompt_sha256` freeze generation text.

### `data/frozen/visbias_comparison_tasks.csv`

One row is one ordered two-image comparison. `comparison_unit_id` links the two order
presentations for an image pair and attribute. `race_a`/`race_b` record order only after
generation. `attribute` is income or education. Image hashes and the exact prompt are
frozen.

## VisBias output

### `data/raw/visbias_responses.csv`

One row is one model/task cell. Description rows preserve full generated paragraphs.
Comparison rows add `parsed_answer` (`A`, `B`, `CANNOT_DETERMINE`, or `UNCLASSIFIED`) and
`selected_race`; `selected_race` is blank for cannot-determine/unclassified responses.
Model/provider, seed, validity, latency, usage, and error fields are retained for both
task kinds.

### Analysis files

- `data/derived/visbias_description_summary.csv`: mean endpoints by model/race.
- `data/derived/visbias_description_significance.csv`: paired effect, 95% interval, sign-flip p-value, BH q,
  and matched-pair count.
- `data/derived/visbias_comparison_summary.csv`: unsupported A/B, cannot-determine, unclassified, and
  directional selection rates.
- `data/derived/visbias_comparison_significance.csv`: order-balanced race-direction effect, interval, p/q,
  and unit count.

The full local analysis also produced scored-description and matched-unit intermediate
tables. They are omitted from this compact package because the raw responses and final
statistical tables are retained.

## Archived August 24 raw data

The minimal row-level IAT, DecodingTrust, and BOLD inputs used for the August 31
significance figures are under `data/raw/archived_*`.

- DecodingTrust `classification` is `AGREE`, `DISAGREE`, `NO_RESPONSE`, or `API_ERROR`;
  `demographic_group`, `topic`, and `regime` define the test cell.
- BOLD `toxic_any` is the thresholded Detoxify endpoint; `sentiment_class` and
  `regard_class` are -1/0/+1. The component Detoxify scores and raw text remain in
  `scored_responses.csv`.
- Every archived significance CSV uses `raw_p_value` for an uncorrected exploratory test
  and `bh_adjusted_q_value` for its Benjamini-Hochberg multiplicity-adjusted value. A q
  value is not an effect size or the probability that the null hypothesis is true.

## Cross-model summary

`data/derived/aug31_model_comparison.csv` has one row for every pinned August 31
model contract, including GPT's intentionally blank final cells. It joins route/revision
metadata with the UTKFace endpoint and the VisBias description/comparison effects,
confidence intervals, raw p-values, BH q-values, validity counts, and behavioral rates.
Blank values mean the pinned provider route did not produce a final benchmark cell; they
must not be interpreted as zero.
