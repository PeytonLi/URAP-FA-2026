# Data Dictionary

## Frozen UTKFace files

### `utkface_iat_evaluation_manifest.csv`

One row is one evaluation image. `image_id` is a hash-derived public identifier;
`filename` and `source_relative_path` reconstruct the private source file. `race`,
`gender`, and their codes are UTKFace metadata. `age_band`, `expression_class`, and
`quality_bin` are exact matching strata. `expression_label` and
`expression_confidence` come from the pinned FER model. `sha256` freezes image bytes;
`match_pair_id` links a Black/White matched pair. `pool_role=evaluation` distinguishes
this pool from all future training images.

### `utkface_finetuning_image_manifest.csv`

Columns match the evaluation manifest. `pool_role=future_finetuning`, and
`finetuning_split` is either `train` or `development`. No SHA-256 occurs in the
evaluation file.

### `utkface_future_dpo_candidate_tasks.csv`

One image produces two rows, identified by `candidate_task_id`: `iat_legacy61` and
`grounded_profile`. `prompt` is the exact proposed user task and `prompt_sha256` freezes
it. `race_metadata_in_prompt=false` confirms that the model is not told the sampling
label. `chosen` and `rejected` are blank; `candidate_generation_complete=false` and
`dpo_training_ready=false` mean the row must not enter DPO training.

## UTKFace IAT output

### `responses.csv`

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

- `model_summary.csv`: Black/White negative rates, signed gap, image-cluster intervals,
  validity, response diversity, entropy, and cost by model.
- `intersectional_metrics.csv`: the same count/rate family by race and recorded gender.
- `word_counts.csv`: every model/race/word frequency, including zeros.
- `exploratory_significance.csv`: image-level effect, raw permutation p-value,
  permutation count/method, and BH-adjusted q-value.
- `metrics.json`: nested machine-readable version of the summaries and analysis contract.

## Frozen VisBias inputs

### `selected_description_images.csv`

One row is a normalized image-description task. `matched_pair_id` links the Black/White
images with the same occupation, gender, and source index. Source and normalized hashes
and dimensions are separate. `prompt` and `prompt_sha256` freeze generation text.

### `selected_comparison_tasks.csv`

One row is one ordered two-image comparison. `comparison_unit_id` links the two order
presentations for an image pair and attribute. `race_a`/`race_b` record order only after
generation. `attribute` is income or education. Image hashes and the exact prompt are
frozen.

## VisBias output

### `responses.csv`

One row is one model/task cell. Description rows preserve full generated paragraphs.
Comparison rows add `parsed_answer` (`A`, `B`, `CANNOT_DETERMINE`, or `UNCLASSIFIED`) and
`selected_race`; `selected_race` is blank for cannot-determine/unclassified responses.
Model/provider, seed, validity, latency, usage, and error fields are retained for both
task kinds.

### Analysis files

- `scored_descriptions.csv`: response text plus word count, VADER compound sentiment,
  negative-word count/rate, race-stereotype lexicon hits, and person-linked identity
  mentions. Bare `black`/`white` color words are excluded from the identity metric.
- `description_race_summary.csv`: mean endpoints by model/race.
- `description_paired_differences.csv`: Black-minus-White values for each matched image
  pair and metric.
- `description_significance.csv`: paired effect, 95% interval, sign-flip p-value, BH q,
  and matched-pair count.
- `comparison_summary.csv`: unsupported A/B, cannot-determine, unclassified, and
  directional selection rates.
- `comparison_order_balanced_units.csv`: both image orders collapsed into one unit.
- `comparison_significance.csv`: order-balanced race-direction effect, interval, p/q,
  and unit count.

## Archived August 24 raw data

The exact row-level IAT, DecodingTrust, and BOLD files used for the archived significance
figures are under `reference/august24/results/{iat12,decodingtrust,bold}/`.

- DecodingTrust `classification` is `AGREE`, `DISAGREE`, `NO_RESPONSE`, or `API_ERROR`;
  `demographic_group`, `topic`, and `regime` define the test cell.
- BOLD `toxic_any` is the thresholded Detoxify endpoint; `sentiment_class` and
  `regard_class` are -1/0/+1. The component Detoxify scores and raw text remain in
  `scored_responses.csv`.
- Every archived significance CSV uses `raw_p_value` for an uncorrected exploratory test
  and `bh_adjusted_q_value` for its Benjamini-Hochberg multiplicity-adjusted value. A q
  value is not an effect size or the probability that the null hypothesis is true.

## Cross-model summary

`results/summary/aug31_model_comparison.csv` has one row for every pinned August 31
model contract, including GPT's intentionally blank final cells. It joins route/revision
metadata with the UTKFace endpoint and the VisBias description/comparison effects,
confidence intervals, raw p-values, BH q-values, validity counts, and behavioral rates.
Blank values mean the pinned provider route did not produce a final benchmark cell; they
must not be interpreted as zero.
