# Methods and Decisions

## Purpose

More responses per face reduce decoding noise while keeping images fixed. They
do not create more independent faces. The two panels remain separate: Neutral12
has 12 deliberately standardized faces; UTKFace has 400 cropped faces arranged
into 200 Black/White pairs matched using recorded age, gender, expression and
quality categories. Matching is approximate, not identical expression or
identity-counterfactual photography. Negative words can accurately describe a
negative expression, so low negativity alone is not evidence of better fairness.
The image-quality heuristic includes brightness/contrast and has not been
validated as race-invariant; matching on it can introduce its own selection effects.

Targets are 50 responses per Neutral12 image (600 per condition) and five per
UTKFace image (2,000 per condition). The legacy validity parser extracts the first
three permitted word matches from the fixed 61-word list in
`src/multimodel_bias/constants.py`. It skips unknown words, collapses consecutive
duplicates, and can retain a repeated word separated by another word. Therefore
validity does not require three unique words or exclude additional prose.
The parser and optional fallback are preserved from prior work. `valid_three`
is a parser-level measure, not proof that the answer is grounded or unbiased;
`exact_format` separately measures strict formatting. A refusal can be invalid
without being an explicit harmful stereotype. This is an IAT-style generative
association task, not a human reaction-time IAT or a direct likelihood score.

## Frozen Conditions and Archive Reuse

`config/study.json` pins model catalog revisions, endpoint routes, generation
parameters and limits. `results/preflight/` preserves the catalog and validated
routes. Fallback routing is disabled; parsing fallback calls use the same route.
Some providers do not honor temperature/top-p or seeds identically. Recorded
effective/requested fields and per-call metadata are authoritative.

The eight hosted models are Qwen 3.5 9B, Kimi K3, GPT-5.6 Luna, Gemini 3.6 Flash,
MiMo-V2.5, MiniMax M3, Step 3.7 Flash, and Grok 4.6. This is five open-weight and
three closed-source models, not eight open-weight plus two closed-source models.
The Qwen2.5-VL-3B compatibility probe is not a fully benchmarked ninth hosted model.

We reuse 840 Neutral12 and 3,200 UTKFace archived response slots after checking
image hashes, source contracts, seed-derived ordering and parsing. Invalid
archived answers remain. Hosted expansion retains the old formula
`base_seed + iteration*1000 + image_index`, with image order reconstructed from
the original saved seeds. A sampled-answer fallback is part of the generation
contract and does not count as an independent response slot.

MiMo's former Neutral12 DeepInfra BF16 route was absent. Its 120 old BF16
responses are preserved as references, **excluded** from the new pooled
condition; 600 fresh FP8 responses were scheduled. This is a new serving
condition, not evidence of a model update or a controlled improvement.

MiniMax's Neutral12 DeepInfra FP8 route returned eleven HTTP 429 errors across
three probes and was excluded for the rest of this session. The 120-slot archive
is retained and labeled incomplete (eight slots are historical API errors, not
valid answers). Its UTKFace Venice FP8 route initially passed, temporarily became
unavailable (catalog status -2), then passed a recovery probe on that same route.
Its expansion resumed with a documented concurrency override. Kimi Neutral12's
DeepInfra BF16 endpoint also became unavailable and was paused. Availability
history and final counts distinguish recovery from still-incomplete cells;
no alternate provider is silently pooled.

A MiMo transport timeout stopped the first expansion invocation. Its charge was
not known from a returned response, so a conservative extra reservation is held
against the session cap and the account usage delta is reconciled. Resumption
keeps completed slots and logs technical retries; three consecutive HTTP or
transport failures stop the affected route. Revision/provider drift stops all
calls. This is not resampling until an answer becomes favorable.

## Local Gemma Base and Adapter

Base: `google/gemma-3-12b-it`, revision
`96b6f1eccf38110c56df3a15bffe176da04bfd80`. Adapter:
`training_v4e_meanlogp_lr2e5_rank8_seed8102026/checkpoints/final`, SHA-256
`aa869029c73588c2cb5ff1bf805466f78e995de3e888ac7ff7eb567f8d421053`.

The local runner checks every base snapshot hash against the preserved September
5 manifest, loads BF16/eager inference under Transformers 4.50.0, and freezes all
parameters. It uses the original bare image-text pipeline prompt, sample budget
30 tokens, temperature/top-p 1.0, and a greedy 20-token parsing fallback. The
model generation configuration records inherited defaults, including top-k.
The adapter is enabled/disabled, never merged into or saved over base weights.
A pure-base response must exactly match the disabled-adapter response for the
same probe seed, and the generation configuration must remain unchanged.

**New local seed contract:** SHA-256 of
`9112026|panel|image_id|iteration`, first four bytes big-endian modulo `2**31`.
Python, NumPy, Torch and CUDA RNGs reset before each sampled response. Base and
tuned conditions use the same task seed. This supports paired inference and
interruption-safe resumption. It differs from the old dataset-wide RNG stream;
historical base bars are not guaranteed to replay. The new matched base is the
proper reference for the new adapter comparison, not a selectively chosen old
baseline. Same seed does not guarantee identical token paths after adaptation.

The historical adapter used 100 FairFace preference rows (80 train/20 development),
70 grounded-profile and 30 controlled three-word pairs in the full freeze. GPT
teacher profiles came from Codex CLI `gpt-5.6-sol`, not an exact OpenAI API
snapshot. Recorded Claude judge IDs were `claude-sonnet-5` for profiles and
`claude-opus-5` for IAT pairs. See `reference/dpo_training_record.json` for the
audited provenance and limitations. Modified mean-response-token DPO used beta
0.05, learning rate 2e-5, rank 8, alpha 16, dropout .05, effective batch 8 and
10 updates. This September task performs no new training and no judge calls.

The adapter was selected retrospectively using the 12-face results, which are
therefore not an untouched confirmatory set. UTKFace evaluation images have no
known exact-file overlap with the FairFace training set; this does not prove
person-level disjointness. Training and evaluation share the IAT task/lexicon.

## Statistical Analysis

An image score is its mean negative-word fraction across valid repetitions.
Main race bars average image scores, giving each face equal weight. UTKFace
uses complete matched Black/White pairs; a pair is omitted from a given model's
gap test if one image has no valid output. Model invalidity and incomplete
sampling remain visible. CSV rates are fractions; graph rates are percentages
and race gaps are percentage points. Word-count plots count selected words,
including all 61 possible words even when their count is zero.

Neutral12: two-sided exact race-label permutation within recorded gender,
with race/gender-stratified image-cluster percentile bootstrap gap intervals.
Individual race-bar intervals resample images within that race.
UTKFace: two-sided sign-flip of paired image differences, with paired bootstrap
intervals. We use 100,000 resamples and seed 9112026; exact sign-flip enumeration
is used for at most 16 units. The permutation tests assume exchangeability
under the null and are exploratory, not causal estimates.

Benjamini-Hochberg correction is applied within each dataset across the available
model-condition race-gap tests. The separate direct Gemma tuning comparison
uses common valid image/iteration/seed tasks, averages tuned-minus-base changes
within images, then tests the change in the signed Black-minus-White gap. Its
two dataset tests form a separate BH family. Significance for base but not tuned
is not itself a significant tuning effect; use the direct paired test.
The individual 61-word bar plots are descriptive; no per-word significance
claim or unreported set of 61 separate hypothesis tests is implied.

A negative tuning delta means the signed Black-minus-White gap decreased, not
automatically that its absolute value improved. Read base/tuned group rates and
diversity alongside it. Entropy, distinct vocabulary and unique response tuples
check for response collapse, not general accuracy. Unique counts depend on
sample size, so incomplete model conditions cannot be compared naively.
No extra sampling is triggered by whether a desired bias result appears.

## Local Execution and Cost Control

The existing network volume is `zecocmq95n`, US-NE-1. Only the newly created own
pod is managed. Container-local Qwen weights are downloaded to `/tmp/qwen-cache`
and need not be preserved; research outputs go to the network volume. No older
pod, base model, dataset, or adapter is deleted.

```bash
.venv/bin/python -B scripts/pod.py deploy
.venv/bin/python -B scripts/pod.py connect
.venv/bin/python -B scripts/pod.py sync
# Start scripts/pod.py watch in another local terminal immediately.
```

On the pod use the compatible existing Python environment, or provision one
from the recorded Linux package lock without upgrading historical environments:

```bash
/workspace/gemma3-fairface-debiasing-august10/.venv/bin/python -B \
  /workspace/research-evaluation-september11/scripts/local_evaluate.py --dry-run
/workspace/gemma3-fairface-debiasing-august10/.venv/bin/python -B \
  /workspace/research-evaluation-september11/scripts/local_job.py launch \
  --deadline-epoch <hard_deadline_epoch_from_runtime_pod_json_minus_480>
```

The watchdog downloads outputs, compares remote and local SHA-256 checksums,
then terminates compute while verifying that the network volume survives.
Keep the coordinating Mac awake and its watchdog running until cleanup is
verified; the local watchdog is not a cloud billing-limit service.
An independent wall-clock deadline enforces the $10 GPU-session ceiling even
when the worker fails. OpenRouter's cap is $20 additional-session spend and its
minimum remaining balance is $1. Per-request cost reservations and reconciliation
with account usage prevent uncontrolled expansion. Provider costs can lag in
account balances; both the response ledger and account snapshots are reported.

## Limitations and Unfinished Work

More generations do not solve the small 12-image cluster count, uncertain
expression matching, model/provider differences, prompt dependence, or biased
checkpoint selection. Neither equal bars nor nonsignificant tests prove that a
model is unbiased. No new DecodingTrust, BOLD, VisBias, human review, DPO training,
general capability evaluation, or Qwen debiasing is claimed in this iteration.

## Dataset References

- Karkkainen, K., and Joo, J. (2021). *FairFace: Face Attribute Dataset for
  Balanced Race, Gender, and Age for Bias Measurement and Mitigation.* WACV,
  1548-1558. [Official dataset and citation](https://github.com/joojs/fairface).
  This is the source of the historical preference-training images, not the
  source of the new UTKFace evaluation images.
- Zhang, Z., Song, Y., and Qi, H. (2017). *Age Progression/Regression by
  Conditional Adversarial Autoencoder.* CVPR. The
  [official UTKFace page](https://susanqq.github.io/UTKFace/) provides the related
  paper citation, filename label coding, and dataset-use terms. The selected
  400-image panel and its exact hashes are frozen in this package.
- The standard 12 image filenames/hashes are preserved in `data/iat12/manifest.csv`.
  The historical archive attributes them to Project Implicit, but lacks exact
  original download URLs. Obtain the authorized original bundle; do not silently
  substitute visually similar faces.
