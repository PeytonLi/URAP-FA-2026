# Gemma 3 DPO/LoRA Fine-Tuning History

Last updated: 2026-08-31. This is a chronological audit, not a claim that any
adapter is deployment-ready. The focal base model remained
`google/gemma-3-12b-it`; later controlled runs pinned Hugging Face revision
`96b6f1eccf38110c56df3a15bffe176da04bfd80`. Training changed only LoRA
adapter weights unless explicitly noted. No run intentionally updated the
base checkpoint.

## Research objective

Reduce unsupported, race-associated negative language about face images while
preserving image grounding, legitimate expression sensitivity, format
validity, identity invariance, and response diversity. Equality produced by
making the better-performing group worse is not accepted. Repeated generations
of the same 12 faces estimate sampling variation, not visual-population
generalization.

## Dataset and model provenance

- Primary image source: FairFace, via `HuggingFaceM4/FairFace`, configuration
  `0.25`. Citation: Karkkainen, K. and Joo, J. (2021), *FairFace: Face
  Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement
  and Mitigation*, IEEE WACV, DOI `10.1109/WACV48630.2021.00152`.
- Protected neutral benchmark: 12 fixed images, six labeled Black and six
  labeled White, balanced by recorded gender. It was repeatedly inspected and
  is now a safeguard/retrospective panel, not an untouched confirmatory set.
- Larger held-out benchmark: 400 FairFace validation images, 100 per
  Black/White by female/male intersection, with 10 generations per image.
- August 3 onward: matched FairFace train-split blocks included expression
  class/fine-label/intensity and image hashes; training/development images were
  checked against 419 frozen evaluation hashes.
- Multimodal teacher used in the August 3/10 candidate pool: Codex CLI
  `gpt-5.6-sol`, Ultra reasoning, Fast mode. This was an authenticated Codex
  execution, not the OpenAI API, so no OpenAI API snapshot/version beyond the
  recorded model name exists. Provenance label:
  `gpt-5.6-sol_codex_cli_multimodal_teacher`.
- Independent judges: direct Anthropic API model IDs `claude-opus-5` for IAT
  comparisons and `claude-sonnet-5` for profile comparisons. Earlier July 27
  work used Vertex `gemini-3.5-flash` as judge. Race metadata was withheld from
  judge prompts; image-present and text-only decisions were repeated with A/B
  order reversed.

## Timeline

### April 1-April 29: execution pilots

- Demonstrated that multimodal Gemma DPO and LoRA code could execute.
- The April 29 run sampled two images from each Black/White by female/male
  group (eight training images), generated eight candidates per image, retained
  only one unique DPO pair, repeated it four times, and ran 16 microsteps with
  gradient accumulation four (four optimizer updates).
- Pair selection emphasized novelty, identity penalties, and valence
  centering. It did not require direct negative-term or identity-leakage
  reduction, and positive-versus-positive pairs were not categorically
  excluded.
- The result is a smoke test, not evidence of debiasing. Preserved source:
  `../URAP/fairface_latest_code/lora_weights/fairface_dpo_default_20260429_104946/`.
  Retrospective audit:
  `../gemma3-fairface-debiasing-july13/results/april29_reanalysis_20260713/`.

### July 6: frozen base reference and pair-policy correction

- No LoRA training was part of the baseline. Base Gemma used 100 FairFace
  validation images per intersection group (400 images), 10 responses per
  image, 4,000 valid responses, and 12,000 selected words.
- Black-photo prompts produced 265/6,000 negative terms (4.42%); White-photo
  prompts produced 117/6,000 (1.95%); signed gap +2.47 points. The image-cluster
  95% interval was `[-0.43, +5.50]` points and exploratory permutation
  `p~0.101`.
- The revised pair objective began requiring direct negative/leakage
  improvement and rejecting unsupported positive-versus-positive pairs.
- Attention backend was not explicitly recorded. SDPA was likely selected by
  Transformers 4.50, but that cannot be asserted as observed fact.
- Preserved reference:
  `../gemma3-fairface-debiasing-july13/results/reference_base_gemma3_20260706/`.

### July 13: strict IAT-pair DPO and the overfit result

- Candidate score penalties: 4.0 per negative term and 8.0 per identity leak;
  novelty weight 0.5 and valence-centering penalty 0.1. Chosen could not be
  worse on either protected metric, every retained pair had to directly
  improve one, and positive-versus-positive was allowed only for direct
  identity-leakage reduction.
- Model-observed and controlled-template provenance were separated; templates
  were capped at 50% and selected round-robin across four intersection groups.
  Pair generation/audit/training/evaluation became separate fail-closed stages;
  training revalidated the exact CSV and stopped below 50 pairs.
- First explicit eager-attention revision. A generic LoRA target accidentally
  included SigLIP vision-tower projections. DPO retained chosen and rejected
  vision graphs, reaching about 78.8 GiB on an H100 80 GB before a 512 MiB OOM.
  This was not an eager-only failure. Excluding/freezing `vision_tower` reduced
  peak allocation/reservation to 42.59/43.96 GiB.
- Research pair audit: 400 FairFace train images, 16 candidates/image (up to
  6,400 generations), 29 observed plus 29 controlled pairs. Training used 58
  unique pairs repeated twice, rank-16 language-side LoRA, 2,000 microsteps,
  accumulation four, and 500 optimizer updates.
- Base-to-adapter signed gap moved +2.283 to +0.033 points and negative terms
  373 to 2, but entropy fell 38.0% and unique responses 71.6%. Loss was below
  0.001 by update 44 and effectively zero by update 59. Decision: overfit,
  reject.
- Evidence:
  `../gemma3-fairface-debiasing-july13/results/research_july13_100pg_20260713_205104/`
  and `../gemma3-fairface-debiasing-july13/docs/FAILED_ATTEMPTS.md`.

### July 19-20: expression-aware AI-only revision

- Rebuilt pairs to preserve visible expression rather than universally force
  positive words. The final run retained validity and diversity much better
  than July 13.
- Held-out Black negative rate moved 4.750% to 5.050%; White 1.950% to 2.833%;
  signed gap narrowed 2.800 to 2.217 points because White worsened more. The
  gap-change interval `[-1.750, +0.383]` included zero.
- On the neutral 12-face panel, Black rose 35.0% to 50.0%, White 3.33% to 5.0%,
  and the gap widened 31.67 to 45.0 points. Decision: primary efficacy failed.
- Evidence:
  `../gemma3-fairface-debiasing-july19/results/research_july19_ai_only_20260720_104900/`.

### July 27: Gemini-judged 50-pair fresh LoRA

- Added 40 FairFace neutral-pool images, generated 3,840 base-Gemma traces,
  and completed 396 Gemini judge calls over 99 candidates. The frozen 50-row
  CSV had 27 model-observed and 23 AI-teacher chosen responses, no overlap with
  protected evaluations, and SHA-256
  `638caf56c926b3007b6f17fef7b042ac5927be06bed8e31fba0e676b19a0cc69`.
- Fresh rank-16/alpha-32/dropout-0.05 LoRA; eager attention; beta 0.05; learning
  rate `1e-5`; accumulation four; 50 gradient examples and 13 optimizer
  updates; peak H100 reservation 44.67 GiB.
- Final held-out Black/White rates were 5.683%/2.433% versus base
  4.750%/1.950%; neutral Black/White were 50.0%/8.33% versus 35.0%/3.33%.
  The held-out Black-rate change interval was entirely above zero.
- Post-run audit found 29/50 fixed-count expression substitutions but only one
  direct negative-count-reduction pair, plus race-conditioned target
  vocabulary (`Pain`, `Sadness`, `Grief`). Decision: reject; do not continue.
- Evidence: `../gemma3-fairface-debiasing-july27/docs/JULY27_TRAINING_RESULTS.md`
  and `../gemma3-fairface-debiasing-july27/results/july19_vs_july27_comparison/`.

### August 3: GPT teacher + Claude judge micro-pilot

- Materialized 4,000 FairFace train images, constructed 80 verified expression
  matches, and generated 320 GPT-5.6 Sol profiles plus Gemma candidates. The
  combined candidate pool had 480 comparisons: 320 grounded-profile and 160
  controlled three-word IAT comparisons.
- Claude judging used image normal/reversed and text-only normal/reversed
  presentations. Opus 5 handled IAT; Sonnet 5 handled profiles.
- A time-limited six-row freeze trained four rows for one optimizer update.
  On the 12-face panel, Black rose 35.0% to 43.33%, White fell 3.33% to 1.67%,
  and the gap widened 31.67 to 41.67 points. Decision: reject; never warm-start
  or scale this adapter.
- Evidence:
  `../gemma3-fairface-debiasing-august3/research_artifacts/august3/micro6_directional_pilot/`.

### August 9-10: expanded Claude audit and expression-conditioned pilots

- Existing Claude judgments were reused by pair ID/model rather than repeated.
  August 10 auditing accumulated 955 calls and estimated `$14.639742`; the
  negative-expression expansion itself used 219 calls (57 Opus 5, 162 Sonnet
  5) and `$3.367927` estimated cost.
- Frozen v4: 100 unique FairFace images, 50/50 race balance, 80 train/20
  development rows, 30 IAT and 70 profile rows, seven negative-expression
  blocks. A later gate correctly found zero negative-expression IAT blocks in
  train because all three such blocks landed in development.
- Five fresh rank-8 LoRAs screened summed or mean-token DPO at learning rates
  `2e-6` through `2e-5`, beta 0.05, one pass, accumulation eight, ten updates,
  SDPA. Eight checkpoints left the development 26.67% Black, 33.33% White,
  30.0% overall rate unchanged; one introduced identity leakage. Decision:
  stop rather than continue a learning-rate sweep.
- Retrospective neutral-12 evaluation found one mean-token `2e-5` adapter that
  moved Black 35.0% to 16.67% and gap 31.67 to 16.67 points, but unique-response
  retention was 81.2%, four other adapters moved the wrong way, and the panel
  had already been observed. It is directional evidence, not confirmation.
- A separate non-confirmatory overfit diagnostic used 192 train rows plus 12
  diagnostic rows with all-positive chosen responses. Rank-8, mean-token DPO,
  beta 0.1, LR `5e-5`, SDPA reached zero negative terms for both races by
  update eight, proving LoRA+DPO can move the metric. It also erased grounded
  negative-expression sensitivity and later collapsed diversity, proving the
  objective was too blunt.
- Evidence:
  `../gemma3-fairface-debiasing-august10/results/august10_negative_expression_pilot_v4/`,
  `../gemma3-fairface-debiasing-august10/results/august10_first_run_neutral12_retrospective_v1/`,
  and `../gemma3-fairface-debiasing-august10/results/august10_dpo_lora_overfit_diagnostic_v2/`.

### August 31: evaluation expansion and future UTKFace task design

- No Gemma inference, DPO, LoRA training, or adapter update occurred in this revision.
- A 400-image UTKFace IAT-style evaluation pool was frozen separately from a
  400-image future fine-tuning pool. Both are balanced across Black/White by
  female/male intersections and matched on age band, automated visible-expression
  class, and quality bin. Exact image hashes do not overlap across the pools.
- The future pool produced 800 prompt-only candidate tasks: one controlled IAT task
  and one grounded-profile task per image. `chosen` and `rejected` remain blank and
  every row is labeled `dpo_training_ready=false`.
- This design prevents evaluation images from entering future preference training,
  but it is not yet a preference dataset. Candidate generation, independent judging,
  block-level race balancing, human/AI audit, and a new freeze are still required
  before any DPO run.
- Evidence: `data/frozen/utkface/utkface_freeze_manifest.json` and
  `docs/METHODS_AUG31.md` in this repository.

## Preference-pair contract at the latest revision

Each CSV row is one multimodal DPO example: image + exact prompt + `chosen` +
`rejected`. Audit columns do not enter the numerical loss; they decide whether
the row is allowed into training. Two race-matched image rows form a block.
IAT chosen/rejected tuples must be shared across the matched Black/White block,
with fewer negative terms in chosen. Profile rows are matched on task,
objective, gender, expression, bounded age/intensity, and grounding/quality.
Claude must be order-stable; profile image/text judgments follow the declared
task policy. Training rechecks hashes, split, block structure, balance,
forbidden overlap, and every chosen/rejected constraint.

## Consolidated conclusions

1. DPO+LoRA can strongly move Gemma's generated-word distribution; the August
   10 canary rules out a completely immovable endpoint.
2. A lower race gap is insufficient if Black negativity does not decrease or
   White negativity increases.
3. Excess updates on a small pair set cause loss saturation and diversity
   collapse. Early checkpoints and development-only selection are mandatory.
4. Expression-accuracy substitutions can preserve grounding but provide little
   direct pressure on total negative rate. Their objective type and exact
   chosen vocabulary must be balanced across race.
5. All-positive targets can erase legitimate sadness/negative-expression
   recognition. Future pairs must be expression-conditioned.
6. The 12-face and 400-image panels have been used repeatedly and cannot serve
   as untouched confirmatory tests. A disjoint, preregistered panel and
   multi-seed replication are needed for a defensible final claim.

## Preserved-artifact rule

The historical paths above are sibling repositories on the research workstation and
were not modified by this benchmark. Key August 10 protocols and result summaries are
copied under `reference/fine_tuning_history/august10/`. Large face images, model
checkpoints, adapters, and full private provider logs remain in their original
repositories or network volume.
