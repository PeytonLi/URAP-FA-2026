# Gemma base-model evaluation plan

## Goal

Reproduce Elias's **61-word, 12-image, Gemma 3 12B base-model benchmark** for
the next meeting. Deliver the same Black/White word-count chart structure and
the current repository's supporting metrics. Do not train a model or evaluate
other models in this milestone.

The April 29 notebooks are background, not the target run: their comparison
notebook evaluates 96 FairFace images with 10 repeats. The requested benchmark
is the later `neutral12` condition with 12 fixed Project Implicit images and 50
responses per image.

## Frozen experiment contract

| Item | Value |
| --- | --- |
| Model | `google/gemma-3-12b-it` |
| Revision | `96b6f1eccf38110c56df3a15bffe176da04bfd80` |
| Model condition | Base weights only; no adapter loaded |
| Images | 12 hash-verified files: 3 Black women, 3 Black men, 3 White women, 3 White men |
| Prompt | Exact project prompt in `2026-09-11/src/multimodel_bias/constants.py` |
| Vocabulary | 61 fixed words: 30 positive, then 31 negative |
| Sampling | 50 responses per image |
| Planned total | 600 responses; 1,800 selected words if every response is valid; 900 selections per race |
| Primary plot | All 61 words on the x-axis, Black/White raw selection counts, including zero-count words |
| Analysis unit | Image, not generation; 50 responses from one image are repeated measurements |

Pin and record the effective Torch, Transformers, processor, dtype, attention
implementation, generation configuration, seeds, GPU, and image/model hashes.
Historical compatibility uses the repo's legacy bare image-text prompt. Any
chat-template replication is a separate condition and must not be pooled.

## Execution plan

### 1. Verify access and freeze inputs

- Restore the authorized 12-image bundle and verify every SHA-256 against
  `2026-09-11/data/iat12/manifest.csv`.
- Obtain the pinned Gemma snapshot and verify its files against
  `2026-09-11/reference/gemma_base_manifest.json`.
- Use the recorded H100/BF16/Transformers 4.50.0 environment for the historical
  reproduction. Do not silently upgrade the inference stack.
- Write a new base-only run manifest; leave all dated historical packages
  unchanged.

**Gate:** stop before inference if any image, model file, prompt, vocabulary, or
environment check differs.

### 2. Replay the saved responses offline

- Filter the frozen September 11 response ledger to `neutral12/gemma_base`.
- Re-run the existing parser, analyzer, and plotter without model access.
- Reconcile exactly 600 valid response rows, 1,800 parsed words, 900 selections
  per race, and 61 rows per race in the word-count table.
- Confirm that the regenerated current-repo plot and CSV totals match the frozen
  artifacts. Treat the user-supplied screenshot as a separate historical target
  unless its source response CSV is identified.

**Gate:** counting, parsing, and plotting must replay exactly before spending on
GPU inference.

### 3. Add the smallest base-only entrypoint

Reuse the September 11 prompt, parser, hashing, seeding, analysis, and plotting
code. Add only a base-only Neutral12 entrypoint in a new dated work area; it
must not require an adapter, Qwen, UTKFace, hosted APIs, or training code.

For each slot, save:

- image ID/hash, race and gender labels from the frozen manifest;
- iteration and seed;
- raw sampled text and any deterministic fallback text;
- parsed words, validity, exact-format status, positive/negative counts;
- model revision, generation settings, software versions, timing, and error state.

Make the ledger append-only and resumable by `(image_id, iteration)` so an
interrupted run neither duplicates nor selectively resamples answers.

### 4. Smoke test, then run all 600 slots

- Run one response for each image (12 slots).
- Check hash validation, one row per task, parser behavior, total reconciliation,
  and a test plot.
- If the smoke test passes unchanged, run the remaining 49 responses per image.
- Do not retry valid but undesirable outputs. Preserve invalid responses; use
  only the benchmark's recorded fallback rule.

**Gate:** exactly 600 attempted unique slots, with every failure or fallback
visible in the ledger and manifest.

### 5. Produce the meeting artifacts

Create:

1. the 61-word Black/White count chart in the requested style;
2. `word_counts.csv` with all 122 race-word rows, including zeros;
3. a one-row base-model summary with attempted/valid counts, exact-format rate,
   negative-term rates by race, and the signed Black-minus-White gap;
4. image-level negative rates, a gender-stratified image-cluster interval and
   exploratory permutation p-value;
5. diversity safeguards: distinct words, unique response triples, entropy, and
   top-three-word share;
6. a manifest containing hashes and the full effective inference configuration;
7. a short comparison of the fresh run with the frozen current-repo baseline.

The current frozen September 11 base result is the reference check: 600/600
valid responses, 39.67% Black-photo negative terms, 3.33% White-photo negative
terms, and a +36.33 percentage-point signed gap. Those numbers are evidence from
that saved run, not values that a fresh stochastic run is forced to reproduce.

## What to report about fine-tuning

- April 29 established the FairFace base-versus-LoRA workflow, but it is a
  different 96-image evaluation and should not be mixed with Neutral12.
- Later DPO experiments showed that lowering negative-word counts alone is not
  enough: all-positive targets can collapse the vocabulary, and a smaller gap
  can result from worsening the White condition rather than improving the Black
  condition.
- The retained August 10 adapter was selected retrospectively from five pilots.
  It was a directional example, not an accepted successful debiasing model, and
  it missed the historical 95% unique-response-retention screen.
- Therefore the base benchmark must be frozen first, with validity, diversity,
  expression sensitivity, and untouched confirmatory data defined before any
  future fine-tuning decision.

## Recommended next step after the meeting

If the reproduction is accepted, freeze a fresh person-disjoint confirmatory
panel and pre-register the primary gap, validity/diversity safeguards, seed
policy, and success/failure thresholds. Only then consider a small new LoRA/DPO
pilot selected on development data. Other models and broader tests remain lower
priority until this base pipeline is trusted.

## Questions for the meeting

1. Does “same counts” mean exact replay of saved responses, or a fresh 600-slot
   run with the same design? A sampled rerun may produce different bar heights.
2. Is the raw response CSV that generated the supplied screenshot available?
3. Are the authorized 12 image files available, and do their hashes match?
4. Should the deliverable use the historical bare prompt only, or also show a
   separately labeled official chat-template replication?
5. Is the current 50-responses-per-image protocol the expected target, or does
   the advisor want an earlier repeat count?
6. For the meeting, is the count plot sufficient, or should the image-cluster
   interval, validity, and diversity diagnostics be presented alongside it?

External provenance and official documentation are collected in
`docs/external-sources.md`.
