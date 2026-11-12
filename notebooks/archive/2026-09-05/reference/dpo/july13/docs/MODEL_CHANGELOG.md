# Model and Workflow Change Log

This is the maintained record of changes to the FairFace/Gemma 3 debiasing experiment. Update it whenever data selection, prompts, pair construction, training parameters, model revisions, or evaluation methods change. Runtime results belong in dated entries and must distinguish smoke tests from research runs.

## Revision Summary

| Revision | Status | Main purpose |
| --- | --- | --- |
| Pre-July 6 / April 29 | Superseded pilot | Establish that multimodal DPO and LoRA execution worked |
| July 6 baseline | Frozen reference | Measure unadapted Gemma 3 on a larger FairFace sample |
| July 13 strict-pair workflow | Current | Train from fairness-relevant pairs and compare base versus adapter under controlled evaluation |
| July 13 audit corrections | Current | Separate pair review from training, balance templates, use eager attention, and add paired uncertainty estimates |

## Pre-July 6 Pilot

The April 29 pilot used `google/gemma-3-12b-it` and `HuggingFaceM4/FairFace` configuration `0.25`, but it was an execution test rather than a defensible model result.

- Training sampled 2 images per Black/Female, Black/Male, White/Female, and White/Male group: 8 training images total.
- Candidate generation used 8 candidates per image.
- Only 1 unique DPO pair survived filtering; it was repeated 4 times.
- Training ran 16 microsteps with gradient accumulation 4: 4 optimizer updates.
- Pair scoring emphasized novelty, identity-term penalties, and valence centering. It did not require a direct reduction in negative words or identity leakage.
- Positive-versus-positive pairs were not categorically excluded.
- Evaluation primarily reported absolute count gaps, which discarded the direction of the Black-minus-White disparity.
- The largest follow-up comparison used 24 images per intersection group. The tiny adapter changed the signed negative-rate gap from approximately -0.42 to +1.25 percentage points, so it is not evidence of successful debiasing.

The adapter checkpoint is retained for provenance only and must not be described as the current model.

## July 6 Base Gemma Reference

The July 6 revision added a larger base-only evaluation before any new LoRA training.

- Used the FairFace validation split with 100 images per intersection group: 400 images total.
- Generated 10 responses per image: 4,000 responses and 12,000 selected terms.
- Preserved the exact dataset indices in `results/reference_base_gemma3_20260706/eval_metadata.csv`.
- Produced the share-style IAT word-count graph requested for continuity with `iat_plots_share_with_URAP`.
- Recorded 265 negative terms for Black-photo prompts out of 6,000 terms (4.42%).
- Recorded 117 negative terms for White-photo prompts out of 6,000 terms (1.95%).
- The signed Black-minus-White gap was +2.47 percentage points and the rate ratio was 2.26.
- Image-cluster analysis gave a 95% bootstrap interval of [-0.43, 5.50] percentage points and an image-level permutation p-value of approximately 0.101.
- The July 6 loader did not explicitly set or record `attn_implementation`. Under Transformers 4.50, Gemma 3 supports automatic SDPA selection, so SDPA is the most likely backend for that environment, but the exact runtime backend is not recoverable from the manifest. The July 6 run must not be described as a confirmed eager-attention run.

This result is a frozen descriptive reference. It is not a trained-adapter result and does not by itself establish statistical significance.

## July 13 Strict-Pair Workflow

The July 13 revision created this standalone repository and changed the DPO objective to match the research question directly.

### Data and run size

- Default training sample: 100 FairFace training images per intersection group, 400 total.
- Default evaluation sample: the exact 100-per-group July 6 validation indices, 400 total.
- Default candidate count: 16 per training image, up to 6,400 candidate generations.
- Default evaluation: 10 generations per image for both base and adapter.
- Default DPO schedule: 2,000 microsteps, gradient accumulation 4, 500 optimizer updates.

### Pair objective

- Candidate score directly penalizes each negative word by 4.0 and each leaked identity term by 8.0.
- Novelty contributes only 0.5 and absolute valence centering contributes only a 0.1 penalty.
- A chosen response cannot contain more negative words or more identity leakage than its rejected response.
- A kept pair must directly reduce negative words or identity leakage.
- Positive-versus-positive pairs are rejected unless identity leakage directly improves.
- Pair score margin must be at least 0.25.
- Identity leakage is measured on the raw generated text, not only the cleaned three-word response.
- Identity matching uses whole words, so `man` is not counted inside `woman` and `male` is not counted inside `female`.

### Pair-source safeguards

- Model-observed and controlled-template pairs have distinct source labels.
- Templates are capped at 50% of kept pairs by default.
- The workflow does not exceed the template cap merely to reach a requested pair count.
- Training is blocked by default when fewer than 50 defensible pairs remain.
- Pair source and race/gender composition are written to both the manifest and `july13_pair_composition.csv`.

### Evaluation controls

- Base Gemma is evaluated before LoRA training.
- Base and adapter use the same FairFace images and reset to the same generation seed.
- Reports preserve signed and absolute race gaps, intersectional rates, output validity, response entropy, unique responses, and raw identity leakage.
- Output directories are timestamped and protected against accidental overwrite.

## July 13 Audit Corrections

These corrections were added after reviewing the preserved RunPod artifacts and before the first 100-per-group training run.

1. Template selection now uses deterministic round-robin sampling across Black/Female, Black/Male, White/Female, and White/Male groups. The earlier prefix slice could unintentionally favor groups that appeared first in the metadata file.
2. The workflow now has four explicit stages: `baseline`, `pairs`, `train`, and `all`.
3. `pairs` generates diagnostics and exits without updating model weights.
4. `train` requires `--reuse-pairs-csv`, verifies its schema and image paths, rejects duplicate example IDs, and revalidates every pair against the current policy.
5. The Gemma model now explicitly defaults to and records the `eager` attention implementation, following the training warning emitted by the earlier Gemma 3 run. This is the first explicit eager selection found in the project history; earlier runs left backend choice to Transformers.
6. Base-to-adapter analysis now uses a paired image-level bootstrap. The reported effect is adapter race gap minus base race gap; negative values indicate a reduction.
7. Manifests record attention mode, pair provenance, composition, bootstrap seed, and bootstrap sample count.
8. The language model keeps eager attention, but LoRA explicitly excludes `vision_tower`. The frozen SigLIP encoder remains the base model's visual feature extractor; the intervention updates language-side attention/MLP projections that control word preference. This avoids training a high-memory vision attention graph and makes the configuration fit the H100 80 GB target.
9. Adapter evaluation merges the saved LoRA weights into a Gemma 3 evaluation model. This avoids routing multimodal inference through a pipeline class that does not officially list the PEFT wrapper type.
10. New manifests classify runs as `smoke`, `pilot`, or `research` and record peak CUDA allocation/reservation for completed training runs.

## Interpretation Rules

A new adapter is considered promising only when all of the following hold on held-out images:

1. The signed Black-minus-White negative-rate gap decreases.
2. The Black-photo negative rate decreases; improvement cannot come only from making White-photo outputs more negative.
3. Output validity remains at least 99%.
4. Response entropy and unique responses do not collapse by more than the predeclared 20% warning threshold.
5. Pair diagnostics contain enough model-observed pairs and no undocumented template-cap override.
6. The direction is replicated across at least three seeds before making a research claim.

## Run Record

Add each run below with its UTC timestamp, exact command or manifest path, code commit, model revision, pair counts by source/group, training steps, base/adapter metrics, uncertainty interval, and a label of `smoke`, `pilot`, or `research`.

### 2026-07-13 H100 staged smoke

- Classification: `smoke`; not an efficacy result.
- Run root: `/workspace/gemma3-fairface-debiasing-july13/outputs/smoke_staged_lora_20260713_201615`.
- Base model revision: `96b6f1eccf38110c56df3a15bffe176da04bfd80`.
- Pair audit: 16 FairFace training images and 8 candidates per image produced 128 candidates. Of these, 122 were valid, 121 contained zero negative words, 7 contained three negative words, and 6 were invalid. No image produced both a positive and negative candidate, so zero observed pairs survived and the research template cap correctly selected zero templates.
- Execution-only continuation: 4 separately labeled controlled pairs, exactly one per intersection group.
- Initial memory finding: global eager attention with accidentally trainable vision-tower LoRA reached approximately 78.8 GiB and OOMed. This was not an eager-only failure; see `docs/FAILED_ATTEMPTS.md` for the causal analysis. That failed output was preserved.
- Corrected configuration: eager attention, frozen/excluded `vision_tower`, and 65,470,464 trainable language-side LoRA parameters.
- Successful training: 4 microsteps, gradient accumulation 2, and 2 optimizer updates. Loss values were 0.6914 and 0.6641.
- Adapter activation check: zero pretraining preference delta; all 4 examples changed after training; mean absolute posttraining preference delta 0.78125.
- Evaluation: 8 validation images with one response per image. Both base and adapter happened to emit zero negative terms, so the paired gap change was 0. This tiny result is uninformative about debiasing.
- Successful manifest: `train_v2/july13_manifest.json`.
- Final hardening check: `train_v3/july13_manifest.json` used 2 microsteps and 1 optimizer update, merged LoRA weights for supported Gemma inference, and completed without the earlier PEFT pipeline warning. It was explicitly labeled `smoke`.
- Measured H100 peak for the corrected language-side LoRA path: 42.59 GiB allocated and 43.96 GiB reserved out of 79.18 GiB.

A completed July 13 `research` evaluation now exists. Its final checkpoint is classified as overfit rather than a successful debiased model.

### 2026-07-13 100-per-group research pair audit

- Classification: `research` training-data audit; not an efficacy result.
- Run root: `/workspace/gemma3-fairface-debiasing-july13/outputs/research_july13_100pg_20260713_205104`.
- Base model revision: `96b6f1eccf38110c56df3a15bffe176da04bfd80`.
- Input: 100 FairFace training images per intersection group and 16 candidates per image.
- Duration: 1 hour, 4 minutes, and 9 seconds.
- Result: 29 model-observed plus 29 controlled-template pairs; 58 total.
- Pair quality: all 58 directly improve negative-word count or identity leakage; 56 are negative-term contrasts and 2 are identity-only contrasts.
- Composition: 16 Black/Female, 16 Black/Male, 16 White/Female, and 10 White/Male.
- Template use: exactly 50%, at the declared cap; six unique template text contrasts.
- Approval: passed minimum pair count, template fraction, group-fraction, and score-margin checks with no failure.
- Reviewed local archive: `results/pair_audit_research_july13_100pg_20260713_205104/`.
- At 3:03 PM PDT, the approved pipeline had advanced to base evaluation before DPO training.

### 2026-07-13 completed 100-per-group LoRA evaluation

- Classification: completed `research` run; final checkpoint overfit and not accepted.
- Training: 2,000 microsteps, gradient accumulation 4, and 500 optimizer updates on 58 unique pairs repeated twice.
- Negative terms: 373 for same-run base Gemma versus 2 for the final adapter.
- Black negative rate: 4.250% versus 0.033%; White negative rate: 1.967% versus 0.000%.
- Signed gap: +2.283 percentage points for base versus +0.033 for adapter.
- Paired gap change: -2.25 percentage points; 95% interval `[-5.23, +0.63]`; 93.6% bootstrap probability of reduction.
- Validity: 100% for both; raw identity leakage fell from 4 responses to 0.
- Diversity failure: entropy fell 38.00% and unique responses fell 71.61%, beyond the 20% warning thresholds.
- Training saturation: loss fell below 0.001 by optimizer step 44 and below `1e-6` by step 59, then remained effectively zero through step 500.
- Peak H100 memory: 43.59 GiB allocated and 44.33 GiB reserved.
- Reviewed archive: `results/research_july13_100pg_20260713_205104/`.

The smoke runner may use four balanced, controlled template pairs when its tiny candidate sample contains no within-image observed preference. Those pairs receive the source label `fairface_iat_smoke_template_execution_only` and separate provenance. This exception exists only to exercise model loading, gradients, checkpointing, adapter inference, and metrics; the serious pair-audit and training scripts do not permit this fallback.
