# Next Iteration Plan

## Current Decision

The July 13 100-per-group pipeline completed, but the final step-500 adapter is overtrained. It reduced total negative terms from 373 to 2 while reducing response entropy by 38.00% and unique responses by 71.61%. The next objective is to retain a meaningful reduction in Black-photo negative rate and the signed Black-minus-White gap without exceeding the predeclared 20% diversity-loss thresholds.

Do not scale to 500 FairFace images per group yet. The immediate limitation is excessive optimization on only 58 unique preference pairs, not GPU capacity or the number of FairFace images alone.

## Frozen Evidence

Use these exact artifacts when isolating the effect of training duration:

- Audited 58-pair CSV: `results/pair_audit_research_july13_100pg_20260713_205104/artifacts/july13_dpo_pairs_kept_combined.csv`
- Pair CSV SHA-256: `5ec43fa66ffd52993b193e8185cf3e04bb32829056c34007c51ed1ef53bf1a6e`
- Training metadata SHA-256: `3e11646215608c4c807d0b5884a91f39b152493b931231c8062a6cc1587c05e6`
- Held-out evaluation metadata: `results/research_july13_100pg_20260713_205104/eval_metadata.csv`
- Evaluation metadata SHA-256: `fb5337b290a9f15ca4b00d3c8428428393374aabc4fadd692275ff06ccc199e0`
- Evaluation generation seed: `2713`
- Base model revision: `96b6f1eccf38110c56df3a15bffe176da04bfd80`
- Saved adapters: `adapters/research_july13_100pg_20260713_205104/`
- Source inventory: `results/research_july13_100pg_20260713_205104/SOURCE_SHA256SUMS`

FairFace images are rematerialized from `HuggingFaceM4/FairFace` using the frozen metadata; they are not committed to Git.

## Phase 1: Evaluate Existing Checkpoints

Add a checkpoint-only evaluation stage that loads an existing adapter and writes the same CSV, metrics JSON, graphs, and paired bootstrap comparison as the completed run. It must not perform DPO updates.

Evaluate `step_0100` first using the exact 400 held-out images, ten responses per image, and generation seed `2713`. If it still exceeds a diversity threshold, evaluating steps 200-500 is unnecessary because the training loss was already effectively zero before step 100. The available checkpoints cannot reveal behavior at the likely useful 10-40 update range.

## Phase 2: Short Schedule Pilot

If step 100 remains collapsed, retrain from the frozen 58-pair CSV with:

- one copy of each unique pair, with no repeat augmentation;
- 40 optimizer updates maximum;
- checkpoints every 10 optimizer updates;
- gradient accumulation 4;
- DPO beta `0.05`;
- learning rate `1e-5`;
- the same LoRA rank, alpha, target modules, eager attention, and frozen vision tower;
- the same evaluation metadata, repeats, generation seed, and bootstrap procedure.

Evaluate checkpoints 10, 20, 30, and 40. Select a checkpoint using held-out negative-rate and diversity criteria, never training loss alone. Record that this pilot changes both duration and learning rate; a later ablation should vary one factor at a time if a publishable causal claim is needed.

## Phase 3: Pair Dataset Revision

Create a new versioned pair CSV only after the frozen-pair pilot establishes whether early stopping is sufficient. A revised pair set should:

- increase the number and lexical diversity of model-observed pairs;
- preserve separate observed/template provenance labels;
- retain only pairs with a direct negative-term or identity-leakage improvement;
- continue rejecting unsupported positive-versus-positive pairs;
- lower the template share below 50% when enough observed pairs survive;
- balance source coverage across Black/White and female/male intersections;
- cap repeated chosen/rejected text combinations so a few positive phrases cannot dominate;
- record all candidate diagnostics, including successful images, not only retained pairs;
- keep exact three-list-word outputs separate from parser failures or explanatory text.

The original audited CSV must remain immutable. A replacement should receive a new filename, manifest, checksum, pair-quality report, and explicit comparison with the original 58 pairs.

## Acceptance Criteria

Predeclare these checks before evaluating a new adapter:

1. Black-photo negative rate decreases from the same-run base condition.
2. The signed Black-minus-White negative-rate gap moves toward zero without hiding an increase in either group's negative rate.
3. Valid exact three-word response rate remains at least 99%.
4. Response entropy and unique-response count each decline by no more than 20% from same-run base.
5. Identity leakage does not increase.
6. Full-response and term concentration are reported so positive-mode collapse is visible.
7. The paired image-level bootstrap and intersectional rates are reported; confidence intervals are not treated as proof when they include zero.
8. Any selected configuration is repeated across at least three training and generation seeds before it is presented as robust.

## Required Code Changes

Before the next paid GPU session:

1. Implement and test a checkpoint-only `evaluate-adapter` stage.
2. Add configurable save intervals and a 10-40 update short-run preset.
3. Add early-stop safeguards based on loss saturation and optional interim diversity evaluation.
4. Add a checkpoint comparison table that applies all acceptance criteria consistently.
5. Keep pair generation, audit, training, and evaluation as separate resumable stages.

The July 13 source, exact pair/evaluation metadata, plots, raw responses, reference-score cache, and all saved checkpoints are backed up locally. The RunPod can therefore remain stopped while these code changes are developed and tested without GPU billing.
