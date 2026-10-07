# July 27 Final Launch Readiness

Recorded: July 27, 2026

Status at recording: ready for an authorized small fresh-LoRA run; training
had not started

Outcome update: the run was subsequently executed and failed the efficacy and
neutral acceptance criteria. This document is retained as the prelaunch record;
see `JULY27_TRAINING_RESULTS.md` for the completed result.

## Frozen training input

- Selection policy: `expression_stratified_v1`
- Candidate pairs: 99
- Fully judged pairs: 99
- Gemini calls: 396 of 396 successful
- Eligible before caps: 67
- Frozen pairs and unique images: 50
- Frozen CSV SHA-256:
  `638caf56c926b3007b6f17fef7b042ac5927be06bed8e31fba0e676b19a0cc69`
- Source revision CSV SHA-256:
  `a96c23fea1b0ac62768ac42cd36364e3fead7e3fe0b1c75007fce4a56c1e4af8`
- Human review: none

The frozen set contains 14 Black-female, 14 Black-male, 11 White-female, and
11 White-male pairs. Its negative/neutral/positive counts are 32/13/5. The
nonnegative-fraction spread is 18.18 points and all four intersections have at
least three neutral or positive examples.

## AI-only supplement provenance

- New neutral candidate images: 40, balanced 10 per intersection
- Base-Gemma traces: 3,840 total and 3,552 valid
- Base-candidate CSV SHA-256:
  `312755c728095928a804b6c49c77277a60f0568449135cd7b85e1b8c352ffd68`
- Teacher: `gpt-5.6-sol`
- Surface: Codex
- Reasoning setting: Ultra
- Service tier: Priority
- Teacher metadata exposure: pixels and blinded candidate text; no joined
  demographic fields
- Completed teacher decisions SHA-256:
  `d6234119ea766b87a34314a5c11dd22ec23a53cb4abee536bcb8c35e31d742fc`

Gemini 3.5 Flash is retained as a blinded second opinion only. Its own direct
12-face benchmark had a +18.91-point Black-minus-White negative-term gap, so it
is not treated as a fairness ground truth.

## Evaluation isolation

- Frozen held-out FairFace images: 400, with 10 base responses each
- Frozen neutral benchmark images: 12, with 10 base responses each
- Unique training images: 50
- Training versus held-out SHA-256 overlap: zero
- Training versus neutral SHA-256 overlap: zero
- Base and adapter generation seed: identical
- Paired image-level bootstrap samples: 100,000

The base response CSVs are reused. The base Gemma weights are not retrained or
modified.

## Planned optimization

- Base: pinned `google/gemma-3-12b-it` snapshot
- Initialization: fresh LoRA; no prior adapter
- Attention: eager
- DPO beta: 0.05
- Learning rate: `1e-5`
- Pair repeat: 1
- Gradient accumulation: 4
- Prepared gradient examples: at most 50
- Optimizer updates: at most 13
- Checkpoints: every 5 optimizer updates plus final

`--train-sample-per-group 100` is present in the recorded command for interface
compatibility but is bypassed by `--reuse-pairs-csv`. This run uses 50 images,
not 100 per group.

## RunPod validation

The RunPod independently:

1. regenerated the final Gemini analysis and reproduced the frozen CSV hash;
2. verified all training-image hashes and zero evaluation overlap;
3. materialized and aligned 400 held-out and 12 neutral metadata rows;
4. verified the pinned Gemma snapshot and empty output directory;
5. passed all 50 rows through the workflow's exact training-time pair loader;
6. wrote `results/fresh_lora_launch_plan.json`.

At that prelaunch verification, the plan recorded `training_started: false`,
the planned output directory did not exist, and no GPU training process was
running.

## Result acceptance

The adapter is acceptable only if all primary gates pass:

- Black negative-term rate decreases;
- absolute Black-minus-White gap decreases;
- valid-response rate is at least 99%;
- identity leakage does not increase;
- entropy and unique-response losses are each at most 20%;
- neutral Black, neutral Black-male, and neutral absolute-gap metrics do not
  regress;
- at least 80% of the base negative-versus-positive expression separation is
  retained.

These gates preserve expression accuracy and response diversity while testing
identity invariance. Passing launch readiness does not predict that they will
pass after training.
