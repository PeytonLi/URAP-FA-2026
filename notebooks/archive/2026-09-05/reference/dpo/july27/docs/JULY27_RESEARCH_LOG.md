# July 27 Research Log

## Scope

This revision begins with a missing methodological control: directly
benchmarking the AI teacher that reviewed expressions and corrected DPO pairs
for the July 20 AI-only pilot.

## Provenance correction

The local Codex rollout record for the July 20 work identifies the model as
`gpt-5.6-sol`. The July 20 result manifests identify the reviewer only as a
Codex AI first pass. Future result manifests must record the exact model,
surface, reasoning setting, prompt hash, image hashes, and run date.

Historical teacher configuration:

- model: `gpt-5.6-sol`
- surface: Codex
- reasoning setting: `ultra`
- human review: false
- independent second rater: false

The direct benchmark pins Standard service speed. Service speed affects
latency and credit consumption, not the intended model or reasoning setting.
Fast mode is intentionally disabled because GPT-5.6 Fast consumes credits at
2.5 times the Standard rate and does not improve the scientific control.

## Why this benchmark is required

Metadata-blind review is a useful control, but it does not establish fairness.
The model still sees each face and can perceive demographic cues. A direct
behavioral benchmark is therefore required before the teacher is treated as a
reliable source of debiasing supervision.

The July 20 LoRA pilot also worsened the fixed neutral benchmark, from a
Black-minus-White negative-term gap of 31.67 percentage points for base Gemma
to 45.00 points with the adapter. This does not prove that the teacher caused
the regression, but it makes teacher and pair-policy audits necessary.

## Direct teacher benchmark protocol

1. Freeze the exact 12 neutral Project Implicit Race IAT face files.
2. Verify each file against its SHA-256 digest.
3. Pass only blinded filenames to the teacher.
4. Use the exact April pipeline prompt and 61-term vocabulary.
5. Start a fresh ephemeral Codex call for every image/iteration.
6. Request 50 responses per image, producing 600 independent calls.
7. Preserve every raw response, including failures and format violations.
8. Join demographic metadata only after generation.
9. Produce the same word-count plot used by the April pipeline.
10. Report validity, negative-term rates, signed race gap, intersections,
    entropy, unique responses, per-image rates, and bootstrap uncertainty.

The call order is deterministically shuffled so race is not confounded with
time, transient service behavior, or model updates during the run.

## Interpretation

The benchmark can detect a disparity under this stimulus set and prompt. It
cannot prove the absence of bias generally.

The following outcomes must remain separate:

- direct teacher generation behavior on the fixed 12-face benchmark;
- teacher expression-label decisions on FairFace;
- teacher acceptance/exclusion decisions for DPO candidates;
- teacher-chosen replacement terms in retained DPO pairs;
- downstream Gemma base-versus-LoRA behavior.

A balanced direct-generation plot would not by itself validate the teacher's
DPO correction decisions.

## Required second audit

After the direct benchmark, reconstruct the July 20 teacher packet and analyze
teacher behavior after demographic metadata is restored:

- item acceptance and exclusion rates by race and gender;
- confidence distributions by intersection;
- chosen and rejected negative-term counts;
- correction magnitude by expression class;
- identity-leakage removals by group;
- chosen-term distributions by group;
- matched Black/White image-pair differences.

This retrospective audit should produce a separately labeled PNG. It must not
be presented as the direct teacher benchmark.

That second audit is now implemented as an independent Gemini 3.5 Flash
four-call judgment protocol in `gemini_pair_judge.py`. It evaluates image fit
and no-image structural behavior separately and repeats each with candidate
order reversed. See `JULY27_CHANGELOG.md` for the exact gates and preserved
failed pilots.

## Current execution status

The direct benchmark completed successfully on July 27, 2026:

- 600 of 600 calls succeeded.
- All 600 responses contained exactly three allowed terms in the requested
  format.
- Black negative-term rate: 8.67%.
- White negative-term rate: 14.33%.
- Black-minus-White gap: -5.67 percentage points.
- Image-bootstrap 95% interval: [-18.0, +6.0] points.

The result does not reproduce base Gemma's anti-Black direction, but its
interval includes zero and the effective stimulus sample remains 12 faces.
See `GPT56_SOL_TEACHER_BENCHMARK_RESULTS.md` for the complete interpretation.

The Gemini pair audit and direct Gemini benchmark are complete. The direct
benchmark found a +18.91-point anti-Black negative-term gap with a wide
image-level interval, so Gemini is not treated as a sole judge.

The first 51-pair audited candidate set failed new expression-distribution
controls: neutral/positive fractions ranged from 7.69% for White male to 64.29%
for Black female. It was preserved but never trained.

A balanced 40-image supplement then produced 3,840 new base-Gemma candidate
traces. Four prioritized errors received blinded GPT-5.6 Sol multimodal review;
three candidate corrections were appended, and all three received Gemini image
and no-image judgments in both orderings. The final 99-candidate analysis
freezes 50 pairs with an 18.18-point nonnegative-fraction spread and at least
three nonnegative pairs in every intersection. The frozen CSV SHA-256 is
`638caf56c926b3007b6f17fef7b042ac5927be06bed8e31fba0e676b19a0cc69`.

The RunPod independently reproduced the final freeze and the exact training
loader accepted all 50 rows. A non-executing fresh-LoRA plan now exists at
`results/fresh_lora_launch_plan.json`. The July 27 LoRA has still not been
trained, so this is dataset readiness rather than an efficacy result.

The missing July 20 `step_0010` evaluation is now complete. It reduced the
held-out signed race gap by only 0.083 percentage points, with a paired 95%
interval of [-0.883, +0.717] points. Neutral aggregate rates were unchanged
and Black-male negativity increased from 60.0% to 66.67%. It is less harmful
than the final adapter but still fails the primary acceptance criteria.

## Fresh LoRA outcome

The fresh July 27 run launched from the pinned base model at
`2026-07-27T14:05:49Z` and finished at `2026-07-27T14:56:25Z`. It completed
all 50 gradient examples and 13 optimizer updates without OOM, NaN, or process
failure. The final adapter was functional on all 16 effect probes.

The final result failed the prespecified efficacy endpoints. Held-out Black
negativity rose from 4.750% to 5.683%, White negativity rose from 1.950% to
2.433%, and the signed race gap rose from 2.800 to 3.250 points. Neutral Black
negativity rose from 35.0% to 50.0%. The final paired image-bootstrap interval
for Black-rate change was [+0.300, +1.683] percentage points.

Optimizer step 10 was evaluated after the primary result and is explicitly
exploratory. It produced held-out Black and White rates of 5.417% and 2.567%
and a 2.850-point gap. Neutral Black negativity was 41.667%. It failed the
same efficacy and neutral safeguards, so step 5 was not tested.

The post-outcome data audit identified a supervision mismatch. July 27 used 29
fixed-count unsupported-negativity substitutions but only one pair that
reduced total negative-word count; its negative-expression targets also used
substantially more `Pain`, `Sadness`, and `Grief` for Black images. This is
consistent with the observed term increases and motivates matched lexical and
fine-expression controls in the next dataset. It does not change the original
failure classification.

All primary, checkpoint, log, plot, and raw-response artifacts were copied
locally and verified against recursive SHA-256 manifests. See
`JULY27_TRAINING_RESULTS.md` and
`results/july19_vs_july27_comparison/`.
