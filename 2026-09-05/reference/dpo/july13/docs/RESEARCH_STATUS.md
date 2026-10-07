# Research Status as of July 13, 2026

## Objective

Train a Gemma 3 12B vision-language LoRA adapter that reduces negative IAT-term generation for Black-photo prompts and narrows the Black-versus-White negative-rate gap on held-out FairFace images.

## What Has Been Established

The July 6 base-only run is the newest completed baseline:

- Base model: `google/gemma-3-12b-it`
- Base revision: `96b6f1eccf38110c56df3a15bffe176da04bfd80`
- Dataset: `HuggingFaceM4/FairFace`, configuration `0.25`
- Evaluation: 100 images per race/gender group, 400 images total
- Repeats: 10 per image, 4,000 responses and 12,000 words
- Valid three-word outputs: 100%
- Black-photo negative rate: 4.42%
- White-photo negative rate: 1.95%
- Signed rate gap: +2.47 percentage points
- Rate ratio: 2.26
- Raw identity-term leakage: 3 of 4,000 responses (0.075%)

Intersectional negative rates were:

| FairFace group | Negative rate |
| --- | ---: |
| Black Female | 4.53% |
| Black Male | 4.30% |
| White Female | 1.00% |
| White Male | 2.90% |

Pain, Sadness, Grief, and Hurtful accounted for the largest Black-minus-White negative count differences.

After clustering repeated generations by image, the 95% bootstrap interval for the race gap was -0.43 to 5.50 percentage points and the two-sided permutation p-value was approximately 0.10. The observed disparity is large enough to motivate training, but additional seeds are needed for a strong inferential claim.

## Earlier Adapter Results

The April 15 larger pilot did not debias the model. It used only six training images per group and 24 kept pairs. Its held-out absolute negative count gap increased from 27 for base to 84 for the adapter.

The April 29 adapter was only a smoke test:

- two training images per group;
- eight training images total;
- one kept DPO pair;
- four repeated training rows;
- sixteen DPO microsteps.

In the later shared-sample comparison, its absolute negative count gap increased from 6 for base to 18 for the adapter. The one kept pair was positive-vs-positive and did not express the intended fairness target.

These adapters demonstrate that the pipeline can update model behavior, but they are not successful debiasing results.

## July 13 Code Changes

- Candidate ranking directly penalizes negative terms and identity leakage.
- Positive-vs-positive pairs are rejected by default.
- Kept pairs require a direct negative-term or identity-leakage improvement.
- Template pairs are labeled and subject to a strict fraction cap.
- The workflow stops when it lacks enough defensible training pairs.
- Base and adapter evaluation reset to the same generation seed.
- Metrics preserve the direction of the race gap and include intersectional rates.
- Output folders are protected from accidental overwrite.
- A standalone image-clustered analysis script is included.

## July 13 Research Pair Audit

The 100-per-group research pair audit completed and passed on July 13:

- 400 balanced FairFace training images and 6,400 requested candidate generations;
- 29 model-observed pairs, giving 7.25% observed-pair image coverage;
- 29 round-robin controlled-template pairs, exactly 50% of the combined data;
- 58 total pairs, all with a direct negative-word or identity-leakage improvement;
- 56 negative-term improvements and 2 identity-only improvements;
- group totals of 16 Black/Female, 16 Black/Male, 16 White/Female, and 10 White/Male;
- no validator threshold failures.

The selected data strongly targets Pain, Sadness, Grief, and Hurtful, the leading July 6 disparity contributors. Half the pairs are templates using six unique text contrasts, so held-out diversity and entropy are required to detect phrase memorization or response collapse.

The complete audit is in `results/pair_audit_research_july13_100pg_20260713_205104/`. This establishes improved training-data alignment, not model efficacy.

## Completed July 13 Research Run

The approved pair file was trained for 2,000 microsteps and 500 optimizer updates. The pipeline completed technically, but the final checkpoint is overfit.

| Metric | Base | Adapter |
| --- | ---: | ---: |
| Black negative rate | 4.250% | 0.033% |
| White negative rate | 1.967% | 0.000% |
| Signed race gap | +2.283 points | +0.033 points |
| Total negative terms | 373 | 2 |
| Response entropy | 5.330 | 3.305 |
| Unique responses | 708 | 201 |

The observed gap change was -2.25 percentage points with a paired image-level 95% interval of `[-5.23, +0.63]`. The interval includes zero. Entropy fell 38% and unique responses fell 72%, so the adapter failed both 20% diversity safeguards.

The next pilot should use substantially fewer optimizer updates, a lower learning rate, frequent early checkpoints, and diversity-aware early stopping. The full record is in `results/research_july13_100pg_20260713_205104/`.

Do not scale to 500 images per group until the 100-per-group objective produces a held-out improvement without response collapse.
