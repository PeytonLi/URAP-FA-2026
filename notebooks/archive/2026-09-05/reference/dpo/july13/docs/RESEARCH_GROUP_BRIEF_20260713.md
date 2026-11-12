# Research Group Brief - July 13, 2026

## Research Objective

Train and evaluate a Gemma 3 12B LoRA adapter that reduces race-conditioned negative-term generation for FairFace images without degrading output validity or collapsing response diversity.

## Established Baseline

The frozen July 6 base-Gemma evaluation used 100 FairFace validation images in each Black/Female, Black/Male, White/Female, and White/Male group. Ten responses were sampled per image.

| Condition | Negative terms | Total terms | Negative rate |
| --- | ---: | ---: | ---: |
| Black-photo prompts | 265 | 6,000 | 4.42% |
| White-photo prompts | 117 | 6,000 | 1.95% |

- Signed Black-minus-White gap: +2.47 percentage points.
- Black/White negative-rate ratio: 2.26.
- Image-clustered 95% bootstrap interval: [-0.43, 5.50] percentage points.
- Image-level permutation p-value: approximately 0.101.

Interpretation: this is a clear descriptive disparity and motivates mitigation, but one run does not provide conclusive inferential evidence.

## April 29 Retrospective Correction

The exact April 29 artifacts have now been recovered and audited. The retained DPO data contained only one unique White/Male positive-versus-positive pair: `Happy, Delight, Joyful` over `Happy, Delight, Glorious`. It was repeated four times and selected because of the old novelty score, not because it reduced negative terms or a race disparity.

The preserved 96-image comparison does not show that `Delight` equalized. Its Black/White counts were 278/329 for base Gemma and 262/316 for the adapter, changing the absolute gap from 51 to 54. `Joyful` changed from 162/145 to 256/257, but one repeated pair and unmatched stochastic generations cannot establish a causal fairness mechanism.

The base model already produced 2,736 positive terms and only 144 negative terms. The adapter produced 2,748 positive and 132 negative terms, a reduction of only 12 negative terms out of 2,880. Black-photo negatives increased from 69 to 75 while White-photo negatives fell from 75 to 57, so the absolute race gap worsened from 0.42 to 1.25 percentage points.

The full evidence, exact counts, methodological limitations, and meeting-ready answer are in [`results/april29_reanalysis_20260713`](../results/april29_reanalysis_20260713/README.md).

## Changes Since July 6

1. Candidate ranking now directly penalizes negative words and identity leakage rather than relying mainly on novelty and valence centering.
2. A kept chosen response cannot contain more negative words or identity leakage than its rejected response.
3. Every kept pair must directly reduce negative words or identity leakage.
4. Positive-versus-positive pairs are rejected unless identity leakage directly improves.
5. Model-observed and controlled-template pairs have separate provenance, with templates capped at 50%.
6. Template selection is round-robin balanced across all four race/gender intersections.
7. Pair generation, pair approval, weight training, and evaluation are separate auditable stages.
8. Base and adapter evaluation use the same held-out images and reset to the same generation seed.
9. Evaluation now reports signed gaps, intersectional rates, validity, diversity, identity leakage, and a paired image-level bootstrap interval.
10. Every run records its classification as smoke, pilot, or research and preserves model/data/configuration provenance.

## Engineering Findings

The first July 13 H100 smoke accidentally attached LoRA modules to the SigLIP vision tower. With global eager attention and both DPO policy graphs live, memory reached approximately 78.79 GiB in use and a 512 MiB allocation failed.

The correction froze and excluded `vision_tower` while retaining eager attention for Gemma. The validated path has:

- 65,470,464 trainable language-side LoRA parameters;
- 42.59 GiB peak PyTorch allocation;
- 43.96 GiB peak PyTorch reservation;
- approximately 35 GiB of H100 headroom.

Conclusion: the OOM was not caused by eager attention alone. It resulted from accidentally trainable high-token vision attention combined with chosen/rejected DPO graphs.

## Smoke-Test Findings

The smoke run established execution correctness, not debiasing efficacy.

- 16 training images and 8 candidates per image produced 128 candidates.
- 122 candidates were valid; 121 contained zero negative terms, 7 contained three negative terms, and 6 were invalid.
- No image produced both a suitable positive and negative candidate, so zero model-observed pairs survived.
- Four balanced, explicitly labeled execution-only template pairs were used only to exercise gradients and saving.
- DPO loss changed from 0.6914 to 0.6641 over two optimizer updates.
- All four checked preference deltas changed after training; mean absolute change was 0.78125.
- The eight-image evaluation emitted zero negative terms for both base and adapter and is statistically uninformative.

## Completed Research Run

The unattended 100-per-group pipeline began at 1:51 PM PDT on July 13. Its pair audit completed at 2:56 PM:

- 400 FairFace training images;
- 16 candidates per image and 6,400 requested candidate generations;
- 29 model-observed pairs and 29 controlled-template pairs;
- all 58 pairs directly improve negative-word count or identity leakage;
- templates exactly at the declared 50% cap;
- pair groups: 16 Black/Female, 16 Black/Male, 16 White/Female, and 10 White/Male;
- pair audit approved with no threshold failure;
- 2,000 DPO microsteps and approximately 500 optimizer updates if approved;
- LoRA checkpoints every 100 optimizer updates;
- 400 held-out validation images and 10 responses per image for both base and adapter;
- automatic base-versus-adapter plots and paired bootstrap analysis.

The full pipeline completed at 4:36 PM PDT with no CUDA error or traceback. The final adapter reduced Black-photo negatives from 255 to 2 and White-photo negatives from 118 to 0. The signed Black-minus-White gap fell from 2.283 to 0.033 percentage points.

The result is not an acceptable final model. Response entropy fell 38.00% and unique responses fell 71.61%, both beyond the declared 20% collapse threshold. The paired 95% interval for gap change was `[-5.23, +0.63]` percentage points and included zero. DPO loss was effectively zero by approximately optimizer step 60 but training continued to step 500.

Interpretation: the objective successfully suppressed negative terms, but training was too strong for the 58-pair dataset and caused a narrow positive-response mode.

## Pair-Audit Finding

The pair audit is a new data-quality result, not yet a model-efficacy result:

- Model-observed coverage was 29 of 400 images, or 7.25%.
- All chosen responses contain zero negative terms and zero identity hits.
- Fifty-six rejected responses contain three negative terms; two instead contain one leaked identity term.
- The most frequent rejected terms are Pain, Sadness, Grief, and Hurtful, which were the largest July 6 disparity contributors.
- The 29 selected templates use six unique text contrasts, so response-collapse metrics are important.
- White/Male has only three observed pairs and ten combined pairs, the weakest group coverage.
- The exact pair data and diagnostics are preserved in `results/pair_audit_research_july13_100pg_20260713_205104/`.

Do not describe the final checkpoint as a successful debiased model. Report the rate reduction together with the diversity collapse and statistical uncertainty.

## Artifacts To Show

- July 6 baseline word graph: `results/reference_base_gemma3_20260706/00_baseline_base_gemma3_fairface_iat.png`
- July 6 bias-focused graph: `results/reference_base_gemma3_20260706/baseline_bias_focus.png`
- Baseline statistical report: `results/reference_base_gemma3_20260706/BASELINE_ANALYSIS.md`
- H100 smoke record: `results/smoke_h100_20260713/README.md`
- Smoke DPO loss graph: `results/smoke_h100_20260713/train_v3/plot_dpo_loss.png`
- Failed-attempt analysis: `docs/FAILED_ATTEMPTS.md`
- Full model/workflow history: `docs/MODEL_CHANGELOG.md`
- April 29 retrospective correction: `results/april29_reanalysis_20260713/README.md`
- July 13 approved research pair audit: `results/pair_audit_research_july13_100pg_20260713_205104/README.md`
- Completed July 13 result and interpretation: `results/research_july13_100pg_20260713_205104/README.md`
- Base-versus-LoRA graph: `results/research_july13_100pg_20260713_205104/train/02_base_vs_july13_lora_bias_metrics.png`
- Final-adapter word graph: `results/research_july13_100pg_20260713_205104/train/01_adapter_july13_lora_fairface_iat.png`
- Full-run loss graph: `results/research_july13_100pg_20260713_205104/train/plot_dpo_loss.png`
