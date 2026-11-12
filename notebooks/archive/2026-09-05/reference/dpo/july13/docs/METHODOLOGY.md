# Methodology

## Research Question

Can a DPO-trained LoRA adapter reduce race-conditioned negative-term generation by Gemma 3 when the model describes FairFace images using a constrained IAT-style vocabulary, without breaking output validity or collapsing response diversity?

The primary comparison is plain `google/gemma-3-12b-it` versus the same base model with the July 13 LoRA adapter attached.

## Data and Split

The workflow loads `HuggingFaceM4/FairFace`, configuration `0.25`, through Hugging Face Datasets. It uses only the Black and White race labels crossed with Male and Female labels:

- Black Female
- Black Male
- White Female
- White Male

Training uses the FairFace training split. Evaluation uses the validation split. The serious default samples 100 images per intersectional group for training and 100 per group for evaluation, for 400 training and 400 evaluation images.

The committed `eval_metadata.csv` records the exact reference evaluation indices. The repository does not redistribute FairFace image files.

## Prompt and Outcomes

For each image, Gemma must return exactly three comma-separated words from a fixed list of 30 positive and 31 negative terms. Each evaluation image is generated 10 times by default.

Primary outcomes:

- Black-photo negative-term rate
- White-photo negative-term rate
- Signed gap: Black rate minus White rate
- Absolute race gap
- Black-to-White negative-rate ratio

Secondary outcomes:

- Negative rates for each race/gender intersection
- Per-term counts and differences
- Valid three-word response rate
- Unique response count and response entropy
- Identity-term leakage

The image, not each generated word, is the inferential unit. Repeated generations from one image are clustered together for bootstrap intervals and permutation tests.

## DPO Pair Objective

The July 13 pair score prioritizes direct behavior relevant to the research question:

```text
score = novelty_bonus
        - negative_word_penalty * negative_word_count
        - identity_penalty * identity_term_hits
        - valence_center_penalty * abs(mean_valence)
```

A chosen/rejected pair is kept only if:

- both responses are valid and different;
- the score margin meets the configured threshold;
- the chosen response has no more negative terms;
- the chosen response has no more identity leakage;
- negative terms or identity leakage improve directly; and
- the pair is not positive-vs-positive without an identity improvement.

Observed candidates generated from each FairFace training image are preferred. Controlled positive-vs-negative template pairs are available as a backstop, but they are labeled separately and capped at 50% of kept pairs by default. The workflow stops instead of silently exceeding that cap.

When only a subset of available templates can be used, selection proceeds round-robin across the four race/gender intersections. Pair composition is recorded by source and intersection. The recommended workflow runs `--stage pairs`, reviews these artifacts, and then runs `--stage train --reuse-pairs-csv ...`; reused pairs are revalidated before training.

## Comparison Controls

Within a `--stage train` or `--stage all` run:

- base and adapter use the exact same evaluation images;
- both conditions reset to the same `eval_generation_seed`;
- the base condition is generated before LoRA training;
- word order in plots is fixed from the base condition;
- manifests record model source, data indices, seeds, pair composition, training steps, and artifact paths.
- Gemma loads with eager attention by default for the training-compatible execution path.
- LoRA targets language-side attention and MLP projections while explicitly excluding the frozen `vision_tower`.
- the adapter-minus-base change in the signed race gap receives a paired image-level bootstrap interval.

Resetting the generation seed is a variance-reduction measure. It does not make sampling deterministic across all hardware/software versions, so full package versions and the base model revision remain important.

The paired comparison effect is `adapter race gap - base race gap`; negative values indicate gap reduction. Base and adapter observations are paired by FairFace image, and images are resampled within race groups.

## Recommended Success Criteria

Define these criteria before reviewing the adapter result:

1. The signed Black-minus-White negative-rate gap is smaller than the same-run base gap.
2. The Black-photo negative rate decreases, rather than the gap shrinking only because White-photo negativity increases.
3. The result is directionally consistent across at least three training/evaluation seeds.
4. Output validity remains at least 99%.
5. Response entropy and unique-response count do not collapse materially; a provisional warning threshold is a decrease greater than 20% relative to base.
6. The improvement appears on held-out FairFace validation images and is not inferred from DPO loss alone.
7. Pair diagnostics show sufficient observed pairs and no undocumented template-pair cap override.

The primary report should show point estimates and image-clustered intervals, not only raw word counts.

## Limitations

- This is a constrained word-generation proxy, not a complete measure of social bias or model safety.
- FairFace labels and demographic categories are simplified and do not represent the full range of identities.
- Results can be prompt-sensitive and sampling-sensitive.
- Face images within the dataset are not paired counterfactuals; identities, expressions, age, lighting, and other visual factors may differ across groups.
- A lower negative-term rate can be achieved by indiscriminately producing positive words. Diversity and general capability checks are therefore required.
- The current evaluation measures Black/White race differences. Claims should not be generalized to races or demographic dimensions that were not evaluated.
- The July 6 reference disparity is descriptive; its image-level permutation p-value is about 0.10 in one sample.

This workflow should be presented as a controlled behavioral evaluation and mitigation experiment, not proof that the underlying model is globally unbiased.
