# April 29 Retrospective Reanalysis - July 13, 2026

## Status

This is a retrospective audit of preserved April 29 artifacts. It is not a new run and it is not evidence that the April LoRA successfully debiased Gemma 3. The audit was created to answer questions about the word-count plots using the exact training manifest, kept DPO pair, generated responses, comparison metrics, plots, and source snapshots.

## Runs Examined

- Training run: `fairface_dpo_default_20260429_104946`
- Larger comparison: `fairface_compare_compare_20260429_111640`
- Base model: `google/gemma-3-12b-it`
- Base revision: `96b6f1eccf38110c56df3a15bffe176da04bfd80`
- Dataset: `HuggingFaceM4/FairFace`, configuration `0.25`

## What Actually Trained

The training sample began balanced, with two images from each Black/Female, Black/Male, White/Female, and White/Male group. That produced eight training images and eight raw candidate pairs. Filtering retained only one unique pair, from one White/Male image:

| Field | Value |
| --- | --- |
| Image | `wm002_nc.png` |
| Chosen | `Happy, Delight, Joyful` |
| Rejected | `Happy, Delight, Glorious` |
| Chosen score | 1.6 |
| Rejected score | -0.4 |
| Chosen/rejected valence | 1.0 / 1.0 |
| Chosen/rejected identity hits | 0 / 0 |

The pair was repeated four times. Training ran for 16 microsteps with gradient accumulation of four, which produced four optimizer updates.

The April score was:

```text
novelty bonus - 5.0 * identity hits - 0.4 * absolute valence
```

Both responses were positive-only and contained no identity terms. The chosen response scored higher only because it received the 2.0 novelty bonus. `Happy, Delight` is shared by both responses, so the distinguishing DPO contrast was principally `Joyful` versus `Glorious`. There was no Black-image example in the final DPO training set and no chosen-versus-rejected reduction in negative words.

## Question 1: Did Delight Equalize?

No. The preserved counts do not support that premise.

The larger comparison used 24 validation images in each of the four race/gender groups, or 96 images total, with ten responses per image. Each response contained three words.

| Word | Base Black | Base White | Base absolute gap | Adapter Black | Adapter White | Adapter absolute gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Delight | 278 | 329 | 51 | 262 | 316 | 54 |
| Joyful | 162 | 145 | 17 | 256 | 257 | 1 |
| Happy | 316 | 291 | 25 | 306 | 286 | 20 |
| Glorious | 2 | 10 | 8 | 0 | 0 | 0 |

`Delight` became slightly less balanced by the absolute-count measure: its gap changed from 51 to 54. The tiny training-time evaluation gives the same direction, with its `Delight` gap changing from 7 to 8.

`Joyful` did become nearly equal in the larger comparison. A plausible hypothesis is that the sole DPO pair directly preferred `Joyful` over `Glorious`, causing a broad increase in `Joyful`. That is not a demonstrated causal fairness mechanism: the pair came from a White/Male image, it was the only unique pair, and the evaluation did not use paired stochastic draws.

The aggregate positive totals were already close because the evaluation allocated exactly 1,440 word slots to each race and the base model selected positive words about 95% of the time. Balanced evaluation exposure and the model's strong positive-output prior explain visually similar positive totals more directly than the training data does.

## Question 2: Why So Few Negative Words?

The large difference between positive and negative selections existed before fine-tuning:

| Model | Black negative | White negative | Total negative | Total positive | Overall negative rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 69 / 1,440 | 75 / 1,440 | 144 / 2,880 | 2,736 / 2,880 | 5.00% |
| April adapter | 75 / 1,440 | 57 / 1,440 | 132 / 2,880 | 2,748 / 2,880 | 4.58% |

The adapter produced only 12 fewer negative terms out of 2,880, a decrease of 0.42 percentage points. It is therefore inaccurate to attribute the positive-versus-negative imbalance to fine-tuning.

The evidence establishes that this imbalance predates fine-tuning. Plausible explanations to test are:

1. Instruction-tuned base Gemma already strongly preferred positive terms under this prompt.
2. The prompt asks for words that "best describe the person." Positive terms such as `Happy` and `Smiling` are more visually and socially plausible face descriptions than many negative options such as `Poison`, `Abuse`, or `Disaster`. This is a construct-validity confound in the word list and prompt.
3. The April DPO pair was positive versus positive. It supplied no direct training signal to reduce negative words.
4. The positive list had 30 terms and the negative list had 31, so the result is not caused by having more positive choices in the list.

The race-specific result also moved in the wrong direction for debiasing. Black-photo negative selections increased from 69 to 75 while White-photo negative selections fell from 75 to 57. The absolute race gap increased from 0.42 to 1.25 percentage points.

## Evaluation Limitations

- The comparison used the same held-out images, which is good.
- Generation was stochastic at temperature 1.0 and top-p 1.0.
- The comparison notebook seeded random number generators once before base evaluation and did not reset them before adapter evaluation. The two conditions therefore did not receive matched stochastic draws.
- No image-clustered confidence interval, paired bootstrap, or permutation test was produced.
- One unique DPO pair is too little data to identify a robust causal mechanism.

The observed 12-term overall decrease may be sampling variation. It should not be reported as a statistically established fine-tuning effect.

## Recommended Research-Group Answer

> The April 29 artifacts do not show that `Delight` became equally associated with Black and White photos. In the larger evaluation, its Black/White counts changed from 278/329 for base Gemma to 262/316 for the adapter, so the absolute gap increased from 51 to 54. `Joyful`, unlike `Delight`, became nearly equal, changing from 162/145 to 256/257. The only retained DPO example was a repeated White/Male positive-versus-positive pair, `Happy, Delight, Joyful` over `Happy, Delight, Glorious`, selected because of novelty. This can motivate a hypothesis about increased `Joyful`, but it cannot establish a race-debiasing mechanism.

> The model selected far more positive than negative words before fine-tuning as well: base Gemma produced 144 negative and 2,736 positive terms, compared with 132 negative and 2,748 positive terms after adaptation. The 12-term change is small, was not based on a direct negative-reduction training pair, and was measured with unmatched stochastic samples. The most defensible explanation for the overall imbalance is the base instruction-tuned model and a prompt/word-list design that makes many positive words more plausible face descriptions than many negative words.

## Artifact Map

- `training_run/fairface_train_manifest.json`: exact training configuration and tiny evaluation metrics.
- `training_run/fairface_iat_dpo_pairs_kept.csv`: the one retained pair.
- `training_run/fairface_iat_dpo_pair_diagnostics_kept.csv`: candidate-level scores and features.
- `training_run/fairface_eval_base.counts.csv`: tiny base word counts.
- `training_run/fairface_eval_adapter.counts.csv`: tiny adapter word counts.
- `larger_comparison/comparison_manifest.json`: 96-image comparison configuration.
- `larger_comparison/comparison_metrics.csv`: aggregate base/adapter counts.
- `larger_comparison/base.csv`: all 960 base responses.
- `larger_comparison/fairface_dpo_default_20260429_104946.csv`: all 960 adapter responses.
- `larger_comparison/base.png`: original base plot.
- `larger_comparison/fairface_dpo_default_20260429_104946.png`: original adapter plot.
- `source_snapshot/train_fairface_iat_dpo_20260429.py`: exact April trainer snapshot.
- `source_snapshot/02_compare_fairface_iat_adapters_20260429.ipynb`: comparison notebook snapshot.
