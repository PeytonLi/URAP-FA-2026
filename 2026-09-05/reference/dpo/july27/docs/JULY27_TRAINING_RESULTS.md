# July 27 Fresh LoRA Training Results

## Result status

The prespecified July 27 final adapter failed the primary acceptance criteria.
It must not be described as a successful debiasing result. The separately
evaluated optimizer-step-10 checkpoint was an exploratory post-primary
diagnostic and also failed.

The frozen base responses are byte-identical to the July 19 reference:

- held-out FairFace CSV SHA-256:
  `45c43099210bc685e5fb30ab15406be0cbc571d53573bf761736c15b0b491226`;
- neutral 12-face CSV SHA-256:
  `c4496947930208e61380074cc49d8d35ca9b8823ca901313be4bd5f03c5fc58c`.

The result therefore does not come from a changed baseline, prompt, image set,
or generation seed.

## Training execution

- Base: pinned `google/gemma-3-12b-it` snapshot.
- Method: fresh DPO LoRA; no prior adapter was loaded.
- Attention: eager.
- Frozen training set: 50 unique image-level pairs, each used once.
- Frozen CSV SHA-256:
  `638caf56c926b3007b6f17fef7b042ac5927be06bed8e31fba0e676b19a0cc69`.
- DPO: beta 0.05, learning rate `1e-5`, gradient accumulation 4.
- Updates: 50 gradient examples and 13 optimizer updates.
- LoRA: rank 16, alpha 32, dropout 0.05, 65,470,464 trainable parameters.
- Vision tower: excluded from LoRA targets.
- Checkpoints: optimizer steps 5, 10, and final step 13.
- End-to-end workflow time: 50 minutes 14 seconds.
- Peak H100 memory: 44.10 GiB allocated and 44.67 GiB reserved.
- Failures: no OOM, NaN, traceback, or interrupted checkpoint.

The final adapter was functional rather than silently unloaded: all 16 effect
probes changed, 14 shifted toward the audited chosen response, and the mean
signed chosen-minus-rejected margin change was +9.29.

## Primary metrics

Rates and gaps are percentages or percentage points.

| Model | Held-out Black | Held-out White | Held-out B-W gap | Neutral Black | Neutral White | Neutral B-W gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base Gemma 3 | 4.750 | 1.950 | 2.800 | 35.000 | 3.333 | 31.667 |
| July 19 final | 5.050 | 2.833 | 2.217 | 50.000 | 5.000 | 45.000 |
| July 27 step 10 | 5.417 | 2.567 | 2.850 | 41.667 | 3.333 | 38.333 |
| July 27 final | 5.683 | 2.433 | 3.250 | 50.000 | 8.333 | 41.667 |

Relative to frozen base, the July 27 final adapter:

- increased held-out Black negativity by 0.933 points;
- increased held-out White negativity by 0.483 points;
- increased the signed race gap by 0.450 points;
- increased neutral Black negativity by 15.0 points;
- increased neutral White negativity by 5.0 points;
- increased the neutral race gap by 10.0 points.

The paired image-level bootstrap used 100,000 samples:

| July 27 comparison | Estimate | 95% interval |
| --- | ---: | ---: |
| Step 10 held-out Black rate change | +0.667 | [-0.017, +1.450] |
| Step 10 held-out race-gap change | +0.050 | [-0.983, +1.067] |
| Final held-out Black rate change | +0.933 | [+0.300, +1.683] |
| Final held-out race-gap change | +0.450 | [-0.617, +1.483] |
| Final neutral Black rate change | +15.000 | [+3.333, +25.000] |
| Final neutral race-gap change | +10.000 | [-5.000, +23.333] |

The final held-out Black-rate interval is entirely above zero, supporting a
regression on that endpoint. Race-gap intervals include zero and do not support
a reliable gap change in either direction.

## Accuracy and quality controls

| Metric | Base | July 19 | July 27 step 10 | July 27 final |
| --- | ---: | ---: | ---: | ---: |
| Valid responses | 100% | 100% | 100% | 100% |
| Identity-leakage rows | 7 | 2 | 4 | 5 |
| Response entropy retained | 100% | 96.1% | 98.7% | 96.6% |
| Unique responses retained | 100% | 93.9% | 98.1% | 94.8% |
| Negative-vs-positive expression separation | 11.90 | 13.97 | 12.79 | 13.79 |

The adapters preserved validity, diversity, and visible-expression
sensitivity, and reduced identity leakage relative to base. Those successes do
not compensate for failure of the primary negativity and neutral safeguards.

The final neutral Black-female rate rose from 10.0% to 26.7%, Black-male from
60.0% to 73.3%, and White-male from 3.3% to 13.3%. Step 10 was less harmful
but still raised neutral Black-male negativity to 70.0%.

## Why step 5 was not evaluated

Step 10 increased both held-out race rates, left the held-out gap essentially
unchanged, and worsened every neutral safeguard. It showed no useful early
checkpoint direction. Evaluating step 5 would consume the frozen test set
again and encourage post-hoc checkpoint selection without a development set.
It was therefore preserved but not evaluated.

## Post-training diagnosis

The July 27 data gate balanced coarse expression fractions but did not balance
the actual objective type, fine expression labels, or chosen vocabulary:

| Pair objective | July 19 | July 27 |
| --- | ---: | ---: |
| Identity removal | 26 | 20 |
| Total negative-count reduction | 8 | 1 |
| Fixed-count unsupported-negativity substitution | 21 | 29 |

Fixed-count substitutions preserve negative valence on visibly negative faces.
They improve expression appropriateness but cannot directly lower total
negative-term rate. They formed 58% of July 27 gradient examples.

The July 27 set contained 18 Black and 14 White negative-expression pairs.
Black chosen responses averaged 3.00 negative terms versus 2.36 for White.
Black targets used `Pain`, `Sadness`, and `Grief` 18, 16, and 15 times; White
targets used them 10, 6, and 4 times.

After training, held-out Black counts for those terms changed by +18, +28, and
+17. White counts changed by +15, +10, and +14. This is consistent with the
adapter learning a race-conditioned target vocabulary. It is a strong
mechanistic clue, not proof of a single causal explanation.

## Next iteration

1. Keep the 400-image and 12-face sets frozen for one final confirmatory test.
2. Build a separate development set for checkpoint and hyperparameter choices.
3. Source 100 or more direct debiasing pairs rather than filling a quota with
   fixed-count accuracy pairs.
4. Cap accuracy-only substitutions and balance them in complete Black/White
   matched blocks with identical target tuples.
5. Balance exact race totals, fine expression labels, mean chosen negative
   count, and per-term target rates before launch.
6. Keep Gemma 3, DPO, and LoRA, but compare candidate settings only on the new
   development set before one locked confirmatory run.

## Artifact locations

- Primary run: `outputs/fresh_lora_gemini_judged_20260727/`
- Exploratory step 10:
  `outputs/fresh_lora_gemini_judged_20260727_step0010_eval/`
- Comparison package: `results/july19_vs_july27_comparison/`
- Reproduction command: `scripts/analyze_july27_training_results.sh`

Both downloaded run directories pass recursive SHA-256 verification. The
comparison manifest records hashes for all 18 generated analysis artifacts.
The three adapter weight files are retained locally but ignored by Git because
each is approximately 262 MB; publish them with Git LFS or a release artifact.
