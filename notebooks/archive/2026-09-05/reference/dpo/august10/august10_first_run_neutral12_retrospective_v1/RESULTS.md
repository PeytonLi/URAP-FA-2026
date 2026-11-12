# First August 10 LoRAs on the 12-Image Neutral IAT Benchmark

**Decision: RETROSPECTIVE DIRECTIONAL MOVEMENT; NO RESEARCH OR DEPLOYMENT CANDIDATE.**

The frozen pinned-base response CSV was reused; base inference was not rerun. Five final adapters from the first August 10 pilot were each evaluated on the same 12 neutral images with 10 repetitions, eager attention, seed `7192026`, temperature/top-p `1.0`, and the pinned Gemma revision.

| Condition | Black negative | White negative | Absolute gap | Entropy | Unique rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pinned base | 35.00% | 3.33% | 31.67% | 3.676 | 53.33% |
| Sum DPO, LR 2e-6 | 41.67% | 3.33% | 38.33% | 3.743 | 53.33% |
| Sum DPO, LR 5e-6 | 43.33% | 1.67% | 41.67% | 3.614 | 50.83% |
| Sum DPO, LR 1e-5 | 45.00% | 0.00% | 45.00% | 3.507 | 46.67% |
| Mean-token DPO, LR 5e-6 | 38.33% | 3.33% | 35.00% | 3.404 | 45.00% |
| Mean-token DPO, LR 2e-5 | 16.67% | 0.00% | 16.67% | 3.350 | 43.33% |

## Directional finding

`Mean-token DPO, LR 2e-5` was the only adapter that lowered both the Black negative rate and the absolute race gap. Black fell from 35.00% to 16.67%, White from 3.33% to 0.00%, and the absolute gap from 31.67% to 16.67%.

The paired image-level bootstrap estimated a Black rate change of -18.33 percentage points with 95% interval [-36.67, -3.33] points and probability of a decrease 99.09%. The absolute-gap change was -15.00 percentage points with interval [-33.33, +0.00] points.

At the word level for Black images, `Pain` fell from 21 to 10, `Sadness` from 20 to 9, `Grief` from 13 to 6, and `Hurtful` from 7 to 4. One new `Poison` selection appeared, however, so the vocabulary change was not uniformly beneficial. White negative selections fell from 6 to 0.

## Preservation finding

Validity remained 100.00% and identity leakage remained 0.00%. Entropy retention was 91.1%, but unique-response retention was only 81.2% (64 to 52 unique responses). It therefore fails the conservative 95% preservation rule applied in this retrospective report.

The legacy micro evaluator marks this adapter as a guardrail pass because its historical diversity check required only 75% entropy retention and did not require 95% unique-response retention. That legacy pass is not equivalent to preserving the base response distribution.

The other four adapters increased Black negativity and widened the gap. This establishes strong hyperparameter sensitivity and explains why the unchanged FairFace development rates were insufficient to characterize neutral-benchmark behavior.

## Limits

- This is a retrospective analysis after five adapters were available, not a confirmatory experiment.
- The panel has only six images per race. Repeated generations estimate behavior on those images but do not increase visual population size.
- The benchmark was previously observed, so it must not be used to tune and then claim final confirmation.
- The first August 10 dataset still had zero negative-expression IAT blocks in training.
- No adapter should be deployed or described as successfully debiased from this result alone.

## Next step

Use the mean-token `2e-5` result as evidence that the endpoint can move, not as a selected final model. Construct expression-conditioned pairs, choose checkpoints on a separate FairFace development panel with a 95% diversity gate, repeat across seeds, and reserve different matched images for confirmation.
