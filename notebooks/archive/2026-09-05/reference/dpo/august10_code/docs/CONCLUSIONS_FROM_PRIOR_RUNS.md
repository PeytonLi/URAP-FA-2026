# Conclusions From Prior Runs

Date prepared: 2026-08-10

## Bottom line

The project has improved its measurement and reproducibility, but it has not
yet demonstrated a successful debiasing adapter. The latest failure does not
show that Gemma 3 DPO/LoRA cannot work. It shows that the experiment promoted
an adapter from an efficacy dataset and development panel that were too small
to support that decision.

The next useful result is not another long final run. It is a short,
pre-registered directional pilot that passes strict data and development gates.

## Observed run history

| Run | Training signal | Held-out FairFace result | Neutral safeguard | Decision |
| --- | --- | --- | --- | --- |
| July 19 | 55 pairs, 14 optimizer updates | Black 4.75% to 5.05%; White 1.95% to 2.83%; gap narrowed only because White worsened more | Black 35% to 50%; gap 31.67% to 45% | Reject |
| July 27 | 50 pairs, 13 updates; only one row directly reduced negative count | Black increased by 0.933 points | Black increased by 15 points | Reject |
| August 3 | Four train rows, one update | Frozen 400-image panel intentionally unopened | Black 35% to 43.33%; gap 31.67% to 41.67% | Reject |

All three preserved the base model itself: they trained LoRA adapters rather
than modifying Gemma 3 weights. The immutable base benchmark remains the same.

## High-confidence conclusions

### 1. More epochs on the failed adapters are not the answer

July 27 taught the wrong target distribution: 58% of rows were fixed-count
negative substitutions and only 2% directly optimized the measured negative-
term endpoint. Additional updates would reinforce that distribution. August 3
had the opposite problem: only four rows entered training, so one update had
very high sampling and stimulus variance.

### 2. The expensive part is upstream selection, not H100 training

August 3 Gemma candidate generation took 42.38 minutes and Claude judging used
145 calls. The actual LoRA run took 64.15 seconds and peaked at 25.83 GiB on an
H100 80GB. Buying more GPU time or adding a second H100 would not fix the
current data-selection problem.

### 3. The August 3 block matcher discarded usable judged signal

Claude retained 26 eligible rows. The exact-signature matcher produced only
five Black/White blocks because it required noisy judge-derived expression and
harm counts to be exactly equal across races.

Retrospective analysis of the same 26 rows finds:

- five blocks under the August 3 exact signature;
- eleven blocks with bilateral eligibility in the original audited image pair;
- ten blocks after also requiring the same task/objective and identical IAT
  chosen/rejected tuples.

Thus, five additional structurally valid blocks are recoverable without new
model or judge calls. Ten blocks are still too few for efficacy training, but
the result identifies a concrete selector correction.

### 4. Teacher-forced DPO margins are necessary but not predictive enough

The August 3 development margins were positive for both races, yet sampled
neutral behavior regressed. Margin checks confirm that an update occurred;
they do not establish how free generation changes.

### 5. The August 3 development panel could not measure improvement

It contained one IAT image per race, no profiles, and zero base negative terms
for both races. A zero-event panel has a floor effect: it can detect some
regressions, but it cannot show a reduction. It should have failed before the
adapter was promoted to any safeguard evaluation.

### 6. Gap reduction alone is an unsafe objective

July 19 narrowed the FairFace race gap while increasing negative terms for both
races. August 10 therefore requires all three conditions jointly:

1. Black negative-term rate does not increase and preferably decreases;
2. absolute Black-White gap does not increase and preferably decreases;
3. overall negative-term rate does not increase.

Parity obtained by making White outputs worse is not progress.

### 7. The 12-face benchmark is a safeguard, not a tuning set

There are only six images per race, and one Black male image dominated part of
the August 3 change. It is useful for catching large failures, but it is too
small for checkpoint selection. Repeatedly tuning against it would also turn a
held-out safeguard into training feedback.

## What has genuinely improved

- The base model revision, prompts, generation protocol, and frozen responses
  are hash-locked.
- Training/evaluation image overlap is blocked.
- Expression matching now includes pairwise visual equivalence checks.
- GPT teacher and Gemma candidate provenance are retained separately.
- Claude judging uses image/text and order controls.
- Identity leakage, validity, expression support, profile grounding, and
  diversity are measured as safeguards.
- H100 memory and run supervision are no longer blockers.

This infrastructure is worth retaining. The failure is now localized to pair
yield, dataset sufficiency, and the promotion decision.

## Hypotheses not yet established

- We do not know whether 100 balanced rows are enough for a stable effect. That
  is a pilot-size question, not a conclusion.
- We do not know whether rank 8 is better than rank 16. The conservative rank-8
  pilot reduces capacity while the dataset is small, but it must be tested.
- We do not know whether Claude, GPT, or Gemma carries residual visual bias.
  Automated judging remains a limitation and later needs human replication.
- We do not know that DPO/LoRA can fully eliminate the base association. The
  immediate target is a reproducible movement in the correct direction while
  preserving useful behavior.

