# August 10 Negative-Expression Pilot Results

**Decision: STOP. No checkpoint is eligible for seed replication or confirmatory evaluation.**

The immutable base is `google/gemma-3-12b-it` revision `96b6f1eccf38110c56df3a15bffe176da04bfd80`. Every candidate started as a fresh rank-8 LoRA; no base weights or prior adapters were updated.

## Frozen data

- Historical v1 dataset gate: **PASS**.
- 100 rows, 50 Black and 50 White, 80 train and 20 development.
- 70 grounded profiles and 30 controlled three-word IAT rows.
- Seven negative-expression blocks: five development and two train.
- Post-run v2 gate: **FAIL** because training contains zero negative-expression IAT blocks; the new minimum is three.
- v2 failed checks: `train_negative_iat_blocks`.

## Development results

The negative-expression challenge base selected 12/45 negative terms for Black images (26.67%) and 15/45 for White images (33.33%), an absolute gap of 6.67 points. This challenge is not the neutral 12-face confirmatory benchmark.

| Checkpoint | Black | White | Overall | Abs. gap | Leakage | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Sum 2e-6, step 1 | 26.67% | 33.33% | 30.00% | 6.67% | 0.00% | `stop_no_joint_directional_improvement` |
| Sum 2e-6, step 5 | 26.67% | 33.33% | 30.00% | 6.67% | 0.00% | `stop_no_joint_directional_improvement` |
| Sum 2e-6, step 10 | 26.67% | 33.33% | 30.00% | 6.67% | 0.00% | `stop_no_joint_directional_improvement` |
| Sum 5e-6, step 5 | 26.67% | 33.33% | 30.00% | 6.67% | 0.00% | `stop_no_joint_directional_improvement` |
| Sum 5e-6, step 10 | 26.67% | 33.33% | 30.00% | 6.67% | 0.00% | `stop_no_joint_directional_improvement` |
| Sum 1e-5, step 10 | 26.67% | 33.33% | 30.00% | 6.67% | 3.33% | `stop_directional_regression` |
| Mean 5e-6, step 10 | 26.67% | 33.33% | 30.00% | 6.67% | 0.00% | `stop_no_joint_directional_improvement` |
| Mean 2e-5, step 10 | 26.67% | 33.33% | 30.00% | 6.67% | 0.00% | `stop_no_joint_directional_improvement` |

Every evaluated checkpoint left Black, White, overall, and absolute-gap rates unchanged. The summed `1e-5` checkpoint also introduced one identity-leakage response (3.33%). Profile validity, subject reference, uniqueness, and bigram diversity otherwise passed their safeguards.

Mean-token DPO removed the extreme profile-versus-IAT margin imbalance, but it did not change the primary generation rates. This narrows the next bottleneck to training coverage and task weighting rather than another learning-rate sweep.

## Claude audit

The negative-expression expansion used 219 calls and an estimated $3.367927 of the authorized $10 session budget. Cumulative August 10 auditing used 955 calls and an estimated $14.639742. Provider account balance remains an estimate.

## Next iteration

1. Add at least six negative-expression IAT blocks so at least three can be frozen into train and three into development without overlap.
2. Keep mean-token scoring available, but prespecify a task-balanced DPO weight or matched accumulation schedule before training.
3. Re-run the base development challenge only if its image panel changes; otherwise reuse the hashed base responses.
4. Require strict primary movement before three-seed replication. Keep the 400-image FairFace and 12-face neutral confirmatory panels closed until replication passes.

See `checkpoint_metrics.csv`, `training_run_metrics.csv`, and `research_summary.json` for machine-readable values.
