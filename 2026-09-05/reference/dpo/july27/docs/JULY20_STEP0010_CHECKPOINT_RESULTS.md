# July 20 Step 10 Checkpoint Results

Status: complete, primary acceptance failed

Evaluation date: July 27, 2026

This is a checkpoint-only evaluation of the July 20 LoRA at optimizer update
10. No model or adapter weights were trained or modified during this run.

## Held-out FairFace

| Metric | Frozen base | Step 10 | Change |
| --- | ---: | ---: | ---: |
| Black negative-term rate | 4.750% | 4.533% | -0.217 points |
| White negative-term rate | 1.950% | 1.817% | -0.133 points |
| Black-minus-White gap | +2.800 points | +2.717 points | -0.083 points |
| Identity-leakage response rate | 0.175% | 0.175% | unchanged |
| Response entropy | 5.383 | 5.265 | -2.18% |
| Unique responses | 735 | 725 | -1.36% |
| Valid response rate | 100% | 100% | unchanged |

The paired image-level 95% interval for the race-gap change was
`[-0.883, +0.717]` percentage points. The estimated probability of a gap
reduction was 58.18%. This is a negligible and statistically unresolved
change.

## Neutral benchmark

| Metric | Frozen base | Step 10 |
| --- | ---: | ---: |
| Black negative-term rate | 35.00% | 35.00% |
| White negative-term rate | 3.33% | 3.33% |
| Black-minus-White gap | +31.67 points | +31.67 points |
| Black-male negative-term rate | 60.00% | 66.67% |

The aggregate neutral rates and gap were unchanged, while the most problematic
Black-male intersection worsened by 6.67 points. The checkpoint therefore
fails the neutral Black-male acceptance criterion.

## Interpretation

Step 10 is less harmful than the July 20 final adapter, which increased
negative rates for both races and widened the neutral gap to 45 points. It is
still not a successful debiasing checkpoint: the held-out improvement is
minimal, uncertainty strongly includes no effect, identity leakage is
unchanged, and the neutral Black-male result regresses.

The complete 32-file result bundle is in
`results/july20_step0010_checkpoint_eval/`. All files copied from the RunPod
were verified byte-for-byte by SHA-256.
