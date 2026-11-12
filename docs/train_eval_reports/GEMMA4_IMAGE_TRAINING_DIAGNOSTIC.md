# Gemma 4 image-training diagnostic: not yet a working route

No model weights have been updated by these diagnostics. No checkpoint was promoted, and no development or 12-image benchmark photos were submitted. All diagnostic trainers have been deleted; native trainer listing returned empty after the fifth attempt.

## Observed evidence

Base model: `accounts/fireworks/models/gemma-4-26b-a4b-it`.

Dedicated LoRA shape: `accounts/fireworks/trainingShapes/gemma-4-26b-a4b-256k-b200-lora`, four NVIDIA B200 180GB accelerators. Rank 8 / alpha 16 model handles, no optimizer calls or inference deployment.

Official local processor: `google/gemma-4-26B-A4B-it`, pinned revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. SDK `fireworks-ai` 1.2.19 and `tinker` 0.23.0, v1 transport.

Six hash-verified historical **training** PNGs were selected solely to validate image transport: one complete Black/White matched block each for positive, neutral and negative expression. The full 54-image dataset is not silently reduced to these six, and they are not the newly supplied 12 benchmark JPEGs. Chosen and rejected continuations give 12 datums per forward batch. The unchanged IAT prompt, official non-thinking generation prefix, three-word continuation and normal turn terminator are preserved. Prompt/image/role positions have zero response loss weight.

| Check | Measured result | Conclusion |
| --- | --- | --- |
| Local processor + datum masks | 256 soft image positions, two explicit BOI/EOI positions, 464-position prompt; response weights sum to one | Local geometry/masks pass; not backend proof |
| Original image-bearing forward, attempt 4 | Each returned vector has two extra positions; e.g. 473 scores vs 471 input/target/weight positions. Vectors are not sparse. | Strict alignment fails; image-swap check was not reached |
| Reconstructed v1 JSON request | PNG checksum, declared image length and equal input/target/mask lengths retained | No dropped positions in this reconstruction; not a captured original HTTP request |
| Full-boundary image-chunk candidate, attempt 5 | Backend rejects: `expected_tokens mismatch: expected 258, got 256 from image` | Candidate is invalid; do not use it for training |

Receipts: [local geometry](../results/gemma4_local_vision_probe.json), [original remote vectors](../results/gemma4_remote_vision_probe_attempt4.json), [wire reconstruction](../results/gemma4_wire_alignment_check.json), [candidate rejection and cleanup](../results/gemma4_remote_vision_probe_attempt5.json).

Two earlier attempts failed local SDK configuration checks before provisioning; attempt 3 returned a forward but did not save vector lengths. Attempt 4 supplied those dimensions. Attempt 5 explicitly tested the boundary-owning hypothesis and contradicted it. Neither a successful HTTP return nor an image-preprocessor error establishes correct image-conditioned response scores.

## What still needs resolution

The provider's image preprocessor confirms 256 soft tokens for these PNGs. The unresolved issue is the two extra **returned score** positions under the original rendering, not a justified change to `expected_tokens`. Do not truncate/pad scores, shift response masks by guesswork, or reuse the rejected 258-position candidate. Obtain the model-specific output-coordinate contract or a supported renderer, then verify exact score/mask alignment and image conditioning before optimization.

The runner now includes an identical-input repeat control. A future passing image-conditioning diagnostic must change at least two response scores by more than both an absolute `1e-4` floor and ten times the largest identical-input repeat difference. It still never calls an optimizer. This control was not reached in either failed remote attempt, so no image-conditioning result is claimed.

This document can support a provider inquiry; no issue, email or external message has been sent. Expiring credentials and numerical raw outputs remain in the ignored private directory. Reports contain no API key or image payload.

## SFT/ORPO preparation, separate from remote validation

The local objectives now use response-token means followed by example means (SFT) or preference-pair means (ORPO), with initial ORPO coefficient 0.1. The inspected cookbook's ORPO implementation sums pair losses; using it unchanged beside mean-reduced SFT would change effective gradient scaling with batch size. The study's local mean reduction avoids that hidden difference and requires no additional server-side gradient normalization. This is a preparation choice, not a measured optimum or a completed remote training run.

Executed local gradient tests check zero prompt/image gradients, masked-NaN isolation, chosen/rejected gradient directions, batch-duplication invariance, and ORPO coefficient zero matching chosen-only SFT. All 27 local study tests pass. A Linux-only cookbook import was avoided by keeping the stable `log(1-exp(x))` arithmetic local rather than pretending Windows has the `resource` module. These tests do not prove remote optimizer, checkpoint, image consumption or benchmark correctness.
