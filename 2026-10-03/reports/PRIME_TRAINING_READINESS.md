# Prime Intellect execution: Gemma 4 SFT complete; ORPO retry pending

Current October 6, 04:00 UTC execution update: the user authorized a separate frozen
**676-train / 180-development resolved-target pilot**, with fixed category loss
weights. The native worker now includes SFT/ORPO for all three models and
matching native base/adapter benchmarks, with owned-pod recovery/cleanup and
credit/time guards. **Gemma 4 SFT completed 507 real updates out of 507 planned**,
with a hash-verified final checkpoint. Gemma 4 ORPO reached 160 updates but failed
with BrokenPipeError after the SSH monitor disconnected; it has no final checkpoint.
No trained-model evaluation is complete; the pod was terminated and ORPO will be
rerun from the untouched base with the monitor fix.

Launch history: a real launch attempt
at 20:21 UTC found no single-GPU H100/H200 offer and stopped before provisioning,
key registration or image upload. A subsequent offer allowed one owned H100
to be provisioned at 20:28 UTC, at $4.29/hour. Setup failed before training on a
CUDA-driver mismatch; termination was independently verified. The corrected
worker pins Python 3.11.15, uses official Torch 2.11 CUDA 12.8 wheels and repeats
architecture checks before loading study weights. All 92 local checks pass on
the corresponding CPU runtime. A second pod timed out before SSH and was
terminated; its absence is verified. The third H100 ran at $4.29/hour and was
terminated after artifact recovery. Wallet credits were $94.0857 before that
launch and $79.2125 after cleanup. Remote checks verified Python
3.11.15, Torch 2.11.0+cu128, BF16 support and 79.19 GiB GPU memory.
[Current pilot, evidence and limitations](BALANCED_PILOT.md). The full-cohort gate
remains unchanged; earlier full-cohort-only statements below are historical.

Checked October 3, 2026 (training catalog: October 4 at 03:16 UTC; draft receipts subsequently through 04:07 UTC).
October 5 read-only refresh at 09:23 UTC: all six catalog queries succeeded, with unchanged model eligibility and no single-GPU H100/H200/B200 offers. [Timestamped receipt](../results/prime/catalogs_20261005T092316.045306_0000.json). A supplementary check at 09:24 UTC found four entries in the unfiltered single-GPU catalog, labeled A10 24GB, A100 40GB or CPU NODE, and zero A100 80GB offers. [Supplementary capacity receipt](../results/prime/capacity_20261005T092452.996532_0000.json). These are returned offers, not reservations, a market-wide capacity claim or evidence that a smaller/quantized experiment matches the intended full study. No hourly total or six-run budget is claimed verified and no resource was provisioned. The previous account balance below is historical, not freshly checked.

Direct photo inspection is complete for all 1,920 train/development records. The user subsequently authorized AI self-approval: 856 resolved proposals are AI-approved, 1,064 neutral/ambiguous targets remain blank, and human approvals remain zero. The full training file is still absent and the neutral-target policy needs a user decision. Self-approval is not independent validation.
The user authorized Prime Lab setup, then draft generation for review, and supplied
$100 of Prime credits. A $100 total cap and $20 drafting sub-cap are recorded.
Paid external-provider Prime gateway inference generated 1,920 image-specific
drafts, with zero human approvals. No Prime GPU resource was provisioned and no
model weights were updated. Unlimited Fireworks spending remains separate.
[Review instructions, provenance and quality cautions](TARGET_DRAFT_REVIEW.md).

## Measured progress

| Milestone | Actual result |
| --- | --- |
| Prime Lab setup | CLI 0.9.2; isolated Python 3.11.15 / Verifiers 0.2.0; all 17 workspace checks pass |
| Curated images | All 2,400 source/export hashes verified; original 1,440/480/480 splits preserved |
| Reviewed targets | 856 AI self-approvals, zero human approvals; 1,064 unresolved; no complete training export |
| Proposed targets | 1,920 image-bearing drafts ready for human review; format/receipt checks pass, visual correctness not established |
| Prime budget | $100 total cap, $20 drafting sub-cap; draft charge estimate $5.21, latest account balance $94.25 |
| Gemma 4 local processor | Explicit non-thinking continuation passes on 96 training images / 192 fixture continuations |
| Qwen local processors | Each native assistant template passes on the same 96 training images / 192 fixtures; no pretrained full-size gradients checked |
| Native optimizer candidate | All three tiny random architecture fixtures pass SFT and ORPO updates, frozen-base equality, image control and adapter save/reload; not study training |
| Real SFT / ORPO | Zero optimizer updates; no adapters, checkpoints or trained-model evaluations |
| Existing base evaluations | Three Fireworks models, 600 benchmark slots each; not Prime-hosted baselines |

The 96-image check uses eight complete published matching sets: one per dataset
and expression stratum, comprising 56 FairFace and 40 UTKFace training images.
It does not create a smaller training cohort, change splits, inspect test photos
for target construction or replace the original 12-image benchmark.

[The local three-model input report](LOCAL_MODEL_INPUT_CHECKS.md) records the
separate Qwen processor revisions and receipts. These passes do not change the
unverified hosted algorithm/modality status in the catalog table below.

## Live Prime catalogs: eligibility is not verified

[Read-only catalog receipt](../results/prime_preflight.json) records all six
successful queries and points to an immutable timestamped copy. The dedicated
cached-model endpoint was queried explicitly; its empty result was not inferred
from a suppressed request error.

| Intended model | Shared training catalog | Dedicated cached-model catalog | Image-bearing SFT / ORPO verified |
| --- | --- | --- | --- |
| google/gemma-4-26B-A4B-it | Not listed | Not listed | Neither |
| Qwen/Qwen3-VL-8B-Instruct | Not listed | Not listed | Neither |
| Qwen/Qwen3.5-9B | Listed | Not listed | Neither |

The shared catalog has 12 models; the dedicated cached-model catalog has zero.
Listing in the shared catalog does not prove support for offline reviewed-answer
SFT, ORPO or images. The latest one-GPU H100/H200/B200 searches returned no
matching resources. An earlier H100 query returned quotes, so neither prices
nor available capacity should be treated as fixed or reserved.

Current official compute documentation also separates base GPU quotes from adjustable resource costs and possible prepaid hours. [Availability pricing](https://docs.primeintellect.ai/api-reference/check-gpu-availability) and [provisioning/pricing rules](https://docs.primeintellect.ai/api-reference/provision-gpu). A catalog minimum is not an all-in training/evaluation cost. Active pods consume credits by the minute, and termination loses instance-local data; preserve verified artifacts before deleting owned compute. [Billing and data persistence FAQ](https://docs.primeintellect.ai/faq). Sandbox idle-timeout settings must not be assumed to apply to self-managed GPU pods. Keep the $100 cap and no-idle-compute rule; do not enable auto-top-up or allocate persistent disks as an implicit workaround.

Prime's documented dedicated hosted SFT is a closed beta requiring account
enablement and a model cached on the owning cluster. It is trainer-only and
uses dataset files on a named volume. Shared LoRA model listings are not valid
evidence of eligibility for that route. Its displayed zero-token / $0.00 SFT
usage fields are a reporting limitation, not proof that training is free.
[Official hosted SFT documentation](https://docs.primeintellect.ai/hosted-training/hosted-sft.md).

The inspected Prime RL algorithms include online teacher-distillation SFT, not
evidence that the required fixed-target multimodal SFT or offline ORPO is ready.
Do not substitute GRPO, distillation, OPD or another objective for the requested
two methods. [Official algorithm documentation](https://docs.primeintellect.ai/prime-rl/algorithms).

A merged Gemma 4 VLM support PR exists, but a historical merge alone does not
establish support in the present code. At inspected Prime RL main commit
`d9adc1e867db9f8e27438fdd65f99e8dc7a2448a`, `VLM_REGISTRY` in
`src/prime_rl/utils/vlm.py` contains `qwen3_5`, `qwen3_5_moe` and `qwen4_exp`,
not `gemma4` or `qwen3_vl`. The inspected file's SHA-256 is
`7198bd5ac4ed74d3d2f65e400aa0108fdda19c5e414a35927d73dec66678dee1`.
This is evidence about that revision, not a claim that every Prime backend or
future version lacks these models.
[Pinned current registry](https://github.com/PrimeIntellect-ai/prime-rl/blob/d9adc1e867db9f8e27438fdd65f99e8dc7a2448a/src/prime_rl/utils/vlm.py),
[historical merged PR](https://github.com/PrimeIntellect-ai/prime-rl/pull/2844).

## Gemma 4 processor verification on the active pool

Pinned processor: `google/gemma-4-26B-A4B-it` at
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`.
Executed locally using the existing isolated runtime: Transformers 5.14.0,
PyTorch 2.13.0, torchvision 0.28.0 and Pillow 11.3.0. CUDA was unavailable.
No model weights were loaded.

The standard complete-assistant template failed the exact-prefix check on all
96 images. Its assistant response omits the empty thought-channel marker that
the official non-thinking generation prefix includes. The strict check rejected
that mismatch rather than silently masking a different training prompt.
[Preserved native-template failure](../results/curated_gemma4_vision_check.json).

The separate explicit continuation format uses the official rendered
non-thinking generation prefix, response text, and verified native turn
terminator. The native processor constructs the full image/text inputs; no
image spans, multimodal types or score mappings are guessed. All 192 fixture
continuations passed exact token-prefix equality, unchanged image tensors and
image-position fields, and response-only label masking.
[Successful continuation receipt](../results/curated_gemma4_continuation_check.json).

`Happy, Joyful, Smiling` and `Pain, Grief, Sadness` are constant grammar/masking
fixtures applied to every diagnostic image, **not chosen/rejected labels**.
They have not been assigned as targets, reviewed or exported for training.
The result verifies local formatting only: it does not verify model gradients,
image conditioning, remote GPU memory, Fireworks alignment or an actual Prime
training backend. The older Fireworks score-length failure remains unresolved.

## Native optimizer candidate: fixtures are not study training

`native_train.py` now provides a single-GPU BF16 candidate for all three intended models, reusing the pinned native processors and existing SFT/ORPO losses. No dependency was installed. It adapts linear projections in language self-attention and Qwen3.5 linear attention only: rank 8, alpha 16, dropout zero. Vision, embeddings, routers and MoE expert tensors stay frozen. This is an attention-only recipe, not all-linear or expert tuning. [PEFT's module versus expert-parameter distinction](https://huggingface.co/docs/peft/main/en/developer_guides/lora).

Native image tensors, grids/positions and multimodal token types are retained. Indexed next-token logits cover only the response predictors, including the native turn terminator; prompt/image tokens receive no target loss. The inspected native APIs support these fields and indexed `logits_to_keep`. [Gemma 4](https://huggingface.co/docs/transformers/main/model_doc/gemma4), [Qwen3-VL](https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl), [Qwen3.5](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5).

Executed checks use tiny, randomly initialized instances of all three actual architecture classes, synthetic pixel tensors and token IDs—not source photos, study targets or pretrained weights. Gemma includes MoE and alternate global attention; Qwen3.5 includes linear and full attention. Both SFT and ORPO passed finite nonzero adapter gradients and changed adapter weights, with every frozen base parameter byte-equal before/after. Adapter save/reload preserved scores. Response-only loss matched each architecture's native shifted-label loss; changing only pixels changed scores. These are six isolated fixture updates and **zero requested-study updates**. Versions: Transformers 5.14.0, Torch 2.13.0, PEFT 0.21.0, Accelerate 1.15.0. CUDA is unavailable. Qwen3.5 uses the Torch fallback, not an optimized GPU kernel.

All 83 current local tests pass. Focused repeat:

```powershell
rtk python 2026-10-03/tests/test_native_train.py
```

The real candidate requires complete approved 1,440-train/480-development targets before weight loading, one visible BF16 CUDA GPU and an explicit run-time ceiling. The actual data-check command refused the incomplete cohort; no pretrained study weights, run directories or paid resources were created. It does not provision or terminate rented compute: the run-time ceiling is not a billing cap.

The recipe uses AdamW at `1e-5`, zero weight decay, logical batch four via streamed microbatches, three epochs, seed zero and ORPO coefficient 0.1: 1,080 updates per complete model/method run. Development receives no gradient updates. Epoch adapters/optimizer states are saved privately with checksums and input/approval provenance; stopped runs remain incomplete. Full-size memory/performance, bounded paid execution, matching native base/adapter benchmarks and measured trained-model outcomes remain unverified. The separate Fireworks alignment failure is unchanged.

## Recommended next route and requirements

A self-managed Prime GPU using the native model/processor plus independently
checked SFT/ORPO response losses is a candidate for the required study. This is
an inference from the catalog limitations, not an executed or verified training
route. The Lab workspace remains useful for reproducible dependencies; its
starter RL/GEPA configs are not this experiment.

Before paid GPU training:

1. Keep total Prime spending within the recorded $100 cap, accounting for
   drafting and account balance. Confirm capacity/pricing at launch time.
   Do not provision an idle GPU while target review is pending.
2. Review/edit the proposed answers, especially the neutral positive-word prior.
   Approve all 1,920 train/development records and rerun the data audit.
   Never treat teacher proposals, demographic or classifier metadata as ground truth.
3. Verify actual image-conditioned forward/backward and response masking on a
   bounded training-only smoke test, with finite losses/gradients and cleanup.
   Verify each comparison model separately; a Gemma processor result does not
   validate either Qwen model.
4. If moving training/evaluation off Fireworks, run an untouched base on that
   same stack, pinned to the same weight/processor revision as its new adapters.
   Fireworks-base versus Prime-adapter differences alone cannot isolate a
   fine-tuning effect. Preserve the original 12-image/50-repeat protocol.
5. Train separate SFT and ORPO adapters from each untouched base, use development
   only for selection, then evaluate frozen checkpoints and produce measured
   per-model reports and all-61-word comparison figures. Do not invent absent
   trained-model results or treat setup as completion of the overall goal.

To repeat only the read-only catalogs:

```powershell
rtk python 2026-10-03/prime_preflight.py
```

To repeat only the processor check, using the existing isolated runtime:

```powershell
rtk proxy 2026-10-03/runtime/train-venv/Scripts/python.exe 2026-10-03/curated_vision_check.py --continuation
```

Both commands avoid uploads, model-weight loading, optimizer updates and paid
resource creation. Source and generated artifacts remain local on `peyton`;
nothing has been pushed.
