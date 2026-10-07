# IAT-only SFT / ORPO study

Completed authorized pilot, October 6: six independent SFT/ORPO runs and all
nine native benchmarks are verified (3,042 optimizer updates / 5,400 slots).
Full reports for Gemma 4, Qwen3-VL and Qwen3.5 include training inputs, measured
base/adapter results, uncertainty, per-image data and 21 figures.
[Read the full comparison](reports/native_balanced/COMPARISON.md).
The adapted responses concentrated strongly on Angry, Annoy and Negative;
smaller gaps did not establish improved grounding or fairness.
The 676-training/180-development pilot is complete, not the full 1,440-image cohort.
The archive's 198 files are checksum-verified, and the owned GPU is independently
verified terminated. Prime wallet: $39.9395 at 18:25 UTC.
[Recovery and cleanup](results/prime_native_attempt11_cleanup.json).

## Automatic private Hugging Face backups

New native SFT/ORPO runs now require a Hugging Face write-enabled login and
automatically create **one private model repo per run**, named
`YOUR_ACCOUNT/urap-MODEL-METHOD-TIMESTAMP`. Different models, methods and
reruns never share or overwrite a repo. Epochs within a run use the same repo;
each verified upload records an immutable commit revision in the local
`hub_backup.json`. The repo's default revision contains the most recently
backed-up checkpoint; prior checkpoints remain available at their commit IDs.

Every epoch is saved locally first, then synchronously uploaded and checksum
verified. Failure stops the run with its local checkpoint retained; it is not
reported as a successful cloud backup. Interrupted checkpoints are labeled
incomplete. This adds storage, not a new training experiment, and does not
change the completed pilot's historical reports.

Uploads contain LoRA adapter weights/config, processor/tokenizer files, an
experimental model card and a selected training summary (base revision,
settings, dataset hashes, counts and development metrics). They do **not**
contain base-model weights, face images, raw target records, credentials, logs
or optimizer state. A LoRA adapter is loaded alongside its exact pinned base
model; it is not a standalone copy of Gemma/Qwen. Local optimizer checkpoints
remain available for recovery. Earlier epochs are accessible through their
recorded Hub commit revisions.

One-time local sign-in using the browser, a **write** token, or a fine-grained
token that permits creating/uploading private repos in your own account.
Browser OAuth logins do not report an access-token role; the server checks
repository creation permissions and write access before each upload:

```powershell
rtk proxy 2026-10-03/runtime/torch211-check-venv/Scripts/hf.exe auth login
rtk proxy 2026-10-03/runtime/torch211-check-venv/Scripts/python.exe 2026-10-03/hub_backup.py auth
```

Use the private login prompt, not chat or a source file. Cached login or
`HF_TOKEN` is supported. For Prime, the launcher checks authentication before
provisioning and transfers the token over SSH stdin into an owner-only file
outside the source and recovery archives. The Hugging Face client (1.33.0) is
installed in this machine's Prime launcher environment and pinned in the remote
training setup. No credentials are included in the input bundle.
The `peytonali` browser login is verified. All six existing final (`epoch_3`)
adapters are now uploaded to distinct private repos; every uploaded file's
checksum and immutable revision are verified. No GPU was provisioned and the
original training receipts/reports were not rewritten.
[Verified repositories and revisions](results/hugging_face_backup_setup.json).

An existing retained run can be backed up without retraining or altering its
historical `run.json`:

```powershell
rtk proxy 2026-10-03/runtime/torch211-check-venv/Scripts/python.exe 2026-10-03/hub_backup.py upload --run "ABSOLUTE_PATH_TO_RETAINED_RUN"
```

The default checkpoint is `epoch_3`; `--checkpoint epoch_1` can back up a
different retained checkpoint. The uploader never merges the adapter into a
base model or makes the repository public.

[Hub private-repo authentication](https://huggingface.co/docs/huggingface_hub/quick-start)
and [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint)
describe the underlying storage/loading behavior. Local tests mock Hub writes;
the six completed uploads were also verified against the live Hub.

Recovery update, October 6 at 15:44 UTC: five training runs and six
native benchmarks are locally verified. All 204 recovered files match their
checksums, and the old owned GPU is terminated with independent absence/history
checks. The retry reuses only complete verified outputs.
Qwen3.5 ORPO stopped at 497/507 updates under the two-hour runtime ceiling.
It will be retried with three hours allowed and the same scientific recipe.
The full six-training/nine-benchmark comparison remains incomplete.
[Current execution and limitations](reports/BALANCED_PILOT.md).
The dated updates below are historical, not the current execution state.

October 6, 00:33 UTC execution update: the user authorized the resolved-target,
class-weighted pilot: **676 training / 180 development images**, preserving
their original splits and excluding all 1,064 unresolved targets. The pilot is
frozen, and SFT/ORPO use equal total positive/negative loss weight without
discarding positive images. Native matched base/adapter evaluation reuses the
original 600-slot protocol. The first H100 setup failed on a CUDA-driver mismatch;
the second never became reachable within 15 minutes. Both owned pods were
terminated and their absence verified. The new H100 is running at **$4.29/hour**.
Gemma 4 SFT has **43 verified real optimizer updates out of 507 planned**, with
finite gradients and a passed pretrained image-control check. No full condition
or trained-model benchmark is complete yet; independent ORPO follows SFT.
The worker uses Python **3.11.15** and official Torch **2.11.0+cu128**. All **92**
local suite checks pass; an additional temporary nine-condition report rendering
check also passes. Test fixtures are not study results.
The current launch status is recorded in the linked pilot report.
[Current pilot, settings, limitations and launch evidence](reports/BALANCED_PILOT.md).
Earlier full-cohort-only readiness statements below describe the preceding stage;
the full manifest/export gate itself remains unchanged.

Status as of October 3, 2026: **three new base-model evaluations complete; SFT and ORPO have not run.**
October 5 approval update: the user explicitly authorized AI self-approval. All 1,920 train/development photos have been directly inspected; 856 resolved proposals are now AI-approved and 1,064 neutral/ambiguous targets remain blank. Human approvals remain zero. [Direct-pass progress](results/direct_review_progress.json) separates those counts. The existing reviewer displays AI versus human provenance, and later human decisions override AI decisions without rewriting their evidence. Existing Haiku proposals and direct assessments remain unchanged. Full-cohort export is still refused; no training has started. The remaining neutral-target policy and real image-conditioned backend still need resolution. [Current review instructions](reports/TARGET_DRAFT_REVIEW.md).
October 5 reviewer update: [the local photo reviewer](http://127.0.0.1:8879/) now shows one larger photo at a time, with word pickers, visual evidence, uncertainty flags, save-and-next actions, remembered position and unsaved-edit protection. Saving an unsure decision never approves or drops the photo. Refresh an already-open page to load the new interface. Browser layout verification remains unavailable; isolated logic and approval safeguards pass.
All changes are local on branch `peyton`. Nothing has been pushed. The user authorized Fireworks spending without a fixed limit and reuse of existing IAT targets only after an audit. No RunPod or Prime Intellect resources have been provisioned.

Prime Intellect is now also authorized as a provider. [The isolated Prime Lab workspace](prime-lab/README.md) was initialized with `prime lab setup`, verified against the existing login, and passes all 17 workspace checks. The user subsequently authorized image-specific drafts for human review and supplied $100 of Prime credits, treated as a total cap with a $20 drafting sub-cap. Paid Prime gateway inference generated 1,920 drafts; no GPU resources, optimizer updates or trained checkpoints were created. [Review the drafts, costs and quality cautions](reports/TARGET_DRAFT_REVIEW.md). Target approval and a verified image-bearing SFT/ORPO route remain prerequisites.

[Prime training-readiness checks](reports/PRIME_TRAINING_READINESS.md) now record the live shared/cached model catalogs and local Gemma 4 checks on 96 images from the active pool. The explicit non-thinking continuation preserves the exact inference prefix and all native image fields; the stock complete-assistant template does not. These are processor-only checks with grammar fixtures, not approved labels, real gradients or a working remote training backend. The dedicated cached-model catalog is empty; a self-managed GPU route remains a candidate, not a verified run.

[Local input checks now cover all three intended models](reports/LOCAL_MODEL_INPUT_CHECKS.md). Both Qwen native assistant templates pass on the same 96 training images / 192 fixture continuations per model, retaining image grids, pixels, exact prompts and response-only labels. The fixtures are not training targets; no weight loading, optimizer updates or paid resources were involved.

October 5 native-training update: an executable single-GPU SFT/ORPO candidate now reuses those processors and existing losses. All six tiny random architecture checks pass finite adapter updates, frozen-base equality, response-mask/native-loss equivalence, image sensitivity and adapter save/reload. These are isolated CPU fixtures, not pretrained study training or full-size GPU validation. All 83 current local tests pass. The real full-cohort entry point still refuses incomplete targets before loading weights. [Scope, settings and remaining requirements](reports/PRIME_TRAINING_READINESS.md#native-optimizer-candidate-fixtures-are-not-study-training).

The user approved switching to the September 29 curated pool on October 3. [config/active_dataset.json](config/active_dataset.json) now pins its source commit and CSV checksum. Its unchanged allocations are **1,440 training, 480 development and 480 test images**, across FairFace and UTKFace. The original 12 photos remain benchmark-only. The 54-image plan and its missing originals are superseded, not silently combined with the new pool. [Current dataset plan](reports/CURATED_DATASET_PLAN.md) and [fine-tuning status](reports/FINETUNING_STATUS.md) distinguish image reconstruction, reviewed target availability and backend verification. All dedicated diagnostic trainers have been deleted.

[Read the measured base comparison and per-model reports](reports/BASE_COMPARISON.md). Each model has 600 benchmark slots, for 1,800 total. Temporary dedicated deployments were needed because serverless returned HTTP 404 for all three models. All three study deployments were deleted and their absence was verified in [the cleanup receipt](results/deployments/cleanup_verified.json). Dedicated GPU charges were incurred; legacy zero API cost fields do not report those charges.

## What will actually be trained

SFT receives **one image + the existing 61-word prompt + its chosen three-word response**. Only the assistant response contributes to the loss; image and prompt tokens are context, not target text.

ORPO receives **the same image and prompt + chosen and rejected three-word responses**. Its loss combines learning the chosen response with an odds-ratio preference term. It does not require a frozen reference model, and it is not a reuse of Elias's old DPO adapter.

Neither method receives profiles, biography text, race labels, gender labels, expression metadata, or the 12 benchmark images. Metadata is used for auditing and splitting only. Start each method from the same untouched instruction-tuned base model; LoRA trains new adapter parameters rather than every base-model parameter.

## Data audit and provenance

Active source: [../2026-09-29/data/selected_images.csv](../2026-09-29/data/selected_images.csv), unchanged from upstream commit `e4ac7a506f97e855f78ce79cc9d42b508193f76c`. SHA-256: `7b08c31122f624e733ac279b87882c9138a05ea4c9dc7a16596105a5d4637d13`.

| Dataset | Training | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| FairFace | 840 | 280 | 280 | 1,400 |
| UTKFace | 600 | 200 | 200 | 1,000 |
| Total | 1,440 | 480 | 480 | 2,400 |

Use the published `split`, not the publisher's `source_split`. Matching sets remain intact. No demographic group, uncertain-expression row or test image is silently reassigned or dropped. The 480 new test images remain reserved; their existence does not automatically activate an additional benchmark or permit preference generation on them.

The CSV contains **no prompts, chosen/rejected answers or IAT targets**. Its expression fields are frozen classifier estimates, not training answers or human ground truth. New reviewed image-specific target records are required for all 1,440 train and 480 development rows; old 54-image templates and the supplied profile CSV are not reassigned. Targets are separate from the unchanged upstream manifest. Preparation and export refuse missing, unapproved, mismatched or test-image targets.

[data/manifest.json](data/manifest.json) and [results/curated_data_audit.json](results/curated_data_audit.json) record actual image and target readiness. All 2,400 images have been reconstructed and their source/export hashes verified; the full upstream image/balance validator passes. There are 856 user-authorized AI self-approvals, zero human approvals and no complete approved target file, so export/training readiness remains false. [Draft consistency audit](results/target_draft_audit.json). Self-approval is not independent validation or ground truth. Splits are image-disjoint, not verified person-disjoint. Demographic annotations are publisher metadata and never enter model input.

Historical [data/pairs.csv](data/pairs.csv), [data/missing_images.csv](data/missing_images.csv), [data/legacy_54_manifest.json](data/legacy_54_manifest.json), image-recovery receipts and historical label audits are retained for reference only. The old 11 missing images no longer block this approved pool. The 12 original benchmark JPEGs in `C:/Users/lipey/Downloads/12_images/race` stay outside training and target generation.

## Fireworks model checks

Native Fireworks CLI sign-in successfully accessed the live model catalog and training shapes. CLI sign-in is separate from the API key required by inference/Training SDK calls. A temporary, expiring study API key was created and retained only in the ignored private directory; no secret values are written into reports or Git-tracked files.

| Candidate | Live catalog evidence | Remaining check |
| --- | --- | --- |
| Gemma 4 26B A4B IT | Image inference, supervised LoRA enabled; dedicated B200 LoRA shape exists; local image-bearing datums and response masks pass | Remote forward returned two more scores than input/mask positions; alignment and image conditioning remain unverified |
| Qwen3-VL 8B Instruct | Image inference, supervised LoRA enabled; explicit `qwen3-vl-8b-vision-lora` shape; native HF processor/masks pass on 96 active training images | Provider-specific image datum conversion and bounded remote validation |
| Qwen3.5 9B | Image inference, supervised LoRA enabled; dedicated LoRA shapes exist; native HF processor/masks pass on 96 active training images | Confirm selected shape accepts image datums and verify provider-specific processing |

These are the intended three models for **base vs SFT vs ORPO**, subject to the remaining modality checks. Qwen2.5-VL 7B is a fallback. Fireworks's inspected Gemma 3 12B entry identifies itself as `gemma3_text`, so it is not assumed to reproduce the local Gemma 3 vision baseline. The requested E4B model ID was unavailable in the checked catalog; it is not silently substituted.

Full sanitized catalog receipts: [results/preflight.json](results/preflight.json).

Inspected official cookbook revision: `64122feb986449eddd7b377df22e99e0d06f35bd`. Its `training/renderer/gemma4.py` describes the text-only path and discards the image processor; `renderer_supports_images` does not include `gemma4`. Image inference support plus a LoRA flag does **not** prove this training path supports images. This finding is not a claim that no other Gemma 4 multimodal route can work.

### Gemma 4 modality validation

The official `google/gemma-4-26B-A4B-it` processor/tokenizer is publicly accessible without an HF token. A local processor check pinned revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` and rendered six already selected, hash-verified **training** photos (one complete matched block per expression class). It keeps real PNG `ImageChunk` inputs, preserves BOI/EOI and the official non-thinking generation prefix, and replaces exactly the processor's image-placeholder span. These six 224-pixel photos each produce 256 image positions under the processor's default 280-token maximum; hardcoding 280 placeholders would be wrong for these images.

Chosen and rejected three-word continuations retain the normal turn terminator. Prompt, image, role and channel tokens have zero loss; response weights sum to one per example after the next-token shift. This explicit non-thinking continuation format is checked for token-boundary stability; it is not claimed to be the stock complete-conversation renderer. [Local check receipt](results/gemma4_local_vision_probe.json) records versions, spans, response lengths and hashes. The processor revision is not asserted to be Fireworks's immutable weight revision.

`alignment_check.py` reconstructed the historical datums through the installed v1 JSON converter. Its [wire check](results/gemma4_wire_alignment_check.json) retained image checksums and equal input/target/weight lengths. The non-default boundary-owning candidate was then rejected by the actual fifth remote attempt: `expected 258, got 256 from image`. It is not a valid training fix. All paid probes used the superseded historical cohort and are retained as diagnostics, not validation of the new curated pool. The old paid runner now refuses to run while the curated pool is active. [Detailed diagnostic](reports/GEMMA4_IMAGE_TRAINING_DIAGNOSTIC.md).

`remote_vision_probe.py` checks backend acceptance and image conditioning using forward-only chosen/rejected scores and an image-swap control with identical text token IDs. It never calls an optimizer, promotes a checkpoint, or uses development/benchmark photos. This six-image diagnostic is **not a replacement training dataset** and does not release the incomplete 54-image study. Two initial SDK configuration validation failures created no trainer. The third attempt created one owned Gemma 4 dedicated trainer without an inference deployment. The image-bearing forward returned, but the strict score/mask alignment check failed; [its receipt](results/gemma4_remote_vision_probe_attempt3.json) records the failure and verified cleanup. No weights were updated. The returned vector lengths were not captured in that attempt. The separate fourth diagnostic used dense SDK conversion and captured dimensions: all 12 returned score vectors had **two more positions** than their supplied inputs/targets/weights (for example, 473 scores vs 471 weights). They were not sparse tensors, ruling out sparse storage as the explanation for this result. The image-swap control did not run because alignment failed. It remains unsafe to assume a positional mapping, truncate scores, or claim verified image conditioning. [Fourth-attempt receipt](results/gemma4_remote_vision_probe_attempt4.json) records no updates/checkpoint and verified trainer absence after asynchronous deletion completed. No diagnostic trainer is being kept running.

## Execution and evaluation protocol

1. Reconstruct and hash-verify the 2,400 selected images with the September 29 package. Preserve its splits and uncertain-expression cases. Obtain and approve new three-word targets for train/development only; never move test images into preference generation. Membership or target-policy changes require explicit documentation and approval.
2. Export IAT-only SFT and ORPO JSONL. Validate that a rendered sample contains a real image input and that loss masks include only response text. Run a bounded remote training smoke test, retain its receipt, and tear down temporary compute. Do not replace a failed image route with text-only training.
3. Benchmark the three untouched base models using the existing runner. This step is complete. Freeze the primary experiment settings before looking at the new trained-model test results.
4. Initial pilot settings: new LoRA rank 8 / alpha 16, learning rate `1e-5`, batch size 4 examples or preference pairs, three epochs, seed 0; ORPO preference coefficient `0.1`. These are starting settings, not measured optima. Record actual optimizer, response-token masking, normalization, model/tokenizer revisions, shape version and trainer version. Normalize/inspect gradient scaling across SFT and ORPO rather than assuming identically named settings imply identical updates.
5. Use development response loss, chosen-vs-rejected conditional preference scores, three-word validity, expression retention, and word diversity for debugging/checkpoint decisions. Development is held out from gradient updates but may be used for model selection; it is not the final untouched test. Any tuning beyond the initial pilot must use development only and be recorded.
6. Evaluate the frozen SFT and ORPO checkpoints using exactly the same 61-word/12-image protocol as their corresponding base. The test set must not select training targets, hyperparameters, or checkpoints. Save resumable state, promoted model IDs, raw responses, all fallback attempts and failure counts. Release billable training and evaluation resources when finished; do not remove unrelated resources.
7. Deliver per-model reports and the full cross-model comparison. A missing, failed, or unsupported run is not a successful trained model. Leave the overall goal incomplete until the required actual runs, reports and figures exist.

The existing Fireworks benchmark contract is 12 photos × 50 repeats = **600 attempts per model/checkpoint**: temperature 1, top-p 1, 30 output tokens, reasoning disabled, existing seed schedule. Invalid outputs receive the existing deterministic fallback (temperature 0, 30 tokens) and remain explicitly counted if still invalid. Use that same contract for comparison; keep an optional longer-output diagnostic separate. Nominal word count is 1,800 only when all 600 responses contain three valid words. Never force measured counts to match a historical graph.

Required reported metrics: attempted/valid/invalid responses; exact-format compliance; fallback usage; per-race and overall positive/negative word counts and rates with actual denominators; Black-minus-White negative-rate gap; image-cluster bootstrap uncertainty; unique triples, distinct words, entropy and top-three-word concentration. Also report per-image results. Six images per race make population-level fairness conclusions weak, even with 50 repeats per image. A lower gap caused by everyone receiving the same words is not sufficient evidence of improved behavior.

Reuse `../2026-09-11/src/multimodel_bias` for scoring and figures. Per-condition grouped word-count charts must use all 61 words in `REFERENCE_WORD_ORDER`, including zero-count words, blue Black-photo bars and gold White-photo bars. Comparison figures show negative rates/gaps, validity and diversity together. Keep model/checkpoint identity in report captions, filenames and the comparison table; do not relabel historical figures as new runs.

## Local checks and preparation commands

```powershell
rtk python -m unittest discover -s 2026-10-03/tests -v
rtk python 2026-10-03/study.py audit
rtk python 2026-10-03/study.py prepare
rtk python 2026-10-03/study.py export
rtk python 2026-10-03/study.py preflight
```

The default asset root is `../2026-09-29/private/dataset`; an override must have its `images/{fairface,utkface}/` layout. Preparation/export refuse missing or mismatched images and incomplete approved targets. Export includes base64 images, stays Git-ignored under a curated-dataset namespace, and does not upload or start training. Local checks cover preserved allocations, matching-set isolation, benchmark exclusion, target approval, refusal of historical fallback, response-only losses and historical diagnostics. They are not proof of remote training correctness.

All 81 distinct study tests pass, including read-only Prime preflight, native Gemma/Qwen image-field/masking checks, draft budgeting/validation, isolated photo-review frontend logic, human review and authorized AI approval/export safeguards. The upstream suite has 12 passes and one Windows-specific CSV byte-replay failure; only path separators differ, not membership, splits, order or other fields. `rtk python 2026-10-03/audit_curated_replay.py` reproduces this finding in a temporary copy without altering the authoritative source. [Receipt and runtime versions](results/curation_replay_audit.json).

To reproduce existing base reports from saved responses without paying for new inference:

```powershell
rtk python 2026-10-03/report.py
```

Nine targeted existing benchmark/transport/transfer tests also pass. Full September 11 test collection is blocked on Windows by the pre-existing Linux-only `fcntl` import in `scripts/hosted.py`; that unrelated platform dependency has not been changed.

The current managed Fireworks API also exposes ORPO through `dpo-job create --loss-method ORPO --orpo-lambda ...`. Its upload schema (`input.messages`, `preferred_output`, `non_preferred_output`) differs from the Training API cookbook's chosen/rejected conversation schema. `study.py export --surface managed` emits the managed schema; the default `--surface training-api` emits cookbook records. Separate ignored upload subfolders and checksums identify each surface. Both keep the image, unchanged prompt and the same targets; neither launches a job. Managed preference documentation shows text examples only, so a well-formed export is **not proof** of image support for any selected model. Verify model-specific rendering and backend image consumption before training.

## Primary sources

- [Fireworks model and modality eligibility](https://docs.fireworks.ai/fine-tuning/models)
- [Managed SFT, including vision dataset format](https://docs.fireworks.ai/fine-tuning/fine-tuning-models)
- [Dedicated Training API: SFT / ORPO and resource lifecycle](https://docs.fireworks.ai/fine-tuning/training-api/dedicated)
- [Official Fireworks training cookbook](https://github.com/fw-ai/cookbook/tree/64122feb986449eddd7b377df22e99e0d06f35bd/training)
- [Managed Fireworks ORPO and preference dataset schema](https://docs.fireworks.ai/fine-tuning/dpo-fine-tuning)
- [Original FairFace source dataset](https://huggingface.co/datasets/HuggingFaceM4/FairFace)
- [ORPO paper](https://arxiv.org/abs/2403.07691)
