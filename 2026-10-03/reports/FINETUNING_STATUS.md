# Fine-tuning status: approved curated image pool

**Next-round model decision — October 6, 2026:** the user requested Gemma 3
for the next training round. Use `google/gemma-3-12b-it` (Gemma 3 12B IT,
image-capable, instruction-tuned), matching the model named in the earlier IAT
notebook, for separate SFT and ORPO runs. This is a future-round decision, not
a replacement of the current Gemma 4 / Qwen study or authorization to launch
additional paid jobs now. Pin its exact weight/processor revision before execution.

Completed pilot, October 6: all six independent SFT/ORPO trainings are locally
verified at 507 updates and three epochs each (3,042 total updates). Qwen3.5
ORPO's final adapter passes all hashes, 304 finite tensors, 152 nonzero LoRA-B
tensors and optimizer states at 507 steps. All three checkpoints are retained.
Final development chosen loss is 0.06654, with zero development optimizer updates.
This conditional scoring result is not free-generation or emotion accuracy.
[Final-checkpoint audit](../results/prime_qwen35_orpo_checkpoint_verified.json).
The first two epochs are locally hash/weight/optimizer verified at 169 and 338 updates.
Their adapter checksums match the earlier attempt's corresponding epochs,
reproducing those stages of the fresh-base retry. Neither is final or benchmark-eligible.
[First-epoch audit](../results/prime_qwen35_orpo_epoch1_verified.json).
[Second-epoch audit](../results/prime_qwen35_orpo_epoch2_verified.json).
All nine benchmarks are complete (5,400 slots). Full per-model reports,
comparison, per-image tables and 21 figures are written and reconciled with the
recovered archive. Strong negative-word concentration prevents interpreting
small gaps as validated improvement. [Full comparison](native_balanced/COMPARISON.md).
The 198 archived files are verified. The owned GPU is terminated, independently
absent from the active list and present in terminated history. No Prime pods
were active at 18:25 UTC; wallet: $39.9395.
[Cleanup evidence](../results/prime_native_attempt11_cleanup.json).

Earlier October 6, 15:44 UTC: five complete trainings and six benchmarks are
locally verified, including Qwen3.5 SFT's 507-update/three-epoch checkpoint,
finite weights, all checkpoint hashes and optimizer state. All 204 recovered
files match the archive checksums. The owned Lambda H100 is terminated and
absent from the active list, with matching terminated history. Wallet: $47.7302.
[Recovery and cleanup](../results/prime_native_attempt10_cleanup.json).
Qwen3.5 ORPO hit its 7,200-second timeout at 497 updates and two completed epochs;
its preserved partial adapter is excluded from benchmark and retry inputs.
A retry is being launched with 10,800 seconds allowed and all scientific
settings unchanged. It reuses the five final adapters and six benchmarks,
restarting only Qwen3.5 ORPO and performing its base/SFT/ORPO evaluations.
The full reports remain incomplete until all six trainings and nine benchmarks.
[Qwen3-VL base benchmark audit](../results/prime_qwen3vl_base_benchmark_verified.json).
[Detailed recovery status](BALANCED_PILOT.md).

Qwen3-VL base has 0% negative words for both groups. SFT has 99.33% for Black
photos and 91.67% for White photos (+7.67 pp gap); ORPO has 99.33% and 92.67%
(+6.67 pp). All 600 responses per condition are valid/exact. Both adapters
concentrate over 91% of words in Annoy/Negative/Angry and have fewer distinct
triples than the base. This is not demonstrated expression grounding or
debiasing success. [SFT audit](../results/prime_qwen3vl_sft_benchmark_verified.json) ·
[ORPO audit](../results/prime_qwen3vl_orpo_benchmark_verified.json).

Earlier October 6, 08:33 UTC: one owned Lambda H100 was active at $3.29/hour.
Pinned-runtime validation accepted both retained Gemma 4 adapters and the
completed base/SFT benchmarks. ORPO evaluation completed all 600 slots and
passed local slot/contract/hash checks. Qwen3-VL SFT completed all 507 updates
and three epochs; all checkpoint hashes, finite final weights and 507-step
optimizer state pass local checks. Its independent ORPO run also completed all
507 updates/three epochs; all checkpoint hashes, finite final weights and
507-step optimizer state pass checks. Native Qwen3-VL base evaluation reached
45/50 rounds. [Final ORPO audit](../results/prime_qwen3vl_orpo_checkpoint_verified.json).
[Verified Qwen3-VL SFT checkpoint](../results/prime_qwen3vl_sft_checkpoint_verified.json).
The preceding resume attempt failed before new study
work and was recovered and terminated; its environment-selection bug is fixed.
[Cleanup evidence](../results/prime_native_attempt9_cleanup.json).
The earlier Massed Compute attempt completed both Gemma 4 SFT and ORPO: each has 507 updates,
three epochs, and a final checkpoint that passed local hash/finite-weight/
optimizer-state checks. Base and SFT benchmarks each completed 600 slots,
with 551 and 600 valid responses. Their contracts, response/manifest/graph hashes
and photo-slot counts passed local checks. SFT's gap is zero because both groups
have 98.33% negative words, predominantly Angry/Annoy/Negative; this alone is not
better grounding. All 49 base invalid responses were refusals for `wm1`.
The earlier ORPO evaluation failed before responses, with no captured underlying traceback;
the corrected continuation completed it. ORPO has 96.33% negative words for
Black photos and 98.67% for White photos, a -2.33 pp gap. Its top three negative
words account for 96.44% of all words; this is not proof of improved grounding.
[Verified ORPO benchmark](../results/prime_gemma4_orpo_benchmark_verified.json).
Both final adapters and complete evaluations are retained; full archive recovery
timed out and intermediate epoch files were not recovered. The runner is patched
for detached execution, retained logs and short PID-verified status checks.
Remaining work is all Qwen evaluations plus both Qwen3.5 trainings. Four of
six trainings and three of nine benchmarks are verified.
Wallet: $71.4132 at 08:33 UTC. Full comparison incomplete.

Earlier October 6, 04:32 UTC update: the user subsequently accepted the **676-train /
180-development resolved-target pilot** with fixed category-balanced loss
weights and requested SFT/ORPO execution. This is an explicitly authorized
subset, not a silent full-cohort export. Training and matched native benchmark
entry points are ready. The initial launch checks found no H100/H200 offer and
created no paid resource; an H100 became available and was provisioned at
20:28 UTC at $4.29/hour. That pod was terminated after setup exposed a CUDA 13
versus driver 12.8 mismatch; no study updates/checkpoints were created.
The second pod timed out before SSH became available; its termination and absence
are verified. The third H100 completed Gemma 4 SFT at $4.29/hour: **507/507
actual optimizer updates**, three epochs, and a hash-verified final checkpoint.
Gemma 4 ORPO reached 160 updates but failed when the SSH monitor disconnected;
it has no final checkpoint. The pod was terminated and its output archive
verified. The monitor and remote worker are patched before rerunning ORPO from the
untouched base. Two later Nebius H200 provisioning attempts failed before SSH and
were terminated/reconciled with no observed wallet change; no trained-model
benchmark exists yet and Qwen runs remain pending.
[Current pilot and evidence](BALANCED_PILOT.md). The dated full-cohort-only
status below is retained as history; its unresolved labels remain blank.

On October 3 the user approved switching to the September 29 `selected_images.csv` pool. The active plan now uses its unchanged 1,440 train / 480 development / 480 test allocation, with the original 12 photos kept benchmark-only. The 54-image plan and proposed 12-image training pilot are superseded, not combined with this cohort. [Current plan and target requirements](CURATED_DATASET_PLAN.md). No SFT or ORPO optimizer updates have run, no fine-tuned checkpoint exists, and the overall requested training/comparison goal remains incomplete.

All 2,400 selected images have been reconstructed and verified against their original-source and exported-PNG hashes. The full upstream image/balance validator passes, and all 19 source-package checksums remain unchanged. The pinned selected-image manifest has no desired or rejected three-word answers. Actual readiness checks are recorded in `../data/manifest.json`; target approvals are separate and now may be explicitly user-authorized AI self-approvals. Automated expression fields are not approved IAT targets. Missing originals from the old cohort no longer block this dataset. The Fireworks alignment failure is a separate unresolved prerequisite.

The user subsequently authorized drafts for review and supplied $100 of Prime credits. Direct Codex photo inspection is complete for all 1,920 train/development photos: 856 tentative three-word pairs and 1,064 unresolved cases, with zero human approvals. Earlier Haiku proposals are retained separately as historical comparisons, not silently used as direct targets. Paid Prime gateway drafting cost an estimated $5.21; the $94.25 account balance was observed October 4 at 04:07 UTC and is not a current quote. No reserved test or benchmark photo was used for target generation, no Prime GPU was provisioned and no training occurred. [Photo review instructions and quality concerns](TARGET_DRAFT_REVIEW.md). The direct pass does not force a positive answer for neutral faces, and completed inspection is not an approved training dataset.

The user's October 5 instruction explicitly authorized AI self-approval. The retained direct visual proposals have now produced 856 AI-approved pairs, with uncertainty flags retained and zero human approvals. Separate approval files and provenance distinguish self-approval from independent human validation; later human decisions override AI decisions. The 1,064 neutral/ambiguous targets remain blank, pending a decision about the fixed vocabulary's lack of neutral/abstention words. No weak targets have been invented or smaller cohort exported.

The October 5, 10:19 UTC readiness audit again verified all 2,400 image hashes and unchanged splits, but found no complete approved target file. The actual export command refused to generate training data. The requested three-model SFT/ORPO runs, matching trained-checkpoint evaluations, per-model reports and comparison graphs remain incomplete. No optimizer updates or trained checkpoints exist. Real image-conditioned backend verification and bounded paid execution remain necessary even after all labels are approved.

## Historical supplied CSV and photos: not the active training source

`C:/Users/lipey/Downloads/dpo_pairs_train_only.csv` contains 4,345 rows and seven columns: `image_file`, `prompt`, `Category`, `Note`, `chosen`, `rejected`, `is_eval`. Prompt/image/chosen/rejected fields are populated, and no chosen response is exactly identical to its rejected response. There are 142 distinct prompts and 36 distinct paired-image filenames. Every prompt begins with "Generate two short profiles"; zero pairs have the requested exact three-word chosen/rejected format. Category counts are 1,834 `general-mkt`, 1,293 `IAT-mkt`, and 1,218 `econ-game`. `IAT-mkt` is a profile-prompt category, not the three-word benchmark. All `is_eval` values are zero; this CSV provides no held-out split.

The CSV contains filenames, not image bytes or the 11 missing FairFace PNGs. Its ordered-pair filenames reference the same 12 person IDs as the benchmark. The newly supplied JPEG attachments match all 12 benchmark SHA-256 values exactly. They can be ingredients for constructing the named pairs, but exact paired-image layout/bytes would need verification against the original pipeline before claiming replication. `Thumbs.db` is not a training image.

Using this CSV would train profile outputs: SFT learns `chosen`, and ORPO learns from `chosen` plus `rejected`. Evaluating a three-word IAT prompt afterward would be a cross-task test, not IAT-only training. Keeping the requested IAT-only task requires new, reviewed three-word targets for these photos; the old FairFace image-specific targets cannot be reassigned to them without review.

Training on all 12 and testing on all 12 can measure a seen-image fit, not performance on unseen faces. A proposed alternative is three-fold identity-held-out evaluation: each fold trains on eight photos and tests on four (one per original race/gender metadata group). If profile-pair training is selected, exclude every composite containing any held-out person, not merely individual CSV rows. Aggregating 50 held-out repeats per photo yields 600 out-of-fold slots per method, but those outputs come from three different adapters and are not the original single-checkpoint benchmark. No such split or training targets have been silently activated.

## What is complete

October 5 native-runner progress: an executable SFT/ORPO candidate reuses the pinned processors and existing objectives for all three models. Six tiny random architecture checks passed finite adapter updates, unchanged frozen weights, native masked-loss equivalence, image sensitivity and adapter save/reload. All 83 current local tests pass. These synthetic CPU fixture updates are not the requested pretrained-model training: the full-cohort gate remains closed, with no paid resource or study checkpoint created. [Exact scope and settings](PRIME_TRAINING_READINESS.md#native-optimizer-candidate-fixtures-are-not-study-training).

- Three real base-model evaluations: Gemma 4 26B A4B IT, Qwen3-VL 8B Instruct, and Qwen3.5 9B. Each completed 600 benchmark slots. [Measured comparison and graphs](BASE_COMPARISON.md).
- Approved September 29 pool activated with unchanged 1,440/480/480 allocations; all 2,400 source/export image hashes and full metadata balance checks pass. [Active dataset audit](../data/manifest.json).
- Historical IAT-only target audit and deterministic split: 38 training images, 16 development images, 27 Black/White matched blocks. Profiles are excluded. [Archived dataset manifest](../data/legacy_54_manifest.json).
- Exact recovery of 43/54 historical images from public FairFace files, checked against both SHA-256 and original metadata. Scans of both public configs and embedded notebook images yielded no remaining matches.
- Local Gemma 4 image-bearing tokenization and response-only mask checks for six available training images. This is a diagnostic, not a reduced training run.
- Prime Lab setup using the existing login; all 17 workspace checks pass. Read-only catalogs do not establish an eligible hosted image SFT/ORPO route. The setup itself launched no paid work; later authorized Prime gateway image inference generated draft targets only, not a training job.
- All 1,920 train/development photos are directly inspected and hash-verified. The user-authorized AI-approval command accepts only resolved, unchanged direct proposals; it cannot fill blank targets, fall back to teacher labels, export a subset or launch training. The existing photo reviewer distinguishes AI versus human decisions and supports human corrections.
- Local Gemma 4 explicit non-thinking continuation checks on 96 active-pool training images / 192 constant grammar fixtures pass, preserving the exact prompt and native image fields. The separate standard complete-assistant template check failed exact prefix equality and remains recorded. Neither result supplies reviewed targets or demonstrates a trained model. [Current readiness evidence and remaining approvals](PRIME_TRAINING_READINESS.md).
- Both Qwen native processors pass on the same 96 active training images / 192 fixtures per model, retaining image grids/pixels and supervising exactly the response and native terminator. No model weights were loaded. [Local model-input comparison](LOCAL_MODEL_INPUT_CHECKS.md).

The study's 81 distinct local tests pass, including draft accounting, isolated photo-review UI logic, human-review/export guards and authorized AI approval, revocation and human overrides. Browser layout verification remains blocked by the app's saved-permission check; no workaround was used. One upstream byte-replay test fails solely because Windows serializes `image_path` with backslashes; a temporary replay confirms identical image membership, splits, row order and every other field. The original CSV remains unchanged. [Replay evidence](../results/curation_replay_audit.json).

## Historical missing originals: no longer required for the active plan

These files would still be needed to reproduce the superseded 54-image plan, but are not prerequisites for the approved September 29 pool:

- `data/august3/raw_pool/images/` — nine missing PNGs.
- `data/august10/negative_expression_expansion_v1/images/` — two missing PNGs.

[The exact 11 filenames and expected SHA-256 checksums](../data/missing_images.csv) identify the requested files. Original bytes matter: re-encoding a visually identical image changes its checksum. An archive containing the full folders is also suitable. The 12 benchmark photos are already available and cannot be used to replace missing training photos.

The remaining 11 images include 10 training and one development example. Dropping incomplete matched blocks would leave only 32 complete-block examples and change the expression balance; that alternative was explicitly declined.

## Backend diagnostic, not fine-tuning

The first two Gemma 4 diagnostic attempts failed local SDK configuration checks before provisioning. The third created a dedicated trainer and returned from an image-bearing forward call, but failed the strict token-score/mask alignment check. Its trainer was deleted and absence verified. That result is not proof of image conditioning.

The fourth diagnostic completed and captured all returned dimensions using dense SDK conversion. Each of its 12 score vectors had two more positions than the supplied input/target/weight vectors (for example, 473 vs 471), and none was sparse. Strict alignment therefore failed; the image-swap control was not reached. No optimizer calls, development/benchmark photos, checkpoints or inference deployments were involved. The temporary trainer was deleted and its absence verified after asynchronous deletion. [Fourth-attempt receipt](../results/gemma4_remote_vision_probe_attempt4.json) records the observed mismatch and cleanup. Image conditioning remains unverified; there is no justified mapping for the extra score positions yet.

An executed local reconstruction exercised the installed SDK/Tinker v1 JSON converter on the same six training images and all 12 continuations. The serialized PNG bytes still match their historical checksums, image expected lengths are retained, and serialized input/target/mask coordinates all agree. This rules out dropped positions in that reconstruction; it is not a captured original HTTP request or proof of actual backend expansion. The boundary-owning candidate passed local coordinate checks but **failed the real fifth attempt**: Fireworks reports `expected 258, got 256 from image`. That contradicts the candidate, not the original 256-token processor count. The fifth trainer was deleted and absence verified; no weight update, checkpoint or inference deployment occurred. Do not activate this rejected variant or guess a score mapping. [Detailed failure analysis and loss-preparation checks](GEMMA4_IMAGE_TRAINING_DIAGNOSTIC.md) records the unresolved backend contract, all receipts, and 27 passing local tests. Real image conditioning remains unverified.

## Archived 54-image execution plan (superseded)

1. Obtain the original PNGs, preserve source-relative paths, and rerun the hash/metadata/contamination checks. Reject mismatches; do not attach old targets to new photos.
2. Visually review the full matched cohort and target suitability before releasing it. Recorded AI judgments are not independent human ground truth; the neutral targets have a positive-word prior, only eight chosen triples occur, and selected negative examples are sad faces only.
3. Resolve and verify real image-conditioned training on each selected model. Do not replace a failed image route with text-only training.
4. Run separate SFT and ORPO LoRA adapters from each untouched base, using only image + unchanged 61-word prompt + existing chosen response (SFT) or chosen/rejected responses (ORPO). No profiles or demographic labels enter the model input.
5. Use development only for checks/selection, freeze the actual checkpoints, then run the same 12-image/600-slot benchmark. Expand per-model reports and comparison plots with measured trained-model results, validity, diversity and actual word denominators.

All work remains local on branch `peyton`; nothing has been pushed. Fireworks spending is authorized, but idle trainers/deployments must not be kept running while waiting for files.
