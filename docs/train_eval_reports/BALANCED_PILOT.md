# Authorized balanced SFT / ORPO pilot

**Completed pilot, October 6:** all six independent SFT/ORPO runs
are locally verified at 507 updates/three epochs each (3,042 total updates).
Qwen3.5 ORPO's final checkpoint passes all file hashes, 304 finite adapter tensors,
152 nonzero LoRA-B tensors and optimizer states at 507 steps. All three epoch
checkpoints are retained; histories have finite, nonzero gradients throughout.
Model/data pins and runtime versions match. The run took 6,832.08 seconds
(113.87 minutes), with 18.12 GiB peak allocated GPU memory on the H100 PCIe.
Final development chosen loss is 0.06654, with 100% conditional chosen-preference
accuracy on 174 positive and six negative examples, and zero development updates.
These scores are not free-generation or emotion accuracy.
[Final-checkpoint audit](../results/prime_qwen35_orpo_checkpoint_verified.json).
All nine benchmarks are complete (5,400 slots). Full per-model reports,
comparison, nine per-image tables and 21 figures are generated from verified
actual outputs. [Read the full comparison](native_balanced/COMPARISON.md).
All six adapters concentrated 91.22%–97.39% of words in Angry, Annoy and Negative;
smaller gaps alone are not a validated debiasing result. The report documents
data limitations and proposed next steps; no additional tuning was performed.
The 748,071,238-byte archive and all 198 files pass checksum checks.
The owned $3.29/hour GPU is terminated, independently verified absent from the
active list and present in terminated history. There were zero active Prime
pods and $39.9395 wallet credits at 18:25 UTC.
[Recovery and cleanup](../results/prime_native_attempt11_cleanup.json).
[Progress snapshot](../results/prime_qwen35_orpo_progress.json).
The report generator's rendering check passes with explicit per-group
rate changes, alongside gap, validity and diversity; temporary fixtures are not
study results. The six-training/nine-benchmark completion gate is unchanged.
Submitted trainer bytes are identical across the Gemma, Qwen3-VL and retry
attempts (SHA-256 `d730a363d09491d80be4c9d9babcc9ac4c12381624ae57e31809ecf378932bce`).
The full report records gradient-norm clipping at 1.0 and the trainer checksum.

The first completed epoch is now retained locally. All eight checkpoint hashes
match; its 304 tensors (9,056,256 parameters) are finite, all 152 LoRA-B tensors
are nonzero, and optimizer states have 169 steps. Its adapter SHA-256 exactly
matches epoch one of the earlier timed-out run, confirming this stage of the
fresh retry reproduces the original recipe. Development chosen loss is 0.09109;
chosen-preference accuracy is 100% on 174 positive and six negative examples,
with zero development updates. This is conditional likelihood, not generated
emotion accuracy. The frozen run snapshot was captured at 173 updates; the
checkpoint itself is at 169. It is intermediate and not benchmark-eligible.
[First-epoch audit](../results/prime_qwen35_orpo_epoch1_verified.json).

The second epoch is also retained and verified locally: all eight file hashes,
304 finite adapter tensors, 152 nonzero LoRA-B tensors and optimizer states at
338 updates pass. Its adapter checksum exactly reproduces the earlier attempt's
second epoch. Development chosen loss is 0.07569, with zero development updates;
conditional preference accuracy remains 100% on 174 positive and six negative
examples. The frozen run snapshot has 343 updates, while this checkpoint has
338. It remains intermediate, not benchmark-eligible; epoch three is now complete.
[Second-epoch audit](../results/prime_qwen35_orpo_epoch2_verified.json).

**Recovery update, October 6 at 15:44 UTC:** five complete trainings and six
complete benchmarks (3,600 slots) are locally verified. The recovered archive
and all 204 files match their hashes. The terminal worker's owned Lambda H100
was deleted at 15:42 UTC; active-list absence and terminated history independently
confirm cleanup. No unrelated resource was removed. The last wallet check was
$47.7302. [Recovery](../results/prime_native_attempt10_recovery.json) ·
[Cleanup](../results/prime_native_attempt10_cleanup.json).

The worker completed Qwen3-VL base/SFT/ORPO and Qwen3.5 SFT before Qwen3.5 ORPO
hit its 7,200-second operational ceiling at 497/507 updates and two complete
epochs. Its partial adapter is preserved in the private recovery but is neither
a completed condition nor a retry input. A retry is being launched using the
five verified final adapters and six benchmarks, skipping their completed work.
Only Qwen3.5 ORPO is restarted from the untouched base, then its model's three
evaluations run. The timeout now permits 10,800 seconds; data, model pins,
optimizer, weights, 507 updates and three epochs are unchanged.

Qwen3.5 SFT has 9,056,256 finite parameters in 304 adapter tensors; all 152
LoRA-B tensors are nonzero and optimizer steps are 507. All three checkpoint
hash sets and its 507 finite/nonzero gradient records pass checks. It took
68.18 minutes and peaked at 18.03 GiB Torch-allocated GPU memory. Final dev
chosen loss was 0.06595, with 100% conditional preference accuracy on 174
positive and six negative examples, not generation accuracy.
[Checkpoint audit](../results/prime_qwen35_sft_checkpoint_verified.json).

### Verified Qwen3-VL native results

| Condition | Valid / 600 | Black negative words | White negative words | Gap (pp) | Unique triples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 600 | 0/900 (0.00%) | 0/900 (0.00%) | 0.00 | 54 |
| SFT | 600 | 894/900 (99.33%) | 825/900 (91.67%) | +7.67 | 29 |
| ORPO | 600 | 894/900 (99.33%) | 834/900 (92.67%) | +6.67 | 30 |

Every final response meets the exact three-word format. SFT's image-cluster gap
interval is +1.67 to +15.67 pp; ORPO's is +1.00 to +14.00 pp. These individual
condition intervals are not paired training-effect intervals. The base uses
only positive words; both adapters shift to mostly Annoy/Negative/Angry, which
account for 91.44% (SFT) and 91.22% (ORPO) of words. Diversity decreases. Neither
the base's zero gap nor this training-induced negative concentration establishes
expression-grounded behavior. All graphs retain the original 61-word order and
blue/gold styling. [Base audit](../results/prime_qwen3vl_base_benchmark_verified.json) ·
[SFT audit](../results/prime_qwen3vl_sft_benchmark_verified.json) ·
[ORPO audit](../results/prime_qwen3vl_orpo_benchmark_verified.json).

**Earlier execution, October 6 at 08:33 UTC:** one owned Lambda H100 was active
at $3.29/hour. Pinned-runtime validation accepted both retained Gemma 4 adapters
and the completed base/SFT benchmarks. ORPO evaluation completed all 600 slots;
the downloaded responses, contract, slots and remote/local hashes pass checks.
Qwen3-VL SFT completed all 507 updates and three epochs. All checkpoint hashes,
finite final weights and 507-step optimizer state pass local checks. Its
independent ORPO run also completed all 507 updates/three epochs, with matching
checkpoint hashes, finite final weights and optimizer steps of 507. There are
now four verified complete trainings and three verified benchmarks. Native
Qwen3-VL base evaluation reached 45/50 rounds; adapter evaluations remain pending.
The preceding resume attempt failed before
new study work, was recovered and terminated, and its environment-selection
bug is fixed. [Cleanup evidence](../results/prime_native_attempt9_cleanup.json).

The earlier Massed Compute H100 is terminated and absent. Gemma 4 SFT and ORPO both
completed **507/507 updates and three epochs**, and both final checkpoints
passed local file-hash, finite-weight and optimizer-state checks. Native base
and SFT evaluations each completed 600 slots (12 photos, 50 per photo), with
551 and 600 valid responses respectively. Their contracts, response hashes
and graph hashes match the retained local copies. Both graphs retain all 61
words in original order and blue/gold styling.

SFT's descriptive race gap is zero, but both groups have **98.33% negative words**;
97.39% of its words are Angry/Annoy/Negative. Unique triples decreased from 21
to 15. This is not by itself evidence of improved image grounding. All 49 base
invalid slots were refusals for `wm1`; validity changes also affect which outputs
enter conditional word rates.

The earlier ORPO evaluation exited before any responses. Its underlying traceback was not
captured by the stale SSH stream; a local replay passes its checkpoint guard.
Full archive recovery timed out, but proactive copies retain both final
adapters, optimizer/processor files, run histories and completed evaluations.
Epoch-one/two files were not recovered. The runner now uses detached jobs,
short PID-verified status checks and retained logs for all children. Existing
completed archives are downloaded without unnecessary recompression.
The active run reuses these verified outputs and must complete both Qwen
studies, with the unchanged scientific recipe. Runtime remains
Python 3.11.15, Torch 2.11.0+cu128, Transformers 5.14.0, PEFT 0.21.0 and
Accelerate 1.15.0. The wallet was $71.4132 at 08:33 UTC.
[Cleanup and retained-output evidence](../results/prime_native_attempt8_cleanup.json).
[SFT checkpoint audit](../results/prime_current_gemma4_sft_checkpoint_verified.json)
and [ORPO checkpoint audit](../results/prime_current_gemma4_orpo_checkpoint_verified.json)
record the verified local copies; [current status](../results/prime_native_launch_status.json)
includes the base evidence. The full comparison remains pending.

### Verified Gemma 4 native results

| Condition | Valid / 600 | Black negative words | White negative words | Gap (pp) | Unique triples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 551 | 858/900 (95.33%) | 471/753 (62.55%) | +32.78 | 21 |
| SFT | 600 | 885/900 (98.33%) | 885/900 (98.33%) | 0.00 | 15 |
| ORPO | 600 | 867/900 (96.33%) | 888/900 (98.67%) | -2.33 | 15 |

ORPO's 95% image-cluster gap interval is -7.33 to +1.33 pp. Angry, Annoy and
Negative account for 1,736/1,800 words (96.44%); only 15 of the 61 words appear.
Its smaller absolute gap accompanies strongly concentrated negativity, not
demonstrated expression-grounded improvement. These are descriptive changes,
not a causal or population-fairness result. The base also has a different valid
output denominator because of refusals on one photo.

[ORPO slot/contract/hash audit](../results/prime_gemma4_orpo_benchmark_verified.json)
records the separately retained live snapshot. Its graph uses the same original
61-word order and blue/gold style. The final nine-condition reports remain gated
on all six complete trainings and nine complete benchmarks.

### Verified Qwen3-VL SFT / ORPO training; benchmarks pending

The final adapter has 7,667,712 finite parameters in 288 LoRA tensors, with all
144 LoRA-B tensors nonzero and all optimizer steps at 507. Training took 22.14
minutes including weight loading/development checks and peaked at 16.67 GiB
Torch-allocated GPU memory on the H100. All three epoch checkpoint hashes
match the downloaded files. Development chosen-response loss was 0.07296;
preference accuracy was 100% on 174 positive and six negative examples.
This compares conditional likelihoods, not generated-answer accuracy.
[Checkpoint and execution audit](../results/prime_qwen3vl_sft_checkpoint_verified.json).
The pre-training pixel control changed response log probabilities by up to
8.95154; it establishes input sensitivity before training, not post-training
grounding. The predeclared final epoch-three adapter is the only eligible
SFT benchmark checkpoint. ORPO starts independently from the same pinned base.
Its first-epoch checkpoint has 288 finite tensors, all 144 LoRA-B tensors
nonzero, matching file hashes and optimizer steps of 169. First-epoch ORPO
development preference accuracy is 100% on 174 positive and six negative
examples, with chosen-response loss 0.11102. This intermediate checkpoint is
not eligible for the benchmark. [First-epoch audit](../results/prime_qwen3vl_orpo_epoch1_verified.json).
The second-epoch checkpoint also passes all eight file hashes, finite-weight
checks and optimizer steps of 338. Its chosen-response development loss is
0.08382 and preference accuracy remains 100% across both categories. The third
epoch subsequently completed. [Second-epoch audit](../results/prime_qwen3vl_orpo_epoch2_verified.json).

Final ORPO took 37.37 minutes including loading/development checks and peaked
at 16.76 GiB Torch-allocated GPU memory. Its 288 final tensors are finite, all
144 LoRA-B tensors are nonzero and optimizer steps are 507. All three checkpoint
file hashes match. Final development chosen-response loss is 0.07253, with 100%
preference accuracy on 174 positive and six negative examples. This is not
generation accuracy. [Final ORPO audit](../results/prime_qwen3vl_orpo_checkpoint_verified.json).

**Earlier completed SFT, October 6 at 04:32 UTC:** Gemma 4 SFT completed all **507
planned optimizer updates across three epochs**. The final checkpoint has eight
hash-verified files, 5,744,640 finite adapter parameters, and all 115 LoRA-B
tensors nonzero. Development preference accuracy was 100% overall, positive
100% and negative 100% (six negative images), with zero development updates.
The final SFT checkpoint has not yet been benchmarked.

Gemma 4 ORPO then reached **160 real updates** but stopped with BrokenPipeError
when the SSH monitor disconnected; no final ORPO checkpoint was promoted. The
pod was recovered and terminated at 03:56 UTC, with output checksums verified.
The monitor and remote worker are patched so transient control-plane loss does
not close child training output. Two later Nebius H200 attempts failed during
provisioning before SSH; both exact pods are now absent from the active list, and
the wallet remains **$79.2125**. The next run will use a different qualifying
provider offer, redo ORPO from the untouched base, then continue with the
remaining models and evaluations. [Machine-readable status](../results/prime_native_launch_status.json).

A follow-up $2.35/hour Massed Compute launch reached the provider's active state,
but its image lacked the `python3-venv` package required for the managed Python
3.11 bootstrap. It failed before model loading and was terminated with zero
additional wallet decrease. The launcher now accepts only the previously
verified Lambda image for this run. The launcher now falls back to the official
unmanaged `uv` installer when `python3-venv` is absent and treats `STOPPED` as a
terminal provider state. [Follow-up cleanup](../results/prime_native_attempt4_cleanup.json).

The next two Nebius H200 attempts were rejected before model loading: neither
exposed SSH, and the second required explicit termination after a contradictory
provider status. Wallet balance did not change. [Attempt 5](../results/prime_native_attempt5_cleanup.json) · [Attempt 6](../results/prime_native_attempt6_cleanup.json).

## Prior launch and verification history

October 6, 2026, 00:22 UTC update. The user accepted excluding unresolved targets and balancing the
retained positive/negative examples, then requested SFT and ORPO now. The pilot
is frozen and the training/evaluation entry points are implemented. An owned
H100 was provisioned at 20:28 UTC, at a checked $4.29/hour quote. Its setup
failed before training: default PyPI Torch 2.13 used CUDA 13, but the driver
reported CUDA 12.8. The pod was terminated at 20:36 UTC; absence and terminated
history were independently verified. **Zero study optimizer updates and no new
checkpoint.** [Current status](../results/prime_native_launch_status.json) and
[cleanup/wallet evidence](../results/prime_native_attempt1_cleanup.json).

The retry pins Python **3.11.15** and uses official Torch **2.11.0 + CUDA 12.8** /
torchvision 0.26.0 builds. Artifact hashing also works with the cloud image's
system Python 3.10, used for recovery. All **92 local tests pass** on the isolated Torch 2.11 CPU
runtime, including six random architecture SFT/ORPO checks and portable cohort
replay. The remote worker repeats the architecture checks, verifies real BF16
CUDA access, then starts the unchanged experiment. This does not establish
pretrained GPU gradients until that actual run succeeds. [Official build index](https://download.pytorch.org/whl/cu128/torch/).

The **20:47 UTC** check returned zero H100/H200 offers. Another H100 offer at
20:49 allowed a second launch, but SSH never became usable within 15 minutes.
That pod was terminated at 21:05; fresh API checks verify its absence and
terminated history, with no observed additional wallet decrease.
[Second-attempt cleanup](../results/prime_native_attempt2_cleanup.json).

A third H100 launched October 6 at 00:20 UTC at **$4.29/hour** using the
corrected Python/CUDA bundle. It produced the completed SFT checkpoint and a
partial ORPO run before cleanup. The launcher retains its spending/time guard;
the cleanup receipt is [here](../results/prime_native_attempt3_cleanup.json).

The launch check at 20:21 UTC returned zero single-GPU H100 and H200 offers.
Those initial attempts created no paid resource, SSH key registration or image
upload. The next launch check found an available H100 and provisioned it with
a dedicated task key; the source credentials are not uploaded.
[Earlier empty-capacity and wallet receipt](../results/prime/native_capacity_2026-10-05T202121.385484+0000.json).
Available credits were $94.2419 before the first launch; after its termination
the wallet held **$94.0857**, a $0.1562 observed wallet decrease. This is not an
isolated invoice; the history endpoint's unverified raw currency units are not
relabeled as dollars. The earlier $100 total authorization remains
unchanged. Earlier H100/H200 quotes were transient, not capacity reservations.

## Data and balancing

| Split | Positive targets | Negative targets | Total retained |
| --- | ---: | ---: | ---: |
| Training | 639 | 37 | 676 |
| Development, no optimizer updates | 174 | 6 | 180 |

Source: the pinned September 29 `selected_images.csv`, with image-specific
direct visual proposals and separately recorded user-authorized AI approvals.
These are expression-association proposals, **not objective emotion/personality
labels or independent human validation**. Human approvals remain zero.

The 1,064 unresolved train/development images are excluded, not relabeled.
Original split assignments and the full 2,400-image manifest are untouched.
All 480 reserved test images remain outside target construction/training.
The original 12 benchmark JPEGs are packaged separately for evaluation only.
No profile targets or demographic text enter the model input.

An additional exact separation audit verified all 2,400 selected export files
and their dimension-prefixed RGB pixel hashes, then compared the 12 reference
JPEGs with original-source hashes, export hashes and decoded-pixel hashes.
There were zero matches. [Audit evidence](../results/benchmark_separation_audit.json).
This does not establish person-disjoint splits or comprehensively exclude
differently cropped, resized or recompressed copies. No identity-generalization
claim or data change follows from the audit.

Each training example receives fixed weight `676 / (2 * category_count)`:
positive **0.5289514867**, negative **9.1351351351**. Each category therefore has
total loss weight **338** per epoch. We retain all 676 images; we do not duplicate
negative images or discard extra positive images. Weights apply to the entire
SFT example or ORPO pair loss before the logical-batch mean, without re-normalizing
within a batch. This balances loss weighting, not the number of distinct examples,
every gradient update, word frequencies or race-group counts.

Important limitations: 37 negative training images and six negative development
images are small and dominated by anger/disgust targets. Strong weighting can
overfit those few examples. Exclusion also breaks the upstream matching balance:
this is a smaller exploratory pilot, not the originally balanced full study.
Neutral/ambiguous faces remain an important evaluation limitation. Reducing the
race gap by making every answer positive or repetitive is not sufficient success.

The retained training targets use only 14 chosen ordered triples and ten of the
61 words. The two most common chosen targets cover 591/676 images (87.43%).
There are seven rejected ordered triples; `Angry, Negative, Sadness` is rejected
for 610/676 images (90.24%). These are AI-proposed counterexamples, not observed
human preferences. Repetition and coarse valence contrasts limit what perfect
development preference accuracy establishes. Earlier gateway-model drafts are
comparison material; all 676 training targets come from the direct visual pass.

[Frozen-policy configuration](../config/balanced_pilot.json) and
[public data/provenance receipt](../results/balanced_pilot_data.json) record exact
counts, weights and checksums. The detailed cohort and image assets stay private.

## Runs and evaluation

Separate SFT and ORPO adapters start from each pinned untouched base:
Gemma 4 26B A4B IT, Qwen3-VL 8B Instruct and Qwen3.5 9B. SFT learns the chosen
three-word answer; ORPO also receives the rejected three-word answer for the
same image and prompt. Response tokens/native terminators alone contribute loss.

The shared recipe is BF16, language-attention-only LoRA rank 8/alpha 16/dropout 0,
AdamW at `1e-5`, logical batch four, three epochs, seed zero, and ORPO coefficient
0.1. Vision, routers, expert tensors, embeddings and base weights remain frozen.
Each complete condition requires **507 optimizer updates**. Development reports
include each target category and a macro average so the six negative examples
are not hidden by the positive majority. The final epoch-three checkpoint is
predeclared; the 12 benchmark photos never select checkpoints or hyperparameters.

Native base, SFT and ORPO each receive the original 12 photos × 50 repeats:
**600 slots per condition, 5,400 across nine conditions**. The existing scorer,
seed schedule, 30-token limit, invalid-output fallback and all-61-word blue/gold
graph style are reused. Invalid final outputs remain counted explicitly. Native
base results must accompany native adapters; prior Fireworks baselines are not
numerically interchangeable with this BF16 Transformers stack.

The local checks cover fixed weighted gradients, native architecture fixtures,
unchanged frozen parameters, image-conditioned scoring, adapter round trips,
fallback/seed/format recording and price-quote refusal. Random CPU fixtures do
not establish full-size GPU memory, performance or trained-model outcomes.
The private input bundle also replays successfully in an extracted workspace.

The native report generator verifies all six complete 507-update trainings and
all nine 600-slot benchmarks before producing final per-model/comparison reports.
It checks checkpoint hashes and matched base/adapter contracts; unfinished
results cannot be presented as a completed comparison. No native trained-model
comparison report exists yet; the completed native Gemma 4 base graph is
verified above. A complete nine-condition rendering check also
passes using synthetic records in a temporary folder; those artifacts are
discarded, not saved as study results. The report-import collision discovered in the joint
test run is corrected; existing benchmark scripts no longer shadow this study's
report helpers.

## Launch and credit guard

The owned-pod launcher requires a live fixed-price single H100/H200 offer,
checks extra resource charges and SSH ownership, and reserves at least $20.
It limits this launch to at most $70 estimated compute/wallet depletion and 16
hours, including setup/evaluation. No top-up or persistent disk is enabled.
These are software guards and conservative reserves, **not a provider-enforced
hard billing cap or a guarantee that all six trainings fit the available credits**.
Partial artifacts are recovered and verified before owned-pod termination;
incomplete runs cannot be evaluated as successful checkpoints.

The existing full-cohort export gate remains closed deliberately; only the
explicit `resolved-balanced` entry point accepts this separately frozen pilot.
Local branch is `peyton`. Nothing has been pushed.
