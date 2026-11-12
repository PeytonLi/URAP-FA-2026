# RunPod Setup and Training

## Hardware

The validated target is one NVIDIA H100 80 GB. The current workflow is
single-process and single-GPU. Attaching a second H100 does not accelerate it
without adding and validating distributed model/training code.

The July 20 successful training path reserved approximately 44.7 GiB at peak.
This is historical guidance, not a guarantee for a future dependency stack.
The launch plan pins the known base snapshot and software versions before GPU
work begins.

## Repository and environment

Place the repository on the network volume:

```bash
cd /workspace/gemma3-fairface-debiasing-july27
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-runpod.txt
```

The currently validated shared environment uses:

- Python environment from the July 13/19 workflow;
- PyTorch 2.4.1 with CUDA 12.4;
- Transformers 4.50.0;
- PEFT 0.14.0;
- Datasets 3.6.0;
- Accelerate 1.4.0.

Do not let the training command silently upgrade packages. The prepared command
uses `--no-install-missing-deps`.

## Base model

The launch gate expects the exact cached snapshot:

```text
/workspace/.hf/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80
```

If that snapshot is unavailable, download Gemma 3 only after confirming Hugging
Face access and then update the pin and manifest deliberately. Do not substitute
a moving model alias during the primary comparison.

## Preflight

From the repository root:

```bash
source .venv/bin/activate
python -m py_compile \
  fairface_iat_core.py \
  fairface_july19_iat_dpo_workflow.py \
  evaluate_adapter_checkpoint.py \
  prepare_fresh_lora_run.py
python prepare_frozen_inputs.py
python prepare_fresh_lora_run.py
```

`prepare_fresh_lora_run.py` performs no training. It validates:

- Gemini audit completion and readiness;
- frozen pair CSV SHA-256;
- at least 50 unique pair images;
- every training image SHA-256;
- all four race/gender intersections;
- at least three neutral/positive pairs in every intersection;
- no more than a 25-point spread in neutral/positive pair fractions;
- frozen base response and metadata alignment;
- exact base model snapshot;
- empty output directory;
- fresh-LoRA initialization with no prior adapter.

When all gates pass, the resulting plan is:

```text
results/fresh_lora_launch_plan.json
```

Review its `shell_command` before execution.

The superseded 51-row candidate set fails the expression-distribution gates and
must not be trained. The final balanced 50-row freeze passes all gates. Its
SHA-256 is
`638caf56c926b3007b6f17fef7b042ac5927be06bed8e31fba0e676b19a0cc69`.
On July 27 the RunPod regenerated that freeze, the exact training loader
revalidated all 50 rows, and the preparation script wrote the launch plan
without starting training.

## Primary run

Training is intentionally not started by any preparation script. When the
frozen audit has passed and the team authorizes the run, execute the exact
`shell_command` recorded in the launch-plan JSON from the repository root.

The run:

- uses Gemma 3, DPO, and LoRA;
- trains each frozen pair at most once;
- uses learning rate `1e-5`, beta `0.05`, rank 16, alpha 32, and dropout 0.05;
- excludes the vision tower from LoRA;
- uses eager attention;
- checkpoints every five optimizer updates;
- reuses the exact frozen base FairFace and neutral response CSVs;
- evaluates the adapter with the same images and generation seed;
- computes a 100,000-sample image-level paired bootstrap.

`--reuse-pairs-csv` bypasses FairFace training sampling. Therefore
`--train-sample-per-group 100` in the recorded command is inert for this run:
the actual exposure is 50 unique image-level pairs, each used once. With
gradient accumulation of four, that is at most 13 optimizer updates. Expected
end-to-end H100 runtime is approximately 40-55 minutes based on the comparable
July 20 run; model loading, reference-score preparation, and evaluation account
for much of that time.

## Monitoring

Use a second SSH session:

```bash
nvidia-smi
tail -f outputs/fresh_lora_gemini_judged_20260727/run.log
```

GPU utilization can be low during CSV validation, image loading, reference
score preparation, plotting, and bootstrap analysis. Confirm the process and
log before treating low utilization as a stopped run.

## Checkpoint selection

Saved checkpoints expose the optimization trajectory and make failure
diagnosis possible. They are not permission to choose the most favorable result
on the 400-image held-out set.

The primary run should use its pre-specified final or loss-based early-stop
checkpoint. If the team wants metric-based checkpoint selection, freeze a
separate validation pool before training and keep the current 400 images
untouched for one final report. Repeatedly choosing on those 400 images would
turn the test set into validation data.
