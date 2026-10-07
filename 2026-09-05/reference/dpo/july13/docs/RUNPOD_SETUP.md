# RunPod Setup

## Storage

Clone the repository under `/workspace`, which should be your RunPod network volume. Model caches, run outputs, and adapters should also remain under `/workspace` so they survive pod stops.

Suggested layout:

```text
/workspace/
├── .hf/
├── gemma3-fairface-debiasing-july13/
│   ├── outputs/
│   └── adapters/
└── other-research-data/
```

Stopping a pod preserves a network volume. Deleting the network volume does not.

## Environment

Start from a RunPod PyTorch image with Python 3.11, CUDA, and a current PyTorch build. An H100 80 GB was sufficient for the reference base evaluation and used about 26 GB of GPU memory.

```bash
cd /workspace/gemma3-fairface-debiasing-july13
bash scripts/setup_runpod.sh
source .venv/bin/activate
```

The setup uses `--system-site-packages` so the virtual environment can reuse the CUDA-enabled PyTorch installation from the RunPod image.

## Model and Dataset Access

Accept any required Hugging Face access terms for:

- `google/gemma-3-12b-it`
- `HuggingFaceM4/FairFace`

Then configure either a token or a local model snapshot:

```bash
export HF_TOKEN="your_huggingface_token"
export HF_HOME=/workspace/.hf
export BASE_MODEL_LOCAL_DIR=/workspace/.hf/hub/models--google--gemma-3-12b-it/snapshots/<revision>
```

Never place the token in a notebook, shell script, manifest, or Git commit.

## Verify Before Spending GPU Time

```bash
python preflight.py --require-gpu
python -m unittest discover -s tests -v
```

The preflight should report CUDA available, bfloat16 support, package versions, and either a valid local model path or token-based access.

## Run Baseline

```bash
bash scripts/run_baseline.sh
```

The 100-per-group, 10-repeat reference evaluation took roughly 35 minutes on the prior H100, including setup/model loading. Sequential generation is the main runtime cost.

## Generate and Audit Pairs

```bash
bash scripts/run_pair_audit.sh
```

The `pairs` stage generates candidates, writes pair diagnostics and group composition, and exits without changing model weights. Review at least:

- `july13_pair_composition.csv`
- `july13_observed_pair_diagnostics.csv`
- `july13_observed_pair_attempts.csv`
- `july13_manifest.json`

## Train and Compare

Train from the approved pair file. The `train` stage revalidates every pair and runs base evaluation before updating the LoRA:

```bash
PAIR_CSV=/workspace/gemma3-fairface-debiasing-july13/outputs/<pair-run>/july13_dpo_pairs_kept_combined.csv \
  PYTHONUNBUFFERED=1 bash scripts/run_train_from_pairs.sh 2>&1 | tee training_july13.log
```

For a small execution-only check:

```bash
bash scripts/run_smoke_train.sh
```

Runtime will depend on how many candidates require fallback generation and on DPO sequence lengths. Expect training plus two 4,000-response evaluations to take substantially longer than the 35-minute base-only run.

## Resume and Backup

Run folders are timestamped and protected against overwrite. Before stopping the pod:

```bash
sync
```

Review and back up these artifacts:

- `july13_manifest.json`
- `july13_dpo_pairs_kept_combined.csv`
- `july13_pair_composition.csv`
- `july13_observed_pair_diagnostics.csv`
- `dpo_loss_history.json`
- `checkpoints/final/`
- base and adapter response CSVs, metric JSON files, and plots
- `03_paired_image_bootstrap.json`

Copy only reviewed, reasonably sized report artifacts into `results/` before committing to GitHub. Store adapter weights on the network volume or a model registry.
