# RunPod Reproduction

## Hardware and model

The verified environment used one NVIDIA H100 80 GB HBM3, Python 3.11,
PyTorch 2.4.1 with CUDA 12.4, Transformers 4.50.0, and PEFT 0.14.0.

Accept the Gemma license on Hugging Face, authenticate on the pod, and pin this
exact snapshot:

```text
google/gemma-3-12b-it
revision 96b6f1eccf38110c56df3a15bffe176da04bfd80
```

Use SDPA. Do not switch to eager attention on a single 80 GB H100 without first
changing the memory design; the prior eager attempt reached about 78.8 GiB and
OOMed.

## Environment

```bash
cd /workspace/gemma3-fairface-debiasing-august10
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-runpod.txt
pytest -q
ruff check .
```

Provider credentials must be environment variables or secret-manager values.
Never add API keys, `.env` files, raw provider checkpoints, or response IDs to
Git.

## FairFace assets

Images are excluded from Git. The historical CSV expects 100 files under:

```text
data/august10/frozen_pilot_v4_negative_expression/images/
```

Each file must match the `image_sha256` in
`results/august10_negative_expression_pilot_v4/dataset/mixed_dpo_pairs_frozen.csv`.
The trainer rejects missing files, path escapes, hash mismatches, repeated
images, and overlap with frozen evaluation hashes.

The historical CSV is not authorized for a new run. Verify the post-run v2
decision:

```bash
python scripts/check_dataset_gate.py \
  --dataset results/august10_negative_expression_pilot_v4/dataset/mixed_dpo_pairs_frozen.csv \
  --config config/pilot_gate.json \
  --output /tmp/postrun_dataset_gate_v2.json
```

Expected exit code: `2`, failed check: `train_negative_iat_blocks`.

## Construct the next dataset

The resumable stages are separate by design:

1. `select_negative_expression_expansion.py`: prespecify matched FairFace images.
2. `run_gemma_response_generation.py`: generate pinned-base Gemma candidates.
3. `run_codex_gpt_teacher_generation.py`: generate independent teacher profiles.
4. `build_candidate_pairs.py`: build cross-model candidate comparisons.
5. `prepare_claude_judging_batch.py`: mechanical prefilter and recoverable-block plan.
6. `run_claude_pair_audit.py`: blinded image/text and order-reversal judging.
7. `freeze_audited_pilot.py`: deterministic matching, split, materialization, and gate.

Run each script with `--help` before a paid or GPU stage. Use dry-run modes,
persist checkpoints under the network volume, and summarize provider usage with
`summarize_claude_usage.py`.

The next freeze must add at least three negative-expression IAT blocks to train
while retaining at least three disjoint blocks in development.

## Base development screen

Generate base responses once after the new split is frozen. Keep the output CSV
and reuse it for every adapter checkpoint:

```bash
python scripts/evaluate_development_generation.py \
  --frozen-csv outputs/august10/frozen_next/mixed_dpo_pairs_frozen.csv \
  --base-only \
  --asset-root /workspace/gemma3-fairface-debiasing-august10 \
  --output-dir outputs/august10/base_development_next \
  --model-id google/gemma-3-12b-it \
  --model-revision 96b6f1eccf38110c56df3a15bffe176da04bfd80 \
  --base-model-local-dir "$BASE_MODEL_LOCAL_DIR" \
  --attn-implementation sdpa \
  --generation-seed 8102026 \
  --iat-repeats 5 \
  --profile-repeats 2 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 8102026
```

Do not train if this base challenge lacks enough negative events or profile
coverage.

## Fresh DPO LoRA

The next candidate should start with mean-token scoring and one pass. The exact
learning rate and any task weighting must be prespecified before looking at its
development generations.

```bash
python scripts/train_mixed_dpo_lora.py \
  --frozen-csv outputs/august10/frozen_next/mixed_dpo_pairs_frozen.csv \
  --freeze-manifest outputs/august10/frozen_next/freeze_manifest.json \
  --gate-config config/pilot_gate.json \
  --forbidden-eval-hashes ../gemma3-fairface-debiasing-august3/reference/frozen_base/frozen_input_manifest.json \
  --repo-root /workspace/gemma3-fairface-debiasing-august10 \
  --output-dir outputs/august10/training_next \
  --model-id google/gemma-3-12b-it \
  --model-revision 96b6f1eccf38110c56df3a15bffe176da04bfd80 \
  --base-model-local-dir "$BASE_MODEL_LOCAL_DIR" \
  --attn-implementation sdpa \
  --dpo-beta 0.05 \
  --learning-rate 5e-6 \
  --gradient-accumulation 8 \
  --max-gradient-steps 80 \
  --save-every-updates 1 \
  --sequence-logp-reduction mean \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --seed 8102026
```

This command is a template, not authorization to reuse the historical CSV.
Run the same command with `--dry-run` first.

## Checkpoint evaluation

Evaluate each candidate with the exact stored base response CSV and the same
generation seed. Then apply the strict gate:

```bash
python scripts/check_checkpoint_gate.py \
  --summary outputs/august10/development_checkpoint_next/development_generation_summary.json \
  --config config/pilot_gate.json \
  --output outputs/august10/development_checkpoint_next/checkpoint_gate_v2.json
```

Only `promote_checkpoint_to_seed_replication` authorizes three fresh training
seeds. Exact ties, floor effects, missing groups, identity leakage, and any
directional regression stop the run. Do not open confirmatory panels after a
failed development gate.

## Public result bundle

Large adapters and raw provider records remain ignored. After all required
outputs are present, regenerate tracked metrics, plots, curated CSVs, and
checksums:

```bash
python scripts/compile_august10_pilot_results.py --replace
```
