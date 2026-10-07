# Next 100-Per-Group Run

Status: historical execution protocol. The 100-per-group run completed on July 13, 2026. Its final adapter was overtrained, so the 500-update settings below must not be reused as the default for the next experiment. See [`NEXT_ITERATION_PLAN.md`](NEXT_ITERATION_PLAN.md) for the current protocol.

For an unattended staged run, `scripts/run_research_pipeline.sh` performs Stage 1, applies the strict pair-policy and composition gate, runs Stage 2 only when approved, and verifies that all expected result plots exist. Its run directory contains `pipeline_status.tsv`, `pair_audit_approval.json`, and `artifact_index.txt` for remote monitoring.

## Stage 1: Generate Research Pairs

```bash
cd /workspace/gemma3-fairface-debiasing-july13
export HF_HOME=/workspace/.hf
export BASE_MODEL_LOCAL_DIR=/workspace/.hf/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80
bash scripts/run_pair_audit.sh
```

Defaults:

- 100 FairFace training images per intersection group, 400 total
- 16 candidates per image, up to 6,400 generations
- direct negative penalty 4.0
- identity leakage penalty 8.0
- template fraction cap 50%
- minimum 50 kept pairs
- no weight updates in this stage

The 128-candidate smoke pair stage took about 2.4 minutes including model loading. A 6,400-candidate audit is therefore expected to take roughly 1.5-2 hours, depending on fallback generation.

## Pair Approval Gate

Do not start training until the following files have been reviewed:

- `july13_manifest.json`
- `july13_pair_composition.csv`
- `july13_observed_pair_diagnostics.csv`
- `july13_observed_pair_attempts.csv`
- `july13_dpo_pairs_kept_combined.csv`

Approval requirements:

1. `training_ready` is true and at least 50 total pairs survived.
2. The template fraction is no greater than 0.50.
3. Every chosen response has no more negative words or identity leakage than its rejected response.
4. Every pair directly improves negative words or identity leakage.
5. No positive-versus-positive pair without identity improvement is present.
6. Source and race/gender composition have been reviewed for severe imbalance.
7. The pair CSV is from the 100-per-group audit, not `smoke_only_balanced_template_pairs.csv`.

## Stage 2: Train and Compare

Replace `<pair-run>` with the approved Stage 1 directory:

```bash
cd /workspace/gemma3-fairface-debiasing-july13
export HF_HOME=/workspace/.hf
export BASE_MODEL_LOCAL_DIR=/workspace/.hf/hub/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80
export PAIR_CSV=/workspace/gemma3-fairface-debiasing-july13/outputs/<pair-run>/july13_dpo_pairs_kept_combined.csv
bash scripts/run_train_from_pairs.sh
```

Training defaults:

- audited pairs repeated twice
- 2,000 DPO microsteps
- gradient accumulation 4, approximately 500 optimizer updates
- unattended pipeline checkpoints every 100 optimizer updates
- learning rate `5e-5`, beta `0.05`
- eager attention with the frozen vision tower excluded from LoRA
- exact July 6 FairFace validation indices, 100 per intersection group
- 10 generations per image for both base and adapter
- same evaluation generation seed for base and adapter
- 100,000-sample paired image bootstrap

The corrected smoke peaked at 42.59 GiB allocated and 43.96 GiB reserved on the 79.18 GiB H100. Increasing the number of training images increases candidate-generation and reference-score time, but it should not materially increase per-step peak GPU memory.

## Result Decision

The adapter is promising only if the signed Black-minus-White gap decreases, the Black negative rate itself decreases, validity remains at least 99%, and diversity does not collapse. The result remains preliminary until its direction is replicated across at least three seeds.
