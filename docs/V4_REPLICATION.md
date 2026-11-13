# Reproduce ORPO V4 from Gemma-3-12B-IT

This private repo now includes a self-contained pack so a collaborator can
re-fine-tune Gemma base with the same V4 matched ORPO recipe, run the frozen
IAT-12 eval, and compare against published V4 numbers.

## What is in Git

| Asset | Path |
| --- | --- |
| Matched ORPO targets (480 train + 160 dev images, 4 variants each) | [`datasets/v4/replication/curated_iat_targets_v4.json`](../datasets/v4/replication/curated_iat_targets_v4.json) |
| Hash-verified train/dev face PNGs (640) | [`datasets/v4/replication/dataset/images/`](../datasets/v4/replication/dataset/images/) |
| Selected-image manifest | [`datasets/v4/selected_images_gated_v4.csv`](../datasets/v4/selected_images_gated_v4.csv) |
| Active trainer config | [`scripts/train_eval/config/active_dataset.json`](../scripts/train_eval/config/active_dataset.json) |
| V4 AI-approval waiver (required by trainer gates) | [`scripts/train_eval/config/prime_budget.json`](../scripts/train_eval/config/prime_budget.json) |
| IAT-12 eval manifest + 12 JPEGs | [`datasets/shared/iat12/iat12/`](../datasets/shared/iat12/iat12/) |
| Published V4 tables / plots | [`results/v4/iat/`](../results/v4/iat/) |
| Plain-language write-up | [`docs/V4_WRITEUP.md`](V4_WRITEUP.md) |
| Checksums for the pack | [`datasets/v4/replication/checksums.json`](../datasets/v4/replication/checksums.json) |

LoRA weights are **not** required to retrain. The private Hugging Face adapter
backup is only for loading the original checkpoint without retraining.

## What you bring

1. Hugging Face access to gated `google/gemma-3-12b-it` (revision pinned in code:
   `96b6f1eccf38110c56df3a15bffe176da04bfd80`).
2. A write-capable HF token (the native trainer creates a private Hub backup per run).
3. One BF16 CUDA GPU (original run: RunPod A40, ~4.3 h, ~$2.90).
4. Python deps used by `scripts/train_eval/` (`torch`, `transformers`, `peft`, …).

## Recipe (must match)

- Method: **ORPO only** on Gemma-3-12B-IT
- LoRA rank 8 / alpha 16 / dropout 0 (language-attention modules)
- AdamW lr `1e-5`, weight decay 0, logical batch 4, 3 epochs, seed 0
- ORPO coefficient `0.1`
- 1,920 train preference pairs → 1,440 optimizer updates
- Development: 640 preference pairs (no gradients)

## Train

From the repo root, with HF login available:

```bash
# Confirm the pack is complete (images + targets + hashes)
python scripts/train_eval/study.py prepare

# Local single-GPU ORPO (same entry the RunPod launcher uses remotely)
python scripts/train_eval/native_train.py --model gemma3_12b --method orpo --cohort full --max-seconds 20000
```

Or use the RunPod launcher if you prefer managed pods:

```bash
python scripts/train_eval/runpod_native.py launch
```

Keep the resulting `epoch_3` adapter directory for evaluation.

## Evaluate on frozen IAT-12

```bash
python scripts/train_eval/native_benchmark.py \
  --model gemma3_12b \
  --method orpo \
  --checkpoint PATH/TO/epoch_3 \
  --image-dir datasets/shared/iat12/iat12/images \
  --max-seconds 20000
```

Optional development check (12 held-out training-pool faces, not the frozen 12):

```bash
python scripts/train_eval/native_benchmark.py \
  --model gemma3_12b \
  --method orpo \
  --checkpoint PATH/TO/epoch_3 \
  --image-dir datasets/shared/iat12/iat12/images \
  --development \
  --max-seconds 20000
```

## Compare to published V4

Reference tables live under:

- Frozen IAT fairness / variety: [`results/v4/iat/iat_stats_v4/`](../results/v4/iat/iat_stats_v4/)
- Variety plots / CSVs: [`results/v4/iat/variety_v4/`](../results/v4/iat/variety_v4/)
- Narrative numbers: [`docs/V4_WRITEUP.md`](V4_WRITEUP.md) and [`docs/REPORT_v1_v4.md`](REPORT_v1_v4.md) §8

Targets for a successful reproduction (frozen IAT-12, image as unit):

| Measure | Published ORPO V4 |
| --- | ---: |
| Valid / 600 | 563 |
| Vocabulary union / 61 | 53 |
| Effective vocabulary size | ~47.6 |
| Median distinct triples / image | 46.5 |
| Black−White negative gap | about −0.9 pp (ns) |

GPU nondeterminism means **bit-identical** adapters are not expected. Treat the
run as a success if validity, vocabulary, variety, and the Black−White gap land
in the same qualitative regime as the table above (large variety, near-zero gap,
modest invalid rate), then drop the per-image CSVs next to
`results/v4/iat/variety_v4/per_image.csv` for a side-by-side check.

## Integrity checks

```bash
python -c "import hashlib,json; from pathlib import Path; c=json.loads(Path('datasets/v4/replication/checksums.json').read_text()); t=hashlib.sha256(Path('datasets/v4/replication/curated_iat_targets_v4.json').read_bytes()).hexdigest(); assert t==c['targets_sha256']; print('targets ok', t)"
python scripts/train_eval/study.py prepare
```

`prepare` must report zero missing images and zero hash mismatches before training.
