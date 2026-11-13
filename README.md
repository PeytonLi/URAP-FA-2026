# URAP-FA-2026

Private Fall 2026 research repository: https://github.com/PeytonLi/URAP-FA-2026.

Can we make a vision-language model less biased without making it worse at other tasks?

This UC Berkeley URAP project studies that question using **Gemma-3-12B-IT**. We fine-tuned with LoRA/QLoRA (SFT / ORPO / KTO across rounds), then tested whether changes carried over to new prompts and benchmarks.

## The idea

In the main IAT-style test, the model sees a face and chooses three words from a fixed 61-word list. We measure how often it chooses negative words for Black versus White faces, plus vocabulary variety, and run independent tests (DecodingTrust, VisBias, BOLD) where available.

## Experiment versions

| Version | What changed | Why |
| --- | --- | --- |
| **Base** | Original `google/gemma-3-12b-it` | Starting point |
| **V1** | Dual-teacher gated labels; SFT + ORPO | First gated IAT fine-tune |
| **V2** | Mixed-valence relabel; SFT + ORPO + KTO | Reduce global negative-word shift |
| **V3** | Coverage / multitask labeling pilot | Design only — no exported adapter |
| **V4** | Matched-block answers; ORPO only | Improve response quality and fairness on the word test |

Prompt sweeps (prompts 2–21) evaluate **base vs saved V4 ORPO** under alternate framings; they are not new training versions.

## What's in this repository?

| Folder | Contents |
| --- | --- |
| [`datasets/`](datasets/) | Shared FairFace/IAT-12 manifests and V1–V4 selected-image CSVs |
| [`notebooks/`](notebooks/) | Base Colab notebook; pre-FA chronology under `notebooks/archive/` |
| [`configs/`](configs/) | V1/V2/V4 dataset configs and prompt-sweep analysis plans |
| [`results/`](results/) | Base baseline, V1–V4 IAT analyses, extras, and prompt-sweep outputs |
| [`scripts/`](scripts/) | Train/eval harness, labeling, analysis, and archived helpers |
| [`docs/`](docs/) | Reports, writeups, and reproduction notes |
| [`tests/`](tests/) | Unit tests for labeling and train/eval |

### Where to start

| Experiment | Open this first |
| --- | --- |
| Notebook index | [`notebooks/README.md`](notebooks/README.md) (`01`–`08` reproducible Jupyter) |
| Narrative (V1–V4) | [`docs/REPORT_v1_v4.md`](docs/REPORT_v1_v4.md) |
| V4 writeup | [`docs/V4_WRITEUP.md`](docs/V4_WRITEUP.md) |
| **Reproduce ORPO V4** | [`docs/V4_REPLICATION.md`](docs/V4_REPLICATION.md) |
| Base Colab (GPU) | [`notebooks/base/neutral12_base_colab.ipynb`](notebooks/base/neutral12_base_colab.ipynb) |
| V4 IAT stats | [`results/v4/iat/iat_stats_v4/`](results/v4/iat/iat_stats_v4/) |
| Prompt sweeps C/D | [`results/v4/prompt_sweeps/C`](results/v4/prompt_sweeps/C), [`…/D`](results/v4/prompt_sweeps/D) |
| Train/eval harness (`.py` kept) | [`scripts/train_eval/`](scripts/train_eval/) |

All training/analysis **Python modules remain under `scripts/`**. Notebooks import them via `scripts/repo_paths.py`; they do not replace the `.py` files.

## Active scope

- Use **Gemma 3 12B-IT only** for new training and evaluations.
- Keep the 61-word / 12-image evaluation protocol; evaluation images must not enter training.
- Next image pool: `C:/Users/lipey/Downloads/FairFace 1400/` (audit before activating).
- Keep private Hugging Face adapter backups (one repo per run).
- Never commit credentials, raw face images, weights, runtimes, or cloud auth files.

## Model files and data access

Small configs and CSVs are in Git. The **ORPO V4 replication pack** is also tracked under [`datasets/v4/replication/`](datasets/v4/replication/) (matched targets + the 640 train/dev faces + IAT-12 eval JPEGs). See [`docs/V4_REPLICATION.md`](docs/V4_REPLICATION.md).

LoRA weights, the full FairFace/UTKFace private pool, and RunPod recoveries stay ignored under paths such as:

- `datasets/shared/fairface_pool_private/` (full pool; V4 subset is copied into `datasets/v4/replication/`)
- `datasets/private_targets/` (legacy local copy; tracked V4 targets live in the replication pack)
- `scripts/train_eval/_data/private/`
- `scripts/train_eval/_runtime/`

## Research notes

Analyses use **12 held-out images, 61 words, 50 generations per image**, with the **photo as the independent unit**. Compare both the Black–White negative gap and overall negative rate / variety.

Layout is top-level `datasets/`, `configs/`, `docs/`, `notebooks/`, `results/`, and `scripts/`, with version folders **base → v4**. Calendar-dated week folders have been removed; older chronology lives under `notebooks/archive/`.
