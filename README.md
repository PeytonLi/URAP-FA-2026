# URAP-FA-2026

Private Fall 2026 research repository: https://github.com/PeytonLi/URAP-FA-2026.
All further project changes and commits belong here, not in Elias's repository.

## Active scope

- Use **Gemma 3 12B-IT only** (`google/gemma-3-12b-it`) for new training and evaluations.
- Retain the 61-word / 12-image evaluation protocol; evaluation images must not enter training.
- The user-supplied **FairFace 1400** folder is the intended next image pool.
  Audit its manifest and preserve train/development/test splits before using it;
  1,400 images in a folder does not mean 1,400 training examples.
- Keep automatic private Hugging Face adapter backups: one model repo per training run.
- Keep credentials, raw training images, runtime environments and model weights out of Git.

The [completed October pilot](2026-10-03/reports/native_balanced/COMPARISON.md)
and [verified Hugging Face backups](2026-10-03/results/hugging_face_backup_setup.json)
are retained as historical results, not new Gemma 3 results. Existing multimodel
scripts/configuration are historical and must not be launched unchanged for the next study.

## Migration

October 7, 2026: this repository starts from a clean snapshot of the local work,
including code, notebooks, manifests, reports and plots. A hard-coded credential
in an old notebook was replaced with a placeholder. Original Git history remains
available locally on `peyton`/`main` and in the `upstream` reference; it is not
pushed into this repository. The active branch is `codex/fa-2026` and `origin`
points only to this private repository. The local workspace remains in place,
including its ignored private data and recovered checkpoints.

## Historical project material

The dated folders preserve Gemma bias-evaluation and DPO/LoRA research snapshots.
The chronology below refers to original upstream commits, not this clean snapshot's history.

## Chronology

| Date | Original commit | Contents |
| --- | --- | --- |
| [`2026-03-04/`](2026-03-04/) | `675d28f` | Initial four-part Gemma debiasing notebook |
| [`2026-03-11/`](2026-03-11/) | `fa1ade6` | Optimized and automated training notebooks |
| [`2026-03-20/`](2026-03-20/) | `518c528` | Adapter comparison and Amazon post-hoc analysis |
| [`2026-04-01/`](2026-04-01/) | `8a55119` | April 1 IAT and summary workflow |
| [`2026-04-29/`](2026-04-29/) | `1f1e5cc` | FairFace DPO/LoRA training and adapter comparison notebooks |
| [`2026-07-27/`](2026-07-27/) | `728f3a7` | FairFace IAT core and July 19 workflow snapshot |
| [`2026-08-10/`](2026-08-10/) | `bde63d3` | Candidate construction, teacher generation, Claude audit, and mixed DPO training |
| [`2026-08-17/`](2026-08-17/) | `38510ba` | Multimodel IAT, BOLD, and DecodingTrust implementation |
| [`2026-08-24/`](2026-08-24/) | `304d91c`, `688950f` | Eight-model reporting, orchestration, route validation, setup, and verification |
| [`2026-08-26/`](2026-08-26/) | current branch | Gemma-inclusive presentation plots and compact reproducibility package |
| [`2026-08-31/`](2026-08-31/) | current branch | UTKFace IAT-style benchmark, image-based VisBias pilot, significance figures, raw data, and compact offline reproduction |
| [`2026-09-05/`](2026-09-05/) | `sep5-evidence-v1` | Formal research report, RA variable responses, supplied-column appendix, pinned Gemma benchmarks, DPO evidence, and standalone reproduction |

## Current Research Package

Start with the [September 5 evidence package](2026-09-05/README.md) and its
[formal research report](2026-09-05/report/FORMAL_RESEARCH_REPORT.md).
It consolidates the pinned Gemma and hosted-model benchmarks, FairFace DPO history,
post-DPO results, limitations and next steps. The appendices explain every column
in the three supplied spreadsheets, answer all 231 RA guideline entries, and
report all 157 exploratory tests. Its standalone offline workflow regenerates
the report and all 15 current figures without a GPU, API key or private images.

The [tagged release](https://github.com/eliaswuberkeley/URAP/tree/sep5-evidence-v1/2026-09-05)
provides stable report links. Earlier dated packages remain unchanged as historical
records; the August 26 folder retains the earlier Gemma-inclusive presentation plots.

Large runtime material is intentionally excluded from Git: model weights, LoRA checkpoints,
virtual environments, caches, secrets, and RunPod offloads should remain in controlled storage.
Historical notebooks and scripts are preserved as research snapshots and may require their
original environments rather than the lightweight August 26 plotting environment.
