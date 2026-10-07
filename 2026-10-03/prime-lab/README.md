# Prime Lab workspace for the URAP IAT study

This workspace is isolated from the existing benchmark and study runtimes.
Prime Lab starter configurations are examples, not approved experiment runs.
Setup does not authorize uploading face images, launching compute, publishing
an environment or replacing SFT/ORPO with reinforcement learning.

## Verified setup

Initialized on October 3, 2026 with the installed Prime CLI 0.9.2:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
rtk proxy "$env:USERPROFILE/.local/bin/prime.exe" --plain lab setup --agents codex --no-interactive --skip-agents-md
```

The command completed successfully, preserving existing repository guidance.
It installed Verifiers 0.2.0 in this folder's isolated Python 3.11.15 environment,
downloaded 15 starter configs and prepared five Prime-managed Codex skills.
The installed CLI pins its skill/template source to Verifiers commit
`6c64ce6a3a01e8edde7c3c0e8e5315fb236e9faa`. Dependencies are frozen in `uv.lock`.
`prime lab doctor` passes all 17 checks, and a local Verifiers import succeeds.
The existing account login was verified; no new API key was created for setup.

An obsolete pyenv launcher currently shadows the working UV-managed CLI when
running bare `prime`. Use the explicit executable above; no system PATH, pyenv
installation or account credentials were replaced. UTF-8 avoids a Windows
console encoding error encountered when displaying token-permission checkmarks.

To check the workspace without launching a run, from this directory:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
rtk proxy "$env:USERPROFILE/.local/bin/prime.exe" --plain lab doctor
rtk proxy uv run --locked python -c "import verifiers; from importlib.metadata import version; print(version('verifiers'))"
```

This setup does not prove that a hosted training model accepts our image-bearing
SFT/ORPO data. The downloaded RL/GEPA/eval examples do not implement the URAP
experiment. Subsequent user authorization supplied $100 of Prime credits, treated
as a total cap with a $20 drafting sub-cap. Paid Prime gateway vision inference
generated 1,920 draft pairs for human review, not a GPU training run.
[Actual drafts, costs and review instructions](../reports/TARGET_DRAFT_REVIEW.md).
Required model/algorithm/image support and launch-time pricing remain unverified.

[The current training-readiness report](../reports/PRIME_TRAINING_READINESS.md)
records actual live model catalogs, the empty dedicated cached-model result,
and a processor-only Gemma 4 check on 96 active-pool training images. It also
distinguishes hosted teacher distillation from the required offline SFT/ORPO.
No training backend is claimed verified by these local checks.

## Study boundaries

The active image pool, frozen research splits, target requirements and current
readiness are defined in [the dataset plan](../reports/CURATED_DATASET_PLAN.md).
Keep the original 12-image benchmark out of training. The new pool contains
1,440 train, 480 development and 480 reserved test images, but no approved
image-specific three-word targets yet. Do not synthesize labels from demographic
metadata or treat automated expression estimates as reviewed answers.

Use the existing account login; do not put credentials into source files or
configs. Local environments and outputs must remain Git-ignored. Nothing should
be pushed from branch `peyton`.

The setup itself uploaded no faces, targets or credentials and launched no paid
work. The later authorized inference step sent only frozen train/development
photos to an external teacher through the Prime gateway; no test/benchmark
photos, credentials, deployment, training job or environment publication were
involved. The approved training target file remains absent until human review.

Official setup reference: [Prime CLI](https://github.com/PrimeIntellect-ai/prime).
