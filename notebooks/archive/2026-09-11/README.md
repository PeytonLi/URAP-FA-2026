# September 11 IAT61 expansion and Gemma LoRA evaluation

This package evaluates existing frozen models. It does **not** train or modify
Gemma's base weights or its saved LoRA adapter. The selected August 10 adapter was
the strongest directional 12-face example among five inspected configurations,
not an accepted successful debiasing model.

## Research Outputs

- `RESULTS.md`: findings, completeness, costs, limitations, and graph guide.
- `plots/`: per-word plots, negative-term rates, race gaps with p/q values,
  validity/diversity, and the direct paired Gemma tuning effect.
- `results/analysis/neutral12_responses.csv` and `utkface_responses.csv`: model responses.
- `results/analysis/summary.csv`, `significance.csv`, and `gemma_paired_effect.csv`:
  numerical results. Rates are fractions in CSVs and percentages in plots.
- `COVERAGE.md`: requested benchmark/model and benchmark/input tables with links.
- `METHODS.md`: sampling, archive reuse, seed changes, and statistical definitions.
- `DATA_DICTIONARY.md`: column definitions and authoritative input references.
- `QWEN_COMPATIBILITY.md`: the Qwen2.5-VL-3B probe and documented DPO/LoRA support.

No private face images, API keys, environments, or model weights belong in GitHub.
Historical August/September packages are preserved rather than overwritten.

## Reproduce Figures Offline

Use Python 3.11. No GPU, paid API, credentials, or images are needed to replay
the released response data into statistics and figures:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -B -m pytest -q
.venv/bin/ruff check scripts tests
.venv/bin/python -B scripts/analyze.py
.venv/bin/python -B scripts/plot.py
.venv/bin/python -B scripts/report.py
.venv/bin/python -B scripts/dictionary.py
.venv/bin/python -B scripts/verify.py
```

Private-input tests skip explicitly when the private bundle is absent. Statistical
tests still run. `verify.py` checks the release inventory and derived totals;
`--freeze` is for creating a new release after intentional changes.
`scripts/export_public.py DESTINATION` copies only the frozen public inventory
to a new, previously nonexistent folder. It never copies private images, weights,
runtime state, credentials, or environments, and it does not make Git commits.
The release was replayed byte-for-byte in the recorded environment. Other
operating systems/fonts may render different PNG bytes despite matching numbers;
verify original checksums before replay and keep originals for comparison.

## Generate New Responses

Read `METHODS.md` before spending money. Archive response replay is exact;
fresh sampled API generations are **not** guaranteed byte-identical, even with
the same requested seed and catalog revision. The provider's backend is not
an independently hashable checkpoint.

Restore the exact 12 neutral images and 400 UTKFace images to the paths in the
two manifests. Use the researcher's authorized private bundle for Neutral12.
UTKFace source and mirror revision are frozen in
`reference/utkface_freeze_manifest.json`; comply with its access/license terms.
Restore the adapter to `data/private/adapter/` and verify its SHA-256 against
`config/study.json`. `scripts/prepare.py` is an optional, recorded local migration
from the old packages; it is not required on a public clone for offline replay.

Set `OPENROUTER_API_KEY` and `RUNPOD_API_KEY` through your secret manager, or use
macOS Keychain services `openrouter-api-key` and `runpod-api-key` for your login
account. Do not put literal keys in source, commands committed to Git, or CSVs.

```bash
.venv/bin/python -B scripts/hosted.py dry-run
.venv/bin/python -B scripts/hosted.py preflight
.venv/bin/python -B scripts/hosted.py probe
.venv/bin/python -B scripts/hosted.py run
.venv/bin/python -B scripts/finalize.py
```

The runner resumes already completed task IDs. API-error attempts are retained
in append-only ledgers; only a last technical retry occupies each analysis slot.
Invalid model answers are not replaced to improve validity or bias rates.
`finalize.py` makes read-only metadata/balance requests after the worker exits;
it audits archive contracts and does not generate any model answers.
`config/execution.json` records unavailable cells. Remove an exclusion only in
an explicitly documented future session after validating that exact endpoint.

For local inference, use an H100 80 GB, the pinned Gemma snapshot and the locked
Linux environment. `scripts/pod.py` deploys into the configured existing network
volume; `local_job.py` starts a detached bounded worker. See `METHODS.md` for the
full command sequence and cleanup requirements. Delete `runtime/pod.json` only
after verifying the old own-pod ID is terminated and archiving the old receipt.

## Pause And Resume

Create `STOP_REQUESTED` in this package to stop hosted tasks at their next
request boundary. For the remote worker the same file must exist in its package
on the network volume. Let active requests finish and retrieve artifacts before
stopping compute. The hard GPU spend watchdog may terminate compute while
outputs remain safely on the network volume. This is inference, not a training
checkpoint: completed task rows are the resumption units.

## Fireworks Gemma 4 Base Benchmark

This is inference against the unchanged `gemma-4-26b-a4b-it` base/instruction
model, not fine-tuning. Because Fireworks lists it as on-demand only, first create
a scale-to-zero deployment (with `firectl` 1.8.8 or newer):

```bash
firectl deployment create accounts/fireworks/models/gemma-4-26b-a4b-it \
  --deployment-shape default --min-replica-count 0 --max-replica-count 1 \
  --scale-up-window 30s --scale-down-window 5m --scale-to-zero-window 5m --wait
```

Put the API key in the runtime's secret manager, never in this repository. Set
`FIREWORKS_MODEL` to the deployment identifier returned by Fireworks, in the form
`accounts/<account>/deployments/<deployment>`. Confirm that sending these face
images to Fireworks is authorized under the project's consent and retention rules.

```bash
export FIREWORKS_API_KEY='use-your-secret-manager-instead'
export FIREWORKS_MODEL='accounts/<account>/deployments/<deployment>'
python scripts/fireworks_gemma4.py dry-run --image-dir /path/to/12_images/race
python scripts/fireworks_gemma4.py smoke   --image-dir /path/to/12_images/race
python scripts/fireworks_gemma4.py run     --image-dir /path/to/12_images/race
```

The smoke test submits the first 12 cells. The full command resumes those cells
and completes all 600. Outputs, the word-count graph, and the other evaluation
metrics are written under
`results/fireworks/gemma4_26b_a4b_fireworks_no_reasoning/`. The API request uses
`reasoning_effort=none` so Gemma 4 spends the 30-token output budget on the required
three-word answer rather than hidden reasoning. Fireworks
bills on-demand deployments by GPU time rather than returning a reliable per-call
cost here; inspect provider billing and delete or stop the deployment afterward.
