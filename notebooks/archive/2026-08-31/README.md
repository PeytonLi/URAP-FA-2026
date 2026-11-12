# August 31 Multimodel Race-Bias Study

This compact package contains the August 31 analyses, frozen row-level data, main plots,
and a standalone offline reproduction script. It does not contain model weights, face
images, API keys, transport logs, or the larger historical workspaces.

## Start here

- [`report/august31_bias_study_report.pdf`](report/august31_bias_study_report.pdf):
  two-page research brief.
- [`docs/RESULTS.md`](docs/RESULTS.md): complete result interpretation.
- [`docs/METHODS.md`](docs/METHODS.md): benchmark, sampling, and statistical methods.
- [`data/derived/aug31_model_comparison.csv`](data/derived/aug31_model_comparison.csv):
  cross-model summary.
- [`plots/`](plots/): nine presentation-ready figures.

## Main findings

- The archived 12-image IAT-style, DecodingTrust, and BOLD analyses had no
  BH-adjusted `q < 0.05` result.
- Seven hosted model routes completed the 400-image UTKFace test. Every observed
  Black-minus-White negative-term gap was positive (`+2.0` to `+10.9` percentage
  points), but none survived correction across models.
- No VisBias image-description or explicit directional comparison survived BH
  correction. Step 3.7 Flash had the highest unsupported explicit-choice rate
  (`36.3%`).
- GPT-5.6 Luna is missing from the new image benchmarks because its pinned Azure EU
  route repeatedly returned `no healthy upstream`; provider fallback was disabled.

These are exploratory, prompt-dependent results. A corrected null does not prove that a
model is unbiased, and statistical significance does not establish practical harm or a
causal race effect.

## Reproduce figures

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python code/reproduce_aug31.py
.venv/bin/pytest -q code/test_reproduce_aug31.py
shasum -a 256 -c CHECKSUMS.sha256
```

`reproduce_aug31.py` first verifies the frozen row counts, headline rates, task counts,
race/gender balance, blank future-DPO preference fields, and evaluation/training hash
separation. It then regenerates all nine public PNGs from the frozen statistical tables.
It makes no network or model API calls.

## Data boundary

`data/raw/` contains the model responses needed to audit the August 31 results. The
small set of `archived_*` files are included only because August 31 recomputed the
IAT-style, DecodingTrust, and BOLD significance figures from those frozen inputs.
`data/frozen/` contains image/task metadata and SHA-256 hashes, not face images.

The optional `code/prepare_private_inputs.sh` retrieves the pinned public dataset
sources and verifies reconstructed selected images against the frozen hashes. Dataset
licenses and terms still apply; private images must not be committed.

## Scope

This folder reproduces the August 31 statistics and figures. It intentionally omits the
paid API orchestration framework, failed compatibility probes, per-model duplicate
plots, previous DPO/LoRA artifacts, virtual environments, caches, and raw API transport
JSONL files. Exact model aliases and provider routes are in
[`config/models_aug31.json`](config/models_aug31.json); final counts and privacy rules
are in [`config/study_manifest.json`](config/study_manifest.json).
