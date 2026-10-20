# 2026-10-20 — Confidence-gated two-pass labeling for SFT/ORPO targets

Builds the training targets (`curated_iat_targets.json`) for the frozen
September-29 image pool: 1,440 train + 480 development images from
`2026-09-29/data/selected_images.csv` (FairFace 1,400 + UTKFace 1,000 total;
480 test images are never labeled — they stay reserved). The 12 benchmark
images remain excluded.

## What changed vs the October pipeline

- The October run used a single teacher pass (Haiku drafts + a direct review
  pass). This package labels every image with **two independent passes** and
  keeps a target only when they agree.
- **Gate** (Peyton's ask: keep only confident labels): identical chosen word
  set on both passes + expression agreement (not neutral/uncertain) + both
  confidences >= 0.8 + no blocking flags. Two-of-three word overlap routes to
  a Devin adjudication queue instead of a forced merge; disjoint sets and
  low-confidence images stay unresolved — nothing is invented to fill them.
- The report includes **per-race/gender/expression resolution rates** so a gate
  that disproportionately drops one demographic cell is visible, plus a
  matched-block divergence audit (`match_set_id`).
- Output schema matches `2026-10-03` targets: `image_id`, `image_sha256`,
  `split`, `prompt` (frozen IAT prompt), `chosen`, `rejected`,
  `review_status: approved`, `reviewer`, `target_provenance`.

## Image source

`2026-09-29/private/dataset/` rebuilt on Linux via
`2026-09-29/scripts/rebuild_images.py` logic: 2,400/2,400 source-image SHA-256
verified, 2,400/2,400 decoded-pixel SHA-256 (`pixel_sha256`) verified.
PNG container bytes differ from the Windows-built originals (platform
libpng/zlib encoding); `rebuild_verification_linux.json` records this, and a
byte-identical rebuild is required before the repo's `study.py prepare`
byte-check accepts them for training. Labeling is unaffected — it reads
decoded pixels.

## Usage

```bash
# pass A (hosted teacher; needs URAP_TEACHER_API_KEY, optional
# URAP_TEACHER_BASE_URL / URAP_TEACHER_MODEL overrides)
python 2026-10-20/labeling/teacher_api.py --pass-tag A --workers 8

# pass B — same caller, different pass tag; store is append-only/resumable
python 2026-10-20/labeling/teacher_api.py --pass-tag B --workers 8

# gate + reports
python 2026-10-20/labeling/gate.py

# export approved targets + audit (refuses on schema errors)
python 2026-10-20/labeling/export.py
```

Adjudication: `gate.py` writes `data/private/target_review/adjudication_queue.jsonl`.
A reviewer (Devin direct-view) writes `adjudication_decisions.jsonl` with the
final `chosen`/`rejected` plus `decision: approved`; `export.py` merges them.

## Tests

`2026-10-20/tests/test_labeling.py` — 15 unit tests over schema validation,
gate rules, and manifest integrity (2,400 rows, splits).
