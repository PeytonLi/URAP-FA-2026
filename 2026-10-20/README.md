# 2026-10-20 — Confidence-gated two-pass labeling for SFT/ORPO targets

Builds the training targets (`curated_iat_targets.json`) for the frozen
September-29 image pool: 1,440 train + 480 development images from
`2026-09-29/data/selected_images.csv` (FairFace 1,400 + UTKFace 1,000 total;
480 test images are never labeled — they stay reserved). The 12 benchmark
images remain excluded.

## Historical graph baseline

The exact saved responses behind Peyton's reference word-count graph are frozen
in [`analysis/historical_baseline_v1/`](analysis/historical_baseline_v1/README.md).
Use that export, not the current v4 base rerun, as the historical reference.
Its Black/White negative rates are 37.89% / 6.67%, with 38/61 vocabulary words.
The export records protocol limitations; comparisons with newer native-chat
runs are descriptive until generation and scoring are aligned. All figures
remain separate, using the same 61-word order.

## V3: ORPO-only coverage and response variety

New training uses **Gemma 3 12B-IT ORPO only**. SFT and KTO implementations
and prior results remain archival; the trainer and launchers reject new runs
of those methods. The frozen 61-word test prompt, 12 photos, seeds, 50
repetitions, temperature, decoding and fallback rules are unchanged.

`labeling/policy_v3.py`, `candidates_v3.py` and `export_v3.py` implement:

- Word-specific visible-expression evidence and subjective confidence from
  two separately queried teachers with recorded model IDs. Their errors may be correlated;
  agreement is not human validation or ground truth.
- A candidate is usable only if both teachers support it at confidence
  >=0.8 without review flags. Absence of support is never repaired by
  assigning an uncovered word. Neutrality does not justify a forced mixed
  triple.
- The existing policy excludes 11 unsupported judgments, leaving 50 eligible
  preferred-label words. All 61 remain in evaluation. Training coverage is
  a hypothesis for improving generation, not a hard ceiling on emitted words.
- The existing source match sets include other race groups. V3 explicitly
  selects their Black/White members, without rematching or changing splits:
  480 train and 160 development images before gating. Reserved test images
  and the original 12-image benchmark receive no target generation.
- At most two coverage-curriculum rounds probe missing/rare **training**
  words on a fixed, seeded image budget. A word is rare when fewer than
  eight training images in either group have support. Development and test
  model outputs never choose curriculum words.
- A deterministic greedy coverage heuristic selects 2–4 genuinely different
  unordered triples per image (default four), solely from supported words.
  Whole Black/White matching blocks survive or are dropped together.
  Each block receives the same positive-word-count schedule across its
  four members, so valence parity is exact in the exported labels.
  This does **not** establish equality of the word distributions; audits
  report their TV distance, entropy and per-group counts separately.
- Selection counts are split-local: development examples cannot affect
  training-example selection. Variant expansion happens after identity,
  approval and complete-block validation. Image counts and preference-example
  counts are recorded separately in trainer receipts.

API responses, including refusals and malformed responses, remain in ignored
private storage with image, request, schema and policy fingerprints. A shared
append-only ledger accounts for token usage at pinned-model published prices,
uses conservative reservations for ambiguous failures, prevents simultaneous
runs from overspending, and fails fast on provider/auth/quota errors. The
v3 labeling ceiling is $20 in total, not $20 per stage. Explicit retries
preserve earlier attempts.

The funded Bitdeer profile uses
`https://api-inference.bitdeer.ai/v1` with
`deepseek-ai/DeepSeek-V4.1-Flash-1017` as teacher A and
`zai-org/GLM-5.3-Flash` as teacher B. Both accept images in Bitdeer's
[official model catalog](https://developers.bitdeer.ai/docs/api/models-catalog).
This changes only labeling teachers: Gemma remains the sole training and
benchmark model. The provider-returned model IDs, token usage and raw responses
are retained; the hosted GLM ID is not proof of an immutable weight revision.
The Bitdeer key is saved as a repo-scoped secret, never in source or manifests.
Its environment binding is `URAP_BITDEER_API_KEY`.

The caller uses Bitdeer's documented `max_tokens`, image content blocks and
strict JSON Schema. A standard user agent is necessary: the API rejected
Python's default urllib user agent with HTTP 403, whereas `curl/8.5.0` works.
Bitdeer accounting conservatively uses the official **non-sale, uncached**
list prices: DeepSeek $0.30 input / $1.20 output per million tokens; GLM Flash
$0.15 / $0.50. These are accounting upper estimates, not a wallet invoice.
The same ledger includes earlier OpenAI attempts and retains conservative
reservations for failures whose usage could not be recovered.

Run from the repo root using the existing Python environment:

```bash
# URAP_BITDEER_API_KEY must be supplied through the secure environment.
python 2026-10-20/labeling/candidates_v3.py pilot --provider bitdeer --limit 48
# Inspect pilot acceptance and raw evidence before full labeling.
python 2026-10-20/labeling/candidates_v3.py full --provider bitdeer
python 2026-10-20/labeling/candidates_v3.py curriculum --provider bitdeer --round 1
# Stop if coverage plateaus; otherwise run the second and final round.
python 2026-10-20/labeling/candidates_v3.py curriculum --provider bitdeer --round 2
python 2026-10-20/labeling/export_v3.py --provider bitdeer --variants 4
```

An `--provider openai` profile is retained for reproducibility, but the saved
OpenAI labeling key currently has exhausted credits. Responses from different
profiles cannot be mixed to manufacture teacher agreement. Bitdeer exports
receive a `_bitdeer` suffix rather than overwrite another profile's targets.

After reviewing the private export audit, activate the generated draft
configuration explicitly; export never silently replaces the active dataset.
The preflight requires complete approved blocks, pinned manifests, at least
three development blocks, and exactly matched expanded-example counts.
The existing RunPod launcher retains its $4 spending ceiling, balance reserve,
owned-pod termination checks and mandatory private Hugging Face backups.
No GPU run should start before the data passes preflight.

The v3 worker evaluates a fixed, hash-selected set of three complete
development blocks (12 photos) before the original frozen test. The model is
the predeclared final epoch-three checkpoint, not one selected using test
outputs. The default v3 run is IAT-only; DT/BOLD/VisBias historical results
are not rerun or relabeled as new results.

`analysis/variety.py` produces the same blue/orange word-count and
image-level-rate graphs plus per-image coverage and unordered-triple plots:

```bash
python 2026-10-20/analysis/variety.py \
  --condition base=PATH_TO_BASE_RESPONSES_CSV \
  --condition orpo_v3=PATH_TO_ORPO_RESPONSES_CSV \
  --training-targets orpo_v3=PATH_TO_V3_TARGETS_JSON \
  --out OUTPUT_DIRECTORY
```

It reports the >=50-word **union** target, both race-group coverage counts,
overlap, frequent-word coverage, effective vocabulary size, entropy normalized
by the fixed 61-word vocabulary, per-image unordered triples and entropy,
dominant-triple concentration, validity, fallback counts, words absent from
training, TV distance and negative-rate gaps. Exact 924-assignment
permutations and 10,000 bootstrap draws use images, not repeated generations,
as the inferential unit. A nonsignificant test is not evidence of equality.
Strict sampled responses and final benchmark-scored responses are separate
views: fallback or word-order permutations cannot manufacture expressiveness.

Current status: OpenAI returned `credit_balance_exhausted`; labeling now uses
Bitdeer. Policy `visible_expression_candidates_v3_2` restricts `review_flags` to
image-level blockers and moves caveats to `review_notes`. Its balanced 48-image
pilot (96 calls, 95 valid) found jointly supported words on 11 images, three or
more on 4, and only four words in total: Smiling, Happy, Joyful, Cheerful. No
matched block was complete, so no export, adapter or v3 benchmark exists yet.
`audit_v3.py --provider bitdeer --out PATH` writes this aggregate audit.
`analysis/variety_r2/` is explicitly a reanalysis of stored round-2 base/ORPO
outputs, not a v3 result.

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
