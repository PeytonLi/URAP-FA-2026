# Work Log

## September 5

- [complete] Copy benchmark sources and selected inputs with 1,046 provenance records.
- [complete] Audit historical DPO evidence and missing metadata; no accepted successful adapter.
- [complete] Freeze 956 local tasks and 36 hosted VisBias12 tasks per model.
- [complete] Validation: 53 unit tests pass in a clean standalone environment; 956-task image hash dry run passes; hosted/local probes pass.
- [complete] Hosted neutral12 VisBias: 288 responses across eight models; reported cost $0.208611944.
- [complete] Recover GPT UTKFace: 400 tasks, 412 calls including 12 fallbacks, 397 valid outputs; account delta $0.03710110.
- [complete] Pinned local Gemma: 956 tasks, 961 generations, 920 valid; all 36 invalid outputs are truncated DecodingTrust responses. No training, no adapter loaded.
- [complete] BOLD neural scoring of 100 local responses; 16/16 archived scorer replay cases match all three classes.
- [complete] Analyze uncertainty/validity/diversity, including 157 exploratory tests and five BH-significant UTK comparisons.
- [complete] Build two-page report, 231 requested-variable entries, 1,099 column entries and 15 final PNG/PDF figure pairs. One legacy flag remains uninterpreted.
- [complete] Isolated public replay without private images or request DOCX: 51 tests passed, two private-image tests skipped; all 61 compared CSV/figure/PDF hashes match. Historical 1,343 copies and public privacy/checksum verification pass.
- [complete] Verify 12 remote artifact hashes and terminate GPU at 08:09 UTC; network volume preserved.
- [superseded] GitHub publication initially required approval; the user later
  explicitly authorized the September 5 release (see follow-up below).

2026-09-05 07:24 UTC: Began read-only audit. RunPod key has RW permissions;
network volume is URAP Network Volume (80 GB, US-NE-1); H100 SXM availability was
low at $3.49/hour. No GPU started yet. OpenRouter balance $17.118872358.

Protocol decisions are recorded before new generation. Completed training is not
equated with successful debiasing. Missing or unrecorded historical metadata will
be marked unknown. DecodingTrust and BOLD are text-only and are not crossed with
face datasets as though they used those images.

Preflight: Qwen DeepInfra BF16 route returned unavailable status -2. Other seven
model routes passed catalog/revision checks. No provider substitution is authorized.
The 956 local tasks consist of 120 neutral IAT continuity responses, 400 UTKFace IAT
responses, 36 neutral-face VisBias tasks, 252 original-stimulus VisBias tasks,
48 DecodingTrust race prompts, and 100 BOLD race prompts. The text tasks preserve
the archived system instructions. All tasks are inference-only.

Audit corrections before final analysis: BOLD has100 prompts but86 subjects, so
uncertainty clusters repeated-subject responses. DecodingTrust clusters four
shared topics. VisBias comparison uncertainty clusters both attributes and both
orders by face pair, not pair-attribute. Historical DPO response entropy is
whole-response entropy in nats, distinct from newer word entropy in bits.
The historical419 forbidden hashes contain412 image hashes plus seven reference
artifact hashes; they are not419 evaluation images.

Neutral12 continuity result:28/120 exact parsed-triple matches with the archive;
new Black/White rates41.6667%/1.6667% versus35%/3.3333%. Prompt hashes and image
ordering agree. Full original effective generation/runtime state was not
recorded, so the exact mismatch cause is not established. This fresh sample is
retained only as an audit, never substituted into historical DPO comparisons.

Final billing snapshot at 08:17:51 UTC: 700 OpenRouter generation calls, including
12 GPT fallbacks, cost $0.251057064. Summed per-call costs match the final account
usage delta; the earlier immediate GPT balance snapshot had not settled fully.
OpenRouter remaining $16.867815294. RunPod account delta $2.0728305953; GPU
deployment-to-termination estimate $2.0634095354 at $3.49/hour. RunPod remaining
$17.0629664033; existing storage continues at $0.008/hour. Total observed account
deltas $2.3238876593. No direct Anthropic/OpenAI/Gemini API calls.

The local DT cell is explicitly not interpretable as a full-response estimate;
the historical hosted Gemma DT archive is also length-limited and not substituted.
The paired UTK tests differ from the earlier unpaired analysis; reused generation
rows have not changed, and the method change is explicitly exploratory.

Visual QA: all 15 figures and both report pages inspected. The first PDF-render
attempt encountered a sandbox font-cache configuration error; rendering succeeded
with a package-local Fontconfig cache. This was a presentation-tool issue, not a
model-generation failure. No scientific outputs were omitted as a result.

## Formal Report And Authorized Publication Follow-Up

- The user removed the two-page limit and authorized pushing the completed
  package to GitHub. The original compact report is not a competing public
  presentation version.
- Preserved all three sep5files exports byte-for-byte. Audited 960 IAT, 384
  DecodingTrust and 800 BOLD rows against their canonical August 24 records.
  The exported BOLD file lost trailing spaces in 448 prompts and initial
  apostrophes in two responses. Numeric metadata were also rounded. These
  alterations are documented; exact replay uses the canonical files.
- Built the 46-page formal write-up: benchmark coverage and findings, methods
  and rationale, FairFace DPO and post-DPO evaluation, failures, limitations,
  next steps, all 122 attached column occurrences, all 231 RA requested-variable
  entries, all 157 significance tests, model versions and full source paths.
- Reviewed all 46 rendered pages. Corrected the significance appendix so BOLD
  rows identify sentiment, regard or toxicity as well as the prompt regime.
- Added attachment and report regression tests: 58 local tests pass; Ruff
  checks pass. Removed an inherited unpublished project from the dependency
  lock so the documented clean installation works.
- Fresh independent public environment: dependency check passed, imports
  verified inside the copied source, 56 tests passed and two private-image
  tests skipped. All 64 compared data, figure, report and appendix hashes
  matched. No API calls, training or GPU use were needed for this follow-up.
- Publication target: `eliaswuberkeley/URAP/2026-09-05`, release tag
  `sep5-evidence-v1`. Only reviewed distributable files and the root README
  are included; earlier dated folders are unchanged. The tag is the
  authoritative publication check; local push verification is kept in runtime.
