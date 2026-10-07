# Coverage And Remaining Work

This is an evidence and inference package, not a new fine-tuning experiment.
All focal-model rows are placed first in the new presentation figures.

## Completed

- The 61-word IAT protocol is retained; it was not changed to exactly 60 words.
- Pinned local Gemma on the 400-image matched UTKFace panel: all 400 valid.
- GPT's previously missing UTKFace cell: 400 attempted, 397 valid, including 12 fallback calls.
- Neutral12 VisBias for local Gemma and all eight hosted models: descriptions
  plus income/education comparisons in both image orders.
- Original 84-image VisBias subset for pinned Gemma, combined with seven
  available hosted archives. GPT on this original subset remains missing.
- Local Gemma BOLD: 100 continuations, 100 scored. Re-scoring 16 archived texts
  reproduced all saved toxicity, sentiment and regard classes.
- Local Gemma neutral12 continuity audit: 120 responses, 28 exact parsed-triple
  matches. Base-file hashes did not change; the frozen baseline is retained.
- DecodingTrust/BOLD race-focused archives consolidated with category/quality
  counts. Statistics use image/pair/topic/subject clusters, not token independence.
- All 157 exploratory tests, raw p-values, BH-adjusted q-values, confidence
  intervals, 15 presentation figures, the formal report and detailed methods.
- All 122 column occurrences in the three supplied sep5files exports defined;
  all 231 RA guideline entries addressed. Export alterations documented without
  modifying the attachments; exact replay uses the canonical records.
- FairFace DPO spreadsheet and available training source/manifests/logs audited;
  100 private image hashes verified; overlap, judge provenance, train/development
  composition and the best directional evaluation documented.
- Requested-variable records, column glossary, failure history, exact paths,
  existing GitHub links, reproducibility commands and checksums.
- H100 outputs downloaded and verified; compute terminated, volume preserved.

## Incomplete Or Not Interpretable

- **Local Gemma DecodingTrust:** 36/48 responses truncated at 150 tokens.
  Saved diagnostic counts are not a reliable full-response estimate. The older
  hosted Gemma run also has 34/48 length-limited outputs and is not a substitute.
- **Exact local baseline replay:** unchanged weights, matching explicit prompt
  and order, but different sampled outputs. The missing full original effective
  runtime state prevents identifying the precise cause.
- **UTKFace fine-tuning pool:** 400 separate images and 800 prompt rows exist,
  but chosen/rejected fields are blank. It is not a ready-to-train DPO dataset.
- **FairFace cross-model matrix:** historical Gemma DPO/evaluation is preserved;
  the eight hosted models have not all been run on FairFace IAT/VisBias.
- **Post-DPO evaluation:** historical neutral12 and FairFace evidence exists;
  no complete post-DPO UTKFace/VisBias/DecodingTrust/BOLD suite is available.
- **Teacher/judge provenance:** recorded model labels are supplied. Exact
  immutable GPT/Claude backend versions and historical executed source commits
  were not recorded and cannot be reconstructed honestly.
- **Independent human validation and capability preservation:** neither the
  AI-only labels nor the model's general factual accuracy have been established.
  No adapter is accepted as successfully debiased.
- **One legacy flag:** `service_test` is false on retained rows but its original
  operational meaning is unverified; current analysis does not use it.
- **Publication:** the user authorized the `2026-09-05` GitHub release at tag
  `sep5-evidence-v1`. No Google Sheet URL was provided and no direct transfer to
  Debbie was requested; the report links the published code and CSV files.

DecodingTrust and BOLD are text-only. Crossing them with 12-face/FairFace/UTKFace
image sets is **not applicable**, not missing image-model execution. The exact
cell-level distinctions are in `benchmark_matrix.csv`.

## Next Steps

1. Share the formal report/release and arrange authorized private image access
   for collaborators who need new image inference, not just plot replay.
2. Resolve deterministic baseline controls; rerun the entire fixed Gemma DT
   panel with a prespecified longer budget before interpreting that endpoint.
3. Freeze a fresh person-disjoint confirmatory panel and preregister endpoints,
   prompt/scorer versions, statistical families and preservation thresholds.
4. Repair per-split expression coverage, verify grounding and judge agreement,
   then run a short fresh-LoRA pilot selected using development data only.
5. Require gap improvement alongside expression sensitivity, output validity,
   diversity and independent capability checks; replicate across seeds before
   making a debiasing claim. Do not force all outputs positive or train on the
   repeatedly inspected evaluation panel.
