# Research Evidence Package September 5

This package consolidates the Gemma FairFace DPO/LoRA evidence and image/text bias
benchmarks requested in `RA_Evidence_Request_Variable_Glossary`. Historical artifacts
are copied with provenance; the original dated projects are never modified.

## Start Here

- [Formal research report with all appendices](report/FORMAL_RESEARCH_REPORT.pdf)
- [Readable report on GitHub](report/FORMAL_RESEARCH_REPORT.md)
- [Definitions for every column in the three supplied sep5files CSVs](report/SEP5FILES_COLUMN_APPENDIX.csv)
- [All 231 RA guideline responses](report/RA_GUIDELINE_RESPONSES.csv)
- [Supplied exports](data/sep5files/) and [audit of export alterations](report/SEP5FILES_AUDIT.json)
- [15 final figures, PNG and PDF](plots/) and [figure/data/code index](report/plot_registry.csv)
- [Raw and scored benchmark CSVs](results/analysis/)
- [All 157 exploratory tests: p, BH q, and 95% intervals](results/analysis/all_significance_tests.csv)
- [Variable coverage requested by Debbie](report/requested_variable_coverage.csv)
- [Column glossary](report/DATA_DICTIONARY.csv) and [uninterpreted legacy field](report/UNINTERPRETED_FIELDS.csv)
- [Exact 100-row FairFace DPO spreadsheet](reference/dpo/august10/august10_negative_expression_pilot_v4/dataset/mixed_dpo_pairs_frozen.csv)
- [Training record](report/dpo_training_record.json), [failure/history review](FINE_TUNING_HISTORY.md), and [coverage/remaining work](report/TASK_STATUS.md)
- [Reproduction instructions](REPRODUCIBILITY.md) and [session costs/call counts](report/SESSION_LEDGER.json)

New inference is complete: 956 local Gemma tasks plus 688 hosted tasks (700 API
calls including fallbacks). The GPU was terminated after verified download; the
network volume is preserved. **No new training occurred; base weights were unchanged.**

## Read Before Interpreting

Five of nine matched UTKFace race-gap tests have exploratory BH q<.05. This uses
a documented paired test instead of the earlier unpaired analysis; it is not
confirmatory evidence. The local Gemma DecodingTrust cell is unreliable because
36/48 outputs were truncated. The neutral12 continuity rerun matched only 28/120
archived triples; the original baseline, not the rerun, is used for historical
comparisons. No prior DPO adapter met the joint bias/diversity preservation goal.

Offline replay requires no API key, GPU or private images. Fresh inference and
exact historical retraining have additional access and reproducibility limits
documented in `REPRODUCIBILITY.md`. Only one final figure version is presented;
older raw evidence is retained separately for provenance, not as competing slides.

## Published Release

This package is published in [URAP/2026-09-05 at sep5-evidence-v1](https://github.com/eliaswuberkeley/URAP/tree/sep5-evidence-v1/2026-09-05).
The formal report's links target that release. No Google Sheet URL was supplied;
the CSV exports and exact canonical originals are downloadable from GitHub.
The three attached exports are not exact frozen inputs: 448 BOLD prompts lost
trailing spaces, two responses lost leading apostrophes, and some numeric
metadata were rounded. Use the canonical files linked in the audit for replay.

Private images, credentials, large weights and temporary deployment files are not
GitHub deliverables. The authorized publication uses only the reviewed list in
`report/PROPOSED_GITHUB_FILES.txt`. Earlier dated GitHub folders are unchanged.
The earlier compact PDF is superseded for presentation by the formal report and
is excluded from this release; historical evidence remains available for audit.
