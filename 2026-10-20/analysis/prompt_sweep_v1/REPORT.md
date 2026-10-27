# Prompts 3, 4 and 5: base versus saved ORPO

Evaluation only: 3,600 sampled attempts, no retraining or extra benchmarks.
Every prompt is reported separately. Prompt 3 changes wording; prompts 4 and 5
change task interpretation. They are not pooled as one task.

See statistics_table.csv for effects, 95% image-bootstrap confidence intervals,
raw p-values and Holm p-values. Holm spans all nine primary hypotheses.
Diagnostic and sensitivity adjustments span prompts within their labeled families.
The photo is the independent unit, not each answer or selected word.

Strict sampled output is primary; fallback output and repeated-word exclusion
are sensitivities. Entropy is normalized to log2(61); triples are unordered.
Graphs/summary.csv within each prompt reports per-group rates and vocabulary
coverage. Word-count graphs are separate and share the fixed reference order.

These are exploratory results on 12 already-inspected photos. Nonsignificance
does not establish equivalence; variety does not establish grounded accuracy.
The historical screenshot baseline is unchanged and excluded from tuning tests.

Reproduce with the recovered checksummed run:
`python 2026-10-20/analysis/prompt_sweep.py --recovered <recovered> --out <new-output>`

Primary rows: 9. Source and script checksums are in provenance.json.
