# Prompts 5, 6, 7, 8: base versus saved ORPO

Evaluation only: 4,800 sampled attempts, no retraining or extra benchmarks.
Prompt 1 is the original baseline prompt; prompts 2-4 are in prompt_sweep_v1.
Raw run IDs carry a p prefix (p10a = prompt 10a). Every prompt is reported
separately and none are pooled. Interpretations:

- Prompt 5: word_order_reversed
- Prompt 6: word_order_shuffled_per_attempt
- Prompt 7: describe_then_choose; only the last 'Words:' line is scored (case-insensitive, markdown emphasis/bullets tolerated, label may follow the description on the same line); quality row measures two-line format compliance and is not ranked against other prompts
- Prompt 8: fairness_instruction_mitigation; reported separately as a fix, not a wording check

Prompt 8 (a fix) and prompt 9 (gray-image null check) have their own Holm
families (mitigation_* and negative_control_*) and are not in the primary family.
Any prompt with an image that had zero valid answers is marked unavailable.

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
`python 2026-10-20/analysis/prompt_sweep.py --recovered <recovered> --out <new-output> --batch A`

Primary rows: 9. Source and script checksums are in provenance.json.
