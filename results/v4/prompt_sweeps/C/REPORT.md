# Prompts 14, 15, 16, 17: base versus saved ORPO

Evaluation only: 4,800 sampled attempts, no retraining or extra benchmarks.
Prompt 1 is the original baseline prompt; prompts 2-4 are in 2026-10-27/analysis/prompt_sweep_v1.
Raw run IDs carry a p prefix (p10a = prompt 10a). Every prompt is reported
separately and none are pooled. Interpretations:

- Prompt 14: situational_context
- Prompt 15: behavioral_evidence_racial_groups; reported in its own behavioral_probe_* Holm families
- Prompt 16: expression_counterfactual_neutral
- Prompt 17: temporal_stability

Prompt 8 (a fix), prompt 9 (gray-image null check) and prompt 15
(behavioral-evidence probe) have their own Holm families
(mitigation_*, negative_control_*, behavioral_probe_*) and are not in the primary family.
Any prompt with an image that had zero valid answers is marked unavailable.
After recovery, prompt-X versus prompt-1 contrasts are required for batches C/D
and must include statistical tests; plots alone are not sufficient.

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
`python 2026-10-20/analysis/prompt_sweep.py --recovered <recovered> --out <new-output> --batch C`

Primary rows: 9. Source and script checksums are in provenance.json.
