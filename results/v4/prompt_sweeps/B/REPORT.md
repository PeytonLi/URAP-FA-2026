# Prompts 9, 10a, 10b, 11, 12, 13: base versus saved ORPO

Evaluation only: 7,200 sampled attempts, no retraining or extra benchmarks.
Prompt 1 is the original baseline prompt; prompts 2-4 are in 2026-10-27/analysis/prompt_sweep_v1.
Raw run IDs carry a p prefix (p10a = prompt 10a). Every prompt is reported
separately and none are pooled. Interpretations:

- Prompt 9: no_face_gray_image_control; race labels belong to the replaced photos, so race contrasts are a null check
- Prompt 10a: race_stated_correct
- Prompt 10b: race_stated_swapped
- Prompt 11: character_traits
- Prompt 12: cannot_tell_option; abstentions are invalid in the word views and reported as their own per-race rate
- Prompt 13: first_instinct

Prompt 8 (a fix) and prompt 9 (gray-image null check) have their own Holm
families (mitigation_* and negative_control_*) and are not in the primary family.
Any prompt with an image that had zero valid answers is marked unavailable.

See statistics_table.csv for effects, 95% image-bootstrap confidence intervals,
raw p-values and Holm p-values. Holm spans all 15 primary hypotheses.
Diagnostic and sensitivity adjustments span prompts within their labeled families.
The photo is the independent unit, not each answer or selected word.

Strict sampled output is primary; fallback output and repeated-word exclusion
are sensitivities. Entropy is normalized to log2(61); triples are unordered.
Graphs/summary.csv within each prompt reports per-group rates and vocabulary
coverage. Word-count graphs are separate and share the fixed reference order.
cannot_tell.json reports prompt 12 abstention rates per race and model.

These are exploratory results on 12 already-inspected photos. Nonsignificance
does not establish equivalence; variety does not establish grounded accuracy.
The historical screenshot baseline is unchanged and excluded from tuning tests.

Reproduce with the recovered checksummed run:
`python 2026-10-20/analysis/prompt_sweep.py --recovered <recovered> --out <new-output> --batch B`

Primary rows: 15. Source and script checksums are in provenance.json.
