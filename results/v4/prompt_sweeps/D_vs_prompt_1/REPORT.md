# Prompts 18, 19, 20, 21 versus prompt 1

Required Batch D follow-up: within base and within saved ORPO separately,
each prompt is compared to the frozen prompt-1 native paired responses on the same
12 images for the three primary metric families (negative Black-White gap via exact
924 race permutations of per-image prompt−prompt-1 differences; entropy and unordered
triples via exact 2^12 sign-flips). Prompt 1 values are listed under Base; the batch
prompt under Tuned; Change is prompt minus prompt 1.

Prompt 15 stays in behavioral_probe_vs_prompt_1_* families and is excluded from the
primary vs-prompt-1 Holm families. Holm adjusts within each model-condition family
across prompts in this batch. Plots are not a substitute for these tests.

Reference: `2026-10-27/analysis/iat_stats_v4/paired_responses.csv`.
Source batch analysis: `C:\Users\lipey\Code\URAP\2026-11-10\analysis\prompt_sweep_D`.

See statistics_table.csv for effects, 95% image-bootstrap CIs, raw and Holm p-values.

Primary vs-prompt-1 rows: 24.
