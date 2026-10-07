# Gemma DPO/LoRA: Audited Research History

Updated September 5, 2026. There is **no accepted successful debiasing adapter**.
This history distinguishes technical execution, directional improvement, and
preservation of useful output behavior. All paths below are relative to this
standalone package; `report/dpo_training_record.json` supplies absolute paths.
Earlier artifacts are preserved under `reference/`, not rewritten.

## Objective And Measurement

Reduce excess race-associated negativity at comparable visible expression while
preserving grounding, legitimate negative-expression recognition, validity and
response diversity. Numerical parity obtained through vocabulary collapse is
not the objective. A higher White negative rate is not automatically an error if
visually warranted, but a smaller gap alone cannot demonstrate improved fairness.
No general-purpose accuracy benchmark or independent human preference validation
has yet established that capability is preserved.

The focal model is `google/gemma-3-12b-it`. Controlled later runs pin snapshot
`96b6f1eccf38110c56df3a15bffe176da04bfd80`. DPO updates language-side LoRA
parameters; the reference/base and vision tower remain frozen. No historical
record reviewed here describes intentional full-model fine-tuning. The September
5 inference run additionally hashes the base files before and after execution.

The principal source is FairFace, transported through
`HuggingFaceM4/FairFace`, configuration `0.25`, in the earlier projects. The final
August 10 freeze does not record a unique upstream dataset revision; exact selected
image hashes are available instead. Cite Karkkainen and Joo (2021), *FairFace:
Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement
and Mitigation*, WACV, pp. 1548-1558. Use the
[publisher record](https://openaccess.thecvf.com/content/WACV2021/html/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.html)
rather than copying unverified DOI strings from older notes.

## Chronology

| Revision | What Changed And Why | Evidence And Outcome |
|---|---|---|
| April 29 | Eight training images, eight candidates/image; novelty/valence-driven selection retained one unique pair repeated four times. Four optimizer updates established execution. | `reference/dpo/july13/` contains the retrospective audit. This is a smoke test, not a population-level debiasing experiment. |
| July 6 | Base-only FairFace reference: 400 validation images, 100 per race/gender intersection, ten outputs/image. Pair policy began emphasizing direct negative/leakage reduction. | Archived audit: Black 265/6000=4.4167%, White 117/6000=1.95%; gap +2.4667 pp, image-cluster CI approximately [-0.43,+5.50], p about .101. Attention backend was not explicitly recorded. See `reference/dpo/july13/`. |
| July 13 | Negative penalty 4, identity penalty 8, novelty .5, valence centering .1; selected response could not worsen either harm count. Required direct harm reduction and separated observed/template provenance, with templates <=50%. | 29 observed+29 template pairs, repeated twice; 500 updates on a small set drove negative terms 373 to 2 but reduced entropy about 38% and unique responses 71.6%. Reject for collapse. `reference/dpo/july13/`. |
| July 13 memory failure | Eager attention plus overly broad LoRA targets and retained chosen/rejected vision graphs exhausted an H100 (~78.8 GiB). Language-only LoRA/vision freezing addressed the accidental vision training. | Peak allocation/reservation fell to 42.59/43.96 GiB. This was **not proof that eager attention alone caused the OOM**. Eager is an attention implementation, not a new transformer architecture. |
| July 19 | Expression-aware AI-only pairs attempted to preserve visible emotion instead of universally preferring positive words. | FairFace Black 4.75% to 5.05%; White 1.95% to 2.833%. Gap narrowed because White increased more. Neutral12 Black 35% to 50%, White 3.333% to 5%; reject. Raw rows and counts: `reference/dpo/july19/`. |
| July 27 | Fresh LoRA from a 50-row Gemini-judged dataset; judge recorded as Vertex `gemini-3.5-flash`. Rank16/alpha32/dropout.05, beta.05, LR1e-5, accumulation4, 13 updates. | FairFace Black/White 5.683%/2.433%; neutral12 50%/8.333%. Post-run audit found mostly fixed-count substitutions and race-conditioned vocabulary rather than direct reduction. Reject. `reference/dpo/july27/`. |
| August 3 | Introduced multimodal Codex GPT teacher and independent Claude image/text judging, each with reversed candidate order. A six-row pilot used four training rows and one update. | Neutral12 Black 35% to 43.333%; White 3.333% to 1.667%. Reject. Detailed chronology preserved in `reference/august31/FINE_TUNING_HISTORY.md`; August10 source snapshot includes the reused August3 helper modules. |
| August 10 v4 | Expanded audited preferences to 100 unique FairFace images, 50 matched blocks, 80 train/20 development; 70 profiles+30 controlled IAT rows. Five fresh rank8 LoRA pilots compared sum/mean token reductions and small learning rates. | Development sampled IAT outputs largely unchanged despite preference-margin changes. A later split audit found all three negative-expression IAT blocks in development, none in train. Strict corrected gate rejects this freeze for new training. `reference/dpo/august10/august10_negative_expression_pilot_v4/`. |
| August 10 diagnostic | Deliberately blunt all-positive targets tested whether LoRA+DPO could move the endpoint at all. | Negative rates reached zero by update8, but negative-expression sensitivity and later diversity failed. This shows controllability, not debiasing. `reference/dpo/august10/august10_dpo_lora_overfit_diagnostic_v2/`. |
| August 10 retrospective neutral12 | Examined all five pilot adapters on the repeatedly inspected 12-face safeguard. Mean-token DPO LR2e-5 was the sole directional improvement. | Black 35% to 16.667%, White 3.333% to 0%, gap31.667 to16.667 pp. Response entropy retention91.1103%; unique responses64 to52 (81.25% retention). Fails retrospective95% preservation. `reference/dpo/august10/august10_first_run_neutral12_retrospective_v1/`. |
| August 17-31 | Expanded image and text benchmarking rather than training. Froze UTKFace400 evaluation and400 disjoint future-training images, with800 prompt-only tasks. | Blank chosen/rejected fields mean **not training-ready**. These rows do not represent new LoRA updates. `data/frozen/utkface/`. |
| September 5 | Reconciled methods/evidence and completed missing local Gemma and 12-face multimodel VisBias evaluations. | Inference only; no adapter or base training. New results are separate from the immutable pre-DPO baseline. See `results/gemma_local/manifest.json` and `report/FORMAL_RESEARCH_REPORT.md`. |

## Representative August 10 Adapter

The retained detailed example is **post-hoc best directional**, not the sole trial
run or an accepted success. Its adapter SHA-256 is
`aa869029c73588c2cb5ff1bf805466f78e995de3e888ac7ff7eb567f8d421053`.

The exact training spreadsheet is
`reference/dpo/august10/august10_negative_expression_pilot_v4/dataset/mixed_dpo_pairs_frozen.csv`.
SHA-256: `589db3c850b383be7c848f15f3fa26e48d9d44af36ea6b6ee947fe69225d67c9`.
Each row contains one image, prompt, chosen response and rejected response. Two
rows form a matched Black/White block; demographic/audit fields select and
validate examples but are not extra inputs to the DPO loss.

For 70 profile rows, chosen provenance is `gpt-5.6-sol_codex_cli_multimodal_teacher`
and rejected provenance is observed base Gemma. The teacher used **Codex CLI**,
not the OpenAI API; an exact OpenAI API snapshot cannot be supplied. For 30 IAT
rows, both texts are controlled race-blind templates, not model-generated
profiles. Recorded judges are direct Anthropic `claude-sonnet-5` for profiles
and `claude-opus-5` for IAT. These are the IDs in the evidence, not a verified
immutable hosted checkpoint. Neither teacher nor judge is assumed unbiased.

The selected variant averages log probabilities over response tokens; this is a
**modified mean-token DPO objective**, not standard sequence-summed DPO. It used
beta.05, LR2e-5, rank8/alpha16/dropout.05, accumulation8, BF16, SDPA, gradient
checkpointing and one-response-graph coefficient replay. One pass through80
training rows gave10 updates. Training took113.1573 seconds (whole process191.0772
seconds), peak allocated/reserved23.7602/25.1992 GiB. Full settings, AdamW defaults,
code paths and exact environment versions are in `report/dpo_training_record.json`.

## Limits And Next Decision

The 12-face panel and earlier FairFace panel have been repeatedly inspected and
cannot be described as untouched confirmatory holdouts. Byte-hash checks exclude
identical files, not the same person photographed twice. The training and IAT
evaluation tasks overlap by design in prompt family and vocabulary; that is
task overlap, distinct from image leakage. There is no UTKFace, VisBias, BOLD or
DecodingTrust post-DPO result for this selected adapter.

Do not resume an old rejected adapter. Fix train/development expression coverage,
construct race-balanced grounded alternatives, validate judges on held-out
examples, and freeze a new untouched evaluation panel before a short fresh-LoRA
pilot. Select on development only and preserve expression sensitivity, response
entropy, uniqueness and validity. More training time is not a substitute for
correct targets and reliable evaluation.
