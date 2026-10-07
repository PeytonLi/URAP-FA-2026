# July 27 Pair Dataset Audit

Status: trained once as the July 27 primary run; the adapter failed the
prespecified efficacy criteria

## Superseded first freeze

All 96 July 20 pre-cap pairs received four blinded Gemini judgments: image and
no-image conditions, each repeated with A/B order reversed. All 384 calls
succeeded. The first policy retained 51 rows, but its expression labels were
confounded with race/gender:

| Intersection | Total | Negative | Neutral | Positive | Neutral/positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Black female | 14 | 5 | 6 | 3 | 64.29% |
| Black male | 13 | 9 | 3 | 1 | 30.77% |
| White female | 11 | 6 | 3 | 2 | 45.45% |
| White male | 13 | 12 | 1 | 0 | 7.69% |

The 56.59-point nonnegative-fraction spread could teach demographic appearance
as a proxy for expression. That CSV is preserved under
`results/gemini-3.5-flash_pair_judge_july20/`, has
`training_ready: false`, and was not trained.

## Balanced supplement

Forty additional FairFace images were selected from the AI-reviewed neutral
pool, exactly 10 per race/gender intersection, with zero SHA-256 overlap against
the 400-image held-out benchmark and 12-face neutral benchmark. Pinned base
Gemma 3 generated 96 traces per image:

- 3,840 candidate traces;
- 3,552 valid three-word responses;
- seed `7272028`, temperature 1.0, top-p 1.0;
- eager attention;
- candidate CSV SHA-256
  `312755c728095928a804b6c49c77277a60f0568449135cd7b85e1b8c352ffd68`.

The error-packet builder ranked identity leakage at 8.0 per hit and
expression-unsupported negativity at 4.0 per term. It selected four blinded
items with safe same-image base-Gemma targets and excluded one error image that
had no such target. A fixed-seed shuffle prevented packet IDs from preserving
filename or group order.

The current Codex session reviewed the pixels and candidate text using
`gpt-5.6-sol`, Ultra reasoning, and Priority service. Demographic metadata was
not joined during review. Three corrections were included; one was excluded
because the visible serious expression did not support any available strongly
positive target. This was AI-only review with no human validation.

## Final Gemini audit

The three included proposals were appended after the original 96 rows, leaving
all historical pair IDs stable. Gemini made 12 incremental calls, so the final
audit contains 396 successful calls over 99 candidates.

One supplemental White-male pair entered the final freeze:

```text
chosen:   Smiling, Happy, Attractive
rejected: Failure, Ugly, Disgust
```

The chosen side is an exact same-image base-Gemma trace. The image-aware Gemini
judgment was order-stable with 0.9 confidence; the text-only control tied. A
Black-male identity correction was eligible but not required by the
prespecified cap, while a Black-female correction was excluded because the
chosen side had lower visible-expression support.

## Final freeze

`expression_stratified_v1` retained 50 of 67 eligible pairs:

| Intersection | Total | Negative | Neutral | Positive | Neutral/positive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Black female | 14 | 8 | 4 | 2 | 42.86% |
| Black male | 14 | 10 | 3 | 1 | 28.57% |
| White female | 11 | 6 | 3 | 2 | 45.45% |
| White male | 11 | 8 | 3 | 0 | 27.27% |

Additional controls:

- 32 negative, 13 neutral, and 5 positive pairs;
- nonnegative-fraction spread: 18.18 percentage points;
- at least three nonnegative pairs per intersection;
- group-count spread: three;
- 27 model-observed and 23 AI-teacher chosen responses;
- 37 unique exact chosen responses;
- largest exact response: 5 of 50, or 10%;
- three complete Black/White expression-match blocks;
- zero SHA-256 overlap with either frozen evaluation set;
- no human-reviewed rows.

Frozen CSV:
`results/gemini-3.5-flash_pair_judge_final_balanced/gemini_judged_dpo_pairs_frozen.csv`

SHA-256:
`638caf56c926b3007b6f17fef7b042ac5927be06bed8e31fba0e676b19a0cc69`

## Training decision

The final summary and manifest both report `training_ready: true`. On the
RunPod, the complete analysis reproduced the same CSV hash and the workflow's
actual `load_audited_pairs` function revalidated all 50 rows under the current
pair policy. `prepare_fresh_lora_run.py` then wrote a launch plan but did not
execute it.

This is authorization of the input data, not evidence that the adapter will
improve fairness. The final outcome must still pass the held-out, neutral-face,
identity-leakage, diversity, validity, expression-sensitivity, and paired
bootstrap evaluations.

## Post-training supervision audit

This section was added after observing the July 27 result. It is an
outcome-informed diagnostic, not a preregistered acceptance rule and not a
retroactive change to the primary endpoint.

The expression-distribution gate fixed one confound but missed two others:
the type of pair-level improvement and the actual chosen vocabulary. DPO gives
each retained pair one gradient example, regardless of whether it reduces
identity leakage, lowers total negative-word count, or only replaces
expression-unsupported negative terms with supported ones at the same count.

| Supervision property | July 19 | July 27 |
| --- | ---: | ---: |
| Total pairs | 55 | 50 |
| Negative-expression pairs | 25 | 32 |
| Identity-removal pairs | 26 | 20 |
| Total negative-count reduction pairs | 8 | 1 |
| Fixed-count unsupported-negativity substitutions | 21 | 29 |
| Black negative-expression pairs | 10 | 18 |
| White negative-expression pairs | 15 | 14 |
| Black mean chosen negative terms per negative pair | 3.00 | 3.00 |
| White mean chosen negative terms per negative pair | 2.40 | 2.36 |

The fixed-count substitutions are legitimate expression-accuracy controls:
they reduce deterministic and Gemini-judged unsupported negativity without
changing visible negative valence. They cannot, however, directly lower the
benchmark's total negative-term rate. They rose from 38.2% of the July 19 set
to 58.0% of the July 27 set, while direct total-count reduction fell from eight
pairs to one.

The July 27 negative-expression targets were also lexically race-conditioned.
All 18 Black rows chose three negative terms. Their chosen terms included
`Pain` 18 times, `Sadness` 16 times, and `Grief` 15 times. The 14 White rows
used those terms 10, 6, and 4 times respectively and averaged 2.36 negative
terms per chosen response. Fine expression labels also differed: 14 of 18
Black rows were labeled `sad`, while the White rows were split among `sad`,
`fear`, `angry`, and `disgust`.

The final adapter increased held-out Black counts for `Pain`, `Sadness`, and
`Grief` by 18, 28, and 17 relative to the frozen base. The same three terms
rose for White images by 15, 10, and 14. This alignment is consistent with the
adapter learning the chosen target vocabulary; it does not by itself prove a
single causal mechanism.

## Prospective correction

The next dataset should restore the primary debiasing signal rather than fill
a 50-row quota with accuracy-only substitutions:

1. Source at least 50 pairs that directly reduce identity leakage or total
   negative-word count on expression-unsupported errors; do not weaken the
   gate to reach 50.
2. Keep fixed-count expression-accuracy pairs in a separately labeled,
   separately capped regularization subset.
3. Balance total race counts exactly and balance fine expression labels, not
   only coarse negative/neutral/positive fractions.
4. For accuracy pairs, prefer complete Black/White matched blocks with the
   same chosen and rejected target tuples.
5. Cap Black-versus-White differences in mean chosen negative-term count and
   per-term target rates before training.
6. Create a separate development set for checkpoint and hyperparameter
   selection. Keep the 400-image FairFace and 12-face neutral sets untouched
   for one final confirmatory evaluation.
