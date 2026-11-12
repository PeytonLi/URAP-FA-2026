# Native base versus ORPO v4: retrospective IAT statistics

No new inference, training, labeling or GPU spending. The same 12 image hashes,
600 attempt slots, prompt, seeds, model/processor revision, package versions,
native decoding parameters and scorer/fallback contract were verified in the
saved manifests and API logs. Only the ORPO adapter differs.

The independent unit is the **image**, not the 50 answers or 150 selected words.
These are retrospective analyses of already-inspected outputs, not preregistered
or confirmatory evidence. They apply to this small fixed stimulus set.

## Main family: three tests

| Test | Base | Tuned | Change | 95% CI | raw p | Holm p |
|---|---:|---:|---:|---|---:|---:|
| negative_gap_change_did | 83.67% | -0.89% | -84.56 points | [-103.0073, -51.4673] | 0.0043290043 | 0.0043290043 |
| word_entropy_change | 0.3297 | 0.8960 | +0.5664 | [0.5510, 0.5816] | 0.00048828125 | 0.0014648438 |
| unordered_triples_change | 3.3333 | 46.6667 | +43.3333 | [42.0833, 44.5833] | 0.00048828125 | 0.0014648438 |

The first row compares Black-minus-White negative rates before and after tuning:
the effect is tuned gap minus base gap. Entropy is normalized by log2(61).
Triple counts are per-image **means** in this test, with permutations of the
same three words collapsed. Their descriptive medians are
3.5 (base) and
46.5 (ORPO).

DiD uses all 924 six-versus-six image-label assignments. Diversity uses all
4,096 paired model swaps. Signed contrasts are two-sided. Holm adjustment is
over these three main tests, not over whichever subset looks favorable.
Confidence intervals are marginal 95% percentile intervals from 10,000 paired
image bootstraps within race, seed 20261027; they are not simultaneous intervals.

## Remaining disparity, validity and sensitivity checks

ORPO's residual negative-gap test has raw p
0.614719, Holm p
0.614719.
Its 61-word total-variation disparity test has raw p
0.160173, Holm p
0.320346.
These belong to a separate four-test diagnostic family including both base tests.
Nonsignificance is **not** equivalence or proof of fairness.

Base has 600/600 exact-valid sampled answers; ORPO has
563/600 (37 strict-invalid attempts).
The paired validity-change test has raw/Holm p 0.000976562.
ORPO's strict invalid count includes malformed sampled answers even when the
final scorer salvaged words. The 28 fallback calls are separately audited;
fallback output never replaces an invalid sample in the main analysis.

Eight ORPO exact-format sampled answers contain a nonadjacent repeated word.
The existing parser accepts these. The main view preserves the previously
published definition rather than silently changing its denominator. An
additional `sampled_distinct` sensitivity excludes repeated-word answers:
555/600 ORPO answers remain.
The exact rarefaction calculation below handles repeated token counts within an
answer, rather than assuming every answer contains three distinct words.

Gender-blocked DiD enumerates 400 assignments while retaining
the observed race counts within each gender: raw p 0.01,
Holm p 0.03 in its five-test sensitivity family. This sensitivity
is reported alongside the unblocked analysis, not selected as a new headline.

Equal-valid-count rarefaction uses between
44 and
50 answers per image, the smaller
valid count for that photo under the two models. Expected entropy and expected
distinct triples are calculated exactly over sampling whole answers without
replacement, not by choosing one lucky random subset. Its two Holm p-values
are 0.000976562 and 0.000976562. This controls answer-count
differences; it cannot recover the content of invalid answers or remove their
selection bias.

Final-scorer results, including deterministic fallbacks and salvaged formats,
are a separate sensitivity family. All main, diagnostic, quality, blocked,
final-output and rarefaction results are in `tests.csv` and the horizontal
copy/paste table `statistics_table.csv`. Adjustment is within each explicitly
labeled family; no global multiplicity guarantee is claimed for the exploratory
sensitivities.

## Limits and tests not run

- Race-label permutations assume exchangeability of image labels under the null;
  paired swaps assume within-image condition exchangeability under the null.
  These are not randomized demographic interventions or population causal claims.
- With six images per race, uncertainty is substantial. Bootstrap intervals and
  exact permutation p-values can disagree at a threshold; neither is hidden.
  Nonnegative TV has finite-sample upward bias, so its bootstrap lower endpoint
  is not a substitute for the permutation test.
- Equivalence was not tested: no scientifically justified acceptable-gap margin
  was agreed in advance. The example of five percentage points was not adopted
  as a fairness threshold after seeing the results.
- Welch and OLS were not added as competing headline p-values. No token-level,
  response-level independent-observation test or unpaired model t-test was run.
- The recovered historical screenshot baseline remains an immutable reference.
  It is not pooled with these runs or used for the tuning contrast: its prompting,
  defaults, parser/fallback records and model/image provenance differ or are
  incomplete. Its separate within-run statistics remain in its own export.
- Matched training targets are shared answer distributions, not literal facial
  descriptions. Improvements on this word-choice task do not establish grounded
  accuracy or transfer to DecodingTrust, BOLD or VisBias.

## Reproduction

`analysis_plan.json` declares the test families. `paired_responses.csv` contains
the minimal saved inputs (no image pixels or credentials); `protocol_audit.json`
records the original source hashes and contract checks. `per_image.csv` exposes
every image-level value and rarefaction denominator.

From the repository root:

```bash
python 2026-10-27/analysis/iat_significance_v4.py \
  --paired-inputs 2026-10-27/analysis/iat_stats_v4/paired_responses.csv \
  --audit 2026-10-27/analysis/iat_stats_v4/protocol_audit.json \
  --plan 2026-10-27/analysis/iat_stats_v4/analysis_plan.json \
  --out /path/to/new/output
```

The frozen paired-input checksum is verified before recomputation. Reproduction
does not silently re-claim that the original private API logs were read; it
relies on their recorded audit plus the frozen input checksum.
