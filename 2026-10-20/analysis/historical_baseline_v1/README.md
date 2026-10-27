# Verified historical screenshot baseline

This is the baseline Peyton selected for future reference. It is **not** the
current v4 untuned rerun and **not** the earlier 39.3% / 6.1% candidate baseline.
Do not substitute either of those datasets for this one.

## Recovery and verification

The private upstream repository was readable using Peyton's local GitHub CLI,
but its current branches did not contain the exact graph. The usable source was
the local download `gemma_3_12b_it_IAT_test_share_with_URAP.ipynb`. Its saved stream
output in cell 15 contains all 600 final answers: 12 filenames, exactly 50 unique
iterations per filename, six Black and six White images.

`historical_baseline.py` reads this output as data. It does not execute the
notebook or copy its authentication cells. Its notebook checksum is pinned.

The answers reproduce the pre-existing chart
`2026-09-11/results/pasted_iat/word_counts_reference_style.png`, which has the same
bar pattern and word order as Peyton's uploaded example. All 122 count positions
were checked against that saved chart: 33 nonzero Black bars, 30 nonzero White
bars, and 59 zero bars. The largest raster height residual was under 0.49 pixels
on the full-resolution chart. Raster inspection was only a cross-check; every
published count and statistic comes from the saved responses, never estimated
bar heights.

Reference chart SHA-256:
`749621cc39777618b39080466ff8de9d25dffa0025ebc166f6b50c2923efba8c`

## Files

- `responses.csv`: the 600 exact saved final answers, with filename, iteration
  and race metadata. `fallback_status` is explicitly `not_recorded`.
- `word_counts_reference_order.csv`: every one of the 61 words, including zero
  counts, in the fixed display order. Positive words run from Delight through
  Magnificent, then negative words from Pain through Sick.
- `image_metrics.csv`: per-image denominators, negative rates and diversity.
- `metrics.json`: group counts, vocabulary, diversity and image-level inference.
- `protocol.json`: the exact final prompt and generation kwargs from the notebook.
- `provenance.json`: source identity, limitations and checksums of the exported
  data and chart.
- `historical_baseline_word_counts.png`: one standalone chart with the original
  counts and ordering, compressed horizontal layout, and green/red valence
  backgrounds. Styling is rerendered, not a pixel-identical copy of the old image.

## Metrics for this exact run

| Metric | Black | White |
|---|---:|---:|
| Saved final answers | 300 | 300 |
| Word selections | 900 | 900 |
| Negative word selections | 341 | 60 |
| Negative-word rate | 37.89% | 6.67% |
| Distinct vocabulary words | 33 | 30 |

Black minus White negative-rate gap: **31.22 percentage points**.
The two-sided image-label permutation test enumerates all 924 assignments:
**p = 62/924 = 0.06710**. The percentile image bootstrap gives a 95% interval
of **[5.33, 60.00] percentage points**, using 10,000 draws and seed 20261027.
The unit is the image, not the 600 repeated answers. The permutation interpretation
assumes exchangeability of the image labels. This small, fixed image set does not
establish population-level fairness or bias, and nonsignificance is not equality.
Bootstrap and permutation results need not agree at a threshold with six images
per group; neither should be hidden.

Overall vocabulary coverage is **38/61 words**, with **25 shared** by both groups.
Effective vocabulary is **15.30 words**. The median image has **24.5 distinct
unordered triples** among its 50 answers. The most frequent unordered triple
accounts for **14%** of all answers. Jensen–Shannon divergence is **0.14861 bits**;
total variation is **0.36222**.

## Comparison rules

Use this immutable export as the historical anchor for future figures and tests.
All future charts must keep the same word order and remain separate image files.
Do not pool this export with newer untuned responses or label it "strict sampled
output": it contains final parsed answers, potentially including unmarked greedy
fallbacks. The original pre-parser generations are unavailable.

The notebook names `google/gemma-3-12b-it`, but does not pin a weight revision,
generation seed, historical library defaults (including top-k), or image content
hashes. The generation cell passes plain `text=PROMPT` to the image-text pipeline,
with a 30-token sampled attempt at temperature 1.0/top-p 1.0 and a 20-token greedy
fallback. The parser takes the first three vocabulary matches. These details
differ from current native-chat evaluations. Matching filenames alone also does
not verify identical image bytes.

Consequently, a v4-versus-historical plot can be descriptive, but must not be
presented as a controlled causal estimate of fine-tuning. Future direct-change
tests need documented alignment of images, prompting, decoding and scoring.
Do not claim a particular protocol difference caused the observed rate difference
without a controlled experiment. No new model evaluations or paid jobs were run
to recover this baseline.

## Reproduction

From the repository root, without the original notebook:

```bash
python 2026-10-20/analysis/historical_baseline.py \
  --frozen-dir 2026-10-20/analysis/historical_baseline_v1 \
  --output /path/to/new/empty/baseline-export
```

The loader verifies the frozen file checksums before recomputing anything.
Data values and word order remain identical; PNG rasterization can vary with the
installed Matplotlib version. Use `--notebook /path/to/the/original.ipynb` instead
of `--frozen-dir` to recover from the original source. A different notebook
checksum is rejected rather than silently changing this baseline.
