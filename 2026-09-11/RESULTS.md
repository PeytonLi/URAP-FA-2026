# September 11 Evaluation Results

## Scope

Inference-only expansion of IAT61 and a paired evaluation of the existing August 10 Gemma LoRA adapter. No new DPO training or judge calls were made. The selected adapter is a retrospective directional example, not a validated best model.

## Response Counts and Race Gaps

Each valid answer contains three permitted words. Race rates below are image-averaged percentages. The gap is Black minus White in percentage points. p values are exploratory two-sided tests; q values are BH-adjusted within each panel.

| Panel | Model | Answered / target | Valid | Black % | White % | Gap [95% CI], pp | p | q |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| neutral12 | Gemma 3 12B base (local) | 600 / 600 | 600 | 39.67 | 3.33 | +36.33 [+27.67, +46.33] | 0.015 | 0.1 |
| neutral12 | Gemma 3 12B + LoRA (local) | 600 / 600 | 600 | 25.11 | 0.67 | +24.44 [+11.00, +47.56] | 0.02 | 0.1 |
| neutral12 | Qwen 3.5 9B | 600 / 600 | 600 | 16.56 | 6.89 | +9.67 [-2.67, +26.44] | 0.385 | 0.55 |
| neutral12 | Kimi K3 | 256 / 600 | 256 | 99.21 | 92.06 | +7.14 [+0.79, +12.70] | 0.2 | 0.3333 |
| neutral12 | GPT-5.6 Luna | 600 / 600 | 600 | 100.00 | 84.78 | +15.22 [+0.33, +41.33] | 0.2 | 0.3333 |
| neutral12 | Gemini 3.6 Flash | 600 / 600 | 600 | 95.67 | 82.33 | +13.33 [-6.00, +40.67] | 0.53 | 0.6625 |
| neutral12 | MiMo-V2.5 | 600 / 600 | 600 | 76.78 | 61.00 | +15.78 [-3.67, +33.11] | 0.175 | 0.3333 |
| neutral12 | MiniMax M3 | 112 / 600 | 112 | 13.89 | 11.44 | +2.45 [-10.28, +13.70] | 0.76 | 0.8444 |
| neutral12 | Step 3.7 Flash | 600 / 600 | 600 | 6.33 | 8.33 | -2.00 [-9.33, +4.33] | 0.845 | 0.845 |
| neutral12 | Grok 4.6 | 600 / 600 | 600 | 85.33 | 53.33 | +32.00 [+15.33, +47.67] | 0.035 | 0.1167 |
| utkface | Gemma 3 12B base (local) | 2000 / 2000 | 2000 | 1.63 | 1.20 | +0.43 [-1.20, +2.03] | 0.647 | 0.7189 |
| utkface | Gemma 3 12B + LoRA (local) | 2000 / 2000 | 2000 | 0.70 | 1.00 | -0.30 [-1.90, +1.00] | 0.843 | 0.843 |
| utkface | Qwen 3.5 9B | 2000 / 2000 | 1995 | 15.77 | 10.40 | +5.38 [+0.52, +10.31] | 0.03292 | 0.05487 |
| utkface | Kimi K3 | 2000 / 2000 | 1972 | 19.91 | 11.88 | +8.03 [+2.97, +13.15] | 0.00209 | 0.006967 |
| utkface | GPT-5.6 Luna | 2000 / 2000 | 1977 | 26.61 | 17.71 | +8.90 [+2.70, +15.11] | 0.0053 | 0.01088 |
| utkface | Gemini 3.6 Flash | 2000 / 2000 | 1990 | 39.30 | 29.14 | +10.16 [+3.06, +17.20] | 0.00544 | 0.01088 |
| utkface | MiMo-V2.5 | 2000 / 2000 | 2000 | 29.77 | 20.33 | +9.43 [+3.67, +15.23] | 0.00161 | 0.006967 |
| utkface | MiniMax M3 | 2000 / 2000 | 1998 | 6.90 | 4.73 | +2.17 [-0.73, +5.17] | 0.1587 | 0.2267 |
| utkface | Step 3.7 Flash | 2000 / 2000 | 1998 | 5.20 | 3.67 | +1.53 [-0.87, +4.10] | 0.2415 | 0.3019 |
| utkface | Grok 4.6 | 2000 / 2000 | 2000 | 23.30 | 13.60 | +9.70 [+4.13, +15.30] | 0.00074 | 0.006967 |

Use [results/analysis/summary.csv](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/results/analysis/summary.csv) and [results/analysis/significance.csv](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/results/analysis/significance.csv) for unrounded values and sample units.

## Cross-Model Findings

**neutral12:** 0 of 10 available conditions have an exploratory race-gap test with BH q < 0.05. Incomplete cells are not full-size estimates, and a nonsignificant test does not establish no bias.

**utkface:** 5 of 10 available conditions have an exploratory race-gap test with BH q < 0.05 (Kimi K3, GPT-5.6 Luna, Gemini 3.6 Flash, MiMo-V2.5, Grok 4.6). Incomplete cells are not full-size estimates, and a nonsignificant test does not establish no bias.

Image panels, expression distributions, sample units and model serving paths affect these measures. Do not interpret cross-model bar height as a general fairness ranking or differences from older plots as evidence that base weights were trained.

## Direct Gemma Base Versus Adapter Comparison

**neutral12:** the absolute observed gap fell from 36.33 to 24.44 pp. The direct paired signed-gap change was -11.89 pp (95% CI -21.78 to +0.00; p=0.175, q=0.175). Adapter word-entropy retention was 94.8% and unique-response retention 93.3% (163 to 152).

**utkface:** the absolute observed gap fell from 0.43 to 0.30 pp. The direct paired signed-gap change was -0.73 pp (95% CI -1.70 to +0.07; p=0.1404, q=0.175). Adapter word-entropy retention was 97.5% and unique-response retention 86.9% (573 to 498).

A smaller signed gap is not automatically a smaller absolute gap. Read both race rates and diversity. These results do not establish preserved general accuracy or causal racial invariance. The direct test uses only tasks valid in both conditions, so missingness can matter.

[Direct paired tuning-effect graph](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/plots/gemma_04_paired_tuning_effect.png)

[Focused base-versus-adapter bars](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/plots/gemma_05_base_vs_lora.png)

[Vocabulary-retention graph](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/plots/gemma_06_diversity_retention.png)

**Research decision:** 0 of 2 direct tuning-effect tests have q < 0.05. The historical 95% diversity-retention screen was not met on: neutral12, utkface. The observed directional changes therefore do not establish a successful accuracy-preserving debiasing result. Low UTKFace baseline negativity also leaves limited room for a reduction; equality alone is insufficient evidence of useful image understanding.

## Figure Guide

All figures place local Gemma conditions first when available. Each figure includes a short interpretation.

- [plots/neutral12_01_negative_term_rates.png](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/plots/neutral12_01_negative_term_rates.png): Group rates: percent of selected words that are negative. Compare uncertainty and missingness, not only bar height.
- [plots/neutral12_02_race_gap_significance.png](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/plots/neutral12_02_race_gap_significance.png): Effect size, 95% interval, and exploratory p/q values. Above zero means more negative associations for Black photos.
- [plots/neutral12_03_validity_diversity.png](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/plots/neutral12_03_validity_diversity.png): Output compliance and vocabulary entropy. Broad vocabulary alone does not demonstrate accurate image understanding.
- Individual neutral12 word-count graphs: `plots/neutral12_<model_key>_word_counts.png`. The 61 x-axis categories include unselected words. Compare rates, not raw counts, when valid sample sizes differ.
- [plots/utkface_01_negative_term_rates.png](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/plots/utkface_01_negative_term_rates.png): Group rates: percent of selected words that are negative. Compare uncertainty and missingness, not only bar height.
- [plots/utkface_02_race_gap_significance.png](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/plots/utkface_02_race_gap_significance.png): Effect size, 95% interval, and exploratory p/q values. Above zero means more negative associations for Black photos.
- [plots/utkface_03_validity_diversity.png](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/plots/utkface_03_validity_diversity.png): Output compliance and vocabulary entropy. Broad vocabulary alone does not demonstrate accurate image understanding.
- Individual utkface word-count graphs: `plots/utkface_<model_key>_word_counts.png`. The 61 x-axis categories include unselected words. Compare rates, not raw counts, when valid sample sizes differ.

## Changes, Failures and Interpretation Limits

- The same image bytes and IAT61 vocabulary are retained. The reuse plan included 4,040 archived slots, of which eight were technical errors rather than model answers. Final retained archive counts are in summary.csv.
- MiMo Neutral12 is a fresh DeepInfra FP8 condition; its old BF16 data are excluded from pooling.
- MiniMax Neutral12's pinned DeepInfra route returned eleven HTTP 429 errors across three probes and was excluded for the rest of this session. Its UTKFace Venice route briefly became unavailable (catalog status -2), then recovered and resumed on the same endpoint. Completed answers were preserved.
- Kimi Neutral12's DeepInfra route also reported unavailable status during expansion. Availability snapshots, recovery attempts and final completion counts are preserved; no alternate provider was substituted.
- One MiMo transport timeout interrupted the first expansion. The uncertain charge is covered by an extra budget reservation and account reconciliation; completed response slots were retained on resume.
- Runpod's container lacked rsync and apt update timed out; archive transfer with no ownership restoration was used. Historical model environments were not upgraded.
- Retrieval initially failed because local Python 3.11.0 lacks the newer tar extraction filter argument. The transfer now explicitly rejects unsafe members; final remote/local checksums matched before compute shutdown.
- Gemma now resets a common seed per image/iteration. Historical dataset-stream baselines can differ without any base-weight changes. The new paired base is the relevant comparator.
- Fifty repetitions do not turn 12 faces into 600 independent subjects. The 12-face panel was used for retrospective adapter selection.
- UTKFace matching uses fallible automated expression and image-quality labels; negative expressions can justify negative words.
- Hosted backends and quantization differ from local inference. Catalog revision/route checks cannot prove immutable hosted weight bytes.
- Sampling was fixed by target counts and technical/budget stops, not repeated until bias became significant. Null results are retained.
- These are IAT-style implicit-association measures only. Existing text-only DecodingTrust/BOLD and image-based VisBias have different constructs and are linked in the coverage tables, not rerun here.

### Incomplete Cells

- neutral12 / Kimi K3: 256 of 600 answered slots; 0 final API-error slots.
- neutral12 / MiniMax M3: 112 of 600 answered slots; 12 final API-error slots.

## Costs and Calls

OpenRouter: 17,028 model-call attempts, including technical errors and parsing fallbacks; $12.128267 reported call cost. Latest account balance: $14.739510. Account usage delta: $12.128305. Timestamps and reconciliation are in [results/hosted/session.json](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/results/hosted/session.json).

| Model | New API-call attempts | Reported cost (USD) |
|---|---:|---:|
| MiniMax M3 | 1721 | 0.142104 |
| Qwen 3.5 9B | 2548 | 0.075211 |
| Gemini 3.6 Flash | 2109 | 2.069699 |
| Step 3.7 Flash | 2161 | 0.666959 |
| GPT-5.6 Luna | 2145 | 0.148454 |
| Kimi K3 | 1867 | 2.097323 |
| MiMo-V2.5 | 2396 | 0.070918 |
| Grok 4.6 | 2081 | 6.857600 |

There were 17,016 successful returned API calls and 1 call with an unknown transport outcome. The reconciled account charge minus reported call costs is $0.00003778; billing timing can contribute to this difference. An uncertainty reservation is a spending safeguard, not an additional confirmed charge.

Local Gemma: 5,200 saved evaluation answers, with 5,279 recorded generation calls including parsing fallbacks. The separate Qwen probe has 12 answers; Gemma's preflight base/disabled-adapter equivalence probes are not included in those response counts.

Runpod compute wall-time estimate: $4.0813. Own pod absent: True; network volume preserved: True. Account balance includes storage and billing timing; see [results/runpod/cleanup.json](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/results/runpod/cleanup.json).

## Next Steps

1. Review the direct paired effect and anti-collapse metrics before choosing any new training objective.
2. Use a new person-disjoint held-out sample and better expression review for a confirmatory study; do not tune on these reported p values.
3. Treat Qwen2.5-VL-3B as a separate future training experiment, not a proven replacement or an already debiased model.
4. Resolve any still-unavailable pinned endpoints before completing partial cells; any new route must be a separately labeled condition.

## Reproducibility

See [README.md](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/README.md), [METHODS.md](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/METHODS.md), [DATA_DICTIONARY.md](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/DATA_DICTIONARY.md), and [COVERAGE.md](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/COVERAGE.md). Run `scripts/analyze.py`, `scripts/plot.py`, and `scripts/report.py` offline to regenerate the reported derived artifacts.
