# Eight-Model Race-Bias Benchmark: Pilot estimates

The IAT-style image test measures implicit positive/negative word association. DecodingTrust and BOLD measure different forms of explicit generated behavior; the three benchmarks must not be combined into one bias score.

![Eight-model overview](eight_model_pilot_overview.png)

## Main estimates

| Model | IAT Black-White gap (95% CI; q) | DT agreement B/U/T | DT no response | BOLD toxicity (95% CI) | Sentiment | Regard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 9B | +33.3 pp [+0.0 pp, +66.7 pp]; q=0.896 | 0.0% / 0.0% / 0.0% | 16.7% | 0.0% [0.0%, 21.5%] | +0.14 | +0.29 |
| Kimi K3 | -16.7 pp [-50.0 pp, +0.0 pp]; q=1.000 | 0.0% / 0.0% / 0.0% | 0.0% | 0.0% [0.0%, 19.4%] | +0.25 | +0.38 |
| GPT-5.6 Luna | +27.8 pp [+0.0 pp, +50.0 pp]; q=0.896 | 0.0% / 0.0% / 0.0% | 0.0% | 0.0% [0.0%, 19.4%] | +0.12 | +0.50 |
| Gemini 3.6 Flash | -16.7 pp [-50.0 pp, +0.0 pp]; q=1.000 | 0.0% / 0.0% / 0.0% | 18.8% | 0.0% [0.0%, 19.4%] | +0.38 | +0.44 |
| MiMo-V2.5 | +50.0 pp [+16.7 pp, +83.3 pp]; q=0.896 | 0.0% / 0.0% / 12.5% | 12.5% | 0.0% [0.0%, 19.4%] | +0.31 | +0.12 |
| MiniMax M3 | -40.0 pp [-80.0 pp, +0.0 pp]; q=0.896 | 0.0% / 0.0% / 0.0% | 8.5% | 0.0% [0.0%, 19.4%] | +0.31 | +0.50 |
| Step 3.7 Flash | +0.0 pp [+0.0 pp, +0.0 pp]; q=1.000 | 0.0% / 0.0% / 6.2% | 0.0% | 0.0% [0.0%, 21.5%] | +0.36 | +0.36 |
| Grok 4.6 | +33.3 pp [-16.7 pp, +83.3 pp]; q=0.896 | 0.0% / 0.0% / 87.5% | 2.1% | 6.2% [1.1%, 28.3%] | +0.12 | +0.19 |

B/U/T means benign, untargeted-adversarial, and group-targeted DecodingTrust conditions. IAT q-values are Benjamini-Hochberg adjusted exact permutation p-values. The BOLD toxicity y-axis is the percentage of responses where any Detoxify `unbiased` score exceeds 0.3; lower means fewer responses were flagged. Sentiment and regard are mean classes from -1 (negative), through 0 (neutral), to +1 (positive).

## Main findings

- MiMo-V2.5 had the largest observed absolute IAT gap (+50.0 pp). The smallest corrected IAT q-value was 0.896 for Qwen 3.5 9B; this is exploratory evidence, not a binary unbiased/biased label.
- Grok 4.6 had the highest targeted DecodingTrust agreement rate (87.5%). The smallest corrected race-group q-value was 1.000, shared by Qwen 3.5 9B, Kimi K3, GPT-5.6 Luna, Gemini 3.6 Flash, MiMo-V2.5, MiniMax M3, Step 3.7 Flash, Grok 4.6.
- BOLD flagged 1 of 124 responses for toxicity. The smallest corrected category q-value across BOLD metrics was 0.758, shared by MiniMax M3.

## Race/category breakdown

- For Grok 4.6, targeted DecodingTrust agreement was Asians 100.0%, Black people 75.0%, Latin Americans 75.0%, White people 100.0%. 0 of 24 corrected model-regime race-group tests reached q < 0.05, so the high targeted rate should be interpreted as broad elicitation susceptibility rather than a detected difference specific to one race group.
- Nonzero BOLD toxicity cells were Grok 4.6 / African Americans. 0 of 24 corrected model-metric category tests reached q < 0.05; sparse flags do not establish a race-category disparity.

## Construct comparison

For this descriptive cross-benchmark classification, an implicit signal requires both an IAT cluster interval excluding zero and corrected q < 0.05. An explicit behavior is counted when any DecodingTrust agreement or BOLD toxicity flag is observed. The latter rule is descriptive, not a significance claim.

- Both signals: none detected.
- Implicit only: none detected.
- Explicit behavior only: MiMo-V2.5, Step 3.7 Flash, Grok 4.6.
- Neither detected by these samples: Qwen 3.5 9B, Kimi K3, GPT-5.6 Luna, Gemini 3.6 Flash, MiniMax M3.

`Neither detected` does not mean unbiased. It means these finite, prompt-dependent samples did not meet the stated detection rule.

## Validity, routing, and cost

| Model | IAT valid | DT valid | BOLD valid | DT refusal/unclassified | Expected route | Observed providers | Reported benchmark cost |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| Qwen 3.5 9B | 12/12 (100.0%) | 48/48 (100.0%) | 14/16 (87.5%) | 16.7% | `deepinfra/bf16` | DeepInfra | $0.0020 |
| Kimi K3 | 12/12 (100.0%) | 48/48 (100.0%) | 16/16 (100.0%) | 0.0% | `deepinfra/bf16` | DeepInfra | $0.2069 |
| GPT-5.6 Luna | 12/12 (100.0%) | 48/48 (100.0%) | 16/16 (100.0%) | 0.0% | `azure/eu` | Azure | $0.0051 |
| Gemini 3.6 Flash | 12/12 (100.0%) | 48/48 (100.0%) | 16/16 (100.0%) | 18.8% | `google-vertex/global` | Google | $0.0289 |
| MiMo-V2.5 | 12/12 (100.0%) | 48/48 (100.0%) | 16/16 (100.0%) | 12.5% | `deepinfra/bf16` | DeepInfra | $0.0107 |
| MiniMax M3 | 9/12 (75.0%) | 47/48 (97.9%) | 16/16 (100.0%) | 8.5% | `deepinfra/fp8` | DeepInfra | $0.0079 |
| Step 3.7 Flash | 12/12 (100.0%) | 48/48 (100.0%) | 14/16 (87.5%) | 0.0% | `deepinfra` | DeepInfra | $0.0451 |
| Grok 4.6 | 12/12 (100.0%) | 48/48 (100.0%) | 16/16 (100.0%) | 2.1% | `xai/zdr` | xAI | $0.2164 |

## Category inference

Demographic/category comparisons are exploratory maximum-minus-minimum tests. DecodingTrust tests are corrected across models within each regime; BOLD tests are corrected across models within each metric. Full effect sizes, sample counts, raw p-values, and corrected q-values are in:

- `results/decodingtrust/decodingtrust_race_eight_model_pilot_20260824_v2/demographic_significance.csv`
- `results/bold/bold_race_eight_model_pilot_20260824_v2/category_significance.csv`

## Pilot versus expanded evidence

The separately archived pilot used one generation per IAT image, 48 DecodingTrust tasks/model, and 16 BOLD prompts/model. Expansion was authorized because technical validity was adequate and the pilot intervals were too broad for useful null interpretation. The primary estimates use ten generations per IAT image and 100 unique BOLD prompts/model. DecodingTrust remains pilot evidence at 48 tasks/model because the pinned Kimi endpoint drifted before expansion; no route substitution was used. The expansion design was frozen before final effect directions were inspected.

Pilot outputs remain in `results/summary/eight_model_pilot_summary.csv` and `results/summary/eight_model_pilot_overview.png`. They are not pooled with or presented as equivalent to the expanded estimates.

## Limitations

The IAT analysis has only 12 image clusters. Hosted routes represent model-plus-provider conditions and may differ from local weights. Automated toxicity, sentiment, regard, and response classifiers can misread context. All results depend on the exact prompts, images, lexicons, and decoding contract. Null results are uncertainty-limited evidence, not proof that bias is absent.
