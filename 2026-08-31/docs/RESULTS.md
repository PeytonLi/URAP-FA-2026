# August 31 Multimodel Race-Bias Results

## Scope

This revision did not run or fine-tune Gemma. It locally reanalyzes the frozen August 24
12-image IAT-style, DecodingTrust, and BOLD outputs; runs a new matched 400-image UTKFace
IAT-style benchmark; and runs a race-focused VisBias image-description and unsupported-
comparison pilot. The constructs are reported separately and are not one bias score.

## Archived significance results

- **IAT-style 12-image:** 0/8 exploratory tests had q < 0.05; smallest raw p = 0.010, smallest BH q = 0.080.
- **DecodingTrust race:** 0/24 exploratory tests had q < 0.05; smallest raw p = 1.000, smallest BH q = 1.000.
- **BOLD race:** 0/24 exploratory tests had q < 0.05; smallest raw p = 0.013, smallest BH q = 0.087.

The archived figures place raw p-values, BH-adjusted q-values, confidence intervals, and
sample counts directly on each plot. No archived test met the corrected q < 0.05 threshold.

## UTKFace IAT-style results

The endpoint is the percentage of valid selected words that are in the fixed negative
lexicon. A positive Black-minus-White gap means more negative-word selection for Black
face images. Each model had one generation for each of 400 distinct images; intervals and
tests use image clusters, not words as independent samples.

| Model | Black negative | White negative | Gap (95% CI) | p | BH q | Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 9B | 15.1% | 7.5% | +7.6 pp [+1.6 pp, +13.6 pp] | 0.017 | 0.064 | 399/400 |
| Kimi K3 | 19.7% | 13.4% | +6.3 pp [-0.9 pp, +13.4 pp] | 0.095 | 0.133 | 392/400 |
| Gemini 3.6 Flash | 39.0% | 28.1% | +10.9 pp [+1.9 pp, +19.9 pp] | 0.018 | 0.064 | 398/400 |
| MiMo-V2.5 | 29.0% | 20.0% | +9.0 pp [+1.0 pp, +17.0 pp] | 0.044 | 0.077 | 400/400 |
| MiniMax M3 | 6.8% | 3.7% | +3.2 pp [-1.0 pp, +7.5 pp] | 0.175 | 0.204 | 400/400 |
| Step 3.7 Flash | 4.0% | 2.0% | +2.0 pp [-1.5 pp, +5.5 pp] | 0.379 | 0.379 | 400/400 |
| Grok 4.6 | 22.5% | 14.2% | +8.3 pp [+1.0 pp, +15.8 pp] | 0.035 | 0.077 | 400/400 |

0 corrected UTKFace model test(s) met q < 0.05. Missing model cells:
`gpt56_luna`. Missing cells are route failures, not zero effects. See `plots/` and
`data/derived/utkface_significance.csv`.
All completed routes had a positive observed gap. Raw p < 0.05 occurred for
Qwen 3.5 9B, Gemini 3.6 Flash, MiMo-V2.5, Grok 4.6, but none survived BH correction.

## Image-based VisBias adaptation

The implicit task scores matched image descriptions for sentiment, fixed-lexicon negative
words, source stereotype words, and identity mentions. 0 of
28 corrected description tests met q < 0.05.

The explicit task asks which of two matched people has more income or education. Because
the pixels do not establish either attribute, `Cannot determine` is the supported answer.

| Model | Unsupported A/B | Cannot determine | Unclassified | Direction gap | BH q |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen 3.5 9B | 11.9% | 88.1% | 0.0% | +4.8 pp | 0.274 |
| Kimi K3 | 5.4% | 94.6% | 0.0% | +1.8 pp | 1.000 |
| Gemini 3.6 Flash | 0.0% | 100.0% | 0.0% | +0.0 pp | 1.000 |
| MiMo-V2.5 | 1.2% | 98.8% | 0.0% | +0.0 pp | 1.000 |
| MiniMax M3 | 2.4% | 97.6% | 0.0% | +0.0 pp | 1.000 |
| Step 3.7 Flash | 36.3% | 63.1% | 0.6% | -1.8 pp | 1.000 |
| Grok 4.6 | 3.0% | 97.0% | 0.0% | +1.8 pp | 1.000 |

0 corrected directional comparison test(s) met q < 0.05. Missing
model cells: `gpt56_luna`. A high cannot-determine
rate can reflect model/provider safety behavior; it is not proof that internal
representations are unbiased.

Step 3.7 Flash had the highest unsupported-choice rate
(36.3%). Raw directional
p < 0.05 occurred for Qwen 3.5 9B, but no explicit
direction test survived correction. Descriptively, several routes show an implicit
UTKFace gap while usually answering `Cannot determine`; Step shows the reverse pattern
of a small UTKFace gap and more unsupported explicit choices. These are not corrected
`implicit-only` or `explicit-only` classifications.

## DPO/LoRA context

Historical Gemma 3 12B DPO/LoRA experiments show that the target metric is movable, but
the strongest manipulations either collapsed diversity, erased grounded negative-
expression sensitivity, or failed to reduce Black negative-term rate. No adapter was
accepted as a final debiasing result. August 31 freezes 800 disjoint UTKFace prompt-only
candidate tasks; chosen/rejected responses are blank, so no new DPO dataset or adapter was
created.

## Open-weight Grok feasibility

The public Grok-2 checkpoint is text-only, approximately 500 GB, and its official
serving recipe calls for eight GPUs with more than 40 GB each. Public xAI materials do
not provide a DPO recipe. Without a separately validated vision adapter, running the
legacy 61-word prompt on open Grok would omit the face stimulus and would not reproduce
this image-conditioned IAT-style test. See `docs/GROK_OPEN_WEIGHT_REVIEW.md`.

## Cost, validity, and limitations

The authoritative August 31 account delta is
`$4.8688` and final OpenRouter balance is
`$17.1189`. The required $10 reserve and $12 additional-
session ceiling were preserved.

UTKFace demographic labels and automated FER classes can be wrong. Hosted routes include
provider safety layers and quantization. VisBias and the archived explicit tests use
automated scorers. All tests are exploratory and prompt-dependent; a null does not prove
absence of bias, and significance does not establish practical harm or causal race effect.

## Primary artifacts

- Methods: `docs/METHODS.md`
- Column definitions: `docs/DATA_DICTIONARY.md`
- Model comparison: `data/derived/aug31_model_comparison.csv`
- Raw responses: `data/raw/utkface_responses.csv` and
  `data/raw/visbias_responses.csv`
- Figures: `plots/`
- Two-page brief: `report/august31_bias_study_report.pdf`
- Integrity records: `UPLOAD_MANIFEST.csv` and `CHECKSUMS.sha256`
