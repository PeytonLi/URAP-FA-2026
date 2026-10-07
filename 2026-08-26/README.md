# August 26 Presentation Figures

This folder is the compact presentation package requested for the Gemma-inclusive model
comparisons. It retains the four original figures and adds six cleaner `v2` figures for slides.
The standalone script regenerates both sets from frozen aggregate tables. Response-level data,
selected DecodingTrust prompts, and model/input provenance are also included. No API keys, image
payloads, model weights, caches, or RunPod offloads are included.

## Recommended V2 Figures

| Figure | View | Gemma route | Grok |
| --- | --- | --- | --- |
| `01_iat_gemma_local_without_grok_v2.png` | IAT-12 negative-term rates | Local pinned `google/gemma-3-12b-it` | Excluded |
| `01_iat_gemma_local_with_grok_v2.png` | IAT-12 negative-term rates | Local pinned `google/gemma-3-12b-it` | Included |
| `02_decodingtrust_gemma_hosted_without_grok_v2.png` | DecodingTrust stereotype agreement | OpenRouter / DeepInfra BF16 | Excluded |
| `02_decodingtrust_gemma_hosted_with_grok_v2.png` | DecodingTrust stereotype agreement | OpenRouter / DeepInfra BF16 | Included |
| `03_iat_black_minus_white_gap_without_grok_v2.png` | IAT-12 Black-minus-White gap | Local pinned `google/gemma-3-12b-it` | Excluded |
| `03_iat_black_minus_white_gap_with_grok_v2.png` | IAT-12 Black-minus-White gap | Local pinned `google/gemma-3-12b-it` | Included |

The original four explanatory figures remain under the same names without the `_v2` suffix. The
`v2` figures use shorter titles, shorter axis labels, and no explanatory footer. The two gap
figures report Black minus White negative-term rates in percentage points; positive values mean
the observed rate was higher for Black face images. Whiskers are 95% confidence intervals.

The IAT figures use the pinned local Gemma revision
`96b6f1eccf38110c56df3a15bffe176da04bfd80`. DecodingTrust is text-only, so its Gemma row uses
the hosted `google/gemma-3-12b-it` route documented in
`gemma_decodingtrust_model_provenance.json`. The different Gemma serving paths are identified in
the plot titles and must not be treated as an identical inference environment.

## Regenerate

Use Python 3.11 from this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build_presentation_plots.py
```

The script reads the eight frozen `*_summary.csv` and `*_significance.csv` tables and replaces
the four original PNGs and six `v2` PNGs in this directory. It validates that the no-Grok family
has eight models and the with-Grok family has nine. Font rendering and PNG binary hashes can vary
across platforms even when plotted values are identical.

## Data Files

- `iat_response_level_data.csv`: the observations underlying both IAT figure families.
- `decodingtrust_response_level_data.csv`: the observations underlying both DecodingTrust
  figure families.
- `decodingtrust_selected_prompts.csv`: the frozen 48-prompt race-focused pilot contract.
- `*_summary.csv`: plotted rates and confidence intervals.
- `*_significance.csv`: exploratory permutation tests and Benjamini-Hochberg adjusted p-values.
- `local_gemma_iat_input_manifest.json`: hashes and provenance for the 12 local IAT images;
  image payloads are intentionally excluded.
- `*_model_provenance.json`: requested model identifiers and frozen provider metadata.
- `presentation_manifest.json`: source and output manifest from the full August 26 pipeline.
- `CHECKSUMS.sha256`: integrity hashes for every publication file except the checksum file itself.

The complete benchmark and API orchestration code remains in `../2026-08-17/` and
`../2026-08-24/`. This folder contains only the presentation-specific plotting layer and its
frozen inputs.

## Interpretation Limits

The IAT comparison contains 12 unique face images with repeated generations, so uncertainty is
clustered by image rather than by response alone. DecodingTrust contains 48 prompts per model and
uses automated response classification. All significance tests are exploratory, and the adjusted
p-values should be interpreted together with effect sizes, confidence intervals, validity, and
refusal rates.
