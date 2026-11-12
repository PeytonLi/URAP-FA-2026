# Notebook workflow

Runnable experiment notebooks (Python modules remain under `scripts/`):

1. [`01_gemma3_baseline.ipynb`](01_gemma3_baseline.ipynb) — frozen base IAT-12 baseline view + Colab pointer
2. [`02_gemma3_v1_gated_orpo.ipynb`](02_gemma3_v1_gated_orpo.ipynb) — V1 config/manifest + analysis index
3. [`03_gemma3_v2_valence_controlled.ipynb`](03_gemma3_v2_valence_controlled.ipynb) — V2 valence-controlled artifacts
4. [`04_gemma3_v3_coverage_pilot.ipynb`](04_gemma3_v3_coverage_pilot.ipynb) — V3 pilot (no adapter)
5. [`05_gemma3_v4_matched_orpo.ipynb`](05_gemma3_v4_matched_orpo.ipynb) — V4 matched labels + launch pointers
6. [`06_gemma3_v4_iat_stats.ipynb`](06_gemma3_v4_iat_stats.ipynb) — offline V4 IAT Holm tables
7. [`07_gemma3_prompt_sweeps.ipynb`](07_gemma3_prompt_sweeps.ipynb) — prompt sweeps 2–21 / C–D vs prompt 1
8. [`08_gemma3_v4_extras_decodingtrust_visbias_bold.ipynb`](08_gemma3_v4_extras_decodingtrust_visbias_bold.ipynb) — extras figures

Also:

- [`base/neutral12_base_colab.ipynb`](base/neutral12_base_colab.ipynb) — self-contained Colab base evaluation (GPU)
- [`archive/`](archive/) — pre-FA chronology packages

Path helper used by notebooks: `scripts/repo_paths.py`.

Public notebooks omit raw face images, tokens, and weights. Private recoveries stay under ignored paths such as `scripts/train_eval/_data/private/` and `datasets/private_targets/`.
