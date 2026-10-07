# Configuration Files

- `models_aug31.json`: active eight-model alias, canonical-revision, route, decoding,
  token, and concurrency contracts. Gemma is intentionally absent.
- `iat_utkface_aug31.json`: active 400-image UTKFace IAT-style generation, budget,
  bootstrap, and privacy contract.
- `visbias_pilot_aug31.json`: active 84-description plus 168-comparison task contract per
  model. `visbias_compatibility_aug31.json` is its low-cost probe variant.
- `models_aug31_deepinfra_conservative.json` and
  `models_aug31_minimax_conservative.json`: route-identical throttled execution
  contracts used to recover provider 429s.
- `models_aug31_step_visbias.json` and `models_aug31_step_visbias_4096.json`: bounded
  Step hidden-reasoning token-cap repairs. Per-row `max_tokens_requested` records which
  cap applied; no route, prompt, image, seed, or scoring contract changed.
- `study_aug31.json`: UTKFace and VisBias source revisions, matching thresholds, study
  scope, account reserve, and additional-session spending limit.
- `models_explicit_aug24.json`, `models_bold_aug24.json`, and `study_aug24.json`: frozen
  historical references used only to reproduce archived August 24 text/image analyses.
- `models.json` and `iat12_pilot.json`: inherited August 17 compatibility references; no
  active August 31 command uses them.
