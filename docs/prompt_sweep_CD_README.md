# 2026-11-10 — Prompt-sweep Batch C (prompts 14–17)

Tuesday week folder for the Batch C evaluation: base vs saved v4 ORPO on
IAT prompts 14–17 (Gemma 3 12B-IT only).

## Contents

- [`analysis/prompt_sweep_C/`](analysis/prompt_sweep_C/) — paired responses,
  Holm tests, and per-prompt word-count / variety graphs (base and tuned)
- [`analysis/prompt_sweep_C_vs_prompt_1/`](analysis/prompt_sweep_C_vs_prompt_1/) —
  required prompt-X versus frozen prompt-1 contrasts (base and ORPO separately)

Shared analysis tooling remains under `2026-10-20/analysis/prompt_sweep.py`.
Earlier batches:

- prompts 2–4 → [`../2026-10-27/analysis/prompt_sweep_v1/`](../2026-10-27/analysis/prompt_sweep_v1/)
- prompts 5–13 → [`../2026-11-03/analysis/`](../2026-11-03/analysis/)
