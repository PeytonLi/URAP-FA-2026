# URAP FA-2026 — Fine-tuning week report (2026-10-20)

Model: `google/gemma-3-12b-it` @ `96b6f1eccf38110c56df3a15bffe176da04bfd80`
Adapters: LoRA SFT + LoRA ORPO, each trained on the same 231 exported targets
(184 train / 47 dev; 29 distinct chosen triples; confidence-gated, dual-teacher
labels over 1,920 FairFace images). Per-epoch checkpoints backed up to private
Hugging Face repos with sha256-verified commits. Training: 138 optimizer updates
each, exit 0, image-conditioning checks passed.

## 1. IAT benchmark — significance tests (black vs white, before vs after)

Unit of inference is the stimulus image (12 images, 50 repeats each; matched
image-level pairing preserved). Tests: block-permutation DiD, cluster-robust OLS
`rate ~ black*tuned` (SE clustered by image), exact sign-flip per-group changes.
Seed 20261020, 25,000 MC draws.

| metric | base | tuned (sft / orpo) | delta | 95% CI | raw p | BH q |
|---|---|---|---|---|---|---|
| Black−White negative-term gap | +0.8400 | +0.0289 | −0.8111 | [−0.993, −0.542] | 0.005 | 0.005 |
| OLS cluster-robust gap change | — | — | −0.8111 | — | <0.0001 | <0.0001 |

Per-group rate changes (exact sign-flip, 2^6):

| group | rate change | raw p | Holm p |
|---|---|---|---|
| Black | −0.0144 | 0.250 | 0.250 |
| White | +0.7967 | 0.031 | 0.063 |

**Interpretation.** The racial disparity on the IAT task is essentially
eliminated (+84.0pp → +2.9pp, significant at p=0.005 and confirmed by OLS).
However, the mechanism is *leveling down*, not lifting up: the Black rate was
already ~99.7% negative and stayed ~98% (−1.4pp, ns), while the White rate rose
~80pp to converge at ~98% negative. Both groups now receive near-uniformly
negative IAT words. Report this to the advisor as "disparity eliminated via
uniform degradation," not as an unqualified fairness win.

## 2. Extra benchmarks — descriptive only (per clarified scope)

All runs on the same pod/model stack; raw responses, scored CSVs, run manifests
and plots preserved under
`2026-10-03/data/private/prime_native/20261020T161536.763422_0000/recovered/results/extra_benchmarks/`.

### DecodingTrust stereotype agreement (288 prompts)

| regime | base | orpo | delta |
|---|---|---|---|
| overall agreement | 26.7% [21.9, 32.1] | 28.8% [23.9, 34.3] | +2.1pp (CIs overlap) |
| benign agreement | 1.0% | 1.0% | 0 |
| untargeted agreement | 15.6% | 16.7% | +1.0pp |
| targeted agreement | 63.5% | 68.8% | +5.2pp |
| no-response rate | 13.2% | 9.7% | −3.5pp |

Per-group agreement (n small): Black people 0%→0% (50%→42% no-response),
White people 12.5%→12.5%, Europeans 30.8%→38.5%. Agreement on
explicitly-targeted stereotypes rose the most — worth flagging.

### BOLD continuations (192 prompts; Detoxify/VADER/regard scorers pinned)

| scorer | base | orpo |
|---|---|---|
| toxicity rate | 0.000 | 0.005 (1 response) |
| mean sentiment (VADER) | +0.109 | +0.156 |
| mean regard | +0.208 | +0.245 |

Race subset (n=9 each, descriptive only): African_Americans VADER
+0.165→+0.340; European_Americans +0.119→+0.188. Both rose; the gap direction
did not flip.

### VisBias (84 image descriptions + 168 paired comparisons)

| outcome | base | orpo |
|---|---|---|
| chose A | 42 | 40 |
| chose B | 48 | 44 |
| CANNOT_DETERMINE | 78 (46.4%) | 84 (50.0%) |
| selected white | 49 | 47 |
| selected black | 41 | 37 |

Selection is near chance (49/41 and 47/37 white/black); uncertainty answers
rose slightly. Descriptions (n=84 per condition) show no group-flag pattern in
spot checks.

## 3. Response quality / refusal / diversity ledger

| benchmark | condition | n | valid | invalid/refusal | format rate | distinct words | unique outputs |
|---|---|---|---|---|---|---|---|
| iat12 | base | 600 | 600 | 0 | 1.000 | 13 | 23 triples |
| iat12 | sft | 600 | 600 | 0 | 1.000 | 35 | 104 triples |
| iat12 | orpo | 600 | 600 | 0 | 1.000 | 35 | 101 triples |
| decodingtrust | base | 288 | 288 | 0 | — | 5,910 | 288 |
| decodingtrust | orpo | 288 | 288 | 0 | — | 5,902 | 288 |
| bold | base | 192 | 192 | 0 | — | 991 | 191 |
| bold | orpo | 192 | 192 | 0 | — | 989 | 191 |
| visbias | base | 252 | 252 | 0 | — | 1,493 | 88 |
| visbias | orpo | 252 | 252 | 0 | — | 1,533 | 88 |

Diversity *increased* after tuning on the IAT task (13→35 distinct words,
23→~100 unique triples) — the closed gap is not vocabulary collapse.

## 4. Provenance / reproducibility

- Analysis code: `2026-10-20/analysis/finetune_significance.py`; outputs in
  `2026-10-20/analysis/out_iat_only/` (summary.md, paired_tests.csv,
  group_rates.csv, response_quality.csv).
- Runs: SFT `gemma3_12b_sft_20261020T051659` + ORPO
  `gemma3_12b_orpo_20261020T052731` (each 138 updates, exit 0); extras run on
  RunPod A40, attempt `20261020T161536.763422_0000`.
- Compute spend this round: ~$0.85 (RunPod A40, ~1.5h). Adapter weights on HF
  (private repos, per-epoch tags).
- DecodingTrust/BOLD/VisBias tasks re-used the repo's frozen 2026-09-05
  sha-pinned task inputs; VisBias normalized images regenerated and
  hash-verified against the frozen manifest.
