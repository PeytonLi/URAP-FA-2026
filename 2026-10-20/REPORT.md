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

## 5. Round 2 — v2 mixed-valence relabeling + KTO (2026-10-21)

### What changed

The v1 dataset's 231 triples were all pure-pole (all-positive or all-negative).
The v2 pipeline relabeled the same pool with two teacher passes
(gpt-5-mini + gpt-4.1-mini, 1,920 images each, 0 failures), a two-class gate,
and a second adjudication pass with a mixed-triple mandate for ambiguous
faces (the first pass's 2-of-3-overlap queue structurally could not produce
mixed triples). Export maximizes word coverage and drops pair-balance
violations. Final dataset `curated_iat_targets_v2.json`: **383 records**
(299 train / 84 dev), **61 mixed-valence triples (16%, was 0)**, 27 distinct
chosen words (was 22), 49 pair-balance violations dropped. Same active
dataset config slot, same train/benchmark separation.

Runs: SFT + ORPO + KTO on the v2 set (KTO added per Peyton's direction;
adapter weights on private HF repos per run). Pod attempts
`20261021T023045.753842_0000` (main) + `20261021T084030.287641_0000`
(resume for the KTO VisBias tail truncated at the 6h deadline). Spend ≈ $3.60.

### IAT-12 — round 2 results

| condition | black neg. rate | white neg. rate | gap | exact perm. p |
|---|---|---|---|---|
| base (rerun) | 0.997 | 0.160 | +0.837 | 0.0022 |
| SFT (v2) | 0.969 | 0.877 | **+0.092** | 0.013 |
| ORPO (v2) | 0.976 | 0.891 | **+0.084** | 0.022 |
| KTO (v2) | 1.000 | 1.000 | **0.000** | 1.000 |

DiD vs base: SFT −74.4pp (p=0.010), ORPO −75.2pp (p=0.005), KTO −83.7pp
(p=0.005); OLS cluster-robust p<1e-5 for all three.

**vs round 1** (v1 data, gap +0.029 ns): the v2 models keep ~90% of the
debiasing but leave a small residual gap that is still significant
(p≈0.01–0.02). The mixed-valence labels bought partial relief of the
leveling-down (white negative rate 95.3% → 87.7–89.1%) and more response
diversity (unique triples ~100 → 138–148), but the output remains
dominantly negative — the end goal (broad vocabulary, equal distributions)
is closer, not met.

### Word-distribution equality (the "fair AND expressive" metric)

| model | distinct words B/W | norm. entropy B/W | TV distance | equality perm p |
|---|---|---|---|---|
| base | 7 / 11 | 0.31 / 0.49 | 0.84 | 0.0022 |
| SFT | 26 / 32 | 0.50 / 0.58 | 0.16 | 0.229 |
| ORPO | 24 / 31 | 0.48 / 0.57 | 0.18 | 0.195 |
| KTO | 3 / 5 | 0.27 / 0.27 | **0.00** | 1.000 |

**KTO is the flattening failure mode made concrete**: it achieves a perfect
zero gap by emitting permutations of "Grief, Pain, Sadness" on ~99% of slots
(8 unique responses total). Binary desirability plus a KL baseline does not
penalize within-condition collapse — maximally fair, maximally flat.

### Extra benchmarks (round 2, descriptive + paired direct-change tests)

- **DecodingTrust**: overall agreement 29.9% → 29.9% (ORPO) / 30.9% (KTO);
  targeted regime +4.2pp / +6.2pp — all changes ns (paired perm p≥0.14).
  Note: base rerun measured 29.9% vs 26.7% in round 1 — ~3pp run-to-run
  variance at temp 1.0.
- **BOLD**: race-category sentiment spread 0.114 → 0.025 under ORPO —
  paired perm p=0.03, the one real extras improvement (less spread between
  race categories). KTO same direction, ns. Toxicity/regard 0 throughout.
- **VisBias**: selection ~50/50 everywhere; all direct-change tests p=1.0.

### Provenance

Analysis: `2026-10-20/analysis/specific_breakdowns_r2.py` (Alica-format
graphs + CSVs in `specific_r2/`), significance suite rerun over 4 IAT + 3
extras conditions → `specific_r2/significance/` (525 tests). v2 labeling
code in `2026-10-20/labeling/{policy,gate_v2,export_v2,adjudicate_v2,
adjudicate_v2b_workflow}.py`.

### Honest round-2 verdict

The mixed-valence data moved expressiveness in the right direction but only
modestly: teachers still judged most benchmark-type faces negative (they
are stern), so 84% of triples stayed pure-pole-negative-ish. SFT/ORPO keep
the debiasing with slightly more variety; KTO "solves" fairness by collapse
and is a counterexample, not a candidate. A real vocabulary expansion would
need either teacher prompts that elicit mid-register words the current
vocab offers (Serious/Calm absent from the 61-word list) or vocab-eligible
labels on genuinely varied face stimuli.

## 6. V3 implementation and a stricter variety baseline

**Execution status:** v3 code is implemented and fixture-tested, but the live
candidate-label pilot received OpenAI `credit_balance_exhausted`. It produced
no accepted labels. Labeling then moved to Bitdeer Model Studio with
DeepSeek V4.1 Flash-1017 and GLM-5.3-Flash as independent teachers.

**Feasibility result (pilot, not a final dataset).** The first Bitdeer pilot
approved nothing because teachers placed general caveats in vetoing review
flags. After flags were restricted to image-level blockers (policy v3_2), the
same balanced 48-image pilot gave 95/96 valid calls. Both teachers supported
at least one word with confidence >=0.8 on 11 images; only 4 images had the
three words needed for one triple. The jointly supported vocabulary was four
smile words: Smiling, Happy, Joyful and Cheerful. DeepSeek alone used 13 words
confidently, GLM four. None of the negative-expression pilot images received
joint support, and no complete Black/White matched block survived.
Honest visible-expression labels therefore look far from the 50-word target;
many vocabulary words (for example Friendship, Disaster, Magnificent) are not
facial expressions. Reaching 50 words would need a different target design,
which is a research decision rather than a labeling fix.
No v3 adapter has been trained or benchmarked. The active
dataset remains v2; these numbers reanalyze the stored round-2 outputs.

Future fine-tuning is **ORPO only**, starting from the pinned Gemma 3 12B-IT
base. SFT/KTO results remain historical evidence, not new training conditions.
The v3 data method is independently supported word candidates, at most two
coverage follow-ups, and multiple non-permutation preference triples per image.
The exact protocol and commands are in `README.md`.

### Baseline using non-permutation diversity

`analysis/variety.py` distinguishes an ordered string from an unordered
three-word combination. It also separates the untouched sampled answer from
the final scorer output, which may contain a deterministic fallback.

| Measure | Base, final scored | ORPO v2, final scored | ORPO v2, strict sampled |
| --- | ---: | ---: | ---: |
| Valid / 600 slots | 600 | 600 | 571 |
| Vocabulary union / 61 | 11 | 34 | 34 |
| Black / White words | 7 / 11 | 24 / 31 | 24 / 31 |
| Shared vocabulary words | 7 | 21 | 21 |
| Median distinct unordered triples per image | 3.5 | 17.0 | 17.0 |
| Dominant unordered triple share | 43.0% | 38.0% | 38.0% |
| Effective vocabulary size, exp(Shannon entropy) | 8.26 | 8.84 | 8.90 |
| Words used at least five times in both groups | 4 | 11 | 11 |
| Image-weighted word-distribution TV | 0.837 | 0.179 | 0.176 |
| Black-minus-White negative-word gap | +83.7 pp | +8.4 pp | +8.6 pp |

ORPO v2 needed fallback on 29 slots, leaving 571 valid strict sampled answers.
The strict check reuses the frozen scorer's parsing and exact-format rules,
including its handling of adjacent duplicate words. Raw samples and final
scorer outputs are both retained; the frozen benchmark's scoring/fallback
rules are not changed by this reanalysis.

Earlier ordered-string counts overstate semantic variety when many answers
are permutations. Coverage also needs frequency context: ORPO v2 has 34 words
in the union, yet an effective vocabulary of only about nine equally frequent
words. The new target remains at least 50 union words, with each group's
coverage, overlap, entropy and concentration reported alongside it. It is not
enough for one group to carry the rare-word tail.

### Corrections and interpretation limits

- Training-label coverage is **not** a hard upper limit on generated
  vocabulary. In the stored v2 run, 5.0% of final responses contain a word
  absent from the training-split chosen labels. Coverage-targeted training
  is an intervention to test, not an already established causal explanation.
- The policy allows 50 preferred-label words, excluding 11 unsupported
  judgments. Evaluation still contains all 61. Reaching the requested target
  cannot justify unsupported face labels or a changed test prompt.
- Exact image-level permutation p-values do not establish equality when
  nonsignificant. TV distance is still nonzero, and six images per group
  provide limited evidence. Race-label exchangeability is an assumption, not
  a randomized causal race intervention.
- Historical extras p-values are exploratory; a small unadjusted p-value
  among many tests is not independently confirmed improvement.
- KTO's observed collapse is established for that run. Attributing the
  collapse to a specific objective mechanism would require controlled
  ablations; it is not proven by the response histogram alone.

Reproducible outputs: `analysis/variety_r2/summary.csv`, `per_image.csv`,
`word_counts.csv`, `provenance.json`, `word_counts.png`,
`per_image_variety.png` and `per_image_rates.png`. Provenance records the
input checksums and analysis source hash. Inference uses all 924 assignments
of six of the 12 images to a group and 10,000 image-cluster bootstrap draws;
repeated generations are not treated as independent people.
