# Benchmark Coverage

## Model Groups

The requested labels '8 other open source' and '2 closed source' do not match the models actually tested. Below, the comparison cohort is five open-weight and three closed-source models. Qwen2.5-VL-3B has only a compatibility probe.

Open-weight cohort: Qwen 3.5 9B, Kimi K3, MiMo-V2.5, MiniMax M3 and Step 3.7 Flash. Closed-source cohort: GPT-5.6 Luna, Gemini 3.6 Flash and Grok 4.6.

| Benchmark | Base Gemma | Other open-weight models (5) | Closed-source models (3) | Tuned Gemma |
|---|---|---|---|---|
| IAT61 | [new 12-face and UTKFace results](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/results/analysis/summary.csv) | [new 12-face and UTKFace results](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/results/analysis/summary.csv) | [new 12-face and UTKFace results](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/results/analysis/summary.csv) | [new 12-face and UTKFace results](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/results/analysis/summary.csv); direct paired effect in results |
| VisBias | [prior image-based results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/visbias_neutral12_responses.csv) | [prior image-based results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/visbias_neutral12_responses.csv) | [prior image-based results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/visbias_neutral12_responses.csv) | Not done |
| DecodingTrust | [prior text-only results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/decodingtrust_responses.csv); local subset has severe truncation limitation | [prior text-only results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/decodingtrust_responses.csv) | [prior text-only results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/decodingtrust_responses.csv) | Not done |
| BOLD | [prior text-only results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/bold_scored_responses.csv) | [prior text-only results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/bold_scored_responses.csv) | [prior text-only results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/bold_scored_responses.csv) | Not done |

Incomplete new IAT cells: Kimi K3 (neutral12); MiniMax M3 (neutral12). Counts and reasons are in RESULTS.md.

## Input Components

Text means the prompt/wrapper accompanying the task, not a claim that image-based tasks were also tested without pictures.

| Benchmark | Text | Image1: standard 12 | Image2: UTKFace |
|---|---|---|---|
| IAT61 | [61-word prompt](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/src/multimodel_bias/constants.py) | [responses](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/results/analysis/neutral12_responses.csv) | [responses](https://github.com/eliaswuberkeley/URAP/blob/sep11-evaluation-v1/2026-09-11/results/analysis/utkface_responses.csv) |
| VisBias | [description/comparison wrappers](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/src/multimodel_bias/visbias.py) | [prior image-based results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/visbias_neutral12_responses.csv) | Not done |
| DecodingTrust | [prior text-only results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/decodingtrust_responses.csv) | N/A: text-only test | N/A: text-only test |
| BOLD | [prior text-only results](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/results/analysis/bold_scored_responses.csv) | N/A: text-only test | N/A: text-only test |

## Prior DPO Evidence

[Audited prior methods, failed attempts, datasets, and outcomes](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/report/FORMAL_RESEARCH_REPORT.md)

[Frozen 100-row FairFace preference dataset (80 train/20 development)](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/reference/dpo/august10/august10_negative_expression_pilot_v4/dataset/mixed_dpo_pairs_frozen.csv)

[Historical training implementation](https://github.com/eliaswuberkeley/URAP/blob/sep5-evidence-v1/2026-09-05/reference/dpo/august10_code/scripts/train_mixed_dpo_lora.py)

The new iteration does not rerun explicit bias benchmarks or add UTKFace training preferences. 'Not done' is not a null result.
