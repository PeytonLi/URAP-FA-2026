# Base vs fine-tuned significance report

Generated: 2026-10-20T16:10:45  |  seed: 20261020  |  MC draws: 25000

## Headline: did tuning change the disparity?

| benchmark | comparison | metric | base | tuned | delta | 95% CI | raw p | BH q |
|---|---|---|---|---|---|---|---|---|
| iat12 | sft_vs_base | black_minus_white_gap_change (DiD) | 0.8400 | 0.0289 | -0.8111 | [-0.993, -0.542] | 0.0050 | 0.0050 |
| iat12 | orpo_vs_base | black_minus_white_gap_change (DiD) | 0.8400 | 0.0289 | -0.8111 | [-0.993, -0.542] | 0.0050 | 0.0050 |

## Secondary tests (per-group changes)

| benchmark | comparison | group | metric | delta | raw p | Holm p | test |
|---|---|---|---|---|---|---|---|
| iat12 | sft_vs_base | Black-White | black_minus_white_gap_change (OLS cluster-robust) | -0.8111 | 0.0000 | 0.0000 | OLS rate~black*tuned, cluster-robust SE by image |
| iat12 | sft_vs_base | Black | negative_term_rate_change | -0.0144 | 0.2500 | 0.2500 | exact_sign_flip_2^6 |
| iat12 | sft_vs_base | White | negative_term_rate_change | +0.7967 | 0.0312 | 0.0625 | exact_sign_flip_2^6 |
| iat12 | orpo_vs_base | Black-White | black_minus_white_gap_change (OLS cluster-robust) | -0.8111 | 0.0000 | 0.0000 | OLS rate~black*tuned, cluster-robust SE by image |
| iat12 | orpo_vs_base | Black | negative_term_rate_change | -0.0144 | 0.2500 | 0.2500 | exact_sign_flip_2^6 |
| iat12 | orpo_vs_base | White | negative_term_rate_change | +0.7967 | 0.0312 | 0.0625 | exact_sign_flip_2^6 |

## Response quality / refusal / diversity ledger

```json
[
  {
    "benchmark": "iat12",
    "condition": "base",
    "responses": 600,
    "valid_responses": 600,
    "invalid_or_refusal": 0,
    "exact_format_rate": 1.0,
    "distinct_words": 13,
    "unique_response_tuples": 23,
    "normalized_word_entropy": 0.515660304956015,
    "errors": 0
  },
  {
    "benchmark": "iat12",
    "condition": "sft",
    "responses": 600,
    "valid_responses": 600,
    "invalid_or_refusal": 0,
    "exact_format_rate": 1.0,
    "distinct_words": 35,
    "unique_response_tuples": 104,
    "normalized_word_entropy": 0.5379099988641644,
    "errors": 0
  },
  {
    "benchmark": "iat12",
    "condition": "orpo",
    "responses": 600,
    "valid_responses": 600,
    "invalid_or_refusal": 0,
    "exact_format_rate": 1.0,
    "distinct_words": 35,
    "unique_response_tuples": 101,
    "normalized_word_entropy": 0.5417847005025751,
    "errors": 0
  }
]
```
