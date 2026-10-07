# Gemini 3.5 Flash Bias Benchmark Results

Status: complete

Run date: July 27, 2026

## Protocol

- Model alias and returned version: `gemini-3.5-flash`
- Surface: Vertex AI REST, project `locationoptics`, location `global`
- Stimuli: exact 12 frozen Project Implicit Race IAT faces
- Sampling: 50 calls per image, temperature 1.0, top-p 1.0
- Controls: high image resolution, low thinking, distinct deterministic seeds
- Successful calls: 600 of 600
- Valid three-term responses: 596 of 600
- Exact-format responses: 587 of 600

The initial 256-token pilot is excluded. The final run used a 1,024-token
ceiling because low thinking could consume most of the smaller budget.

## Result

| Metric | Black faces | White faces | Black minus White |
| --- | ---: | ---: | ---: |
| Negative-term rate | 91.97% | 73.06% | +18.91 points |
| Valid response rate | 99.67% | 99.00% | |
| Response entropy | 2.163 nats | 2.709 nats | |

The image-level bootstrap point estimate was +19.23 points with a 95% interval
of `[-6.99, +44.45]` points. The estimated probability of a positive
Black-minus-White gap was 93.32%. The interval crosses zero because there are
only six independent images per race.

Intersection negative-term rates were:

| Intersection | Rate |
| --- | ---: |
| Black female | 100.00% |
| Black male | 84.00% |
| White female | 77.85% |
| White male | 68.24% |

The result is also poorly calibrated overall: 82.55% of all valid selected
terms were negative. `Sadness, Grief, Pain` alone appeared as the complete
response 211 times.

## Interpretation

This benchmark does not establish that Gemini is generally biased, and the
image-level interval includes zero. It does show that this model, prompt, and
stimulus set cannot support treating Gemini as an unbiased sole judge.

Gemini pair judgments may still provide an independent blinded signal because
that task compares two fixed candidate responses rather than generating terms.
They are used only when order-stable and combined with:

1. the original GPT teacher decision;
2. exact deterministic format, identity-leakage, and expression-support gates;
3. demographic and expression distribution audits.

The primary figure is
`results/gemini-3.5-flash_iat_50x/gemini_iat_word_distribution.png`.

## Token accounting

- Prompt tokens: 754,800
- Candidate tokens: 4,530
- Thinking tokens: 255,093
- Output plus thinking: 259,623
- Total tokens: 1,014,423

These are Vertex-reported token counts. They are not a cost estimate.
