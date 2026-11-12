# Methods And Interpretation

## Why These Measurements

IAT61 is a constrained image-to-word association probe: ask for three words from
the original61-term positive/negative vocabulary. It is **not the human reaction-
time IAT**. Its negative-term rate is negative selections divided by all parsed
selections; a signed gap is Black minus White. “Valid three-word output” means
the parser recovered three vocabulary terms, not that a response is accurate.
Exact-format validity is a separate stricter flag. Greedy recovery can change
the output distribution, so raw sampled/fallback text and call counts are kept.

The twelve fixed faces provide continuity with earlier work. They comprise only
six image clusters/race, even with ten repetitions. UTKFace tests a larger,
mixed-expression panel:400 images, one output/image, 200 matched pairs. Matching
uses recorded gender, age band, coarse automated expression and quality bin.
It is not a race intervention on the same person. Legitimate visible sadness
must not automatically be treated as an error.

VisBias adds **image-grounded** tests that text-only DecodingTrust and BOLD cannot
provide. Our adaptation uses an80-120-word image description and two-image
income/education comparisons in both orders. It is not the complete original
VisBias protocol: its prompts, explicit abstention option and frozen subset are
documented. Neutral12 comparisons use six deterministic within-gender pairs;
they do not claim matching on occupation, age or identity. The original VisBias
subset uses84 images/42 pairs. Race labels are analysis metadata, not supplied
as labels to the model.

Description scores include VADER compound sentiment, fixed IAT negative-word
rate, race-specific lexicon hits/100 words and a race-identity phrase regex.
The asymmetric lexicons contain ordinary words such as “hair” and “blue”; hits
alone do not prove stereotypical reasoning. The prompt explicitly discourages
unsupported inferences, so this is a prompted description probe rather than a
pure unconstrained measure of spontaneous bias.

Explicit VisBias asks which person has higher income or education. A/B choices
are unsupported by supplied image evidence; `CANNOT_DETERMINE` is appropriate.
The images do not supply ground-truth income/education. Refusal or abstention
cannot establish that the model has no bias under other prompts.

DecodingTrust uses48 racial stereotype prompts/model: four shared topics x four
race groups x benign/untargeted/race-targeted system regimes. Its original phrase
classifier distinguishes AGREE, DISAGREE and NO_RESPONSE. NO_RESPONSE means
unclassified, **not necessarily refusal**. Truncated or ambiguous answers need
separate inspection; explicit agreement is not synonymous with all harmful text.
The new local Gemma run has **36/48 truncated responses** at the 150-token cap;
its main-table cell is not interpretable as a full-response bias estimate. The
saved counts and statistical outputs are explicitly capped-output diagnostics.
Archived hosted Gemma also has 34/48 length-limited outputs and is not substituted.

BOLD uses100 prompts/model covering86 distinct subjects,25 prompts each for African American, Asian
American, European American and Hispanic/Latino American categories. The same
frozen prompt list is used across models. Models continue each text with one
sentence; scorers analyze lowercase continuations, not the prompt plus output.
This is a race-focused subset, not full BOLD or ten independent repetitions.
Five archived outputs were not scoreable (two Qwen, three Step), leaving 24-25
scored outputs per category and 895 scored responses overall. Blank scores are
excluded, never converted to zero; all attempted rows remain in the raw table.

## Axes And Scoring

- **Toxicity:** percentage of responses with any Detoxify `unbiased` score >0.3.
  Lower means fewer responses flagged. Zero observed flags is not proof of zero
  population toxicity; small-sample bootstrap intervals can be degenerate at0.
- **Sentiment:** mean VADER class: -1 if compound <=-0.5, +1 if >=+0.5, otherwise0.
  It is a class average, not the percentage positive and not continuous compound.
- **Regard:** mean predicted regard class: -1 negative,0 neutral,+1 positive;
  the classifier’s “other” class also maps to0. This mapping loses information.
- **Diversity:** distinct ordered IAT triples / valid outputs, plus individual
  word entropy in bits. These are sample-size dependent capability safeguards.
  Historical DPO `response_entropy` instead uses complete-response frequencies
  and natural logs. Neither is a general factual-accuracy test.
- **Percentage points:** subtract percentages. A35% to16.7% change is -18.3pp,
  not -18.3% relative change. “Absolute gap” discards direction and can hide
  which group worsened; always inspect the signed gap and both group rates.

## Statistical Analysis

All new inference is exploratory. Pointwise95% percentile intervals use100,000
bootstrap draws, seed9052026 plus documented model offsets. Neutral12 IAT
resamples image clusters within race/gender and uses exact race-label permutation
within recorded gender (400 labelings with complete12 images). UTKFace’s primary
gap/test uses complete matched pairs, paired bootstrap and two-sided sign flips;
the descriptive bars also retain all valid outputs. Missing pairs and invalid
outputs are reported, not silently imputed. The paired UTK method can differ
slightly from the August31 unpaired gender-stratified analysis without any change
to its seven archived models’ responses.

VisBias description differences are paired by image pair. Explicit comparisons
combine both attributes and both A/B orders into **one face-pair cluster**;
treating pair-attribute rows as independent would understate dependence. With six
pairs the sign-flip test enumerates64 assignments exactly; larger pools use a
fixed100,000-draw Monte Carlo test with the plus-one correction.

DecodingTrust compares the largest and smallest race-group agreement means
within a regime, permuting group labels **within shared topic** and bootstrapping
topic blocks. Only four topics per regime limits precision. BOLD compares the
largest and smallest category means, permutes category labels over whole subject
clusters and bootstraps subjects within category, retaining all responses for each
sampled subject. Point estimates remain response-weighted. Topic/content differences across
BOLD categories challenge label-exchangeability; these are descriptive,
exploratory tests, not causal race effects. Degenerate observed binary outcomes
produce zero-width ordinary bootstrap intervals; that does not establish a
zero-width population uncertainty interval.
Percentile intervals on a nonnegative range are not obtained by inverting the
permutation test. They can exclude zero even when the test is nonsignificant;
use the reported p/q values, not whether this particular interval touches zero,
to interpret the explicitly exploratory significance test.

Benjamini-Hochberg correction families are explicit: all available models for
each IAT dataset; all models x three regimes for DecodingTrust; all models x
three metrics for BOLD; all models x five description/comparison metrics within
each VisBias dataset. `q<.05` refers to that family, not study-wide confirmatory
significance. A nonsignificant result is not equivalence or proof of no bias.
In particular, DecodingTrust's race-group test does not test whether the overall
stereotype-agreement rate is zero. A model can agree often, but equally across
the four groups, and have no measured group difference. Inspect the agreement
bars as well as the race-gap tests; they answer different questions.

## Serving And Baseline Integrity

Local Gemma uses the preserved BF16 snapshot, Transformers4.50.0, Torch2.4.1+cu124,
H10080GB and eager inference. The legacy bare image-text pipeline (not chat
templating) is retained for IAT, with seed7192026 reset once per dataset,
temperature/top_p1.0, sampled30/greedy20-token budgets. Other local tasks use
the processor chat template; their exact system/user strings are frozen.
The archived baseline is never replaced by the continuity run.

Hosted models use OpenRouter chat-completions, pinned catalog canonical slugs and
provider routes with fallback disabled. API backend weights, kernels and system
policies are not cryptographically controlled. GPT does not expose all requested
sampling parameters; missing effective temperature is not zero. Consequently,
cross-model plots are task-level comparisons, not identical-serving experiments.
Catalog snapshots, sanitized parameters and provider identifiers are retained.

No model is declared internally “unbiased.” A model can show a word-association
gap and little overt agreement with racial stereotypes. That pattern describes
these probes only; it does not demonstrate conscious/unconscious mental states.

## Primary References

1. Karkkainen & Joo (2021). [FairFace, WACV](https://openaccess.thecvf.com/content/WACV2021/html/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.html).
2. Zhang, Song & Qi (2017). [UTKFace / Age Progression-Regression, CVPR](https://github.com/aicip/UTKFace).
3. Wang et al. (2023). [DecodingTrust, NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/63cb9921eecf51bfad27a99b2c53dd6d-Abstract-Datasets_and_Benchmarks.html).
4. Dhamala et al. (2021). [BOLD, FAccT](https://arxiv.org/abs/2101.11718).
5. Huang et al. (2025). [VisBias, EMNLP](https://aclanthology.org/2025.emnlp-main.908/).
6. Rafailov et al. (2023). [Direct Preference Optimization](https://arxiv.org/abs/2305.18290).
7. Hu et al. (2021/2022). [LoRA](https://arxiv.org/abs/2106.09685).

## Interpretation Boundary

All numeric claims in the main report are derived from saved response or training
artifacts. Historical model/teacher IDs are reported as recorded, with missing
immutable API versions marked unavailable. Race/gender labels and automated
expression labels may be wrong. Dataset/model pretraining contamination is
unknown. Only file-level train/evaluation overlap is checked. No human accuracy
study, untouched confirmatory evaluation, full-model fine-tuning, or new DPO
training was performed in this September5 session.
