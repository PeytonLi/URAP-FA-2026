# Review of Image-Based Bias Benchmarks

## Decision

The August 31 pilot uses a race-focused adaptation of
[VisBias](https://aclanthology.org/2025.emnlp-main.908/) because it directly tests
current generative vision-language APIs, includes both explicit and implicit constructs,
has public code and images, and can be run with pinned black-box routes. The source is
Huang et al., EMNLP 2025, DOI `10.18653/v1/2025.emnlp-main.908`; code is pinned from
[the public repository](https://github.com/limenlp/VisBias).

This selection is not a claim that a live citation index proves VisBias is the single
most-cited multimodal bias paper. Citation rankings change and no bibliometric snapshot
was available in the workspace. It is the best directly implementable match among the
screened benchmarks; the more comprehensive MultiTrust source requires a gated dataset,
which prevents a fully reproducible unattended run here.

## Screened alternatives

| Benchmark | Image-based bias coverage | Strength | Reason not used as the pilot |
| --- | --- | --- | --- |
| VisBias (EMNLP 2025) | Explicit comparisons and implicit descriptions/forms | Direct fit for generative VLM APIs; public task images and code | Selected, but adapted rather than exactly reproduced |
| MultiTrust (NeurIPS 2024) | Broad multimodal trustworthiness including fairness | Wider safety and robustness coverage | Full dataset is gated and large; access/redistribution could not be automated |
| VLStereoSet (AACL 2022) | Image-text stereotype association | Older established controlled association design | Primarily caption-selection/embedding oriented, less direct for chat-generation APIs |
| Bias in the Picture (NeurIPS 2025 workshop) | Open-ended news-image social cues | Rich realistic contexts and human-verified judge rubric | Newer workshop benchmark; substantially more judge infrastructure and prompt/image pairs |

## Implemented adaptation

The implicit task compares descriptions of Black- and White-presenting images matched by
occupation, recorded gender, and source index. The explicit task asks models to choose
which of two matched people has more income or education. Pixels do not establish those
attributes, so `Cannot determine` is the supported response. Every pair is shown in both
orders to expose position bias.

The adaptation deliberately avoids asking a model to identify race. Race labels are used
only after generation for aggregate analysis. The source's complete form-completion and
single-image multiple-choice tasks are outside this pilot.

## Interpretation

Description differences are called implicit here because the model is performing a
neutral assistance task rather than being asked to endorse a stereotype. Unsupported
income/education choices are called explicit because the prompt directly asks for a
social inference from faces. These operational labels follow VisBias; they do not imply
that a model has human awareness, intent, or unconscious cognition.

Automated sentiment and lexicon hits are limited proxies. The comparison task is easier
to interpret behaviorally, but refusal or `Cannot determine` can reflect provider safety
policy rather than model weights alone. Results therefore identify model-route behavior,
not an intrinsic or universal bias score.
