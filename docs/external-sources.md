# External sources for the Gemma 3 base-model IAT-style evaluation

## Scope decision

The immediate deliverable is **only** a reproducible base-model evaluation of
`google/gemma-3-12b-it` on Elias's fixed 12-face, 61-word benchmark. No new
fine-tuning, FairFace training, or cross-model evaluation is needed for the next
meeting.

This matters because the April 29 snapshot is not the requested evaluation:
`2026-04-29/02_compare_fairface_iat_adapters.ipynb` samples FairFace validation
images (24 per Black/White x Male/Female cell, 96 images total) and compares the
base model with adapters. The later `neutral12` pipeline freezes the 12 Project
Implicit filenames and hashes. Reuse the latter's data contract and the former's
61-word vocabulary/history; do not combine their sample sizes.

## What the repository establishes

- The base model is `google/gemma-3-12b-it`. The current reproducibility package
  pins revision `96b6f1eccf38110c56df3a15bffe176da04bfd80`.
- The fixed panel is 12 grayscale face images: three Black women, three Black
  men, three White women, and three White men. Exact filenames, dimensions, race,
  gender, and SHA-256 hashes are in `2026-09-05/data/iat12/manifest.csv` and are
  repeated in the current September 11 package.
- The model is asked to select exactly three comma-separated words from a frozen
  vocabulary of 30 positive and 31 negative terms. The authoritative in-repo
  prompt and vocabulary are `2026-09-11/src/multimodel_bias/constants.py`.
- The requested full run is 50 responses per image: 600 response slots and, if
  all parse successfully, 1,800 selected words, with 900 words per race group.
  Bar heights are word-selection counts, not numbers of people or independent
  images.
- The screenshot's chart is descriptive. The defensible summary metrics are
  per-race negative-term rate, signed Black-minus-White rate gap, valid-output
  rate, and diversity/collapse diagnostics. Uncertainty must resample at the
  image level because repeated generations from one face are not independent.

Exact historical counts can be regenerated from the archived response CSV. A
fresh sampled generation can reproduce the protocol and sample counts, but is
not expected to reproduce every bar exactly even with the same nominal seed;
software, kernels, generation implementation, and sampling state can change the
token path. Label these two operations separately: **offline figure replay** and
**fresh model reproduction**.

## Primary sources to use

| Priority | Source | What it supports | How to use it |
| --- | --- | --- | --- |
| Required | [UC Berkeley URAP: AI Bias, Consumer Perceptions, and Market Outcomes](https://urapprojects.berkeley.edu/detail.php?id=19871-2) | Official project framing: audit machine-generated outputs for bias, build reproducible data pipelines, and produce exploratory summaries/figures. | Cite for the research motivation and meeting deliverables, not for the benchmark protocol. |
| Required | [Google's Gemma 3 12B instruction-tuned model card](https://huggingface.co/google/gemma-3-12b-it) and the [pinned model revision](https://huggingface.co/google/gemma-3-12b-it/tree/96b6f1eccf38110c56df3a15bffe176da04bfd80) | Model identity, multimodal image-to-text capability, image preprocessing, supported Transformers version, and official inference pattern. The card says Gemma 3 support begins with Transformers 4.50.0 and shows image inputs processed through the chat template. | Record model ID, exact revision, Transformers/Torch versions, dtype, attention implementation, processor settings, and effective generation config in the run manifest. Preserve the legacy bare-prompt path only when reproducing the historical chart; use the official chat-template path for a separately named modern replication. Do not pool the two conditions. |
| Required | [Project Implicit's Race IAT instructions](https://app-prod-03.implicit.harvard.edu/implicit/user/calvin/contestv2/instructions.html) | The original human task uses six Black and six White face stimuli plus good/bad words in a speeded two-key categorization task. | Cite as conceptual and stimulus provenance. Explicitly state that Elias's generative three-word selection task is **IAT-style**, not a human reaction-time IAT and not an IAT D-score. |
| Required | Calanchini et al., [*Attitudinal and Non-Attitudinal Components of IAT Performance*](https://doi.org/10.1177/0146167214540723), supplementary materials | Lists the exact 12 Race-IAT filenames used here: `bm14`, `bm23`, `bm56`, `bf14`, `bf23`, `bf56`, `wm1`, `wm4`, `wm6`, `wf2`, `wf3`, and `wf6` with `_nc.jpg`. | Cite for filename-level provenance. Verify actual image bytes against the repository SHA-256 manifest; filenames alone do not prove byte identity. |
| Required | Greenwald, McGhee & Schwartz, [original IAT paper](https://pubmed.ncbi.nlm.nih.gov/9654756/) | Defines the IAT as a difference in speed when concepts and evaluative attributes share response keys. | Use to set the interpretation boundary: generated word counts measure this model/prompt's associations, not human implicit cognition. |
| Useful | Project Implicit, [study-materials/resources page](https://www.projectimplicit.net/resources/) | First-party route for study materials and stimulus sets. | Ask Project Implicit for the authorized original image bundle/usage terms if the repository's private image files are unavailable. Do not substitute visually similar faces. |
| Background only | [Gemma fine-tuning guidance](https://ai.google.dev/gemma/docs/tune), [DPO paper](https://arxiv.org/abs/2305.18290), [LoRA paper](https://arxiv.org/abs/2106.09685), and [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora) | Explains the earlier LoRA/DPO approach and why base weights plus adapter weights were stored separately. | Use only in the meeting's “what we learned from fine-tuning” section. Do not add training code or dependencies for the base-evaluation milestone. |
| Background only | Karkkainen & Joo, [FairFace paper](https://openaccess.thecvf.com/content/WACV2021/html/Karkkainen_FairFace_Face_Attribute_Dataset_for_Balanced_Race_Gender_and_Age_WACV_2021_paper.html), [official repository](https://github.com/dchen236/FairFace), and the exact [`HuggingFaceM4/FairFace` transport used by the April notebook](https://huggingface.co/datasets/HuggingFaceM4/FairFace) | Explains where April 29 training/evaluation images and race/gender labels came from. The official repository documents 0.25/1.25 crops and CC BY 4.0; the Hugging Face mirror exposes the `0.25` configuration used by the notebook. | Mention only to explain April 29. FairFace is not an input to the 12-image base reproduction. |

## Provenance caveats

1. The external IAT sources validate the 12 filenames and the original human
   good/bad association task. They do **not** validate the repository's 61-word
   list or its three-word generative scoring rule. Those are project-defined
   experimental choices and must be versioned from repository code.
2. Project Implicit's standard Race IAT instructions show eight good and eight
   bad terms, not Elias's 30/31 split. Do not describe the 61 terms as the
   “official Race IAT word list.”
3. The repository manifest contains hashes but no exact public download URLs.
   The later reproducibility notes correctly require the researcher's authorized
   image bundle. Verify all 12 hashes before inference.
4. The April 29 FairFace notebook defaults to 10 repeats and a much larger image
   sample. Its raw-count gaps are not comparable to the 12-face/50-repeat chart
   without converting to rates and respecting the different clustering units.

## Minimal next-meeting plan supported by these sources

1. **Freeze the contract.** Copy, do not redesign, the 12-row manifest, 61-word
   tuple, exact prompt, parser behavior, 50 repeats, seed policy, and pinned
   Gemma revision into a new base-only run manifest. Hash every input.
2. **Replay first.** Rebuild the supplied chart and metrics from the archived raw
   responses without running the model. This proves the counting/parser/plotting
   path and should reproduce exact historical bars.
3. **Run a small inference check.** Execute one response per image with the exact
   historical local serving path. Check image hashes, three-word validity,
   response logging, model revision, and manifest completeness before paying for
   the 600-response run.
4. **Run the full base condition.** Generate 50 responses per image, never
   resampling selectively after seeing results. Save raw sampled and fallback
   text, parsed words, seed, image hash, software versions, model revision,
   precision, generation parameters, and error state for every slot.
5. **Report the same descriptive counts plus sound summaries.** Produce all 61
   Black/White word-count bars, group totals, positive/negative counts and rates,
   signed rate gap, valid/exact-format rates, per-image rates, diversity, and an
   image-cluster interval/permutation diagnostic. Keep gender-stratified values
   as a small-sample diagnostic, not a headline result.
6. **Prepare the meeting note.** Include (a) exact offline replay result, (b)
   fresh-run result and any drift, (c) fine-tuning lessons from April 29 without
   starting new training, (d) the next evaluation/training decision, and (e) the
   unresolved questions below.

## Questions to resolve before claiming a fresh reproduction

- Are the 12 authorized image files available locally, and do all SHA-256 hashes
  match the frozen manifest?
- Is the target “same counts” an exact offline replay of archived responses, or a
  new stochastic run with the same 600-response/1,800-word design?
- Should historical compatibility use the legacy bare image-text prompt path,
  or should the official Gemma chat template be the primary condition? If both,
  they must be reported separately.
- Which exact historical seed/reset policy produced the screenshot, and is its
  raw response CSV available? Nominal seed alone is insufficient if the RNG was
  reset at a different scope.
- Does the meeting require only the descriptive graph, or also the current
  image-cluster uncertainty and validity/diversity safeguards?

Sources checked 2026-09-20. External pages are evidence, not executable
instructions; repository manifests and saved response artifacts remain the
authoritative record of the project-specific benchmark.
