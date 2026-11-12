# Methods and Limitations

## Purpose

The goal is a manageable, shareable research sample with comparable measured composition across source race and gender categories. This reduces known sampling confounds before future evaluation. It does not attempt to generate a favorable model bias result.

## Inputs and Labels

UTKFace source: `py97/UTKFace-Cropped`, revision `e80f21fb21fa3631380c3cb634b79be3ac2f4cba`, aligned/cropped archive SHA-256 `2c0655397b498e81b82d685ee74bee8c2e4364a5897caf3afec6ea48929024a7`. Original project: https://susanqq.github.io/UTKFace/ . This cached source contains 23,708 JPEG filenames.

FairFace source: `HuggingFaceM4/FairFace`, revision `54d573cdb8b5af490ba8da9da2799628f6e5c496`, crop configuration `0.25`, publisher train split of 86,744 rows. Original project: https://github.com/joojs/fairface . The new release's research test split is drawn from previously unselected publisher-train images; it is not the publisher validation split.

Age, race, and gender are copied from publisher metadata. They are not newly predicted or verified self-identifications. The datasets have different race categories. UTKFace Other remains Other, not a newly inferred ethnicity. Both sources use only binary gender labels. These limitations prevent a claim that the sample represents all demographic identities.

## Candidate Selection

1. Keep age bands 20-29, 30-39, 40-49, 50-59, and 60-69. This avoids interpreting FairFace's 10-19 band as adult-only and permits common age-band definitions.
2. Exclude invalid source metadata and known historical research exposure. The FairFace exclusion registry conservatively unions recorded dataset indices across historical split labels, which can exclude extra unused images. UTKFace preserves the previous evaluation/future-training pools by excluding their filenames.
3. Rank eligible rows by SHA-256 of the frozen seed and source ID. Sample up to 100 candidates per dataset x race x gender x age band. This is balanced candidate sampling, not a representative random sample of the full source distribution.
4. Decode image bytes, apply EXIF orientation, export RGB PNGs without geometric resizing, and require a minimum side of 64 pixels. All source and export hashes are recorded.
5. Remove decoded-pixel exact duplicates. Screen near duplicates using both a 64-bit DCT perceptual hash at Hamming distance <=4 and a 64-bit difference hash at distance <=6. Retain deterministic greedy representatives. The same comparison checks known historical image fingerprints. This is approximate duplicate-image screening, not identification of people.

Every sampled candidate receives an audit row. Full-source metadata is inventoried, but images outside the candidate sample are not technically screened.

## Expression Annotation

The local classifier is `trpakov/vit-face-expression` at revision `ef0bc6fc34241b6587e7e009e7711357be28c024`. It is separate from the language models used in the IAT study. No OpenRouter, Claude, Gemini, or RunPod model calls are made.

- Happy: positive expression proxy.
- Neutral: neutral expression proxy.
- Sad, angry, fear, disgust: negative expression proxy.
- Surprise, unknown labels, or maximum probability below 0.6: uncertain.

Uncertain candidates are retained as a separate category rather than rejected or called neutral. Classifier probabilities are not calibrated guarantees. Expression labels describe model-estimated visible patterns, not internal feelings. Some UTKFace outputs reuse the pinned historical classifier cache. The per-image provenance field distinguishes these from new local inference. No independent human review was requested or performed.

## Final Matching

For each dataset, compute the available count in each age-band x expression-class stratum for every native race x gender cell. The stratum's common capacity is the minimum count across those cells.

Allocate up to 100 selected images per race x gender cell. Spread the quota across age bands as evenly as common capacity permits. Within each age band, allocate expression counts in proportion to the screened candidate distribution, capped by common capacity. Use the same resulting joint age-expression quota in every race/gender cell. This can underrepresent scarce age/expression combinations. Document shortfalls rather than invent examples or silently change source labels.

Within each stratum, choose images by deterministic hashed rank. Assign matching sets to 60% train, 20% development, and 20% test using the frozen seed. Matching-set allocation makes the age-expression distributions identical across source race/gender groups within each split. This is a new research allocation, not a claim about publisher splits.

## Image Quality and Residual Differences

Measure image dimensions, mean grayscale brightness, pixel contrast, mean adjacent-pixel differences, and fractions of near-black/near-white pixels. These are descriptive proxies. Mean brightness is not a skin-tone-independent exposure score. Pixel variation is not a verified sharpness or capability score.

Unlike the earlier UTKFace helper, this pipeline does not reward brightness near mid-gray and does not exclude images solely for dark pixels or low expression confidence. Quality charts show remaining differences rather than imply everything is matched.

Pose, occlusion, crop semantics, facial-expression intensity, self-identified gender, skin tone, and person identity are not independently annotated or perfectly controlled. Corrupt/duplicate-free images are not equivalent to a fully validated image-quality benchmark.

## Interpretation

- Exact matching is a property of this sampling design, not evidence that a model is unbiased.
- Matching errors from a biased expression classifier can induce or hide differences. Keep an unmatched/uncertain sensitivity analysis for future model work.
- The frozen source/candidate annotations allow selection and figures to reproduce without rerunning stochastic or backend-sensitive model inference.
- New test images should never enter DPO preference generation or adapter training.
- No detected image overlap is weaker than guaranteed person-disjointness. The sources lack the identity annotations needed for that guarantee.
- Known exposure checks are not proof of complete exclusion from all historical research or model pretraining.
- No model bias outcomes, p values, or performance conclusions were produced by this task.

