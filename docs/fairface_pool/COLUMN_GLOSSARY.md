# CSV Column Glossary

All demographic values below are publisher annotations, not new inferences or self-identification. Empty cells mean unavailable or not applicable, never zero. Boolean CSV values are `True`/`False`. Fields reused across files retain the definitions below. Hashes are lowercase hexadecimal. CSVs use UTF-8 and a header row.

## File Units

| File | One row represents |
| --- | --- |
| `data/source_inventory.csv` | One source image's metadata, before candidate sampling |
| `data/candidate_audit.csv` | One sampled candidate and its screening outcome |
| `data/expression_annotations.csv` | One eligible candidate's frozen classifier result |
| `data/selected_images.csv` | One final image; same schema as private `manifest.csv` |
| `data/selection_decisions.csv` | One eligible candidate's final inclusion decision |
| `data/common_support_quotas.csv` | One dataset/age/expression stratum shared across demographic cells |
| `data/frozen/excluded_fairface_indices.csv` | One conservatively excluded source index |
| `data/frozen/excluded_utkface_filenames.csv` | One known historical source filename |
| `data/frozen/historical_image_fingerprints.csv` | One available historical image fingerprint; entries can repeat |
| `analysis/*.csv` | One aggregate group or comparison, identified by the grouping columns |

## Source and Selection Fields

| Column | Meaning |
| --- | --- |
| `dataset` | `utkface` or `fairface`; categories are not mapped between sources |
| `source_id` | Original UTK filename or FairFace `train:NNNNNN` row identifier; historical files retain their archived IDs |
| `source_index` | Zero-based FairFace source-train row index; blank for UTKFace |
| `source_filename` | UTKFace source filename; blank for FairFace embedded image bytes |
| `source_split` | Publisher split: FairFace `train`; UTKFace `unspecified` |
| `source_revision` | Exact Hugging Face dataset repository commit |
| `source_race` | Source race category; UTKFace `Other` is not equated with any FairFace category |
| `source_gender` | Publisher binary `Female`/`Male` label; no nonbinary coverage or identity verification |
| `source_age` | UTKFace age in years or FairFace source age-band text |
| `age_band` | Derived decade for UTKFace or FairFace native age bin; only 20-69 admitted |
| `inventory_status` | First applicable metadata decision: `eligible`, `age_out_of_scope`, `historical_exposure`, `invalid_source_metadata` |
| `image_id` | Stable release-local image identifier, not a person identifier |
| `image_path` | Relative exported-PNG path: project-relative in candidate audit; relative to private dataset root in selected manifest |
| `screening_status` | `eligible`, `unreadable`, `undersized`, `exact_duplicate`, `near_duplicate`, `historical_exact_duplicate`, or `historical_near_duplicate` |
| `duplicate_of` | Retained current-candidate identifier responsible for duplicate exclusion. Blank for historical matches: their corresponding fingerprints can be looked up in `data/frozen/historical_image_fingerprints.csv`, rather than assuming a unique source ID. |
| `automated_review` | Automated screening marker; not human verification |
| `human_reviewed` | Whether independent human review occurred; always False in this release |
| `review_status` | Final review scope, `automated_only` |
| `selection_status` | `selected` or `eligible_not_selected`; excludes candidates already failing technical screening |
| `reason` | Selection decision (`common_support_matched_quota` / `outside_frozen_quota_or_rank`) or screening-exclusion category in aggregate tables |
| `split` | Newly assigned research `train`, `development`, or `test`, distinct from publisher split |
| `match_set_id` | One image per race/gender cell within an age/expression stratum, assigned together to a split; not the same person or a counterfactual pair |
| `common_capacity_per_cell` | Minimum available candidates across race/gender cells for that age/expression stratum |
| `selected_per_cell` | Number drawn from that stratum for every race/gender cell |

## Image Integrity and Descriptive Metrics

These are whole-image diagnostics, not validated face quality or skin-tone measures. Only unreadability and minimum side length determine technical rejection; brightness is not a rejection criterion.

| Column | Meaning |
| --- | --- |
| `source_image_sha256` | SHA-256 of original compressed source image bytes |
| `image_sha256` | SHA-256 of exported RGB PNG bytes |
| `pixel_sha256` | SHA-256 of width/height/color-mode prefix plus decoded RGB bytes; detects re-encoded identical pixels |
| `perceptual_hash` | 64-bit DCT perceptual hash, calculated as in `image_metrics` |
| `difference_hash` | 64-bit left/right adjacent brightness comparison hash |
| `width` | Exported image width in pixels, no resizing |
| `height` | Exported image height in pixels, no resizing |
| `brightness` | Mean 8-bit grayscale value divided by 255, range 0-1 |
| `contrast` | Population standard deviation of grayscale values divided by 255 |
| `edge_energy` | Sum of mean absolute adjacent grayscale differences along both axes, range 0-2; a detail proxy only |
| `clipped_dark_fraction` | Fraction of grayscale pixels <=5/255 |
| `clipped_light_fraction` | Fraction of grayscale pixels >=250/255 |

## Automated Expression Fields

| Column | Meaning |
| --- | --- |
| `expression_label` | Highest-probability fine label from the pinned FER model |
| `expression_confidence` | Softmax probability of the top label, 0-1; not calibrated accuracy |
| `expression_class` | `positive` for happy; `negative` for sad/angry/fear/disgust; `neutral` for neutral; `uncertain` for surprise or confidence <0.6 |
| `expression_model_id` | Hugging Face classifier identifier |
| `expression_model_revision` | Exact classifier repository commit |
| `expression_provenance` | `historical_same_revision_cache` or `local_new_inference`; cached results were not newly human-reviewed |

## Aggregate Analysis Fields

| Column | Meaning |
| --- | --- |
| `stage` | Named sampling stage or comparison population; denominators vary by stage and dataset |
| `images` | Integer image count for the row's dataset/group/stage |
| `percent` | 100 times count in the category divided by all images in the specified dataset/stage |
| `metric` | Underlying diagnostic being summarized: brightness, contrast, edge_energy, or expression_confidence |
| `n` | Number of selected images summarized |
| `mean` | Arithmetic mean of the diagnostic |
| `std` | Sample standard deviation, denominator n-1 |
| `median` | 50th percentile |
| `q25` | 25th percentile; not a confidence bound |
| `q75` | 75th percentile; not a confidence bound |
| `group_a` | First publisher race/gender cell in a descriptive pairwise comparison |
| `group_b` | Second publisher race/gender cell in the comparison |
| `standardized_mean_difference` | (mean A - mean B) / sqrt((sample variance A + sample variance B)/2); blank if denominator is zero; descriptive, not a significance test or model-bias measure |

`data/validation.json` records image-byte and balance checks. `data/selection_report.json` records realized quotas and split sizes. `analysis/summary.json` records totals and limitations. No p/q values, language-model responses, preference labels, or DPO training targets are produced by this dataset-only task.
