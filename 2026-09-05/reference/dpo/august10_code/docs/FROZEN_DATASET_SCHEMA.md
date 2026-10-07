# Frozen Dataset Schema

`mixed_dpo_pairs_frozen.csv` has one DPO preference example per row. Two rows,
one Black and one White image, form a matched block through `match_block_id`.
`chosen` and `rejected` are the texts used by DPO; the remaining columns make
selection, matching, judging, and leakage checks auditable.

## Identity, image, and split

| Column | Meaning |
| --- | --- |
| `pair_id` | Unique row identifier. |
| `match_block_id` | Active two-row Black/White matched block used by the freeze. |
| `image_path` | Repository-relative materialized image path. |
| `image_sha256` | Image integrity and overlap key. |
| `split` | `train` or held-out `development`. |
| `task_type` | `iat_3word` or `grounded_profile`. |
| `prompt` | Exact multimodal user prompt supplied during DPO scoring. |
| `source_image_id` | Source dataset/image identifier retained for provenance. |

## FairFace and expression labels

| Column | Meaning |
| --- | --- |
| `fairface_race` | Audit stratum (`Black` or `White` in this pilot), never shown to the judge. |
| `fairface_gender` | FairFace gender label used for matching. |
| `age_bucket` | FairFace age bucket used for bounded rematching. |
| `expression_class` | Coarse `positive`, `neutral`, or `negative` expression class. |
| `expression_fine` | Fine expression label. |
| `expression_intensity` | Ordinal visible-expression intensity. |
| `expression_confidence` | Confidence in the row's expression label. |
| `expression_pair_equivalent` | Whether the original expression audit treated the pair as equivalent. |
| `expression_pair_min_confidence` | Lower confidence of the original paired images. |
| `expression_pair_max_cue_difference` | Largest allowed/observed expression-cue difference in the source pair. |
| `expression_pair_audit_version` | Version of the expression audit that produced these fields. |

## Source pairing and candidates

| Column | Meaning |
| --- | --- |
| `sampling_match_id` | Sampling pair identifier used for within-block checks. |
| `source_sampling_match_id` | Original sampling ID before profile rematching. |
| `source_match_block_id` | Original block ID before profile rematching. |
| `candidate_1_provenance` | Model/template source of candidate 1. |
| `candidate_2_provenance` | Model/template source of candidate 2. |
| `candidate_1_response_id` | Stable candidate-1 generation identifier. |
| `candidate_2_response_id` | Stable candidate-2 generation identifier. |
| `candidate_construction` | Method used to form the comparison. |
| `controlled_template_id` | Template identifier for controlled IAT rows; blank for profiles. |
| `chosen` | Preferred response optimized upward by DPO. |
| `rejected` | Dispreferred response optimized downward by DPO. |
| `chosen_response_provenance` | Source model/template of the selected chosen text. |
| `rejected_response_provenance` | Source model/template of the selected rejected text. |

## Claude audit

| Column | Meaning |
| --- | --- |
| `judge_model` | Claude model used for the audit. |
| `judge_image_consensus` | Consensus across image-present orderings. |
| `judge_text_consensus` | Consensus across text-only orderings. |
| `judge_order_stable` | Whether candidate reversal preserved the substantive choice. |
| `judge_image_confidence` | Judge confidence for the image-grounded comparison. |
| `judge_audit_source` | Paid/resumed audit cycle supplying the accepted judgment. |

## Mechanical validity and scoring

| Column | Meaning |
| --- | --- |
| `chosen_format_valid` | Chosen satisfies task format. |
| `rejected_format_valid` | Rejected satisfies task format. |
| `chosen_identity_leakage_count` | Identity-label mentions in chosen. |
| `rejected_identity_leakage_count` | Identity-label mentions in rejected. |
| `chosen_grounded_detail_count` | Visible, image-grounded details in chosen. |
| `rejected_grounded_detail_count` | Visible, image-grounded details in rejected. |
| `chosen_visible_expression_support` | Support for the visible expression in chosen. |
| `rejected_visible_expression_support` | Support for the visible expression in rejected. |
| `chosen_person_negative_count` | Negative descriptors applied to the person in chosen. |
| `rejected_person_negative_count` | Negative descriptors applied to the person in rejected. |
| `chosen_unwarranted_negative_count` | Unsupported negative claims in chosen. |
| `rejected_unwarranted_negative_count` | Unsupported negative claims in rejected. |
| `chosen_unsupported_trait_count` | Unsupported personality/trait claims in chosen. |
| `rejected_unsupported_trait_count` | Unsupported personality/trait claims in rejected. |
| `chosen_invented_biography_count` | Invented biographical claims in chosen. |
| `rejected_invented_biography_count` | Invented biographical claims in rejected. |
| `chosen_quality_score` | Judge quality score for chosen. |
| `rejected_quality_score` | Judge quality score for rejected. |
| `objective_type` | Direct pair objective, such as negative-count or unsupported-trait reduction. |
| `block_direct_harm_reduction` | Whether both rows in the active block directly reduce measured harm. |

## Rematching audit

| Column | Meaning |
| --- | --- |
| `matching_provenance` | Original IAT pair or deterministic profile-rematching method. |
| `match_age_bucket_difference` | Age-bucket distance across the active block. |
| `match_expression_intensity_difference` | Expression-intensity distance across the block. |
| `match_min_expression_confidence` | Lower expression confidence in the active block. |

The trainer revalidates the CSV, freeze manifest, image hashes, block structure,
race balance, task coverage, and forbidden evaluation overlap before loading
Gemma. These columns are audit evidence; DPO's numerical loss is computed from
`prompt`, image, `chosen`, and `rejected`, while the gate determines whether a
row is allowed to reach training.
