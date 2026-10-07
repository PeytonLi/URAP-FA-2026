# July 27 DPO Training CSV Data Dictionary

## Authoritative file

The July 27 run loaded:

`results/gemini-3.5-flash_pair_judge_final_balanced/gemini_judged_dpo_pairs_frozen.csv`

- SHA-256:
  `638caf56c926b3007b6f17fef7b042ac5927be06bed8e31fba0e676b19a0cc69`
- Logical CSV rows: 50
- Unique training images: 50
- Frozen columns: 118
- Pair repeat: 1
- Maximum gradient examples: 50

Quoted multiline responses make the file 353 physical text lines. It must be
read with a CSV parser, not counted with `wc -l`.

The workflow wrote two derived copies:

- `outputs/fresh_lora_gemini_judged_20260727/july19_dpo_pairs_reused.csv`
  is a pandas-normalized copy of the 50 loaded rows.
- `outputs/fresh_lora_gemini_judged_20260727/july19_dpo_training_pairs.csv`
  contains the same 50 example IDs in seeded shuffled order and adds
  `train_repeat_idx=0`.

The historical `july19_` output names are inherited filenames. They contain the
July 27 frozen data in this run.

## What directly trains the adapter

For each row, DPO directly builds two multimodal sequences from:

1. `image_path`
2. `prompt`
3. `chosen`
4. `rejected`

`example_id` keys the precomputed base-model reference scores. The remaining
columns control validation, pair selection, balancing, and provenance; their
numeric values are not supplied to the DPO loss as weights.

The loss is:

`-logsigmoid(beta * ((policy_chosen - policy_rejected) - (base_chosen - base_rejected)))`

Every retained row has equal example weight. Race, gender, expression,
candidate scores, Gemini confidence, and teacher confidence do not directly
scale its loss.

## Core pair and image columns

| Column | Meaning | Training role |
| --- | --- | --- |
| `example_id` | Unique preference-example identifier. | Required; deduplicates rows and keys reference scores. |
| `pair_source` | Pair-generation lineage, including whether the chosen response was model-observed or teacher-produced. | Required provenance; recorded with reference scores, not a loss weight. |
| `image_path` | Path to the image supplied to Gemma for both alternatives. | Direct DPO input. |
| `prompt` | Multimodal three-word IAT prompt supplied with the image. | Direct DPO input. |
| `chosen` | Preferred assistant response. | Direct DPO target. |
| `rejected` | Dispreferred assistant response. | Direct DPO comparison target. |
| `filename` | Stable source-image filename. | Required identifier and audit field. |
| `fairface_race` | FairFace race label, restricted to Black or White in this study. | Balancing/audit only; not inserted into the prompt. |
| `fairface_gender` | FairFace gender label. | Balancing/audit only; not inserted into the prompt. |
| `age` | Original FairFace age category. | Matching and audit only. |
| `age_band` | Collapsed age stratum such as `20_39`. | Matching and audit only. |
| `expression_class` | Coarse visible-expression label: negative, neutral, or positive. | Required; used to revalidate whether negative terms are expression-supported. |
| `expression_fine` | Fine label such as sad, fear, angry, disgust, neutral, or happy. | Required; selects the allowed expression-supported vocabulary during validation. |
| `expression_confidence` | Confidence assigned to the final expression annotation. | Required provenance; not a loss weight. |
| `expression_provenance` | Method that produced the final expression label. | Required provenance; not a loss weight. |
| `quality_score` | Precomputed scalar image-quality/matching score. | Matching/audit only. |
| `quality_bin` | Low, medium, or high bin derived from image quality. | Matching/audit only. |
| `match_pair_id` | Identifier for a candidate Black/White matched block. | Final-selection priority and audit only. |
| `match_stratum` | Matching stratum, generally gender, age band, and expression. | Matching/audit only. |

## Chosen-versus-rejected diagnostics

| Column | Meaning | Training role |
| --- | --- | --- |
| `chosen_word_tuple` | Lowercase normalized chosen words joined by commas. | Duplicate/concentration audit; recomputable from `chosen`. |
| `rejected_word_tuple` | Lowercase normalized rejected words joined by commas. | Audit; recomputable from `rejected`. |
| `chosen_score` | Pair-policy score for chosen text: expression support minus unsupported-negativity and identity penalties, plus any configured novelty bonus. | Pair selection only; not used in DPO loss. |
| `rejected_score` | Same deterministic score for rejected text; invalid text receives a very negative sentinel. | Pair selection only; not used in DPO loss. |
| `chosen_valid` | Whether chosen parses to exactly three allowed words. | Gate/audit; recomputed by the training loader. |
| `rejected_valid` | Whether rejected parses to exactly three allowed words. | Gate/audit; invalid rejected text is allowed only for a direct identity-leakage correction. |
| `chosen_format_compliant` | Whether chosen contains only one compliant response line. | Gate/audit; recomputed. |
| `rejected_format_compliant` | Whether rejected contains only one compliant response line. | Gate/audit. |
| `chosen_valence` | Chosen response valence: positive or negative for the three-word list. | Gate/audit; not a loss weight. |
| `rejected_valence` | Rejected response valence; may be `NaN` for invalid text. | Gate/audit; not a loss weight. |
| `chosen_identity_hits` | Count of identity-bearing terms in chosen raw text. | Pair gate; chosen cannot increase identity leakage. |
| `rejected_identity_hits` | Count of identity-bearing terms in rejected raw text. | Pair gate; a reduction can justify the preference. |
| `chosen_negative_word_count` | Number of chosen words in the fixed negative lexicon. | Pair gate and distribution audit; not a loss weight. |
| `rejected_negative_word_count` | Number of rejected words in the fixed negative lexicon. | Pair gate and distribution audit. |
| `chosen_positive_word_count` | Number of chosen words in the fixed positive lexicon. | Audit; complementary to the negative count for valid responses. |
| `rejected_positive_word_count` | Number of rejected words in the fixed positive lexicon. | Audit. |
| `chosen_unsupported_negative_count` | Chosen negative words not supported by the row's expression annotation. | Core pair-selection gate. |
| `rejected_unsupported_negative_count` | Rejected negative words not supported by the expression annotation. | Core pair-selection gate; chosen must directly improve this or identity leakage. |
| `chosen_expression_supported_count` | Chosen words supported by the fine expression vocabulary. | Pair gate; chosen cannot reduce deterministic expression support. |
| `rejected_expression_supported_count` | Rejected words supported by the fine expression vocabulary. | Pair gate. |
| `chosen_response_provenance` | Origin of chosen text, such as an observed base-Gemma trace or AI-teacher correction. | Provenance and 50% teacher cap; not a loss weight. |
| `rejected_response_provenance` | Origin of rejected text. | Provenance only. |
| `teacher_item_id` | Blinded teacher-review item ID when a teacher supplied the chosen side. | Provenance only; blank for non-teacher rows. |
| `teacher_rationale` | Teacher explanation for the correction. | Audit only; never shown to Gemma during DPO. |
| `teacher_confidence` | Teacher confidence in the correction. | Audit only; not a loss weight. |
| `human_reviewed` | Whether a human validated the pair. All July 27 rows are false. | Governance metadata only. |
| `pair_filter_reason` | Final deterministic loader result, normally `keep` or the allowed invalid-rejected identity exception. | Gate/audit. |

## Sparse upstream pair diagnostics

These fields were inherited from some candidate-generation branches and are
populated for only a subset of the 50 rows. They are not authoritative final
inclusion flags; every row in the frozen CSV was retained.

| Column | Meaning | Training role |
| --- | --- | --- |
| `score_margin` | Upstream chosen score minus rejected score. | Legacy audit only; current loader recomputes the margin. |
| `chosen_abs_valence` | Absolute chosen valence in the upstream diagnostic. | Legacy audit only. |
| `rejected_abs_valence` | Absolute rejected valence in the upstream diagnostic. | Legacy audit only. |
| `direct_improvement` | Upstream flag that chosen reduced unsupported negativity or identity leakage. | Legacy audit; current loader recomputes it. |
| `keep` | Upstream candidate keep decision. | Legacy field, not the final frozen-row decision. |
| `filter_reason` | Upstream candidate rejection/retention reason. | Legacy provenance only. |

## Source image and expression metadata

| Column | Meaning | Training role |
| --- | --- | --- |
| `dataset_index` | Original source-dataset row index. | Provenance only. |
| `service_test` | Source split/test flag inherited from the earlier FairFace sampling workflow. | Provenance only. |
| `split` | Source dataset split, such as train. | Provenance only. |
| `dataset_role` | Intended source role, such as train. | Provenance only. |
| `brightness` | Precomputed image brightness statistic. | Quality matching/audit only. |
| `contrast` | Precomputed image contrast statistic. | Quality matching/audit only. |
| `edge_energy` | Precomputed edge/detail statistic. | Quality matching/audit only. |
| `fer_label` | Fine expression predicted by the FER classifier. | Annotation provenance. |
| `fer_confidence` | FER confidence for its winning label. | Annotation provenance. |
| `fer_probabilities` | JSON map of FER probabilities for all expression labels. | Annotation audit only. |
| `gemma_expression_class` | Coarse expression class assigned by the Gemma annotation pass. | Annotation provenance. |
| `gemma_expression_fine` | Fine expression assigned by Gemma. | Annotation provenance. |
| `gemma_expression_confidence` | Gemma expression-annotation confidence. | Annotation provenance. |
| `gemma_expression_raw` | Compact raw Gemma annotation record. | Annotation provenance. |
| `expression_ambiguous` | Whether the automated expression decision was ambiguous. | Pool construction/audit. |
| `fer_expression_class` | FER fine label collapsed to a coarse class. | Annotation comparison/audit. |
| `annotation_agreement` | Whether FER, Gemma, or an AI override supported the final annotation. | Annotation audit. |
| `matched_for_training` | Whether the source image was selected into an earlier matched training pool. | Source-pool provenance only. |

## Upstream candidate-text diagnostics

These columns describe an upstream candidate response used while constructing
some teacher pairs. They are not additional DPO responses.

| Column | Meaning | Training role |
| --- | --- | --- |
| `raw` | Cleaned upstream candidate text. | Candidate audit only. |
| `words` | Parsed list of candidate words. | Candidate audit only. |
| `word_tuple` | Normalized candidate tuple. | Candidate audit only. |
| `valid` | Whether that upstream candidate parsed to three allowed words. | Candidate construction only. |
| `format_compliant` | Whether the upstream candidate obeyed the exact one-line format. | Candidate construction only. |
| `valence` | Upstream candidate valence. | Candidate audit only. |
| `positive_word_count` | Positive-list count for the upstream candidate. | Candidate audit only. |
| `negative_word_count` | Negative-list count for the upstream candidate. | Candidate audit only. |
| `identity_term_hits` | Identity-bearing term count in the upstream candidate. | Candidate ranking/construction. |
| `expression_supported_count` | Expression-supported word count for the upstream candidate. | Candidate ranking/construction. |
| `unsupported_negative_count` | Unsupported negative count for the upstream candidate. | Candidate ranking/construction. |
| `score` | Upstream candidate policy score. | Candidate ranking only. |
| `_source_index` | Internal source-response index used to trace the candidate. | Provenance only. |
| `_rejected_raw` | Original rejected candidate text before final pair assembly. | Provenance only. |
| `same_image_positive_word_pool` | Positive words observed from base Gemma on this same image. | Teacher construction and audit. |
| `same_image_zero_negative_tuples` | Zero-negative three-word tuples actually observed from base Gemma on the same image. | Pair validation exception for a larger nonnegative valence shift. |
| `teacher_pair_source` | Teacher-pair construction branch that produced the proposal. | Provenance only. |

## Image integrity and AI-teacher review

| Column | Meaning | Training role |
| --- | --- | --- |
| `source_image_sha256` | SHA-256 of the actual training image. | Critical preflight integrity and train/evaluation overlap check. |
| `blinded_image_path` | Path used in the blinded teacher packet. | Review provenance; not a DPO input path. |
| `blinded_image_sha256` | SHA-256 of the blinded packet image. | Review integrity. |
| `review_id` | AI-review record ID. | Provenance only. |
| `ai_visible_expression_class` | AI reviewer's coarse visible-expression judgment. | Teacher-pool selection/audit. |
| `ai_visible_expression_fine` | AI reviewer's fine visible-expression judgment. | Teacher-pool selection/audit. |
| `ai_review_confidence` | AI reviewer's confidence. | Audit only; not a loss weight. |
| `ai_visible_cues` | Short description of visible cues supporting the AI judgment. | Audit only. |
| `ai_include_as_neutral` | AI decision to include the image in the neutral pool. | Pool construction only. |
| `ai_reviewer_type` | Reviewer type, here AI multimodal. | Governance provenance. |
| `ai_reviewer_label` | Versioned reviewer label. | Governance provenance. |
| `ai_review_provenance` | Full AI-review method/version string. | Governance provenance. |

## Same-image response generation

| Column | Meaning | Training role |
| --- | --- | --- |
| `candidate_index` | Candidate number among repeated generations for that image. | Provenance only. |
| `candidate_seed` | Seed used to generate that candidate. | Reproducibility only. |
| `response_source` | Whether generation came from sampling or fallback decoding. | Provenance only. |
| `generator_valid_three` | Whether the generator produced three list words. | Candidate construction/audit. |
| `model_response_raw` | Complete raw base-Gemma response before cleaning. | Audit only; not shown as separate DPO input. |
| `candidate_score` | Policy score assigned to the generated candidate. | Candidate ranking only. |
| `teacher_used_same_image_model_word` | Whether the teacher target reused words observed from base Gemma on that image. | Teacher-grounding audit only. |

## Independent Gemini pair-judge columns

| Column | Meaning | Training role |
| --- | --- | --- |
| `gemini_pair_id` | Stable blinded pair ID used for Gemini calls. | Audit/deduplication only. |
| `gemini_selected_source_side` | Which original side Gemini preferred after order normalization. | Final pair-selection evidence. |
| `gemini_image_confidence` | Mean/combined confidence from the image-aware judgments. | Selection ordering and audit; not a DPO weight. |
| `gemini_text_consensus` | Result of text-only, order-reversed control judgments: chosen or tie. | Bias-control audit and eligibility evidence. |
| `gemini_selected_visible_support` | Aggregated visible-expression support score for selected text. | Pair eligibility evidence. |
| `gemini_rejected_visible_support` | Aggregated visible-expression support score for rejected text. | Pair eligibility evidence. |
| `gemini_selected_unwarranted_negative_terms` | Gemini-estimated unsupported negative terms on selected side. | Pair gate; selected cannot be worse. |
| `gemini_rejected_unwarranted_negative_terms` | Same estimate for rejected side. | Pair gate and direct-improvement evidence. |
| `gemini_selected_raw_allowed_negative_terms` | Raw negative-list count on selected side. | Audit and valence-preservation gate. |
| `gemini_rejected_raw_allowed_negative_terms` | Raw negative-list count on rejected side. | Audit and valence-preservation gate. |
| `gemini_judge_model` | Judge model identifier. | Provenance only. |
| `gemini_judge_date` | Date of the judge run. | Provenance only. |

## Effective-training-only column

| Column | Meaning | Training role |
| --- | --- | --- |
| `train_repeat_idx` | Zero-based pair repetition index added to `july19_dpo_training_pairs.csv`. It is zero for all July 27 rows because each pair was used once. | Audit only; does not enter the loss. |

## Training-critical information not stored in the CSV

The CSV is not a complete training specification. The following are external:

- image pixels referenced by `image_path`;
- pinned `google/gemma-3-12b-it` snapshot;
- base chosen/rejected log probabilities in
  `outputs/fresh_lora_gemini_judged_20260727/july19_dpo_reference_scores.pt`;
- LoRA rank 16, alpha 32, dropout 0.05, target modules, and vision-tower exclusion;
- DPO beta 0.05, learning rate `1e-5`, gradient accumulation 4, warmup 8,
  gradient clipping 1.0, seeds, token limits, and checkpoint schedule;
- software versions and attention implementation;
- loss history, adapter-effect probes, held-out metrics, bootstrap results, and
  acceptance decisions.

The launch specification is
`results/fresh_lora_launch_plan.json`. The executed run configuration and
outcomes are in
`outputs/fresh_lora_gemini_judged_20260727/july27_manifest.json`.
