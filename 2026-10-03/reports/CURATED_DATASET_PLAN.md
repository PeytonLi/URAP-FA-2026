# Approved September 29 dataset switch

The user approved this switch on October 3, 2026. The current experiment uses the September 29 image pool instead of the incomplete 54-image historical cohort or a training pilot on the 12 benchmark photos. This changes the training-data source, not the required SFT/ORPO comparisons or primary IAT evaluation.

## Frozen image membership

Source commit: `e4ac7a506f97e855f78ce79cc9d42b508193f76c`.

Manifest: `2026-09-29/data/selected_images.csv`.

SHA-256: `7b08c31122f624e733ac279b87882c9138a05ea4c9dc7a16596105a5d4637d13`.

| Dataset | Train | Development | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| FairFace | 840 | 280 | 280 | 1,400 |
| UTKFace | 600 | 200 | 200 | 1,000 |
| Total | 1,440 | 480 | 480 | 2,400 |

Preserve the published `split` and `match_set_id` assignments. Do not substitute `source_split` or pool training and test images. Age/expression composition is matched across the publisher race/gender cells within each dataset. Publisher race categories differ between datasets; no new demographic inference or remapping is made. Uncertain expressions stay in the pool and require review, not silent relabeling or exclusion.

The upstream manifest and all 19 package files passed their published checksums after import and after execution. All 2,400 images have now been reconstructed: every original-source hash and exported-PNG hash matches. The full upstream validator (without `--skip-image-files`) passes its physical-file, balance and duplicate checks. Its report still correctly states that person-disjointness and human validation are not established. The source package remains unchanged, and generated face images/source archives stay under its ignored `private/` folder.

Source revisions: UTKFace `py97/UTKFace-Cropped` at `e80f21fb21fa3631380c3cb634b79be3ac2f4cba`; FairFace `HuggingFaceM4/FairFace` at `54d573cdb8b5af490ba8da9da2799628f6e5c496`, crop `0.25`, publisher train. Read the upstream source terms and preserve attribution. This plan grants no redistribution rights. Python/Pillow and other executed versions must be recorded; matching output hashes is not a claim that the entire runtime matches the upstream environment.

Executed reconstruction/audit environment: Windows, Python 3.13.11, Pillow 11.3.0, pyarrow 24.0.0, NumPy 2.4.4, SciPy 1.17.1 and huggingface_hub 1.21.0. Pillow matches the upstream lock; other versions and Python do not all match. The reconstructed hashes, rather than a claimed environment match, establish byte identity of the images.

The study's 73 distinct local tests pass. The upstream suite has 12 passes and one byte-equality failure on Windows: `str(Path(...))` in upstream `select` writes backslashes in `image_path`, whereas the committed CSV uses forward slashes. A replay in a temporary copy confirms all 2,400 IDs, research splits, row order and other fields are identical; only those 2,400 path separators change. [Replay audit receipt](../results/curation_replay_audit.json). The authoritative CSV was not regenerated, normalized or edited to hide this failure.

## Three-word answers are still required

The new CSV has 38 metadata columns and **zero prompt/desired-response/preference columns**. Its classifier expression estimate is not a three-word answer or independent human ground truth. There are no approved targets for these new images yet. Neither the historical 54-image answers nor the 4,345 profile-preference rows are reassigned to them.

The user has now authorized draft generation for human review. Image-bearing Claude Haiku 4.5 through the Prime gateway produced 1,920 proposed pairs under a visible-expression policy; none is human-approved. [Review policy, provenance, costs and positive-prior concerns](TARGET_DRAFT_REVIEW.md). Human target approval must precede optimization. The reviewed target file will be a JSON list at `2026-10-03/data/private/curated_iat_targets.json`, with one record for each of the 1,920 train/development images. Each record needs:

- `image_id`, `image_sha256`, and the unchanged research `split`;
- `chosen` and `rejected`, each exactly three distinct canonical words separated by comma-space;
- `review_status: approved`, a nonempty `reviewer`, and `target_provenance`;
- if a `prompt` is recorded, exactly the existing 61-word IAT prompt.

These fields record approval and provenance; their presence does not by itself certify visual correctness. Record reviewer scope, teacher/model revision, image access, audit method and disagreements when producing labels. Do not use race, gender, invented personality or biography as model-input text. Giving all images positive words could reduce a measured gap while teaching a positive-word prior; the current neutral drafts display exactly that concern. The teacher policy and alias are documented, not treated as objective labels or an immutable teacher-weight revision.

No preference/target records may be generated for the 480 test images. The original 12 benchmark images likewise remain excluded. Development labels support checks and selection, not gradient updates. Label shortages, mismatches, duplicate records, altered splits or missing approval block export rather than activate the old cohort or a smaller sample.

## Training and evaluation

SFT: image + unchanged prompt + approved chosen three-word response. ORPO: the same image/prompt + approved chosen/rejected responses. No profile training. Only response tokens contribute to loss. Use the same frozen dataset/targets for Gemma 4 26B A4B IT, Qwen3-VL 8B Instruct and Qwen3.5 9B, with separate SFT and ORPO adapters from each untouched base, subject to actual image-training verification.

The primary benchmark remains the original 12 images x 50 repeats, using the existing generation/fallback, scoring and all-61-word chart contract. Existing base-model results remain applicable to that same primary benchmark. The 480 new held-out photos are reserved; an additional evaluation on them needs a separately defined protocol and must not select training targets or checkpoints.

The Fireworks Gemma 4 image-training mismatch is unresolved. The old diagnostic receipts remain valid historical evidence, not validation of this new cohort. The old paid runner refuses to launch while the curated pool is active. Do not truncate score vectors, use the rejected 258-position image-chunk variant, replace images with text-only training, or claim trained checkpoints until a supported image-conditioned route has actually passed.

Prime Intellect is an additional approved provider. Its isolated [Lab workspace](../prime-lab/README.md) passes 17 health checks. Setup itself performed no paid work; subsequent authorized Prime gateway inference sent train/development image bytes for drafts under a $100 total cap and $20 drafting sub-cap. This does not verify Gemma 4 or other model image-bearing SFT/ORPO support or substitute another training objective. Model/algorithm/modality validation and a launch-time price/capacity check remain required before GPU training.

[Current Prime and processor checks](PRIME_TRAINING_READINESS.md) preserve the live empty dedicated-model catalog and distinguish a failed standard Gemma 4 assistant template from a successful explicit non-thinking continuation check on 96 active training images. The two constant three-word fixtures are not targets. No weights were loaded and no remote training support or image-conditioned gradients have been established. A change of serving stack also requires a matching untouched-base evaluation, not attribution of cross-provider differences to training alone.

Both intended Qwen processors now pass native assistant-prefix, image-grid/pixel preservation and exact response-only continuation checks on those same 96 training photos. [Per-model local input evidence](LOCAL_MODEL_INPUT_CHECKS.md). This adds no approved targets and makes no claim about trained-model behavior or provider-specific backend compatibility.

## Local artifacts and readiness

`config/active_dataset.json` selects the pinned cohort. `study.py prepare`, `audit` and `export` now use it by default. `data/manifest.json` and `results/curated_data_audit.json` record actual image/target readiness. Exported payloads are private, separated by dataset and API surface, and include train/development only. Historical pairs, missing-file list, image-recovery records and `data/legacy_54_manifest.json` remain available for provenance; they are no longer the active training source.

Dataset selection, image reconstruction, approved targets, backend correctness and actual optimizer updates are separate milestones. No SFT/ORPO training result is claimed by this switch.
