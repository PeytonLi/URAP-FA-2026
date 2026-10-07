# Local image/response checks: all three intended models

These are processor-only checks, not fine-tuning results. They loaded no model
weights, created no gradients/adapters/checkpoints, uploaded no faces and launched
no paid resource. Separate, subsequently authorized Prime gateway inference has
now generated 1,920 drafts for review under a $100 total spending cap; it is not
training. [Draft-review status](TARGET_DRAFT_REVIEW.md). The SFT/ORPO study remains incomplete.

## Measured results

| Model | Train images checked | Fixture continuations checked | Working local format |
| --- | ---: | ---: | --- |
| Gemma 4 26B A4B IT | 96 | 192 | Explicit continuation of the official non-thinking generation prefix |
| Qwen3-VL 8B Instruct | 96 | 192 | Native complete-assistant template |
| Qwen3.5 9B | 96 | 192 | Native complete-assistant template, thinking explicitly disabled |

Each model used the same 96 frozen training images: eight complete published
matching sets across both datasets and all four expression strata, including
uncertain cases. They contain 56 FairFace and 40 UTKFace images. No development,
reserved test or original benchmark photos were used. This diagnostic does not
change the full 1,440-image training membership or generate labels for it.

Each photo was rendered with the unchanged 61-word prompt and both constant
grammar fixtures, `Happy, Joyful, Smiling` and `Pain, Grief, Sadness`. These are
**not image-specific labels or approved preferences**. They test coordinates
and grammar only; no visual judgment is inferred from them.

All checks require exact generation-prefix token equality, unchanged pixels,
preserved native image-coordinate/modality fields and ignored prompt/image
labels. Qwen additionally requires the decoded supervised text to be exactly
the fixture response plus native `<|im_end|>` and newline, rejecting extra
thinking/control text. Native `image_grid_thw`, `pixel_values` and
`mm_token_type_ids` are retained, not reconstructed by guessing image spans.
For example, the first Qwen photo has grid `[1,16,16]`, pixel tensor shape
`[256,1536]` and 64 merged image placeholders. Those are measured outputs,
not a general rule to copy into Gemma or a provider-specific training API.

Gemma's stock complete-assistant template failed exact prefix equality on all
96 images because its empty thought-channel marker differs from generation.
That failure remains preserved; the separate explicit continuation passes
without changing image fields or guessing a backend score mapping.

## Pinned processors and actual receipts

| Repository | Processor/template revision | Receipt |
| --- | --- | --- |
| google/gemma-4-26B-A4B-it | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` | [Continuation pass](../results/curated_gemma4_continuation_check.json); [native-template failure](../results/curated_gemma4_vision_check.json) |
| Qwen/Qwen3-VL-8B-Instruct | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | [Qwen3-VL check](../results/curated_qwen3vl_vision_check.json) |
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | [Qwen3.5 check](../results/curated_qwen35_vision_check.json) |

Qwen receipts were generated October 4, 2026 at 03:29 UTC (October 3 locally).
Both resolve to `Qwen3VLProcessor`; that shared class does not establish
identical weights, architectures or serving settings. Qwen3-VL Instruct has no
thinking switch in its pinned template, so none is passed. Qwen3.5 receives
`enable_thinking=False`; its empty thinking block stays in the ignored prompt.

Qwen template SHA-256 values are respectively
`3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4`
and `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`.
The isolated runtime is Transformers 5.14.0 / PyTorch 2.13.0; Qwen also records
torchvision 0.28.0 / Pillow 11.3.0. Processor pins are not asserted to equal
Fireworks's immutable serving-weight revisions.

The authoritative dataset CSV remains unchanged, SHA-256
`7b08c31122f624e733ac279b87882c9138a05ea4c9dc7a16596105a5d4637d13`.
Every diagnostic photo was hash-verified. Prompt SHA-256:
`427f09c21a3fd30765904b95e8a80b864567028453dcf00d408ee0a9396a9280`.

## Remaining requirements

The current data audit still has 0 approved target records of 1,920 required
for train/development. All 1,920 proposals now exist and require explicit human
review/editing. Neither teacher proposals, classifier expression estimates nor
demographic metadata establish ground-truth targets. Profiles are excluded.

Local processing does not prove image-conditioned model gradients, LoRA
correctness, memory feasibility or a Fireworks/Prime training backend. The
Fireworks Gemma score-length mismatch is unresolved. Prime's cached-model
catalog is empty; a $100 total Prime limit is now recorded, but actual GPU
capacity/pricing and image-training support still require verification.
[Provider readiness](PRIME_TRAINING_READINESS.md).

After target approval and backend verification, train separate SFT and ORPO
adapters from each untouched base, select only on development, then evaluate
frozen checkpoints on the original 12-image / 50-repeat benchmark and produce
measured reports/figures. A new serving stack requires its matching untouched
base benchmark first, to separate provider changes from fine-tuning effects.

## Repeat the checks

```powershell
rtk proxy 2026-10-03/runtime/train-venv/Scripts/python.exe 2026-10-03/curated_vision_check.py --continuation
rtk proxy 2026-10-03/runtime/train-venv/Scripts/python.exe 2026-10-03/qwen_vision_check.py --model qwen3vl
rtk proxy 2026-10-03/runtime/train-venv/Scripts/python.exe 2026-10-03/qwen_vision_check.py --model qwen35
rtk python -m unittest discover -s 2026-10-03/tests -q
```

All 73 distinct study tests pass. Qwen tests cover pinned identity, retained
grid/pixels/masks, rejected missing or changed image fields, model-specific
thinking controls and rejected extra supervised control text. These rejection
tests use mocks; the receipts separately record actual processing of real photos.

Official references: [Qwen3-VL](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_vl),
[Qwen3.5](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5).
Context7 had no matching docs for this specific template question; official
pages, pinned templates and the installed implementation were inspected instead.
Documentation itself is not treated as evidence of successful training.
