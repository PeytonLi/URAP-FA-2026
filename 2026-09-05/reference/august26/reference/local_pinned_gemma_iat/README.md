# Frozen Base Reference

These response CSVs are the immutable pipeline-generated Gemma 3 baseline
carried forward from July 27. They are copied byte-for-byte so the August 3
adapter is compared with the same base outputs, images, prompt, repetitions,
and generation seed rather than with a newly sampled baseline.

The archived July run records the comparison protocol as the Transformers
`image-text-to-text` pipeline, eager attention, generation seed `7192026`, ten
repetitions, temperature/top-p `1.0`, and sampled/fallback limits of 30/20 new
tokens. `evaluate_frozen_base_vs_adapter.py` enforces those values and verifies
the files below against `SHA256SUMS` before generating adapter responses.

The metadata files identify the 400-image FairFace evaluation set and the
12-image neutral Project Implicit safeguard. The image files are intentionally
not committed. Materialize them under the paths recorded in the metadata and
verify them against `frozen_input_manifest.json` before evaluation.

`profile_eval_metadata.csv` is a deterministic 100-image subset of the same
frozen FairFace panel. It contains 50 Black and 50 White images in 50
race-matched blocks, with exact matches on gender, age band, coarse and fine
visible-expression labels, and quality bin. Its construction receipt is
`profile_eval_panel_manifest.json`. It contains no generated profiles; base
and adapter profiles are generated only after the final checkpoint is locked.

Do not use any evaluation image for training, training-candidate generation,
training-pair judging, hyperparameter tuning, or checkpoint selection.
