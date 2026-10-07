# Neutral12 base-model Colab reproduction

Open `neutral12_base_colab.ipynb` in Google Colab and run the cells in order.
It is self-contained and does not require GitHub access or a repository clone.

## Runtime

Use an **H100** if available. An **A100 with at least 40 GB VRAM** should fit,
but differs from the historical hardware. T4 and L4 runtimes are intentionally
rejected because the pinned BF16 evaluation does not fit without changing the
experiment through quantization or offloading.

## Before running

1. Accept access to `google/gemma-3-12b-it` on Hugging Face.
2. Add a private Colab secret named `HF_TOKEN`, or use the login widget when the
   notebook asks.
3. Select all 12 JPG files from `C:/Users/lipey/Downloads/12_images/race/` when
   the upload cell opens. Their hashes are checked before inference.
4. Allow the notebook to mount Google Drive. Results are written to
   `MyDrive/URAP/neutral12_gemma_base_colab_v1/` so an interrupted run resumes.

## Outputs

- `responses.jsonl`: append-only raw response ledger.
- `run_manifest.json`: model, image, environment, and generation provenance.
- `word_counts.csv` and `word_counts.png`: the requested 61-word Black/White chart.
- `summary.csv` and `image_scores.csv`: evaluation metrics and image-level units.
- `MEETING_BRIEF.md`: the four requested next-meeting sections.
- `neutral12_base_results.zip`: downloadable result bundle.

The notebook performs no fine-tuning and never loads an adapter.
