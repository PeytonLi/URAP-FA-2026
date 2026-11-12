# September 29: Reconstruct the Curated Image Sample

This is the minimal sample-reproduction package, not a new bias benchmark or
training run. It recreates the same **1,000 UTKFace and 1,400 FairFace images**
selected on September 29, 2026. No face images, model weights, API keys, slide
tools, or historical model results are distributed here.

## What Is Reproduced

Each dataset has 100 images per publisher race/gender cell. The shared
age-band/expression quotas match across those cells **within each dataset**.
The research split is 60 train, 20 development, and 20 test images per cell.
These are image-disjoint allocations, not verified person-disjoint splits.

The sample uses ages 20-69, source demographic labels, automated duplicate
screening, and frozen expression estimates. It is not human-certified and
does not control every photographic characteristic or establish model fairness.
Read [methods and limitations](docs/METHODS.md) and the
[CSV column glossary](docs/COLUMN_GLOSSARY.md).

## Setup

Use Python 3.11. Run commands from this folder. No GPU or paid API is needed.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python scripts/verify_checksums.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
```

The small lock is limited to reconstruction, frozen selection, and tests. It
does not install Torch or Transformers. Binary wheel availability can differ
by platform; exported PNG checksums catch image-encoding differences.

## Get the Exact Image Sample

Read the source terms below before downloading. This fetches the pinned source
archives, extracts only the selected images into the output sample, and checks
both original-image and exported-PNG SHA-256 hashes:

```bash
.venv/bin/python scripts/rebuild_images.py --download
.venv/bin/python src/curation.py validate
```

Images are written to `private/dataset/images/{utkface,fairface}/` and remain
Git-ignored. `data/selected_images.csv` is the manifest: join on `image_id`
or resolve its `image_path` relative to `private/dataset/`.
`private/dataset/rebuild_verification.json` records the reconstruction check.
Downloads are larger than the final sample (FairFace train Parquet alone is
about 500 MB). Access depends on continued source availability and source
permissions; the script stops on missing or changed image bytes.

Existing source copies can be used instead:

```bash
.venv/bin/python scripts/rebuild_images.py \
  --utk-dir /path/to/UTKFace \
  --fairface-parquet-dir /path/to/fairface/0.25
```

`--fairface-arrow-dir` also accepts the pinned original Arrow train shards.
Do not substitute a different crop, dataset revision, or reordered source.

## Reproduce the Selection Without Downloading Images

```bash
.venv/bin/python src/curation.py select --metadata-only
.venv/bin/python src/curation.py validate --skip-image-files
.venv/bin/python scripts/verify_checksums.py
```

The frozen candidate audit and expression annotations recreate the same
manifest, quotas, and splits. `tests/test_reproduction.py` also checks the
recreated manifest against the committed version in an isolated directory.
Metadata-only validation does not claim that image files were checked.
Generated selection decisions, quotas, and validation reports are Git-ignored.

The original `prepare` and `annotate` implementations remain in `curation.py`
for method transparency, but depend on the original research workspace and
additional inference packages. They are **not needed or supported entry points
for this compact package**. The standalone entry points are frozen `select`,
`validate`, and `rebuild_images.py`. Classifier outputs are intentionally frozen
so backend-sensitive expression predictions do not change sample membership.

## Main Files

| File | Purpose |
| --- | --- |
| [src/curation.py](src/curation.py) | Screening method, deterministic matched selection, splits, and validation |
| [scripts/rebuild_images.py](scripts/rebuild_images.py) | Reconstruct and byte-verify selected images from pinned sources |
| [data/selected_images.csv](data/selected_images.csv) | Exact final membership, labels, splits, source references, and hashes |
| [data/candidate_audit.csv](data/candidate_audit.csv) | Frozen candidate pool and technical/duplicate decisions |
| [data/expression_annotations.csv](data/expression_annotations.csv) | Frozen classifier estimates and provenance |
| [config/curation.json](config/curation.json) | Seed, quotas, thresholds, and exact dataset/model revisions |
| [data/frozen/](data/frozen/) | Source provenance and historical image fingerprints used in validation |
| [tests/](tests/) | Algorithm and exact-selection reproduction checks |

Historical references in `data/frozen/historical_sources.json` are provenance
only; collaborators do not need those old folders. The glossary retains
definitions for the wider curation project, including analysis files not
included in this deliberately minimal publication. Plots and Slides are out
of scope for this package.

## Publication Checks

Before publication, a fresh Python 3.11 environment installed this lock without
dependency conflicts; all 13 tests and Ruff lint passed. Frozen selection
reproduced the exact committed manifest. Rebuilding from existing UTKFace
JPEGs and pinned FairFace Arrow shards verified all 2,400 source/export image
hashes, followed by full image/balance validation. Both pinned source revisions
were reachable without authentication. The complete network-download path was
not rerun for this compact publication. The source and selection algorithms,
configuration, and frozen CSVs are unchanged from the September 29 curation.

## Dataset Terms and Citations

**UTKFace:** [official source and terms](https://susanqq.github.io/UTKFace/)
specify non-commercial research use and original-owner copyright. This package
does not grant redistribution rights. Download and use images under those terms;
do not treat access to a mirror as permission for unrestricted redistribution.

Zhang, Z., Song, Y., and Qi, H. (2017). *Age Progression/Regression by Conditional
Adversarial Autoencoder*. Proceedings of the IEEE Conference on Computer Vision
and Pattern Recognition.

**FairFace:** the [official dataset repository](https://github.com/joojs/fairface)
lists CC BY 4.0. Retain attribution, source/license references, and modification
notes; third-party rights and privacy considerations still apply.

Karkkainen, K., and Joo, J. (2021). *FairFace: Face Attribute Dataset for Balanced
Race, Gender, and Age for Bias Measurement and Mitigation*. Proceedings of the
IEEE/CVF Winter Conference on Applications of Computer Vision, 1548-1558.

Our image modifications are EXIF orientation correction and RGB PNG export
without resizing. Sampling, duplicate exclusions, automated expression estimates,
and new research splits are described in the methods. Keep face images in
authorized research storage rather than committing them to this repository.
