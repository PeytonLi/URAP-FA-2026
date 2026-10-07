"""Rebuild byte-verified selected images from pinned source archives, without old repos."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curation import read_csv, sha, write_json


def extract_safe(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for m in members:
            p = (dest / m.name).resolve()
            if not p.is_relative_to(dest.resolve()) or m.issym() or m.islnk() or m.isdev():
                raise ValueError("Unsafe archive member")
        tf.extractall(dest, members=members, filter="data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="Download public-source archives after accepting their research-use terms.")
    p.add_argument("--utk-dir", type=Path)
    p.add_argument("--fairface-parquet-dir", type=Path)
    p.add_argument("--fairface-arrow-dir", type=Path, help="Optional existing HF Arrow cache instead of Parquet.")
    p.add_argument("--which", choices=["selected", "candidates"], default="selected")
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    config = json.loads((ROOT / "config/curation.json").read_text())
    if args.download:
        from huggingface_hub import snapshot_download
        utk = Path(snapshot_download("py97/UTKFace-Cropped", repo_type="dataset", revision=config["utkface_revision"],
                                     allow_patterns=["*.tar.gz"], local_dir=ROOT / "private/sources/utkface"))
        archives = list(utk.rglob("*.tar.gz"))
        matching = [a for a in archives if sha(a) == config["utkface_archive_sha256"]]
        if len(matching) != 1:
            raise ValueError("Pinned UTKFace archive SHA-256 mismatch or missing")
        extract_safe(matching[0], utk / "extracted")
        roots = list((utk / "extracted").rglob("UTKFace"))
        args.utk_dir = roots[0] if len(roots) == 1 else utk / "extracted"
        ff = Path(snapshot_download("HuggingFaceM4/FairFace", repo_type="dataset", revision=config["fairface_revision"],
                                    allow_patterns=["0.25/train-*.parquet"], local_dir=ROOT / "private/sources/fairface"))
        args.fairface_parquet_dir = ff / "0.25"
    if not args.utk_dir or not (args.fairface_parquet_dir or args.fairface_arrow_dir):
        p.error("Provide --download or both UTKFace images and a FairFace source directory")
    if args.fairface_arrow_dir:
        import pyarrow.ipc as ipc
        tables = []
        for file in sorted(args.fairface_arrow_dir.glob("fair_face-train-*.arrow")):
            with pa.memory_map(str(file), "r") as f:
                tables.append(ipc.open_stream(f).read_all())
    else:
        tables = [pq.read_table(f) for f in sorted(args.fairface_parquet_dir.glob("train-*.parquet"))]
    if not tables:
        raise ValueError("No FairFace train source shards found")
    fair = pa.concat_tables(tables)
    metadata = ROOT / ("data/selected_images.csv" if args.which == "selected" else "data/candidate_audit.csv")
    rows = read_csv(metadata)
    rows = [r for r in rows if r.get("screening_status") == "eligible"]
    output = args.output_dir or ROOT / ("private/dataset" if args.which == "selected" else "private/rebuilt_candidates")
    verified = 0
    for r in rows:
        if r["dataset"] == "utkface":
            raw = (args.utk_dir / r["source_filename"]).read_bytes()
        else:
            raw = fair["image"][int(r["source_index"])].as_py()["bytes"]
        import hashlib
        if hashlib.sha256(raw).hexdigest() != r["source_image_sha256"]:
            raise ValueError(f"Source image drift: {r['image_id']}")
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            image = ImageOps.exif_transpose(im).convert("RGB")
        rel = Path(r["image_path"]) if args.which == "selected" else Path(r["dataset"]) / (r["image_id"] + ".png")
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, format="PNG", optimize=False)
        if sha(dest) != r["image_sha256"]:
            raise ValueError(f"Rebuilt PNG mismatch: {r['image_id']}; use locked Pillow version")
        verified += 1
    write_json(output / "rebuild_verification.json", {"verified_images": verified, "which": args.which,
                                                        "source_images_and_exported_png_hashes_match": True})
    print(f"Verified {verified} rebuilt {args.which} images")


if __name__ == "__main__":
    main()
