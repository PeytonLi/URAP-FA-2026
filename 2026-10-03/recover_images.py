"""Recover only byte-identical historical training images from public FairFace."""
import hashlib
import io
import argparse

import datasets
from PIL import Image, __version__ as pillow_version

import study


def main():
    if study.dataset_config():
        raise RuntimeError("Historical image recovery is superseded. Use 2026-09-29/scripts/rebuild_images.py for the active pool.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("0.25", "1.25"), default="0.25")
    args = parser.parse_args()
    rows = study.read_csv(study.ROOT / "data/pairs.csv")
    wanted = {row["image_sha256"]: row for row in rows}
    asset_root = study.ROOT / "data/private/assets"
    wanted = {digest: row for digest, row in wanted.items()
              if not (asset_root / row["image_path"]).is_file()
              or study.sha(asset_root / row["image_path"]) != digest}
    info = study.request_json("https://huggingface.co/api/datasets/HuggingFaceM4/FairFace")
    revision = info["sha"]
    report = {"started_at": study.now(), "dataset": "HuggingFaceM4/FairFace", "config": args.config,
              "recovery_revision": revision, "historical_revision_known": False,
              "pillow_version": pillow_version, "datasets_version": datasets.__version__,
              "match_rule": "Exact SHA-256 of original RGB PNG encoding; no approximate substitutions",
              "initial_verified_images": len(rows) - len(wanted),
              "matches": [], "splits_scanned": [], "training_performed": False}
    report_path = study.ROOT / ("results/image_recovery.json" if args.config == "0.25"
                               else "results/image_recovery_1p25.json")
    for split in ("train", "validation"):
        stream = datasets.load_dataset(report["dataset"], report["config"], split=split,
                                       revision=revision, streaming=True, token=False)
        stream = stream.cast_column("image", datasets.Image(decode=False))
        scanned = candidates = 0
        for index, item in enumerate(stream):
            scanned += 1
            race = stream.features["race"].int2str(item["race"])
            if race not in ("Black", "White"):
                continue
            candidates += 1
            raw = item["image"]["bytes"]
            if raw is None:
                raise RuntimeError("Expected embedded FairFace image bytes")
            with Image.open(io.BytesIO(raw)) as image:
                encoded = io.BytesIO()
                image.convert("RGB").save(encoded, format="PNG")
            png = encoded.getvalue()
            digest = hashlib.sha256(png).hexdigest()
            row = wanted.get(digest)
            if row:
                gender = stream.features["gender"].int2str(item["gender"])
                age = stream.features["age"].int2str(item["age"])
                if (race, gender, age) != (row["fairface_race"], row["fairface_gender"], row["age_bucket"]):
                    raise RuntimeError("Exact image match has inconsistent historical FairFace metadata")
                target = asset_root / row["image_path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and study.sha(target) != digest:
                    raise RuntimeError("Refusing to overwrite a mismatched image")
                if not target.exists():
                    target.write_bytes(png)
                report["matches"].append({"pair_id": row["pair_id"], "image_sha256": digest,
                                           "source_split": split, "source_index": index,
                                           "image_path": row["image_path"]})
                del wanted[digest]
                study.save_json(report_path, report)
                print(f"Recovered exact image {len(report['matches'])}/{len(rows)}", flush=True)
            if candidates % 1000 == 0:
                print(f"FairFace {split}: scanned {scanned} rows, {len(wanted)} exact images still missing", flush=True)
            if not wanted:
                break
        report["splits_scanned"].append({"split": split, "rows": scanned, "candidate_images": candidates})
        study.save_json(report_path, report)
        if not wanted:
            break
    report["finished_at"] = study.now()
    report["missing_hashes"] = sorted(wanted)
    report["all_images_recovered"] = not wanted
    study.save_json(report_path, report)
    print(f"Exact image recovery: {len(rows) - len(wanted)}/{len(rows)}", flush=True)
    if not wanted:
        return study.prepare(asset_root)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
