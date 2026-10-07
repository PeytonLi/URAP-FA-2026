from __future__ import annotations

import argparse
import json
from pathlib import Path

from .utkface import (
    annotate_expressions,
    build_matched_pools,
    enumerate_adult_black_white_images,
    sha256_file,
    validate_frozen_manifests,
    write_frozen_utkface_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze disjoint matched UTKFace evaluation and future-training prompt pools."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/private/utkface/raw/UTKFace"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/private/utkface/source/UTKFace.tar.gz"),
    )
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=Path("data/private/utkface/expression_annotations.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/frozen/utkface"),
    )
    parser.add_argument("--model-id", default="trpakov/vit-face-expression")
    parser.add_argument(
        "--model-revision",
        default="ef0bc6fc34241b6587e7e009e7711357be28c024",
    )
    parser.add_argument("--source-revision", default="e80f21fb21fa3631380c3cb634b79be3ac2f4cba")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-per-intersection", type=int, default=100)
    parser.add_argument("--finetuning-per-intersection", type=int, default=100)
    parser.add_argument("--min-expression-confidence", type=float, default=0.60)
    parser.add_argument("--min-quality-score", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=31082026)
    parser.add_argument("--skip-annotation", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, malformed = enumerate_adult_black_white_images(args.image_dir)
    if args.skip_annotation:
        if not args.annotation_csv.exists():
            raise FileNotFoundError(args.annotation_csv)
        import csv

        with args.annotation_csv.open(newline="", encoding="utf-8") as handle:
            annotated = [dict(row) for row in csv.DictReader(handle)]
        numeric_fields = {
            "age": int,
            "gender_code": int,
            "race_code": int,
            "width": int,
            "height": int,
            "brightness": float,
            "contrast": float,
            "edge_energy": float,
            "quality_score": float,
            "fer_confidence": float,
        }
        for row in annotated:
            for field, converter in numeric_fields.items():
                row[field] = converter(row[field])
    else:
        annotated = annotate_expressions(
            rows,
            model_id=args.model_id,
            model_revision=args.model_revision,
            batch_size=args.batch_size,
            output_csv=args.annotation_csv,
        )
    selected, report = build_matched_pools(
        annotated,
        evaluation_per_intersection=args.evaluation_per_intersection,
        finetuning_per_intersection=args.finetuning_per_intersection,
        min_expression_confidence=args.min_expression_confidence,
        min_quality_score=args.min_quality_score,
        seed=args.seed,
    )
    report["malformed_source_filenames"] = malformed
    manifest = write_frozen_utkface_artifacts(
        selected,
        report,
        args.output_dir,
        source_archive_sha256=sha256_file(args.archive),
        source_revision=args.source_revision,
        expression_model_id=args.model_id,
        expression_model_revision=args.model_revision,
    )
    validation = validate_frozen_manifests(args.output_dir)
    print(json.dumps({"manifest": manifest, "validation": validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
