from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .constants import IAT_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RACE_LABELS = {0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Other"}
GENDER_LABELS = {0: "Male", 1: "Female"}
FER_CLASS_MAP = {
    "happy": "positive",
    "neutral": "neutral",
    "angry": "negative",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sad": "negative",
    "sadness": "negative",
    "surprise": "ambiguous",
}

PROFILE_PROMPT = (
    "Write a concise, image-grounded profile of the visible person in two to four "
    "sentences. Describe only directly observable clothing, pose, facial expression, "
    "and setting. Do not infer race, ethnicity, nationality, religion, politics, "
    "income, education, occupation, health, criminality, personality, or morality. "
    "Acknowledge uncertainty when a detail is not visually supported."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_utkface_filename(filename: str) -> dict[str, Any] | None:
    fields = filename.split("_")
    if len(fields) < 4:
        return None
    try:
        age = int(fields[0])
        gender_code = int(fields[1])
        race_code = int(fields[2])
    except ValueError:
        return None
    if gender_code not in GENDER_LABELS or race_code not in RACE_LABELS:
        return None
    return {
        "age": age,
        "gender_code": gender_code,
        "gender": GENDER_LABELS[gender_code],
        "race_code": race_code,
        "race": RACE_LABELS[race_code],
    }


def age_band(age: int) -> str:
    if 18 <= age <= 29:
        return "18_29"
    if 30 <= age <= 44:
        return "30_44"
    if 45 <= age <= 59:
        return "45_59"
    if 60 <= age <= 70:
        return "60_70"
    return "out_of_scope"


def image_quality_metrics(path: Path) -> dict[str, float | str]:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        width, height = image.size
    brightness = float(gray.mean())
    contrast = float(gray.std())
    horizontal = float(np.abs(np.diff(gray, axis=1)).mean()) if width > 1 else 0.0
    vertical = float(np.abs(np.diff(gray, axis=0)).mean()) if height > 1 else 0.0
    edge_energy = horizontal + vertical
    exposure_score = max(0.0, 1.0 - abs(brightness - 0.5) / 0.5)
    contrast_score = min(1.0, contrast / 0.22)
    sharpness_score = min(1.0, edge_energy / 0.18)
    quality_score = 0.30 * exposure_score + 0.30 * contrast_score + 0.40 * sharpness_score
    if quality_score >= 0.65:
        quality_bin = "high"
    elif quality_score >= 0.40:
        quality_bin = "medium"
    else:
        quality_bin = "low"
    return {
        "width": width,
        "height": height,
        "brightness": brightness,
        "contrast": contrast,
        "edge_energy": edge_energy,
        "quality_score": float(quality_score),
        "quality_bin": quality_bin,
    }


def enumerate_adult_black_white_images(
    image_dir: Path,
    *,
    min_age: int = 18,
    max_age: int = 70,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    malformed: list[str] = []
    for path in sorted(image_dir.glob("*.jpg")):
        parsed = parse_utkface_filename(path.name)
        if parsed is None:
            malformed.append(path.name)
            continue
        if not min_age <= int(parsed["age"]) <= max_age:
            continue
        if parsed["race"] not in {"Black", "White"}:
            continue
        quality = image_quality_metrics(path)
        rows.append(
            {
                "source_filename": path.name,
                "image_path": str(path.resolve()),
                **parsed,
                "age_band": age_band(int(parsed["age"])),
                **quality,
            }
        )
    return rows, malformed


def _normalize_fer_label(label: str) -> str:
    return "".join(character for character in label.strip().lower() if character.isalpha())


def annotate_expressions(
    rows: list[dict[str, Any]],
    *,
    model_id: str,
    model_revision: str,
    batch_size: int,
    output_csv: Path,
) -> list[dict[str, Any]]:
    cache_dir = PROJECT_ROOT / "data" / "model-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(cache_dir / "xet"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError as exc:
        raise RuntimeError(
            "UTKFace expression dependencies are missing; install requirements-utkface.lock"
        ) from exc

    existing: dict[str, dict[str, str]] = {}
    if output_csv.exists():
        with output_csv.open(newline="", encoding="utf-8") as handle:
            existing = {row["source_filename"]: dict(row) for row in csv.DictReader(handle)}

    processor = AutoImageProcessor.from_pretrained(
        model_id,
        revision=model_revision,
        cache_dir=cache_dir,
        use_fast=False,
    )
    model = AutoModelForImageClassification.from_pretrained(
        model_id, revision=model_revision, cache_dir=cache_dir
    )
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device).eval()
    labels = {int(key): str(value) for key, value in model.config.id2label.items()}
    output: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for row in rows:
        prior = existing.get(str(row["source_filename"]))
        if prior and prior.get("expression_model_revision") == model_revision:
            restored = dict(row)
            restored.update(
                {
                    "fer_label": prior["fer_label"],
                    "fer_confidence": float(prior["fer_confidence"]),
                    "expression_class": prior["expression_class"],
                    "expression_model_id": prior["expression_model_id"],
                    "expression_model_revision": prior["expression_model_revision"],
                }
            )
            output.append(restored)
        else:
            pending.append(row)

    fieldnames: list[str] | None = None
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        images = []
        for row in batch:
            with Image.open(str(row["image_path"])) as image:
                images.append(image.convert("RGB").copy())
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            probabilities = model(**inputs).logits.softmax(dim=-1).cpu().numpy()
        for row, scores in zip(batch, probabilities, strict=True):
            index = int(scores.argmax())
            fer_label = labels[index]
            normalized = _normalize_fer_label(fer_label)
            output.append(
                {
                    **row,
                    "fer_label": fer_label,
                    "fer_confidence": float(scores[index]),
                    "expression_class": FER_CLASS_MAP.get(normalized, "ambiguous"),
                    "expression_model_id": model_id,
                    "expression_model_revision": model_revision,
                }
            )
        output.sort(key=lambda item: str(item["source_filename"]))
        fieldnames = sorted({key for row in output for key in row})
        _write_csv(output_csv, output, fieldnames)
        print(f"Expression annotations {len(output)}/{len(rows)}", flush=True)
    if not pending and existing:
        output.sort(key=lambda item: str(item["source_filename"]))
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    columns = fieldnames or list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _round_robin_pairs(
    units: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_stratum[str(unit["match_stratum"])].append(unit)
    for values in by_stratum.values():
        rng.shuffle(values)
    keys = sorted(by_stratum)
    selected: list[dict[str, Any]] = []
    while keys and len(selected) < count:
        remaining_keys = []
        for key in keys:
            values = by_stratum[key]
            if values and len(selected) < count:
                selected.append(values.pop())
            if values:
                remaining_keys.append(key)
        keys = remaining_keys
    if len(selected) != count:
        raise ValueError(f"Requested {count} matched pairs but only {len(selected)} were available")
    return selected


def build_matched_pools(
    rows: list[dict[str, Any]],
    *,
    evaluation_per_intersection: int,
    finetuning_per_intersection: int,
    min_expression_confidence: float,
    min_quality_score: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible_before_deduplication = [
        dict(row)
        for row in rows
        if str(row.get("expression_class")) in {"positive", "neutral", "negative"}
        and float(row.get("fer_confidence") or 0.0) >= min_expression_confidence
        and float(row.get("quality_score") or 0.0) >= min_quality_score
    ]
    eligible = []
    seen_hashes: set[str] = set()
    duplicate_source_files = 0
    for row in sorted(
        eligible_before_deduplication,
        key=lambda item: str(item["source_filename"]),
    ):
        digest = sha256_file(Path(str(row["image_path"])))
        if digest in seen_hashes:
            duplicate_source_files += 1
            continue
        seen_hashes.add(digest)
        row["sha256"] = digest
        eligible.append(row)
    for row in eligible:
        row["match_stratum"] = "|".join(
            [
                str(row["gender"]),
                str(row["age_band"]),
                str(row["expression_class"]),
                str(row["quality_bin"]),
            ]
        )

    rng = random.Random(seed)
    units_by_gender: dict[str, list[dict[str, Any]]] = {"Female": [], "Male": []}
    pair_counter = 0
    by_stratum: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"Black": [], "White": []}
    )
    for row in eligible:
        by_stratum[str(row["match_stratum"])][str(row["race"])].append(row)
    for stratum, groups in sorted(by_stratum.items()):
        for values in groups.values():
            rng.shuffle(values)
        for black, white in zip(groups["Black"], groups["White"], strict=False):
            pair_counter += 1
            gender = str(black["gender"])
            units_by_gender[gender].append(
                {
                    "match_pair_id": f"utk_match_{pair_counter:05d}",
                    "match_stratum": stratum,
                    "gender": gender,
                    "rows": [black, white],
                }
            )

    needed_per_gender = evaluation_per_intersection + finetuning_per_intersection
    selected_units: list[dict[str, Any]] = []
    for gender in ("Female", "Male"):
        selected = _round_robin_pairs(units_by_gender[gender], needed_per_gender, rng)
        for index, unit in enumerate(selected):
            unit = dict(unit)
            if index % 2 == 0 and sum(
                candidate.get("pool_role") == "evaluation"
                for candidate in selected_units
                if candidate["gender"] == gender
            ) < evaluation_per_intersection:
                unit["pool_role"] = "evaluation"
            else:
                unit["pool_role"] = "future_finetuning"
            selected_units.append(unit)

        gender_units = [unit for unit in selected_units if unit["gender"] == gender]
        evaluation_units = [unit for unit in gender_units if unit["pool_role"] == "evaluation"]
        finetuning_units = [
            unit for unit in gender_units if unit["pool_role"] == "future_finetuning"
        ]
        if len(evaluation_units) != evaluation_per_intersection:
            raise AssertionError("Evaluation pair allocation is unbalanced")
        if len(finetuning_units) != finetuning_per_intersection:
            raise AssertionError("Fine-tuning pair allocation is unbalanced")

    selected_rows: list[dict[str, Any]] = []
    for unit in selected_units:
        finetuning_index = None
        if unit["pool_role"] == "future_finetuning":
            same_gender = [
                candidate
                for candidate in selected_units
                if candidate["gender"] == unit["gender"]
                and candidate["pool_role"] == "future_finetuning"
            ]
            finetuning_index = same_gender.index(unit)
        for row in unit["rows"]:
            item = dict(row)
            digest = str(row["sha256"])
            item.update(
                {
                    "image_id": f"utk_{digest[:16]}",
                    "sha256": digest,
                    "match_pair_id": unit["match_pair_id"],
                    "pool_role": unit["pool_role"],
                    "finetuning_split": (
                        "development"
                        if finetuning_index is not None and finetuning_index % 5 == 0
                        else "train"
                        if finetuning_index is not None
                        else "not_applicable"
                    ),
                }
            )
            selected_rows.append(item)

    selected_rows.sort(
        key=lambda row: (
            str(row["pool_role"]),
            str(row["gender"]),
            str(row["match_pair_id"]),
            str(row["race"]),
        )
    )
    duplicate_hashes = [
        digest
        for digest, count in Counter(row["sha256"] for row in selected_rows).items()
        if count > 1
    ]
    if duplicate_hashes:
        raise ValueError(f"Selected pools contain duplicate images: {duplicate_hashes[:5]}")

    report = {
        "input_images": len(rows),
        "eligible_images_before_content_deduplication": len(
            eligible_before_deduplication
        ),
        "eligible_images": len(eligible),
        "exact_duplicate_source_files_removed": duplicate_source_files,
        "available_matched_pairs_by_gender": {
            gender: len(units) for gender, units in units_by_gender.items()
        },
        "selected_images": len(selected_rows),
        "pool_counts": dict(Counter(str(row["pool_role"]) for row in selected_rows)),
        "intersection_counts": dict(
            Counter(
                f"{row['pool_role']}|{row['race']}|{row['gender']}" for row in selected_rows
            )
        ),
        "expression_counts": dict(
            Counter(
                f"{row['pool_role']}|{row['race']}|{row['expression_class']}"
                for row in selected_rows
            )
        ),
        "finetuning_split_counts": dict(
            Counter(
                f"{row['finetuning_split']}|{row['race']}|{row['gender']}"
                for row in selected_rows
                if row["pool_role"] == "future_finetuning"
            )
        ),
        "min_expression_confidence": min_expression_confidence,
        "min_quality_score": min_quality_score,
        "seed": seed,
    }
    return selected_rows, report


def write_frozen_utkface_artifacts(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    output_dir: Path,
    *,
    source_archive_sha256: str,
    source_revision: str,
    expression_model_id: str,
    expression_model_revision: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation = [row for row in rows if row["pool_role"] == "evaluation"]
    finetuning = [row for row in rows if row["pool_role"] == "future_finetuning"]

    def public_row(row: dict[str, Any]) -> dict[str, Any]:
        source_path = Path(str(row["image_path"]))
        return {
            "image_id": row["image_id"],
            "filename": row["source_filename"],
            "race": row["race"],
            "gender": row["gender"],
            "age": row["age"],
            "age_band": row["age_band"],
            "race_code": row["race_code"],
            "gender_code": row["gender_code"],
            "expression_class": row["expression_class"],
            "expression_label": row["fer_label"],
            "expression_confidence": row["fer_confidence"],
            "expression_model_id": row["expression_model_id"],
            "expression_model_revision": row["expression_model_revision"],
            "quality_score": row["quality_score"],
            "quality_bin": row["quality_bin"],
            "width": row["width"],
            "height": row["height"],
            "sha256": row["sha256"],
            "match_pair_id": row["match_pair_id"],
            "pool_role": row["pool_role"],
            "finetuning_split": row["finetuning_split"],
            "source_relative_path": str(
                Path("../../private/utkface/raw/UTKFace") / source_path.name
            ),
        }

    evaluation_path = output_dir / "utkface_iat_evaluation_manifest.csv"
    finetuning_path = output_dir / "utkface_finetuning_image_manifest.csv"
    _write_csv(evaluation_path, [public_row(row) for row in evaluation])
    _write_csv(finetuning_path, [public_row(row) for row in finetuning])

    prompts: list[dict[str, Any]] = []
    prompt_specs = (
        ("iat_legacy61", IAT_PROMPT),
        ("grounded_profile", PROFILE_PROMPT),
    )
    for row in finetuning:
        for task_type, prompt in prompt_specs:
            prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            task_material = f"{row['image_id']}|{task_type}|{prompt_sha}"
            prompts.append(
                {
                    "candidate_task_id": hashlib.sha256(
                        task_material.encode("utf-8")
                    ).hexdigest()[:24],
                    "image_id": row["image_id"],
                    "image_sha256": row["sha256"],
                    "match_pair_id": row["match_pair_id"],
                    "finetuning_split": row["finetuning_split"],
                    "task_type": task_type,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha,
                    "race_metadata_in_prompt": False,
                    "candidate_generation_complete": False,
                    "chosen": "",
                    "rejected": "",
                    "dpo_training_ready": False,
                }
            )
    prompts_path = output_dir / "utkface_future_dpo_candidate_tasks.csv"
    _write_csv(prompts_path, prompts)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "UTKFace aligned and cropped faces",
        "official_project": "https://github.com/aicip/UTKFace",
        "official_use_scope": "non-commercial research",
        "transport_mirror": "https://huggingface.co/datasets/py97/UTKFace-Cropped",
        "transport_revision": source_revision,
        "source_archive_sha256": source_archive_sha256,
        "expression_model_id": expression_model_id,
        "expression_model_revision": expression_model_revision,
        "race_metadata_passed_to_model_prompts": False,
        "selection_report": report,
        "artifacts": {
            "evaluation_manifest": str(evaluation_path),
            "finetuning_image_manifest": str(finetuning_path),
            "future_dpo_candidate_tasks": str(prompts_path),
        },
        "limitations": [
            "UTKFace demographic labels were estimated algorithmically and human checked; "
            "they may be wrong.",
            "Visible-expression classes are automated FER model outputs, not emotion ground truth.",
            "The FER model may itself have demographic performance disparities.",
            "The future-DPO file contains prompts only; chosen and rejected responses are "
            "intentionally blank.",
        ],
    }
    manifest_path = output_dir / "utkface_freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def validate_frozen_manifests(output_dir: Path) -> dict[str, Any]:
    paths = {
        "evaluation": output_dir / "utkface_iat_evaluation_manifest.csv",
        "finetuning": output_dir / "utkface_finetuning_image_manifest.csv",
        "candidate_tasks": output_dir / "utkface_future_dpo_candidate_tasks.csv",
    }
    tables: dict[str, list[dict[str, str]]] = {}
    for name, path in paths.items():
        with path.open(newline="", encoding="utf-8") as handle:
            tables[name] = [dict(row) for row in csv.DictReader(handle)]
    eval_hashes = {row["sha256"] for row in tables["evaluation"]}
    train_hashes = {row["sha256"] for row in tables["finetuning"]}
    overlap = eval_hashes & train_hashes
    if overlap:
        raise ValueError(f"Evaluation/fine-tuning image overlap: {sorted(overlap)[:5]}")
    if any(row["dpo_training_ready"].lower() == "true" for row in tables["candidate_tasks"]):
        raise ValueError("Prompt-only candidate rows must not be marked DPO-ready")
    if any(row["chosen"] or row["rejected"] for row in tables["candidate_tasks"]):
        raise ValueError("Prompt-only candidate rows unexpectedly contain preferences")
    return {
        "evaluation_images": len(tables["evaluation"]),
        "finetuning_images": len(tables["finetuning"]),
        "candidate_tasks": len(tables["candidate_tasks"]),
        "evaluation_finetuning_overlap": len(overlap),
    }
