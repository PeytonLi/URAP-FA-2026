"""Deterministic image-sample curation. No generative-model API calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
EXPR = ("neutral", "positive", "negative", "uncertain")
UTK_RACES = ("White", "Black", "Asian", "Indian", "Other")
UTK_GENDERS = ("Male", "Female")
FER_CLASSES = {"neutral": "neutral", "happy": "positive", "sad": "negative",
               "angry": "negative", "disgust": "negative", "fear": "negative"}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stable(seed, value):
    return digest_bytes(f"{seed}|{value}".encode())


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({k for r in rows for k in r})
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def pixel_hash(image):
    rgb = image.convert("RGB")
    return digest_bytes(f"{rgb.width}x{rgb.height}|RGB|".encode() + rgb.tobytes())


def image_metrics(image):
    image = image.convert("RGB")
    g = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    small = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    coeff = dctn(small, type=2, norm="ortho")[:8, :8]
    mask = coeff > np.median(coeff.ravel()[1:])
    phash = int("".join("1" if b else "0" for b in mask.ravel()), 2)
    dh = np.asarray(image.convert("L").resize((9, 8)), dtype=np.int16)
    bits = dh[:, 1:] > dh[:, :-1]
    dhash = int("".join("1" if b else "0" for b in bits.ravel()), 2)
    return {
        "width": image.width, "height": image.height,
        "pixel_sha256": pixel_hash(image),
        "perceptual_hash": f"{phash:016x}", "difference_hash": f"{dhash:016x}",
        "brightness": float(g.mean()), "contrast": float(g.std()),
        "edge_energy": float(np.abs(np.diff(g, axis=0)).mean() +
                             np.abs(np.diff(g, axis=1)).mean()),
        "clipped_dark_fraction": float((g <= 5 / 255).mean()),
        "clipped_light_fraction": float((g >= 250 / 255).mean()),
    }


def historical_exclusions(workspace):
    """Record explicit historical input references, not prior model outcomes."""
    files = [
        "gemma3-fairface-debiasing-august3/data/august3/raw_pool/fairface_training_pool.csv",
        "gemma3-fairface-debiasing-july13/results/pair_audit_research_july13_100pg_20260713_205104/artifacts/train_metadata.csv",
        "gemma3-fairface-debiasing-july13/results/reference_base_gemma3_20260706/eval_metadata.csv",
        "gemma3-fairface-debiasing-july19/results/research_july19_pretraining_20260719/inputs/fairface_train_eval_metadata.csv",
        "gemma3-fairface-debiasing-july19/results/research_july19_500pg_20260720_071215/inputs/fairface_train_eval_metadata.csv",
        "multimodel-bias-benchmark-august31/data/frozen/utkface/utkface_iat_evaluation_manifest.csv",
        "multimodel-bias-benchmark-august31/data/frozen/utkface/utkface_finetuning_image_manifest.csv",
        "research-evaluation-september11/data/iat12/manifest.csv",
    ]
    indices, filenames, pixels = set(), set(), set()
    frozen = []
    provenance = []
    for rel in files:
        p = workspace / rel
        if not p.exists():
            raise FileNotFoundError(f"Required historical registry source missing: {rel}")
        rows = read_csv(p)
        provenance.append({"source": rel, "sha256": sha(p), "rows": len(rows)})
        for row in rows:
            if row.get("dataset_index"):
                indices.add(int(row["dataset_index"]))
            if "utkface" in rel:
                filenames.add(row["filename"])
            path = None
            if row.get("source_relative_path"):
                path = (p.parent / row["source_relative_path"]).resolve()
            elif "raw_pool" in rel:
                path = workspace / "gemma3-fairface-debiasing-august3" / row["image_path"]
            if path and path.is_file():
                with Image.open(path) as im:
                    rgb = ImageOps.exif_transpose(im).convert("RGB")
                    metrics = image_metrics(rgb)
                pixels.add(metrics["pixel_sha256"])
                frozen.append({"source_id": row.get("image_id", row.get("filename")),
                               **{k: metrics[k] for k in ["pixel_sha256", "perceptual_hash", "difference_hash"]}})
    write_csv(ROOT / "data/frozen/historical_image_fingerprints.csv", frozen)
    write_csv(ROOT / "data/frozen/excluded_fairface_indices.csv",
              [{"source_index": i} for i in sorted(indices)])
    write_csv(ROOT / "data/frozen/excluded_utkface_filenames.csv",
              [{"source_filename": s} for s in sorted(filenames)])
    write_json(ROOT / "data/frozen/historical_sources.json", {
        "sources": provenance,
        "fairface_index_exclusion_policy": "Conservative union of indices across historical train/eval tables; split ambiguity can overexclude.",
        "complete_training_history_verified": False,
        "identity_overlap_verified": False,
    })
    return indices, filenames, pixels


def inventory(workspace, config):
    indices, excluded_files, _ = historical_exclusions(workspace)
    out = []
    files = sorted((workspace / "multimodel-bias-benchmark-august31/data/private/utkface/raw/UTKFace").glob("*.jpg"))
    for p in files:
        row = {"dataset": "utkface", "source_id": p.name, "source_index": "",
               "source_split": "unspecified", "source_filename": p.name,
               "source_race": "", "source_gender": "", "age_band": "",
               "source_age": "", "source_revision": config["utkface_revision"],
               "inventory_status": "eligible"}
        try:
            age, sex, race = map(int, p.name.split("_")[:3])
            if not (0 <= age <= 116 and sex in (0, 1) and race in range(5)):
                raise ValueError("invalid label")
            row.update(source_age=age, source_race=UTK_RACES[race], source_gender=UTK_GENDERS[sex])
            row["age_band"] = f"{age // 10 * 10}-{age // 10 * 10 + 9}"
            if row["age_band"] not in config["age_bands"]:
                row["inventory_status"] = "age_out_of_scope"
            elif p.name in excluded_files:
                row["inventory_status"] = "historical_exposure"
        except (ValueError, IndexError):
            row["inventory_status"] = "invalid_source_metadata"
        out.append(row)
    fair_dir = workspace / ("gemma3-fairface-debiasing-august3/data/august3/huggingface_cache/datasets/"
                            "HuggingFaceM4___fair_face/0.25/0.0.0/" + config["fairface_revision"])
    info = json.loads((fair_dir / "dataset_info.json").read_text())
    import pyarrow as pa
    import pyarrow.ipc as ipc
    fair_tables = []
    for p in sorted(fair_dir.glob("fair_face-train-*.arrow")):
        with pa.memory_map(str(p), "r") as f:
            fair_tables.append(ipc.open_stream(f).read_all())
    table = pa.concat_tables(fair_tables)
    meta = table.select(["age", "race", "gender"]).to_pylist()
    for index, r in enumerate(meta):
        age = info["features"]["age"]["names"][r["age"]]
        row = {"dataset": "fairface", "source_id": f"train:{index:06d}",
               "source_index": index, "source_split": "train", "source_filename": "",
               "source_race": info["features"]["race"]["names"][r["race"]],
               "source_gender": info["features"]["gender"]["names"][r["gender"]],
               "source_age": age, "age_band": age,
               "source_revision": config["fairface_revision"], "inventory_status": "eligible"}
        if age not in config["age_bands"]:
            row["inventory_status"] = "age_out_of_scope"
        elif index in indices:
            row["inventory_status"] = "historical_exposure"
        out.append(row)
    write_csv(ROOT / "data/source_inventory.csv", out)
    write_json(ROOT / "data/frozen/source_files.json", {
        "fairface_arrow": [{"filename": p.name, "sha256": sha(p)} for p in sorted(fair_dir.glob("fair_face-train-*.arrow"))],
        "fairface_dataset_info": info,
        "utkface_source_count": len(files),
        "source_revisions": {d: config[d + "_revision"] for d in ("utkface", "fairface")},
    })
    return out, table


def capped_allocation(total, capacities, weights=None):
    """Deterministic capped apportionment; no fabricated fill for scarce cells."""
    if total < 0 or total > sum(capacities.values()):
        raise ValueError("Insufficient common support for allocation")
    weights = weights or {k: 1.0 for k in capacities}
    result = {k: 0 for k in capacities}
    for _ in range(total):
        available = [k for k in capacities if result[k] < capacities[k]]
        k = min(available, key=lambda k: ((result[k] + 1) / max(weights.get(k, 0), 1e-9), str(k)))
        result[k] += 1
    return result


class HashIndex:
    """Five exact hash blocks guarantee retrieval within four Hamming bits."""
    def __init__(self):
        self.buckets = defaultdict(set)
        self.items = []

    def add(self, row):
        i = len(self.items)
        self.items.append(row)
        h = int(row["perceptual_hash"], 16)
        for shift, width in [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]:
            self.buckets[(shift, (h >> shift) & ((1 << width) - 1))].add(i)

    def matches(self, row, max_phash=4, max_dhash=6):
        if not 0 <= max_phash <= 4 or not 0 <= max_dhash <= 64:
            raise ValueError("HashIndex supports perceptual distance 0-4 and difference distance 0-64")
        if max_phash > 4:
            raise ValueError("Index only guarantees Hamming radius <= 4")
        h, d = int(row["perceptual_hash"], 16), int(row["difference_hash"], 16)
        ids = set()
        for shift, width in [(0, 13), (13, 13), (26, 13), (39, 13), (52, 12)]:
            ids.update(self.buckets.get((shift, (h >> shift) & ((1 << width) - 1)), ()))
        return [self.items[i] for i in sorted(ids)
                if (h ^ int(self.items[i]["perceptual_hash"], 16)).bit_count() <= max_phash
                and (d ^ int(self.items[i]["difference_hash"], 16)).bit_count() <= max_dhash]


def prepare(workspace, config):
    rows, table = inventory(workspace, config)
    grouped = defaultdict(list)
    for r in rows:
        if r["inventory_status"] == "eligible":
            grouped[(r["dataset"], r["source_race"], r["source_gender"], r["age_band"])].append(r)
    candidates = []
    for _, group in sorted(grouped.items()):
        candidates.extend(sorted(group, key=lambda r: stable(config["seed"], r["dataset"] + r["source_id"]))
                          [:config["candidate_per_age_race_gender"]])
    hist = HashIndex()
    historical_pixels = set()
    for h in read_csv(ROOT / "data/frozen/historical_image_fingerprints.csv"):
        hist.add(h)
        historical_pixels.add(h["pixel_sha256"])
    current = HashIndex()
    distances = {"max_phash": config["perceptual_hash_max_distance"],
                 "max_dhash": config["difference_hash_max_distance"]}
    pixels = {}
    out = []
    for row in sorted(candidates, key=lambda r: stable(config["seed"], r["dataset"] + r["source_id"])):
        r = dict(row)
        r.update(image_id=r["dataset"][:2] + "_" + stable(config["seed"], r["source_id"])[:20],
                 screening_status="eligible", duplicate_of="", automated_review=True,
                 human_reviewed=False)
        dest = ROOT / "private/candidates" / r["dataset"] / (r["image_id"] + ".png")
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if r["dataset"] == "utkface":
                source = workspace / "multimodel-bias-benchmark-august31/data/private/utkface/raw/UTKFace" / r["source_filename"]
                raw = source.read_bytes()
            else:
                im_field = table["image"][int(r["source_index"])].as_py()
                raw = im_field["bytes"]
                if raw is None:
                    raise ValueError("FairFace image bytes missing")
            r["source_image_sha256"] = digest_bytes(raw)
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                image = ImageOps.exif_transpose(im).convert("RGB")
            if min(image.size) < config["min_side_pixels"]:
                r["screening_status"] = "undersized"
            else:
                r.update(image_metrics(image))
                if r["pixel_sha256"] in historical_pixels:
                    r["screening_status"] = "historical_exact_duplicate"
                elif hist.matches(r, **distances):
                    r["screening_status"] = "historical_near_duplicate"
                elif r["pixel_sha256"] in pixels:
                    r["screening_status"] = "exact_duplicate"
                    r["duplicate_of"] = pixels[r["pixel_sha256"]]
                else:
                    near = current.matches(r, **distances)
                    if near:
                        r["screening_status"] = "near_duplicate"
                        r["duplicate_of"] = near[0]["image_id"]
                if r["screening_status"] == "eligible":
                    image.save(dest, format="PNG", optimize=False)
                    r["image_sha256"] = sha(dest)
                    r["image_path"] = str(dest.relative_to(ROOT))
                    pixels[r["pixel_sha256"]] = r["image_id"]
                    current.add(r)
        except (OSError, ValueError) as exc:
            r["screening_status"] = "unreadable"
            r["screening_error"] = type(exc).__name__
        out.append(r)
        if len(out) % 1000 == 0:
            print(f"Screened {len(out)}/{len(candidates)}", flush=True)
    write_csv(ROOT / "data/candidate_audit.csv", out)
    print(json.dumps({"screening": dict(Counter(r["screening_status"] for r in out)), "candidates": len(out)}), flush=True)


def annotate(workspace, config, limit=None):
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    model_dir = workspace / ("multimodel-bias-benchmark-august31/data/model-cache/models--trpakov--vit-face-expression/"
                             "snapshots/" + config["expression_model_revision"])
    path = ROOT / "data/expression_annotations.csv"
    prior = {r["image_id"]: r for r in read_csv(path)} if path.exists() else {}
    old = {r["source_filename"]: r for r in read_csv(workspace / "multimodel-bias-benchmark-august31/data/private/utkface/expression_annotations.csv")}
    candidates = [r for r in read_csv(ROOT / "data/candidate_audit.csv") if r["screening_status"] == "eligible"]
    todo = []
    for r in candidates:
        if r["image_id"] in prior:
            if prior[r["image_id"]]["image_sha256"] != r["image_sha256"]:
                raise ValueError("Cached annotation image drift")
            continue
        cached = old.get(r["source_filename"]) if r["dataset"] == "utkface" else None
        if cached and cached["expression_model_revision"] == config["expression_model_revision"]:
            prior[r["image_id"]] = expression_record(r, cached["fer_label"], float(cached["fer_confidence"]), config, "historical_same_revision_cache")
        else:
            todo.append(r)
    write_csv(path, list(prior.values()))
    if limit:
        todo = todo[:limit]
    if not todo:
        print("No new expression annotations needed", flush=True)
        return
    torch.set_num_threads(4)
    torch.manual_seed(20260929)
    processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(str(model_dir), local_files_only=True).eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    write_json(ROOT / "data/frozen/expression_model.json", {
        "model_id": config["expression_model_id"], "revision": config["expression_model_revision"],
        "files": [{"filename": p.name, "sha256": sha(p)} for p in model_dir.iterdir() if p.is_file()],
        "device": device, "torch_version": torch.__version__,
        "historical_cache_sha256": sha(workspace / "multimodel-bias-benchmark-august31/data/private/utkface/expression_annotations.csv"),
        "historical_cache_caveat": "Same pinned model and named source files; historical inference was not rerun. Cached values are explicitly labeled.",
    })
    start = time.monotonic()
    batch = config["expression_batch_size"]
    for i in range(0, len(todo), batch):
        rows = todo[i:i + batch]
        images = []
        for r in rows:
            with Image.open(ROOT / r["image_path"]) as im:
                images.append(im.convert("RGB").copy())
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            probs = model(**inputs).logits.softmax(dim=-1).cpu().numpy()
        for r, prob in zip(rows, probs, strict=True):
            ix = int(prob.argmax())
            prior[r["image_id"]] = expression_record(r, model.config.id2label[ix], float(prob[ix]), config, "local_new_inference")
        if i % (batch * 8) == 0 or i + batch >= len(todo):
            write_csv(path, sorted(prior.values(), key=lambda r: r["image_id"]))
            done = min(i + batch, len(todo))
            print(f"Expression {done}/{len(todo)} new; {len(prior)} total; {done / max(time.monotonic()-start, .001):.2f} images/s", flush=True)


def expression_record(row, label, confidence, config, provenance):
    label = label.lower()
    klass = FER_CLASSES.get(label, "uncertain")
    if confidence < config["expression_confidence_for_class"]:
        klass = "uncertain"
    return {"image_id": row["image_id"], "image_sha256": row["image_sha256"],
            "expression_label": label, "expression_confidence": confidence,
            "expression_class": klass, "expression_provenance": provenance,
            "expression_model_id": config["expression_model_id"],
            "expression_model_revision": config["expression_model_revision"]}


def select(config, materialize=True):
    annotation_rows = read_csv(ROOT / "data/expression_annotations.csv")
    annotations = {r["image_id"]: r for r in annotation_rows}
    if len(annotations) != len(annotation_rows):
        raise ValueError("Duplicate annotation IDs")
    candidates = [r for r in read_csv(ROOT / "data/candidate_audit.csv") if r["screening_status"] == "eligible"]
    missing = [r["image_id"] for r in candidates if r["image_id"] not in annotations]
    if missing:
        raise ValueError(f"Missing expression annotations for {len(missing)} eligible images")
    for r in candidates:
        ann = annotations[r["image_id"]]
        if ann["image_sha256"] != r["image_sha256"]:
            raise ValueError("Annotation image hash drift")
        if any(ann[k] != config[k] for k in ("expression_model_id", "expression_model_revision")):
            raise ValueError("Annotation model revision drift")
        if ann["expression_class"] not in EXPR or not 0 <= float(ann["expression_confidence"]) <= 1:
            raise ValueError("Invalid expression annotation")
        r.update(ann)
    selected, quotas, reports = [], [], {}
    for dataset, races in config["race_labels"].items():
        subset = [r for r in candidates if r["dataset"] == dataset]
        cells = [(race, sex) for race in races for sex in config["genders"]]
        groups = defaultdict(list)
        for r in subset:
            groups[(r["source_race"], r["source_gender"], r["age_band"], r["expression_class"])].append(r)
        capacities = {(a, e): min(len(groups[(race, sex, a, e)]) for race, sex in cells)
                      for a in config["age_bands"] for e in EXPR}
        age_caps = {a: sum(capacities[a, e] for e in EXPR) for a in config["age_bands"]}
        target = min(config["target_per_race_gender"], sum(age_caps.values()))
        if target < 50:
            raise ValueError(f"Common support too small for {dataset}: {target} per cell")
        age_counts = capped_allocation(target, age_caps)
        q = {}
        for age in config["age_bands"]:
            weights = {e: sum(len(groups[(race, sex, age, e)]) for race, sex in cells) for e in EXPR}
            expr_counts = capped_allocation(age_counts[age], {e: capacities[age, e] for e in EXPR}, weights)
            q.update({(age, e): n for e, n in expr_counts.items()})
        # Assign identical within-stratum splits to every group by shared match sets.
        units = [((age, expr), i) for (age, expr), n in sorted(q.items()) for i in range(n)]
        ordered_units = sorted(units, key=lambda u: stable(config["seed"], dataset + repr(u)))
        ntrain = math.floor(target * config["split_fractions"]["train"])
        ndev = math.floor(target * config["split_fractions"]["development"])
        splits = {unit: "train" if i < ntrain else "development" if i < ntrain + ndev else "test"
                  for i, unit in enumerate(ordered_units)}
        for (age, expr), n in sorted(q.items()):
            quotas.append({"dataset": dataset, "age_band": age, "expression_class": expr,
                           "common_capacity_per_cell": capacities[age, expr], "selected_per_cell": n})
            for race, sex in cells:
                pool = sorted(groups[(race, sex, age, expr)], key=lambda r: stable(config["seed"], r["image_id"]))
                for i, row in enumerate(pool[:n]):
                    r = dict(row)
                    r["split"] = splits[((age, expr), i)]
                    r["match_set_id"] = dataset + "_" + stable(config["seed"], f"{age}|{expr}|{i}")[:12]
                    r["review_status"] = "automated_only"
                    dest = ROOT / "private/dataset/images" / dataset / (r["image_id"] + ".png")
                    if materialize:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(ROOT / r["image_path"], dest)
                    r["image_path"] = str(Path("images") / dataset / dest.name)
                    r.pop("screening_error", None)
                    selected.append(r)
        reports[dataset] = {"target_per_cell": config["target_per_race_gender"], "selected_per_cell": target,
                            "total_selected": target * len(cells), "group_cells": len(cells),
                            "age_quota_per_cell": age_counts, "common_capacity_per_cell": sum(age_caps.values()),
                            "image_disjoint_splits": {"train": ntrain, "development": ndev, "test": target-ntrain-ndev},
                            "not_person_disjoint_verified": True}
    selected.sort(key=lambda r: (r["dataset"], r["source_race"], r["source_gender"], r["image_id"]))
    write_csv(ROOT / "data/selected_images.csv", selected)
    write_csv(ROOT / "private/dataset/manifest.csv", selected)
    write_csv(ROOT / "data/common_support_quotas.csv", quotas)
    chosen_ids = {r["image_id"] for r in selected}
    write_csv(ROOT / "data/selection_decisions.csv", [
        {"image_id": r["image_id"], "dataset": r["dataset"],
         "selection_status": "selected" if r["image_id"] in chosen_ids else "eligible_not_selected",
         "reason": "common_support_matched_quota" if r["image_id"] in chosen_ids else "outside_frozen_quota_or_rank"}
        for r in candidates])
    write_json(ROOT / "data/selection_report.json", reports)
    print(json.dumps(reports, indent=2), flush=True)


def validate(config, skip_image_files=False):
    rows = read_csv(ROOT / "data/selected_images.csv")
    errors = []
    for key in ("image_id", "image_sha256", "pixel_sha256"):
        if len({r[key] for r in rows}) != len(rows):
            errors.append(f"Non-unique {key}")
    history = read_csv(ROOT / "data/frozen/historical_image_fingerprints.csv")
    idx = HashIndex()
    for r in history:
        idx.add(r)
    for r in rows:
        p = ROOT / "private/dataset" / r["image_path"]
        if not skip_image_files and (not p.is_file() or sha(p) != r["image_sha256"]):
            errors.append(f"Image absent or drifted: {r['image_id']}")
        if idx.matches(r, max_phash=config["perceptual_hash_max_distance"],
                       max_dhash=config["difference_hash_max_distance"]):
            errors.append(f"Near duplicate among release/history: {r['image_id']}")
        idx.add(r)
    balance = []
    for dataset, races in config["race_labels"].items():
        cells = [(r, g) for r in races for g in config["genders"]]
        for split in ["all", "train", "development", "test"]:
            subset = [r for r in rows if r["dataset"] == dataset and (split == "all" or r["split"] == split)]
            distributions = [Counter((r["age_band"], r["expression_class"]) for r in subset
                                     if (r["source_race"], r["source_gender"]) == cell) for cell in cells]
            matched = all(c == distributions[0] for c in distributions)
            if not matched:
                errors.append(f"Joint balance failure: {dataset}/{split}")
            balance.append({"dataset": dataset, "split": split, "joint_age_expression_equal": matched,
                            "images_per_race_gender_cell": sum(distributions[0].values())})
    report = {"passed": not errors, "errors": errors, "selected_images": len(rows), "balance_checks": balance,
              "image_files_verified": not skip_image_files,
              "person_disjointness_verified": False, "human_validated": False, "bias_benchmarks_run": 0,
              "config_sha256": sha(ROOT / "config/curation.json")}
    write_json(ROOT / "data/validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise ValueError("Release validation failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "annotate", "select", "validate"])
    p.add_argument("--workspace", type=Path, default=ROOT.parent)
    p.add_argument("--limit", type=int)
    p.add_argument("--metadata-only", action="store_true", help="Reproduce frozen selection without private image files.")
    p.add_argument("--skip-image-files", action="store_true", help="Validate metadata only; does not claim image-byte verification.")
    args = p.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = json.loads((ROOT / "config/curation.json").read_text())
    if args.stage == "prepare":
        prepare(args.workspace, config)
    elif args.stage == "annotate":
        annotate(args.workspace, config, args.limit)
    elif args.stage == "select":
        select(config, materialize=not args.metadata_only)
    else:
        validate(config, skip_image_files=args.skip_image_files)


if __name__ == "__main__":
    main()
