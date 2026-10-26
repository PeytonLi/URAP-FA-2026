"""IAT-only SFT/ORPO study preparation and read-only provider checks."""
from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "2026-09-11" / "src"))
from multimodel_bias.constants import IAT_PROMPT, IAT_WORDS

SOURCE = REPO / "2026-09-05/reference/dpo/august10/august10_negative_expression_pilot_v4/dataset"
MODEL_IDS = ("gemma-4-26b-a4b-it", "gemma-4-e4b-it", "qwen3-vl-8b-instruct",
             "qwen3p5-9b", "gemma-3-12b-it", "qwen2p5-vl-7b-instruct")
FIRECTL = ROOT / "runtime/firectl.exe"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha(path):
    """Decoded-pixel digest: sha256(f"{w}x{h}|RGB|" + RGB bytes), matching
    2026-09-29/src/curation.py::pixel_sha256."""
    from PIL import Image
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return hashlib.sha256(
            f"{rgb.width}x{rgb.height}|RGB|".encode() + rgb.tobytes()).hexdigest()


def image_matches_manifest(row, image):
    """Identity proof for a stored face image. Accepts either the PNG byte
    digest (image_sha256) or, when the file was re-encoded on another
    platform (byte-different, pixel-identical), the decoded-pixel digest
    (pixel_sha256) recorded by the September-29 manifest."""
    image = Path(image)
    if not image.is_file():
        return False
    if sha(image) == row["image_sha256"]:
        return True
    expected = row.get("pixel_sha256")
    if not expected:
        return False
    try:
        return pixel_sha(image) == expected
    except Exception:
        return False


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


MULTITRIPLE_DATASETS = ("image-curation-2026-09-29-v3", "image-curation-2026-09-29-v4")


def dataset_config():
    path = ROOT / "config/active_dataset.json"
    if not path.is_file():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("dataset_id") not in ("image-curation-2026-09-29-v1",
                                        "image-curation-2026-09-29-v2",
                                        "image-curation-2026-09-29-v3",
                                        "image-curation-2026-09-29-v4"):
        raise ValueError("Unknown active dataset; refusing to fall back to historical training data")
    return config


def read_ini(path):
    parser = configparser.ConfigParser(interpolation=None)
    if path.exists():
        raw = path.read_text(encoding="utf-8-sig")
        parser.read_string(raw if raw.lstrip().startswith("[") else "[default]\n" + raw)
    return {key: value for section in parser.sections() for key, value in parser[section].items()}


def request_json(url, token="", *, body=None, headers=None):
    request_headers = {"User-Agent": "URAP-IAT-Study/1.0", **(headers or {})}
    if token:
        request_headers["Authorization"] = "Bearer " + token
    payload = None if body is None else json.dumps(body).encode()
    if payload is not None:
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=payload, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Never propagate auth request/response bodies into logs.
        raise RuntimeError(f"HTTP {exc.code} at {urllib.parse.urlsplit(url).hostname}") from None


def fireworks_auth():
    if os.environ.get("FIREWORKS_API_KEY"):
        return os.environ["FIREWORKS_API_KEY"].strip(), "environment"
    auth = read_ini(Path.home() / ".fireworks/auth.ini")
    if auth.get("api_key"):
        return auth["api_key"], "firectl_api_key"
    private = ROOT / "data/private/fireworks_auth.json"
    if private.is_file():
        saved = json.loads(private.read_text(encoding="utf-8"))
        if saved.get("api_key"):
            return saved["api_key"], "study_key"
    raise RuntimeError("A Fireworks API key is required for REST/SDK calls; CLI sign-in works separately")


def ensure_api_key():
    try:
        return fireworks_auth()
    except RuntimeError:
        pass
    expiry = (datetime.now(timezone.utc) + timedelta(days=14)).strftime("%Y-%m-%d")
    result = cli_json("api-key", "create", "--key-name", "URAP IAT study 2026-10-03",
                      "--expire-time", expiry)
    private = ROOT / "data/private/fireworks_auth.json"
    # Retain the response privately before parsing so an unexpected schema never loses the key.
    save_json(private, result)
    key = next((value for value in result.values()
                if isinstance(value, str) and value.startswith("fw_")), None)
    if not key:
        raise RuntimeError("API key response schema needs inspection; private response retained, not logged")
    save_json(private, {"api_key": key, "expires_on": expiry, "purpose": "URAP IAT study"})
    return key, "study_key"


def cli_json(*args):
    result = subprocess.run([str(FIRECTL), *args, "-o", "json"], capture_output=True,
                            text=True, encoding="utf-8", timeout=90)
    if result.returncode:
        raise RuntimeError(f"firectl {' '.join(args[:2])} failed (exit {result.returncode})")
    return json.loads(result.stdout)


def preflight():
    report = {"checked_at": now(), "paid_resources_created": False, "models": []}
    try:
        if FIRECTL.is_file():
            token, source = "", "official_firectl_login"
        else:
            token, source = fireworks_auth()
        report["credential_source"] = source
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        report["auth_error"] = str(exc)
        save_json(ROOT / "results/preflight.json", report)
        print(json.dumps(report, indent=2))
        return 2
    fields = ("name", "hugging_face_url", "state", "supports_image_input", "tunable",
              "supports_lora", "use_training_v2", "supervised_lora_tunable",
              "supervised_full_parameter_tunable", "supports_serverless", "training_context_length")
    for model_id in MODEL_IDS:
        entry = {"requested_model": model_id}
        try:
            if FIRECTL.is_file():
                model = cli_json("model", "get", "accounts/fireworks/models/" + model_id)
            else:
                raw = request_json("https://api.fireworks.ai/v1/accounts/fireworks/models/" + model_id, token)
                import re
                model = {re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower(): value for key, value in raw.items()}
            entry.update({key: model[key] for key in fields if key in model})
            entry["base_model_details"] = {key: value for key, value in model.get("base_model_details", {}).items()
                                           if key in ("model_type", "parameter_count", "tunable", "moe")}
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            entry["error"] = str(exc)
        report["models"].append(entry)
        print(f"Checked {model_id}: {'available' if 'error' not in entry else entry['error']}", flush=True)
    try:
        if FIRECTL.is_file():
            shapes = cli_json("training-shape", "list")
            report["training_shapes_next_page_token"] = ""
        else:
            catalog = request_json("https://api.fireworks.ai/v1/accounts/fireworks/trainingShapes?pageSize=200", token)
            shapes = catalog.get("trainingShapes", [])
            report["training_shapes_next_page_token"] = catalog.get("nextPageToken", "")
        report["training_shapes"] = [s for s in shapes if any(
            word in json.dumps(s).lower() for word in ("gemma", "qwen2p5-vl", "qwen3-vl", "qwen3p5-9b"))]
    except RuntimeError as exc:
        report["training_shapes_error"] = str(exc)
    save_json(ROOT / "results/preflight.json", report)
    print(json.dumps(report, indent=2))
    return 0


def exact_target(text):
    if not isinstance(text, str):
        return False
    words = text.split(", ")
    return len(words) == 3 and len(set(words)) == 3 and all(word in IAT_WORDS for word in words)


def label_failures(row, forbidden):
    failures = []
    for key in ("chosen", "rejected"):
        if not exact_target(row[key]):
            failures.append(key + "_format")
    if row["chosen"] == row["rejected"]:
        failures.append("identical_responses")
    # Typed image input replaces the old pipeline's literal image marker.
    if row["prompt"].strip().removeprefix("<start_of_image> ") != IAT_PROMPT.strip():
        failures.append("prompt_mismatch")
    if row["image_sha256"] in forbidden:
        failures.append("benchmark_image_overlap")
    if row.get("judge_image_consensus") != "chosen" or row.get("judge_order_stable", "").lower() != "true":
        failures.append("historical_image_judge_did_not_pass")
    for key in ("chosen_identity_leakage_count", "rejected_identity_leakage_count"):
        if float(row.get(key) or 0) != 0:
            failures.append(key)
    relative = Path(row["image_path"])
    if relative.is_absolute() or ".." in relative.parts:
        failures.append("unsafe_source_path")
    return failures


def match_and_split(rows):
    """Keep compatible original pairs, then pair remaining compatible rows."""
    ages = ("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70")
    age_index = {age: index for index, age in enumerate(ages)}

    def compatible(black, white):
        fields = ("fairface_gender", "expression_class", "expression_fine", "chosen", "rejected")
        return (black["fairface_race"] == "Black" and white["fairface_race"] == "White"
                and all(black[key] == white[key] for key in fields)
                and black["age_bucket"] in age_index and white["age_bucket"] in age_index
                and abs(age_index[black["age_bucket"]] - age_index[white["age_bucket"]]) <= 1
                and abs(float(black["expression_intensity"]) - float(white["expression_intensity"])) <= 1
                and min(float(black["expression_confidence"]), float(white["expression_confidence"])) >= .7)

    groups = defaultdict(list)
    for row in rows:
        groups[row["match_block_id"]].append(row)
    paired, used = [], set()
    for block_id, block in sorted(groups.items()):
        if len(block) != 2:
            continue
        block = sorted(block, key=lambda row: row["fairface_race"])
        if compatible(*block):
            paired.append((block_id, block))
            used.update(row["image_sha256"] for row in block)
    remaining = sorted((row for row in rows if row["image_sha256"] not in used),
                       key=lambda row: row["image_sha256"])
    for black in remaining:
        if black["image_sha256"] in used or black["fairface_race"] != "Black":
            continue
        whites = [row for row in remaining if row["image_sha256"] not in used and compatible(black, row)]
        if not whites:
            continue
        white = min(whites, key=lambda row: (abs(age_index[black["age_bucket"]] - age_index[row["age_bucket"]]),
                                            row["image_sha256"]))
        digest = hashlib.sha256((black["image_sha256"] + white["image_sha256"]).encode()).hexdigest()[:20]
        paired.append(("oct03-rematch-" + digest, [black, white]))
        used.update(row["image_sha256"] for row in (black, white))
    strata = defaultdict(list)
    for block in paired:
        strata[block[1][0]["expression_class"]].append(block)
    if len(strata["negative"]) < 6:
        raise ValueError("Need six disjoint negative-expression blocks: three train, three development")
    prepared = []
    for expression, blocks in sorted(strata.items()):
        blocks.sort(key=lambda block: hashlib.sha256(("oct03-iat-only-v1|" + block[0]).encode()).hexdigest())
        development_count = max(3 if expression == "negative" else 1, round(len(blocks) * .25))
        if len(blocks) <= development_count:
            raise ValueError(f"Too few {expression} blocks to cover both splits")
        for index, (block_id, block) in enumerate(blocks):
            for row in block:
                prepared.append({**row, "original_match_block_id": row["match_block_id"],
                                 "match_block_id": block_id, "source_split": row.get("split", ""),
                                 "split": "development" if index < development_count else "train",
                                 "prompt": IAT_PROMPT})
    excluded = sorted(row["pair_id"] for row in rows if row["image_sha256"] not in used)
    return prepared, excluded


def prepare(asset_root=None):
    if not dataset_config():
        raise ValueError("Active dataset configuration is missing; refusing historical fallback")
    import curated
    return curated.prepare(asset_root)


def prepare_legacy(asset_root=None):
    """Archived 54-image preparation; not the default after the approved switch."""
    recovered = ROOT / "data/private/assets"
    asset_root = asset_root or (recovered if recovered.is_dir() else REPO)
    source = SOURCE / "all_claude_eligible_pairs.csv"
    all_rows = read_csv(source)
    forbidden = {row["sha256"] for row in read_csv(REPO / "2026-09-11/data/iat12/manifest.csv")}
    rows = [row for row in all_rows if row["task_type"] == "iat_3word"]
    failures = {row["pair_id"]: label_failures(row, forbidden) for row in rows}
    eligible = [row for row in rows if not failures[row["pair_id"]]]
    if len({row["image_sha256"] for row in eligible}) != len(eligible):
        raise ValueError("Repeated training images require an explicit resolution")
    prepared, unmatched = match_and_split(eligible)
    output = ROOT / "data/pairs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prepared[0]))
        writer.writeheader()
        writer.writerows(prepared)
    missing, verified = [], []
    for row in prepared:
        image = asset_root / row["image_path"]
        if image_matches_manifest(row, image):
            verified.append(row["image_sha256"])
        else:
            missing.append({"pair_id": row["pair_id"], "image_path": row["image_path"],
                            "image_sha256": row["image_sha256"]})
    train_hashes = {row["image_sha256"] for row in prepared if row["split"] == "train"}
    dev_hashes = {row["image_sha256"] for row in prepared if row["split"] == "development"}
    if train_hashes & dev_hashes or (train_hashes | dev_hashes) & forbidden:
        raise ValueError("Training/development/benchmark images overlap")
    manifest = {"created_at": now(), "source_file": str(source.relative_to(REPO)),
                "asset_root": str(asset_root.resolve()),
                "source_sha256": sha(source), "pairs_sha256": sha(output),
                "profile_rows_used": 0, "source_iat_rows": len(rows), "selected_rows": len(prepared),
                "split_counts": dict(Counter(row["split"] for row in prepared)),
                "split_expression_counts": dict(Counter(row["split"] + "/" + row["expression_class"] for row in prepared)),
                "split_race_counts": dict(Counter(row["split"] + "/" + row["fairface_race"] for row in prepared)),
                "excluded_label_failures": {key: value for key, value in failures.items() if value},
                "excluded_unmatched_pairs": unmatched, "verified_images": len(verified),
                "missing_images": missing, "image_hash_overlap": 0,
                "training_ready": not missing, "training_performed": False,
                "target_origin": "Existing controlled templates judged by the recorded claude-opus-5 image judge",
                "limitations": ["AI-judged templates, not human ground truth",
                                "Neutral targets favor positive words",
                                "Selected negative-expression rows are sad faces only; angry/fear rows were unmatched",
                                "Exact image hashes are disjoint; identity-level and perceptual deduplication are not established",
                                "This is a new IAT-only pilot, not the old mixed-profile DPO dataset"],
                "split_policy": "Matched race blocks; same target/gender/expression; age buckets and intensity differ by <=1; expression confidence >=0.7; deterministic expression-stratified split, >=3 negative blocks in each split",
                "training_inputs": {"sft": ["image", "prompt", "chosen"],
                                    "orpo": ["image", "prompt", "chosen", "rejected"]}}
    save_json(ROOT / "data/manifest.json", manifest)
    missing_path = ROOT / "data/missing_images.csv"
    with missing_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_id", "image_path", "image_sha256"])
        writer.writeheader()
        writer.writerows(missing)
    print(json.dumps({key: value for key, value in manifest.items() if key != "missing_images"}, indent=2))
    print(f"Missing or mismatched training images: {len(missing)}")
    return 0 if not missing else 2


def training_example(row, image_data_url, method, surface="training-api"):
    if surface not in ("training-api", "managed"):
        raise ValueError("Surface must be training-api or managed")
    if row["task_type"] != "iat_3word" or row["prompt"] != IAT_PROMPT:
        raise ValueError("Only the unchanged IAT prompt and IAT targets may be exported")
    if not exact_target(row["chosen"]) or not exact_target(row["rejected"]):
        raise ValueError("Invalid three-word target")
    user = {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": IAT_PROMPT},
    ]}
    chosen = {"messages": [user, {"role": "assistant", "content": row["chosen"]}]}
    if method == "sft":
        return chosen
    if method == "orpo":
        if surface == "managed":
            # This is the managed preference schema, not evidence of model-specific image support.
            return {"input": {"messages": [user], "tools": []},
                    "preferred_output": [{"role": "assistant", "content": row["chosen"]}],
                    "non_preferred_output": [{"role": "assistant", "content": row["rejected"]}]}
        return {"chosen": chosen, "rejected": {"messages": [
            user, {"role": "assistant", "content": row["rejected"]}]}}
    raise ValueError("Method must be sft or orpo")


def export(asset_root=None, surface="training-api"):
    if surface not in ("training-api", "managed"):
        raise ValueError("Surface must be training-api or managed")
    config = dataset_config()
    recovered = ROOT / "data/private/assets"
    asset_root = asset_root or (REPO / config["asset_root"] if config
                                else recovered if recovered.is_dir() else REPO)
    # Fail before embedding or uploading anything unless every selected image is verified.
    if prepare(asset_root) != 0:
        print("Export refused: active image verification and approved IAT targets must both pass.")
        return 2
    if config:
        import curated
        rows = curated.training_rows(config)
    else:
        rows = read_csv(ROOT / "data/pairs.csv")
    output = ROOT / "data/upload" / (config["dataset_id"] if config else "legacy54") / surface
    output.mkdir(parents=True, exist_ok=True)
    exports = {}
    for method in ("sft", "orpo"):
        for split in ("train", "development"):
            path = output / f"{method}_{split}.jsonl"
            count = 0
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    if row["split"] != split:
                        continue
                    image = (asset_root or REPO) / row["image_path"]
                    if image.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                        raise ValueError("Unsupported training image type")
                    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
                    data_url = f"data:{mime};base64," + base64.b64encode(image.read_bytes()).decode("ascii")
                    handle.write(json.dumps(training_example(row, data_url, method, surface)) + "\n")
                    count += 1
            exports[str(path.relative_to(ROOT))] = {"rows": count, "sha256": sha(path), "surface": surface}
    manifest_path = ROOT / "data/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("exports", {}).update(exports)
    save_json(manifest_path, manifest)
    print(json.dumps({"exports": exports, "uploaded": False, "training_performed": False}, indent=2))
    return 0


def audit(asset_root=None):
    if not dataset_config():
        raise ValueError("Active dataset configuration is missing; refusing historical fallback")
    import curated
    return curated.prepare(asset_root)


def audit_legacy(asset_root=None):
    """Archived historical label audit, kept separate from the active cohort."""
    recovered = ROOT / "data/private/assets"
    asset_root = asset_root or (recovered if recovered.is_dir() else REPO)
    source_files = ("mixed_dpo_pairs_frozen.csv", "all_claude_eligible_pairs.csv")
    evaluation = read_csv(REPO / "2026-09-11/data/iat12/manifest.csv")
    forbidden = {row["sha256"] for row in evaluation}
    report = {"checked_at": now(), "training_performed": False, "profile_targets_used": 0,
              "prompt_sha256": hashlib.sha256(IAT_PROMPT.encode()).hexdigest(), "sources": []}
    for filename in source_files:
        path = SOURCE / filename
        all_rows = read_csv(path)
        rows = [row for row in all_rows if row["task_type"] == "iat_3word"]
        entries = []
        for row in rows:
            failures = label_failures(row, forbidden)
            relative = Path(row["image_path"])
            image = (asset_root / relative) if asset_root else REPO / relative
            exists = image.is_file()
            matches = exists and sha(image) == row["image_sha256"]
            entries.append({"pair_id": row["pair_id"], "match_block_id": row["match_block_id"],
                            "image_sha256": row["image_sha256"], "source_image_path": row["image_path"],
                            "race": row["fairface_race"], "gender": row["fairface_gender"],
                            "expression": row["expression_class"], "original_split": row.get("split", ""),
                            "chosen": row["chosen"], "rejected": row["rejected"],
                            "label_provenance": row.get("chosen_response_provenance", ""),
                            "judge_model": row.get("judge_model", ""),
                            "historical_label_checks_pass": not failures, "label_failures": failures,
                            "image_present": exists, "image_hash_verified": matches})
        blocks = defaultdict(list)
        for row in rows:
            blocks[row["match_block_id"]].append(row)
        negative_train = {row["match_block_id"] for row in rows
                          if row["split"] == "train" and row["expression_class"] == "negative"}
        report["sources"].append({"file": str(path.relative_to(REPO)), "sha256": sha(path),
                                  "all_rows": len(all_rows), "iat_rows": len(rows),
                                  "excluded_profile_rows": len(all_rows) - len(rows),
                                  "unique_images": len({row["image_sha256"] for row in rows}),
                                  "unique_chosen_targets": len({row["chosen"] for row in rows}),
                                  "expressions": dict(Counter(row["expression_class"] for row in rows)),
                                  "original_split_counts": dict(Counter(row["split"] for row in rows)),
                                  "original_negative_train_blocks": len(negative_train),
                                  "complete_two_row_blocks": sum(len(b) == 2 for b in blocks.values()),
                                  "historical_label_checks_pass": sum(e["historical_label_checks_pass"] for e in entries),
                                  "verified_images": sum(e["image_hash_verified"] for e in entries),
                                  "training_ready": False,
                                  "limitations": ["AI-judged controlled templates, not human ground truth",
                                                  "Targets favor positive terms even for neutral faces",
                                                  "A fresh expression-stratified block split is required",
                                                  "Images must be restored and hash-verified before training"],
                                  "rows": entries})
    save_json(ROOT / "results/data_audit.json", report)
    print(json.dumps({"sources": [{k: v for k, v in s.items() if k != "rows"} for s in report["sources"]]}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "audit", "prepare", "export", "auth"))
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--surface", choices=("training-api", "managed"), default="training-api",
                        help="Export format only; does not upload data or launch training")
    args = parser.parse_args()
    if args.action == "preflight":
        return preflight()
    if args.action == "auth":
        _, source = ensure_api_key()
        print(json.dumps({"api_key_available": True, "source": source, "secret_logged": False}))
        return 0
    if args.action == "export":
        return export(args.asset_root, args.surface)
    return prepare(args.asset_root) if args.action == "prepare" else audit(args.asset_root)


if __name__ == "__main__":
    raise SystemExit(main())
