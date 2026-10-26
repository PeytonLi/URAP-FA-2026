"""Prepare the approved September 29 image pool without inventing IAT labels."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import study
from multimodel_bias.constants import POSITIVE_IAT_SET

FORBIDDEN_CHOSEN = {"Sick", "Poison", "Evil", "Selfish", "Abuse", "Failure",
                    "Dirty", "Rotten", "Hatred", "Hate", "Humiliate"}


def preference_variants(target):
    return target.get("preference_variants", [
        {branch: target.get(branch) for branch in ("chosen", "rejected")}])


def variant_failures(target, require_word_support=True):
    variants = preference_variants(target)
    if (not isinstance(variants, list) or not 2 <= len(variants) <= 4
            or not all(isinstance(v, dict) for v in variants)):
        return ["invalid_preference_variants"]
    failures, seen = [], set()
    evidence = target.get("approved_candidates", {})
    if not isinstance(evidence, dict):
        return ["missing_candidate_evidence"]
    for variant in variants:
        if set(variant) != {"chosen", "rejected"}:
            failures.append("variant_schema")
        if not all(study.exact_target(variant.get(b, "")) for b in ("chosen", "rejected")):
            failures.append("variant_format")
            continue
        chosen, rejected = (set(variant[b].split(", ")) for b in ("chosen", "rejected"))
        if chosen & FORBIDDEN_CHOSEN:
            failures.append("unsupported_preferred_judgment")
        if chosen == rejected:
            failures.append("variant_identical_responses")
        key = tuple(sorted(chosen))
        if key in seen:
            failures.append("permutation_only_variant")
        seen.add(key)
        if chosen & rejected:
            failures.append("chosen_rejected_overlap")
        if not require_word_support:
            continue
        for word in chosen:
            support = evidence.get(word)
            if (not isinstance(support, list) or len(support) != 2
                    or not all(isinstance(s, dict) for s in support)):
                failures.append("missing_independent_word_support")
                continue
            teachers = [s.get("teacher") for s in support]
            if not all(isinstance(t, str) for t in teachers) or len(set(teachers)) != 2:
                failures.append("missing_independent_word_support")
            for item in support:
                confidence = item.get("confidence")
                if (isinstance(confidence, bool) or not isinstance(confidence, (float, int))
                        or not 0.8 <= confidence <= 1
                        or item.get("word") != word
                        or not all(isinstance(item.get(k), str) and item[k].strip()
                                   for k in ("visible_evidence", "api_id", "request_id", "teacher"))):
                    failures.append("invalid_word_support")
    if any(target.get(b) != variants[0].get(b) for b in ("chosen", "rejected")):
        failures.append("primary_variant_mismatch")
    return sorted(set(failures))


def validate_rows(rows, config, forbidden=()):
    """Check frozen allocations; never rematch, resplit, or filter the pool."""
    if Counter(row["dataset"] for row in rows) != config["dataset_counts"]:
        raise ValueError("Published dataset counts changed")
    if Counter(row["split"] for row in rows) != config["split_counts"]:
        raise ValueError("Published research split counts changed")
    for field in ("image_id", "image_sha256", "pixel_sha256"):
        if len({row[field] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate {field} in selected pool")
    forbidden = set(forbidden)
    blocks = defaultdict(list)
    for row in rows:
        relative = PurePosixPath(row["image_path"])
        if ("\\" in row["image_path"] or relative.is_absolute() or ".." in relative.parts
                or len(relative.parts) != 3 or relative.parts[:2] != ("images", row["dataset"])
                or relative.name != row["image_id"] + ".png"):
            raise ValueError("Unsafe or inconsistent selected image path")
        if row["image_sha256"] in forbidden:
            raise ValueError("Original 12-image benchmark overlaps the selected pool")
        blocks[row["match_set_id"]].append(row)
    for block in blocks.values():
        if len({row["split"] for row in block}) != 1:
            raise ValueError("A published matching set crosses research splits")
        if len({(row["dataset"], row["age_band"], row["expression_class"]) for row in block}) != 1:
            raise ValueError("A matching set has inconsistent composition")
        if len({(row["source_race"], row["source_gender"]) for row in block}) != len(block):
            raise ValueError("A matching set repeats a demographic cell")


def selected_rows(config):
    source = study.REPO / config["manifest_path"]
    if study.sha(source) != config["manifest_sha256"]:
        raise ValueError("Pinned selected_images.csv checksum changed; refusing silent dataset drift")
    rows = study.read_csv(source)
    benchmark = study.read_csv(study.REPO / config["primary_benchmark"])
    validate_rows(rows, config, {row["sha256"] for row in benchmark})
    return rows


def ai_approval_authorization(source=None):
    """Fingerprint the user's explicit waiver; never treat it as human review."""
    try:
        permission = json.loads((study.ROOT / "config/prime_budget.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None  # fail closed: no waiver file, no AI approval
    if permission.get("human_target_approval_required") is not False:
        return None
    authorization = (permission.get("ai_target_approval_candidates_v3", {})
                     if source == "two_pass_candidates_v3" else
                     permission.get("ai_target_approval_matched_v4", {})
                     if source == "matched_block_answers_v4" else
                     permission.get("ai_target_approval_gated_two_pass")
                     or permission.get("ai_target_approval", {}))
    if (authorization.get("authorized") is not True
            or not authorization.get("authorization_source", "").startswith("User:")):
        return None
    return hashlib.sha256(json.dumps(authorization, sort_keys=True).encode()).hexdigest()


def target_failures(target, image, ai_authorization_sha256=None):
    failures = []
    if image["split"] == "test":
        failures.append("heldout_test_target_forbidden")
    for key in ("image_id", "image_sha256", "split"):
        if target.get(key) != image[key]:
            failures.append(key + "_mismatch")
    reviewer = target.get("reviewer")
    if (target.get("review_status") != "approved"
            or not isinstance(reviewer, str) or not reviewer.strip()):
        failures.append("approval_missing")
    kind = target.get("reviewer_kind", "human")
    if kind == "ai":
        if (not ai_authorization_sha256
                or target.get("approval_authorization_sha256") != ai_authorization_sha256):
            failures.append("ai_approval_not_authorized")
        if target.get("draft_source") not in (
                "codex_direct_visual_review", "two_pass_gated_teachers",
                "two_pass_gated_teachers_v2", "two_pass_candidates_v3",
                "matched_block_answers_v4"):
            failures.append("ai_approval_requires_direct_inspection")
    elif kind != "human":
        failures.append("unknown_reviewer_kind")
    provenance = target.get("target_provenance")
    if not isinstance(provenance, str) or not provenance.strip():
        failures.append("target_provenance_missing")
    if target.get("prompt", study.IAT_PROMPT) != study.IAT_PROMPT:
        failures.append("prompt_mismatch")
    for branch in ("chosen", "rejected"):
        if not study.exact_target(target.get(branch, "")):
            failures.append(branch + "_format")
    if (study.exact_target(target.get("chosen", "")) and study.exact_target(target.get("rejected", ""))
            and set(target["chosen"].split(", ")) == set(target["rejected"].split(", "))):
        failures.append("identical_responses")
    if target.get("draft_source") == "matched_block_answers_v4":
        failures.extend(variant_failures(target, require_word_support=False))
    elif "preference_variants" in target or target.get("draft_source") == "two_pass_candidates_v3":
        failures.extend(variant_failures(target))
    return failures


def reviewed_targets(rows, path):
    """Require complete approved train/dev targets; no labels for untouched test."""
    wanted = {row["image_id"]: row for row in rows if row["split"] in ("train", "development")}
    if not path.is_file():
        return {}, {"file_present": False, "approved_records": 0, "required_records": len(wanted),
                    "missing_records": len(wanted), "errors": [], "ready": False}
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Target file must be a JSON list of image-specific records")
    images = {row["image_id"]: row for row in rows}
    approved, seen, errors = {}, set(), []
    sources = {
        record.get("draft_source")
        for record in records
        if isinstance(record, dict) and record.get("reviewer_kind") == "ai"
        and isinstance(record.get("draft_source"), str)}
    ai_authorizations = {source: ai_approval_authorization(source) for source in sources}
    for index, target in enumerate(records):
        if not isinstance(target, dict):
            errors.append({"index": index, "failures": ["record_not_object"]})
            continue
        identity = target.get("image_id")
        if not isinstance(identity, str):
            errors.append({"index": index, "failures": ["image_id_not_string"]})
            continue
        if identity not in images:
            failures = ["unknown_image_id"]
        else:
            source = target.get("draft_source")
            failures = target_failures(
                target, images[identity],
                ai_authorizations.get(source) if isinstance(source, str) else None)
        if identity in seen:
            failures.append("duplicate_image_id")
        seen.add(identity)
        if failures:
            errors.append({"image_id": identity, "failures": failures})
        else:
            approved[identity] = target
    return approved, {"file_present": True, "sha256": study.sha(path),
                      "approved_records": len(approved), "required_records": len(wanted),
                      "human_approved_records": sum(t.get("reviewer_kind", "human") == "human" for t in approved.values()),
                      "ai_approved_records": sum(t.get("reviewer_kind") == "ai" for t in approved.values()),
                      "missing_records": len(wanted.keys() - approved.keys()), "errors": errors,
                      "ready": not errors and approved.keys() == wanted.keys()}


def training_rows(config):
    rows = selected_rows(config)
    targets, audit = reviewed_targets(rows, study.REPO / config["targets_path"])
    if not audit["ready"]:
        raise ValueError("Approved image-specific train/development targets are incomplete")
    if config["dataset_id"] in study.MULTITRIPLE_DATASETS:
        validate_multitriple_blocks(rows, targets, config)
    return [{**row, "task_type": "iat_3word", "prompt": study.IAT_PROMPT,
             "chosen": variant["chosen"],
             "rejected": variant["rejected"],
             "reviewer_kind": targets[row["image_id"]].get("reviewer_kind", "human"),
             "target_provenance": targets[row["image_id"]]["target_provenance"],
             "approval_authorization_sha256": targets[row["image_id"]].get("approval_authorization_sha256")}
            for row in rows if row["split"] in ("train", "development")
            for variant in preference_variants(targets[row["image_id"]])]


def validate_multitriple_blocks(rows, targets, config):
    if config.get("training_methods") != ["orpo"]:
        raise ValueError("Candidate dataset requires ORPO-only training")
    parent = study.read_csv(study.REPO / config["parent_manifest_path"])
    if study.sha(study.REPO / config["parent_manifest_path"]) != config["parent_manifest_sha256"]:
        raise ValueError("Candidate parent manifest changed")
    if config.get("target_races") != ["Black", "White"]:
        raise ValueError("Candidate race subset must be explicit")
    original = defaultdict(set)
    parent_rows = {r["image_id"]: r for r in parent}
    selected = defaultdict(list)
    for row in parent:
        if row["source_race"] in config["target_races"]:
            original[row["match_set_id"]].add(row["image_id"])
    for row in rows:
        if row != parent_rows.get(row["image_id"]):
            raise ValueError("Candidate source metadata or published split changed")
        selected[row["match_set_id"]].append(row)
        if targets[row["image_id"]].get("image_pixel_sha256") != row["pixel_sha256"]:
            raise ValueError("Candidate target pixel identity mismatch")
        if len(preference_variants(targets[row["image_id"]])) != config["variants_per_image"]:
            raise ValueError("Candidate variants per image changed")
    for sid, members in selected.items():
        if {r["image_id"] for r in members} != original[sid]:
            raise ValueError("Partial matching block in candidate dataset")
        races = Counter(r["source_race"] for r in members)
        if set(races) != {"Black", "White"} or races["Black"] != races["White"]:
            raise ValueError("Race-unbalanced candidate block")
        schedules = [tuple(len(set(v["chosen"].split(", ")) & POSITIVE_IAT_SET)
                           for v in preference_variants(targets[r["image_id"]]))
                     for r in members]
        if len(set(schedules)) != 1:
            raise ValueError("Valence-unbalanced candidate variants")
        if config.get("identical_answers_within_block"):
            answer_sets = {tuple(frozenset(v["chosen"].split(", "))
                                 for v in preference_variants(targets[r["image_id"]]))
                           for r in members}
            if len(answer_sets) != 1:
                raise ValueError("Matched block members have different answers")
    dev_blocks = sum(members[0]["split"] == "development"
                     for members in selected.values())
    if dev_blocks < 3:
        raise ValueError("At least three complete development blocks required")
    actual = Counter(r["split"] for r in rows for _ in preference_variants(targets[r["image_id"]]))
    if actual != config.get("example_split_counts"):
        raise ValueError("Candidate example counts changed")


def prepare(asset_root=None):
    config = study.dataset_config()
    if not config:
        raise ValueError("No active curated dataset configured")
    rows = selected_rows(config)
    asset_root = Path(asset_root or study.REPO / config["asset_root"]).resolve()
    missing, mismatched, verified = [], [], []
    for row in rows:
        image = (asset_root / row["image_path"]).resolve()
        if not image.is_relative_to(asset_root):
            raise ValueError("Selected image path escapes the asset root")
        if not image.is_file():
            missing.append(row["image_id"])
        elif not study.image_matches_manifest(row, image):
            mismatched.append(row["image_id"])
        else:
            verified.append(row)
    _, target_audit = reviewed_targets(rows, study.REPO / config["targets_path"])
    report = {"checked_at": study.now(), "dataset_id": config["dataset_id"],
              "source_commit": config.get("source_commit"), "source_file": config["manifest_path"],
              "source_sha256": config["manifest_sha256"], "asset_root": str(asset_root),
              "selected_rows": len(rows), "split_counts": config["split_counts"],
              "dataset_counts": config["dataset_counts"],
              "verified_images": len(verified),
              "verified_split_counts": dict(Counter(row["split"] for row in verified)),
              "missing_images": missing, "mismatched_images": mismatched,
              "published_splits_preserved": True, "profile_rows_used": 0,
              "benchmark_images_used_for_training": 0,
              "prompt_sha256": hashlib.sha256(study.IAT_PROMPT.encode()).hexdigest(),
              "target_audit": target_audit,
              "data_ready": not missing and not mismatched and target_audit["ready"],
              "training_ready": False, "backend_verification_required": True,
              "training_performed": False, "superseded_pairs": config["superseded_pairs"],
              "limitations": ["Publisher demographic annotations, not inferred identities",
                              "Expression labels are automated, not human-approved IAT targets",
                              "Image-disjoint splits do not establish person-disjointness",
                              "Exact benchmark checksum exclusion is not an identity-overlap guarantee"],
              "training_inputs": config["training_inputs"]}
    rebuild_receipt = asset_root / "rebuild_verification.json"
    if rebuild_receipt.is_file():
        report["reconstruction_receipt"] = {
            "path": str(rebuild_receipt), "sha256": study.sha(rebuild_receipt),
            "reported_result": json.loads(rebuild_receipt.read_text(encoding="utf-8")),
        }
    active_path = study.ROOT / "data/manifest.json"
    archive = study.ROOT / "data/legacy_54_manifest.json"
    if active_path.is_file() and not archive.exists():
        previous = json.loads(active_path.read_text(encoding="utf-8"))
        if previous.get("dataset_id") != config["dataset_id"]:
            study.save_json(archive, previous)
    study.save_json(active_path, report)
    study.save_json(study.ROOT / "results/curated_data_audit.json", report)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("missing_images", "mismatched_images")}, indent=2))
    print(f"Missing images: {len(missing)}; hash mismatches: {len(mismatched)}")
    return 0 if report["data_ready"] else 2
