"""Explicitly authorized resolved-target pilot; never rewrite the upstream pool."""
import argparse
import json
import math
from collections import Counter

import curated
import review_targets
import study
from multimodel_bias.constants import NEGATIVE_IAT_SET, POSITIVE_IAT_SET

POLICY = study.ROOT / "config/balanced_pilot.json"
FROZEN = study.ROOT / "data/private/balanced_pilot/cohort.json"


def balance_weights(counts):
    if set(counts) != {"positive", "negative"} or any(
            isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in counts.values()):
        raise ValueError("Both target categories require positive integer training counts")
    total = sum(counts.values())
    return {kind: total / (2 * count) for kind, count in counts.items()}


def current_cohort():
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if (policy.get("authorized") is not True or policy.get("authorization_source") !=
            "User: do the recommendation and ORPO/SFT now" or
            policy.get("test_or_benchmark_training_allowed") is not False):
        raise ValueError("Resolved balanced pilot requires its explicit authorization")
    store = review_targets.ReviewStore()
    drafts, decisions = store.drafts(), store.decisions()
    wanted = {r["image_id"] for r in store.rows if r["split"] in ("train", "development")}
    if len(drafts) != len(wanted) or {d["image_id"] for d in drafts} != wanted:
        raise ValueError("Every train/development image must retain its direct inspection")
    retained, excluded = [], []
    for draft in drafts:
        if draft.get("draft_source") != "codex_direct_visual_review":
            raise ValueError("No historical teacher fallback in this pilot")
        if draft.get("target_unresolved"):
            excluded.append({"image_id": draft["image_id"], "split": draft["split"],
                             "reason": draft["teacher_expression"]})
        elif decisions.get(draft["image_id"], {}).get("status") == "approved":
            if draft["teacher_expression"] not in ("positive", "negative"):
                raise ValueError("Pilot needs a resolved positive/negative direct target")
            retained.append(draft)
        else:
            raise ValueError("A resolved target lost approval; do not silently shrink the pilot")
    ids = {d["image_id"] for d in retained}
    images = [r for r in store.rows if r["image_id"] in ids]
    records = review_targets.approved_records(
        images, retained, {i: decisions[i] for i in ids}, require_direct=True,
        ai_authorization_sha256=curated.ai_approval_authorization())
    by_id = {r["image_id"]: r for r in records}
    categories = {d["image_id"]: d["teacher_expression"] for d in retained}
    counts = {split: dict(Counter(categories[r["image_id"]] for r in images if r["split"] == split))
              for split in ("train", "development")}
    if (dict(Counter(r["split"] for r in images)) != policy["split_counts"] or
            counts["train"] != policy["training_class_counts"] or
            counts["development"] != policy["development_class_counts"]):
        raise ValueError("Authorized pilot counts changed; explicit re-freeze is required")
    weights = balance_weights(counts["train"])
    asset_root = (study.REPO / store.config["asset_root"]).resolve()
    rows = []
    for image in images:
        target, category = by_id[image["image_id"]], categories[image["image_id"]]
        if not set(target["chosen"].split(", ")).issubset(
                POSITIVE_IAT_SET if category == "positive" else NEGATIVE_IAT_SET):
            raise ValueError("Preferred target words disagree with the balancing category")
        path = (asset_root / image["image_path"]).resolve()
        if not path.is_relative_to(asset_root) or study.sha(path) != image["image_sha256"]:
            raise ValueError("Retained pilot image hash mismatch")
        rows.append({**image, **target, "task_type": "iat_3word", "balance_class": category,
                     "training_weight": weights[category] if image["split"] == "train" else 1.0})
    return {"dataset_id": policy["dataset_id"], "policy_sha256": study.sha(POLICY),
            "source_manifest_sha256": store.config["manifest_sha256"],
            "direct_reviews_sha256": study.sha(review_targets.drafting.PRIVATE / "codex_direct_reviews.json"),
            "ai_approval_authorization_sha256": curated.ai_approval_authorization(),
            "split_counts": policy["split_counts"], "class_counts": counts, "class_weights": weights,
            "rows": rows, "excluded": excluded, "original_splits_preserved": True,
            "original_matching_balance_preserved": False, "test_images_used": 0,
            "profile_targets_used": 0, "targets_are_ground_truth": False}


def prepare():
    cohort = current_cohort()
    if FROZEN.exists():
        if json.loads(FROZEN.read_text(encoding="utf-8")) != cohort:
            raise ValueError("Frozen pilot changed; existing cohort must not be overwritten")
    else:
        study.save_json(FROZEN, cohort)
    receipt = {k: v for k, v in cohort.items() if k not in ("rows", "excluded")}
    receipt.update(checked_at=study.now(), cohort_sha256=study.sha(FROZEN),
                   excluded_images=len(cohort["excluded"]), optimizer_updates=0,
                   training_started=False, cohort_path=FROZEN.relative_to(study.REPO).as_posix(),
                   total_training_weight_by_class={k: cohort["class_counts"]["train"][k] * w
                                                   for k, w in cohort["class_weights"].items()})
    if any(not math.isclose(v, 338.0) for v in receipt["total_training_weight_by_class"].values()):
        raise ValueError("Each training category must have equal total loss weight")
    study.save_json(study.ROOT / "results/balanced_pilot_data.json", receipt)
    print(json.dumps(receipt, indent=2))


def training_rows():
    if not FROZEN.is_file():
        raise ValueError("Prepare and freeze the authorized balanced pilot first")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen != current_cohort():
        raise ValueError("Frozen pilot labels, membership, approvals or policy changed")
    config = {**study.dataset_config(), "dataset_id": frozen["dataset_id"],
              "targets_path": FROZEN.relative_to(study.REPO).as_posix(),
              "split_counts": frozen["split_counts"], "pilot_class_weights": frozen["class_weights"],
              "pilot_policy_sha256": frozen["policy_sha256"], "full_cohort_study": False}
    return config, frozen["rows"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare",))
    parser.parse_args()
    prepare()
