"""Verify retained drafts and paid-request evidence; export a private review CSV."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from decimal import Decimal

import curated
import draft_targets as drafting
import study
import review_targets as review
from multimodel_bias.constants import POSITIVE_IAT_SET, NEGATIVE_IAT_SET


def main():
    config = study.dataset_config()
    rows = curated.selected_rows(config)
    eligible = {row["image_id"]: row for row in rows if row["split"] in ("train", "development")}
    drafts_path = drafting.PRIVATE / "draft_targets.json"
    drafts = json.loads(drafts_path.read_text(encoding="utf-8"))
    if len(drafts) != 1920 or {row["image_id"] for row in drafts} != eligible.keys():
        raise ValueError("Full frozen train/development coverage is required")
    asset_root = (study.REPO / config["asset_root"]).resolve()
    expression_words = defaultdict(Counter)
    chosen_words, rejected_words = Counter(), Counter()
    same_word_sets = []
    for draft in drafts:
        image = eligible[draft["image_id"]]
        path = (asset_root / image["image_path"]).resolve()
        if (not path.is_relative_to(asset_root) or study.sha(path) != image["image_sha256"]
                or any(draft[field] != image[field] for field in ("image_sha256", "split", "image_path"))
                or draft["review_status"] != "draft" or draft["reviewer"] != ""
                or draft["prompt"] != study.IAT_PROMPT):
            raise ValueError("Draft/image/approval/prompt mismatch")
        receipt_path = (study.ROOT / draft["teacher_receipt"]).resolve()
        if not receipt_path.is_relative_to(drafting.PRIVATE.resolve()):
            raise ValueError("Teacher receipt is outside the private review folder")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt["image_sha256"] != image["image_sha256"]
                or receipt["policy_sha256"] != drafting.POLICY_SHA
                or receipt["requested_model"] != drafting.TEACHER
                or receipt["response_schema_sha256"] != drafting.SCHEMA_SHA
                or receipt["finish_reason"] != "stop"):
            raise ValueError("Teacher evidence is inconsistent")
        answer = drafting.validate_draft(json.loads(receipt["content"]))
        for branch in ("chosen", "rejected"):
            if draft[branch] != ", ".join(answer[branch]):
                raise ValueError("Stored words do not match the actual teacher response")
        if (draft["teacher_expression"] != answer["expression"]
                or draft["visible_evidence"] != answer["visible_evidence"]
                or draft["rejected_reason"] != answer["rejected_reason"]):
            raise ValueError("Stored explanatory evidence does not match the response")
        if set(answer["chosen"]) == set(answer["rejected"]):
            same_word_sets.append(image["image_id"])
        chosen_words.update(answer["chosen"])
        rejected_words.update(answer["rejected"])
        expression_words[draft["teacher_expression"]].update(answer["chosen"])
    ledger = json.loads((drafting.PRIVATE / "cost_ledger.json").read_text(encoding="utf-8"))
    if any(row["image_id"] not in eligible for row in ledger):
        raise ValueError("A paid draft request touched an unknown or held-out image")
    if any(row["status"] == "reserved_or_unobserved" for row in ledger):
        raise ValueError("Outstanding requests must be reconciled before handoff")
    approved_path = study.REPO / config["targets_path"]
    decisions_path = drafting.PRIVATE / "human_review_decisions.json"
    decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    current_decisions = review.ReviewStore().decisions()
    wallet = json.loads((drafting.PRIVATE / "wallet_after.json").read_text())
    budget_config = json.loads((study.ROOT / "config/prime_budget.json").read_text())
    csv_path = drafting.PRIVATE / "draft_targets_for_review.csv"
    fields = ["image_id", "split", "dataset", "image_path", "image_sha256", "chosen", "rejected",
              "teacher_expression", "visible_evidence", "rejected_reason", "subjective_confidence",
              "review_flags", "review_status", "reviewer", "teacher_receipt", "teacher_policy_sha256"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for draft in drafts:
            writer.writerow({**draft, "review_flags": " | ".join(draft["review_flags"])})
    def word_stats(counts):
        return {"total_words": sum(counts.values()),
                "positive_words": sum(n for w, n in counts.items() if w in POSITIVE_IAT_SET),
                "negative_words": sum(n for w, n in counts.items() if w in NEGATIVE_IAT_SET),
                "distinct_words": len(counts), "top_words": counts.most_common(10)}
    audit = {"checked_at": study.now(), "manifest_sha256": config["manifest_sha256"],
             "drafts_sha256": study.sha(drafts_path), "review_csv_sha256": study.sha(csv_path),
             "drafts": len(drafts), "split_counts": dict(Counter(d["split"] for d in drafts)),
             "all_images_and_retained_receipts_verified": True, "canonical_word_checks_pass": True,
             "actual_review_decisions": len(current_decisions), "actual_human_review_decisions": len(decisions),
             "actual_ai_review_decisions": sum(d.get("reviewer_kind") == "ai" for d in current_decisions.values()),
             "approved_file_present": approved_path.exists(),
             "human_approved_records": sum(d.get("status") == "approved" and d.get("reviewer_kind", "human") == "human"
                                            for d in current_decisions.values()),
             "ai_approved_records": sum(d.get("status") == "approved" and d.get("reviewer_kind") == "ai"
                                         for d in current_decisions.values()),
             "ai_approval_authorization_sha256": curated.ai_approval_authorization(),
             "paid_draft_requests": len(ledger), "unique_images_requested": len({r["image_id"] for r in ledger}),
             "invalid_request_counts": dict(Counter(r.get("validation_failure", r.get("error_type"))
                                                    for r in ledger if r["status"] == "failed_or_invalid")),
             "unknown_charge_reservations": sum(r.get("unknown_charge_reserved_conservatively", False)
                                                for r in ledger),
             "charge_upper_estimate_usd": str(sum((Decimal(r["charged_upper_estimate_usd"])
                                                  for r in ledger), Decimal(0))),
             "wallet_remaining_usd": wallet["balance_usd"], "wallet_checked_at": wallet["checked_at"],
             "draft_subcap_usd": budget_config["draft_targets_cap"],
             "total_user_spending_cap_usd": budget_config["total_spending_cap"],
             "test_or_benchmark_images_requested": 0, "profiles_used": 0,
             "teacher_expression_counts": dict(Counter(d["teacher_expression"] for d in drafts)),
             "flagged_images": sum(bool(d["review_flags"]) for d in drafts),
             "review_flag_counts": dict(Counter(flag for d in drafts for flag in d["review_flags"])),
             "chosen_word_stats": word_stats(chosen_words), "rejected_word_stats": word_stats(rejected_words),
             "chosen_word_stats_by_teacher_expression": {key: word_stats(value)
                                                         for key, value in expression_words.items()},
             "same_word_set_response_pairs": len(same_word_sets),
             "validity_is_not_visual_correctness": True, "teacher_revision_immutable": False,
             "labels_are_ground_truth": False, "training_started": False,
             "browser_preview_verified": False,
             "browser_preview_limitation": "Codex browser saved-permission verification unavailable; no workaround used"}
    study.save_json(study.ROOT / "results/target_draft_audit.json", audit)
    direct = [row for row in review.ReviewStore().drafts()
              if row.get("draft_source") == "codex_direct_visual_review"]
    direct_csv = drafting.PRIVATE / "codex_direct_reviews_for_review.csv"
    with direct_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["draft_source", "target_unresolved"], extrasaction="ignore")
        writer.writeheader()
        for row in direct:
            writer.writerow({**row, "review_flags": " | ".join(row["review_flags"])})
    progress = {"checked_at": study.now(), "required_images": len(eligible),
                "directly_inspected": len(direct), "remaining_images": len(eligible) - len(direct),
                "proposed_three_word_pairs": sum(not row["target_unresolved"] for row in direct),
                "unresolved_targets": sum(row["target_unresolved"] for row in direct),
                "manifest_sha256": config["manifest_sha256"], "direct_policy_sha256": review.DIRECT_POLICY_SHA,
                "direct_csv_sha256": study.sha(direct_csv), "image_hashes_verified": True,
                "method": "Codex direct visual inspection via view_image(original); exact model revision unavailable",
                "human_approved_records": audit["human_approved_records"], "ai_approved_records": audit["ai_approved_records"],
                "all_images_complete": len(direct) == len(eligible),
                "training_started": False, "test_or_benchmark_images_inspected": 0,
                "image_membership_or_splits_changed": False, "original_teacher_drafts_preserved": True}
    study.save_json(study.ROOT / "results/direct_review_progress.json", progress)
    print(json.dumps(audit, indent=2))
    print(json.dumps(progress, indent=2))


if __name__ == "__main__":
    main()
