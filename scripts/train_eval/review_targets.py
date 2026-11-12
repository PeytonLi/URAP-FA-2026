"""Loopback-only photo review; approvals do not silently become training."""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import curated
import draft_targets as drafting
import study

PAGE = study.ROOT / "target_review.html"
DECISIONS = drafting.PRIVATE / "human_review_decisions.json"
AI_DECISIONS = drafting.PRIVATE / "ai_review_decisions.json"
DIRECT_POLICY = ("Codex inspects the actual frozen image directly, without demographic/classifier labels. "
                 "Describe visible facial features, not actual feelings, health or personality. "
                 "Propose three distinct canonical words per branch only when reasonably grounded; "
                 "leave neutral or ambiguous targets unresolved rather than force positivity. "
                 "Every proposal needs independent human review; no test/benchmark images or profiles.")
DIRECT_POLICY_SHA = hashlib.sha256(DIRECT_POLICY.encode()).hexdigest()


def direct_draft(draft, assessment):
    if any(assessment.get(key) != draft[key] for key in ("image_id", "image_sha256")):
        raise ValueError("Direct inspection does not match the frozen image")
    unresolved = assessment.get("chosen") is None and assessment.get("rejected") is None
    if not unresolved:
        drafting.validate_draft(assessment)
    else:
        if (assessment.get("expression") not in ("neutral", "uncertain", "negative", "positive")
                or any(not isinstance(assessment.get(k), str) or not assessment[k].strip()
                       or len(assessment[k]) > 600 for k in ("visible_evidence", "rejected_reason"))
                or isinstance(assessment.get("confidence"), bool)
                or not isinstance(assessment.get("confidence"), (int, float))
                or not 0 <= assessment["confidence"] <= 1
                or not isinstance(assessment.get("review_flags"), list)
                or not all(isinstance(flag, str) and len(flag) <= 100 for flag in assessment["review_flags"])):
            raise ValueError("Malformed unresolved direct inspection")
    flags = set(assessment["review_flags"])
    if unresolved:
        flags.add("insufficiently_grounded_three_word_target")
    if assessment["expression"] == "neutral":
        flags.add("neutral_vocabulary_gap")
    fingerprint = hashlib.sha256(json.dumps(assessment, sort_keys=True).encode()).hexdigest()
    return {**draft, "chosen": "" if unresolved else ", ".join(assessment["chosen"]),
            "rejected": "" if unresolved else ", ".join(assessment["rejected"]),
            "visible_evidence": assessment["visible_evidence"], "rejected_reason": assessment["rejected_reason"],
            "teacher_expression": assessment["expression"], "subjective_confidence": assessment["confidence"],
            "review_flags": sorted(flags), "review_status": "draft", "reviewer": "",
            "draft_source": "codex_direct_visual_review", "direct_review_pending": False,
            "target_unresolved": unresolved, "draft_revision_sha256": fingerprint,
            "comparison_teacher_receipt": draft["teacher_receipt"],
            "teacher_receipt": "data/private/target_review/codex_direct_reviews.json#" + draft["image_id"],
            "teacher_policy_sha256": DIRECT_POLICY_SHA,
            "target_provenance": "Codex assistant direct visual inspection via view_image(original); "
                                 "exact model revision unavailable; pending human review"}


def validate_decision(value, draft, ai_authorization_sha256=None):
    if not isinstance(value, dict) or value.get("image_id") != draft["image_id"]:
        raise ValueError("Unknown image decision")
    if value.get("status") not in ("approved", "needs_revision"):
        raise ValueError("Choose approve or needs revision")
    kind = value.get("reviewer_kind", "human")
    if kind == "ai":
        if (not ai_authorization_sha256
                or value.get("approval_authorization_sha256") != ai_authorization_sha256):
            raise ValueError("AI approval requires the current explicit user authorization")
        if (draft.get("draft_source") != "codex_direct_visual_review" or draft.get("target_unresolved")
                or any(value.get(branch) != draft.get(branch) for branch in ("chosen", "rejected"))):
            raise ValueError("AI approval requires a resolved, unchanged direct proposal")
    elif kind != "human":
        raise ValueError("Unknown reviewer kind")
    if not isinstance(value.get("reviewer"), str) or not value["reviewer"].strip():
        raise ValueError("Enter your reviewer name first")
    if len(value["reviewer"]) > 120:
        raise ValueError("Reviewer name is too long")
    if value.get("image_seen") is not True:
        raise ValueError("Confirm that you examined the photo")
    if value["status"] == "approved" and draft.get("direct_review_pending"):
        raise ValueError("This image still awaits the required direct Codex inspection")
    for field in ("chosen", "rejected"):
        if value["status"] == "needs_revision" and value.get(field) == "":
            continue
        if not study.exact_target(value.get(field, "")):
            raise ValueError("Each answer needs exactly three distinct canonical words, separated by comma-space")
    if value["chosen"] and set(value["chosen"].split(", ")) == set(value["rejected"].split(", ")):
        raise ValueError("Preferred and rejected answers must use different word sets")
    if value["status"] == "approved" and set(value["chosen"].split(", ")) & drafting.FORBIDDEN_CHOSEN:
        raise ValueError("The approved drafting policy excludes health/moral judgments as preferred targets")
    notes = value.get("notes", "")
    if not isinstance(notes, str) or len(notes) > 2000:
        raise ValueError("Notes must be text of at most 2,000 characters")
    if value["status"] == "approved" and draft["review_flags"] and value.get("flags_reviewed") is not True:
        raise ValueError("Confirm that you considered the review flags")
    return {"image_id": draft["image_id"], "image_sha256": draft["image_sha256"],
            "split": draft["split"], "chosen": value["chosen"], "rejected": value["rejected"],
            "status": value["status"], "reviewer": value["reviewer"].strip(),
            "reviewer_kind": kind, "approval_authorization_sha256":
            value.get("approval_authorization_sha256") if kind == "ai" else None,
            "image_seen": True, "flags_reviewed": value.get("flags_reviewed") is True,
            "notes": notes, "reviewed_at": study.now(),
            "draft_policy_sha256": draft["teacher_policy_sha256"],
            "draft_revision_sha256": draft.get("draft_revision_sha256"),
            "draft_teacher_receipt": draft["teacher_receipt"]}


def approved_records(rows, drafts, decisions, require_direct=False, ai_authorization_sha256=None):
    expected = {row["image_id"]: row for row in rows if row["split"] in ("train", "development")}
    by_id = {record["image_id"]: record for record in drafts}
    if len(by_id) != len(drafts) or by_id.keys() != expected.keys():
        raise ValueError("All 1,920 train/development drafts must exist; no smaller cohort is exported")
    if decisions.keys() - expected.keys():
        raise ValueError("Unknown or held-out images must not have review decisions")
    if require_direct and any(draft.get("draft_source") != "codex_direct_visual_review" for draft in drafts):
        raise ValueError("Every eligible image requires direct Codex inspection; no teacher fallback")
    records = []
    for image in expected.values():
        draft = by_id[image["image_id"]]
        value = decisions.get(image["image_id"])
        if not value or value.get("status") != "approved":
            raise ValueError("Every image requires an explicit human approval or user-authorized AI approval before training export")
        if (draft.get("image_sha256") != image["image_sha256"]
                or draft.get("split") != image["split"]
                or draft.get("review_status") != "draft"
                or value.get("image_sha256") != image["image_sha256"]
                or value.get("split") != image["split"]
                or value.get("draft_policy_sha256") != draft["teacher_policy_sha256"]
                or value.get("draft_revision_sha256") != draft.get("draft_revision_sha256")
                or value.get("draft_teacher_receipt") != draft["teacher_receipt"]):
            raise ValueError("Review does not match the frozen image/split/policy")
        decision = validate_decision(value, draft, ai_authorization_sha256)
        record = {key: decision[key] for key in ("image_id", "image_sha256", "split", "chosen", "rejected", "reviewer",
                                                "reviewer_kind", "approval_authorization_sha256")}
        provenance = "AI self-approved under explicit user authorization; no independent human review" if decision["reviewer_kind"] == "ai" else "human reviewed"
        record.update(review_status="approved", prompt=study.IAT_PROMPT,
                      reviewed_at=value.get("reviewed_at"), review_notes=decision["notes"],
                      target_provenance=draft["target_provenance"].replace("pending human review", provenance)
                      + "; reviewer=" + decision["reviewer"],
                      teacher_receipt=draft["teacher_receipt"],
                      teacher_policy_sha256=draft["teacher_policy_sha256"],
                      draft_source=draft.get("draft_source", "prime_haiku_draft"),
                      draft_revision_sha256=draft.get("draft_revision_sha256"),
                      comparison_teacher_receipt=draft.get("comparison_teacher_receipt"))
        if curated.target_failures(record, image, ai_authorization_sha256):
            raise ValueError("Approved target failed the training audit")
        records.append(record)
    return records


class ReviewStore:
    def __init__(self):
        self.config = study.dataset_config()
        self.rows = curated.selected_rows(self.config)
        self.images = {row["image_id"]: row for row in self.rows if row["split"] != "test"}
        permission = json.loads((study.ROOT / "config/prime_budget.json").read_text())
        self.primary_direct = permission.get("primary_draft_source") == "codex_direct_visual_review"
        self.lock = threading.Lock()

    def drafts(self):
        path = drafting.PRIVATE / "draft_targets.json"
        records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        seen = set()
        for row in records:
            image = self.images.get(row["image_id"])
            if (not image or row["image_id"] in seen or row["image_sha256"] != image["image_sha256"]
                    or row["split"] != image["split"] or row["review_status"] != "draft"):
                raise ValueError("Draft data does not match frozen train/development images")
            seen.add(row["image_id"])
        if getattr(self, "primary_direct", False):
            direct_path = drafting.PRIVATE / "codex_direct_reviews.json"
            direct = json.loads(direct_path.read_text(encoding="utf-8")) if direct_path.exists() else []
            by_id = {row["image_id"]: row for row in direct}
            if len(by_id) != len(direct) or by_id.keys() - seen:
                raise ValueError("Unknown, held-out or duplicate direct inspection")
            records = [direct_draft(row, by_id[row["image_id"]]) if row["image_id"] in by_id else
                       {**row, "draft_source": "Previous Haiku draft; direct inspection pending",
                        "direct_review_pending": True} for row in records]
        return records

    def decisions(self):
        human = json.loads(DECISIONS.read_text(encoding="utf-8")) if DECISIONS.exists() else {}
        if any(v.get("reviewer_kind", "human") != "human" for v in human.values()):
            raise ValueError("Human review file must not contain AI approvals")
        ai = json.loads(AI_DECISIONS.read_text(encoding="utf-8")) if getattr(self, "primary_direct", False) and AI_DECISIONS.exists() else {}
        if any(v.get("reviewer_kind") != "ai" for v in ai.values()):
            raise ValueError("AI review file must identify AI provenance")
        # A subsequent human decision takes precedence, without copying sources between files.
        decisions = {**ai, **human}
        authorization = curated.ai_approval_authorization() if ai else None
        current = {row["image_id"]: row for row in self.drafts()}
        if decisions.keys() - current.keys():
            raise ValueError("Unknown or held-out image decision")
        for identity, value in decisions.items():
            draft = current.get(identity)
            if draft and (value.get("draft_revision_sha256") != draft.get("draft_revision_sha256")
                          or value.get("draft_teacher_receipt") != draft["teacher_receipt"]
                          or value.get("draft_policy_sha256") != draft["teacher_policy_sha256"]
                          or value.get("image_sha256") != draft["image_sha256"]
                          or value.get("split") != draft["split"]
                          or value.get("reviewer_kind") == "ai" and (
                              not authorization or value.get("approval_authorization_sha256") != authorization)):
                decisions[identity] = {**value, "status": "needs_revision", "stale_previous_status": value["status"]}
            elif value.get("status") == "approved":
                try:
                    validate_decision(value, draft, authorization)
                except ValueError:
                    decisions[identity] = {**value, "status": "needs_revision", "stale_previous_status": "approved"}
        return decisions

    def save(self, value):
        with self.lock:
            if value.get("reviewer_kind", "human") != "human":
                raise ValueError("The manual review form saves human decisions only")
            draft = next((row for row in self.drafts() if row["image_id"] == value.get("image_id")), None)
            if not draft:
                raise ValueError("Draft not yet available")
            if (draft.get("draft_revision_sha256")
                    and value.get("draft_revision_sha256") != draft["draft_revision_sha256"]):
                raise ValueError("The direct proposal changed; refresh and inspect it before saving")
            decision = validate_decision(value, draft)
            decisions = json.loads(DECISIONS.read_text(encoding="utf-8")) if DECISIONS.exists() else {}
            decisions[draft["image_id"]] = decision
            study.save_json(DECISIONS, decisions)
            return decision

    def approve_direct_proposals(self):
        """Accept only retained, resolved visual assessments; never fill blank targets."""
        with self.lock:
            authorization = curated.ai_approval_authorization()
            if not authorization or not getattr(self, "primary_direct", False):
                raise ValueError("Direct AI approval has not been explicitly authorized")
            drafts = self.drafts()
            if len(drafts) != len(self.images) or {d["image_id"] for d in drafts} != self.images.keys():
                raise ValueError("The entire frozen train/development cohort must remain present")
            decisions = json.loads(AI_DECISIONS.read_text(encoding="utf-8")) if AI_DECISIONS.exists() else {}
            if decisions.keys() - self.images.keys():
                raise ValueError("Unknown or held-out AI decision")
            asset_root = (study.REPO / self.config["asset_root"]).resolve()
            unresolved = 0
            for draft in drafts:
                image = self.images[draft["image_id"]]
                path = (asset_root / image["image_path"]).resolve()
                if not path.is_relative_to(asset_root) or study.sha(path) != image["image_sha256"]:
                    raise ValueError("Image hash mismatch during AI approval")
                if draft.get("draft_source") != "codex_direct_visual_review":
                    raise ValueError("No teacher fallback or uninspected images may be AI approved")
                if draft.get("target_unresolved"):
                    unresolved += 1
                    continue
                value = {"image_id": draft["image_id"], "status": "approved", "reviewer": "Codex (AI self-approval)",
                         "reviewer_kind": "ai", "approval_authorization_sha256": authorization,
                         "chosen": draft["chosen"], "rejected": draft["rejected"], "image_seen": True,
                         "flags_reviewed": True, "notes": "Accepting the retained direct visual assessment and its uncertainty flags under explicit user authorization. Tentative expression associations, not ground truth or independent human validation."}
                decision = validate_decision(value, draft, authorization)
                previous = decisions.get(draft["image_id"])
                if previous and {**previous, "reviewed_at": None} == {**decision, "reviewed_at": None}:
                    decision = previous
                decisions[draft["image_id"]] = decision
            study.save_json(AI_DECISIONS, decisions)
            return {"ai_approved_records": sum(d["status"] == "approved" for d in self.decisions().values()
                                                if d.get("reviewer_kind") == "ai"),
                    "unresolved_targets": unresolved, "required_records": len(self.images),
                    "authorization_sha256": authorization, "path": str(AI_DECISIONS), "training_started": False}

    def export(self):
        with self.lock:
            records = approved_records(self.rows, self.drafts(), self.decisions(),
                                       require_direct=getattr(self, "primary_direct", False),
                                       ai_authorization_sha256=curated.ai_approval_authorization())
            target = study.REPO / self.config["targets_path"]
            if target.exists():
                # Preserve existing approved records rather than overwrite them.
                raise ValueError("An approved target file already exists; inspect it before replacing")
            study.save_json(target, records)
            return {"approved_records": len(records), "sha256": study.sha(target),
                    "training_started": False, "path": str(target)}


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def allowed(self):
            host = f"127.0.0.1:{self.server.server_address[1]}"
            origin = self.headers.get("Origin")
            return (self.headers.get("Host") == host
                    and (not origin or origin == "http://" + host)
                    and self.headers.get("Sec-Fetch-Site") != "cross-site")

        def send(self, value, status=200, mime="application/json"):
            raw = value if isinstance(value, bytes) else json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'self'")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if not self.allowed():
                return self.send({"error": "Local review origin required"}, 403)
            route = urlsplit(self.path).path
            try:
                if route == "/":
                    return self.send(PAGE.read_bytes(), mime="text/html; charset=utf-8")
                if route == "/api/targets":
                    return self.send({"drafts": store.drafts(), "decisions": store.decisions(),
                                      "required": 1920, "words": list(study.IAT_WORDS)})
                if route.startswith("/image/"):
                    image = store.images.get(unquote(route.removeprefix("/image/")))
                    if not image:
                        return self.send({"error": "Unknown training image"}, 404)
                    asset_root = (study.REPO / store.config["asset_root"]).resolve()
                    path = (asset_root / image["image_path"]).resolve()
                    if not path.is_relative_to(asset_root) or study.sha(path) != image["image_sha256"]:
                        raise ValueError("Image hash mismatch")
                    return self.send(path.read_bytes(), mime="image/png")
                return self.send({"error": "Not found"}, 404)
            except (ValueError, OSError):
                return self.send({"error": "Local review data unavailable or invalid"}, 409)

        def do_POST(self):
            if not self.allowed() or self.headers.get("Origin") != f"http://127.0.0.1:{self.server.server_address[1]}":
                return self.send({"error": "Same-origin review required"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid review request size")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("Review request must be a JSON object")
                if self.path == "/api/decision":
                    return self.send(store.save(value))
                if self.path == "/api/export" and value.get("confirm_all_reviewed") is True:
                    return self.send(store.export())
                return self.send({"error": "Unknown or unconfirmed action"}, 400)
            except (ValueError, KeyError, TypeError) as exc:
                return self.send({"error": str(exc)}, 400)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument("--approve-direct-proposals", action="store_true",
                        help="Record user-authorized AI approvals of resolved direct proposals; no export or training")
    args = parser.parse_args()
    if args.approve_direct_proposals:
        print(json.dumps(ReviewStore().approve_direct_proposals(), indent=2))
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(ReviewStore()))
    print(f"Photo review ready at http://127.0.0.1:{args.port}/; no training will start from draft approval alone", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
