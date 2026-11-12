import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import draft_targets as drafting
import review_targets as review
import study


class ReviewTargetTests(unittest.TestCase):
    def setUp(self):
        self.row = {"image_id": "fixture", "image_sha256": "fixture-hash", "split": "train",
                    "image_path": "images/fairface/fixture.png", "dataset": "fairface",
                    "expression_class": "positive"}
        value = {"chosen": ["Happy", "Joyful", "Smiling"], "rejected": ["Angry", "Pain", "Sadness"],
                 "expression": "positive", "visible_evidence": "Mouth corners are raised.",
                 "rejected_reason": "The alternative contradicts the smile.", "confidence": 0.8,
                 "review_flags": []}
        self.draft = drafting.draft_record(self.row, value, "fixture-receipt.json")
        self.value = {"image_id": "fixture", "status": "approved", "reviewer": "test-fixture-only",
                      "chosen": "Happy, Joyful, Smiling", "rejected": "Angry, Pain, Sadness",
                      "image_seen": True, "flags_reviewed": True, "notes": "Fixture, not a real review."}

    def decision(self):
        return review.validate_decision(self.value, self.draft)

    def test_explicit_human_name_photo_and_flag_acknowledgment_are_required(self):
        for change in ({"reviewer": ""}, {"image_seen": False}, {"status": "draft"},
                       {"image_id": "test"}, {"chosen": "Neutral, Happy, Smiling"},
                       {"chosen": "Sick, Evil, Selfish"},
                       {"rejected": "Smiling, Happy, Joyful"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                review.validate_decision({**self.value, **change}, self.draft)
        flagged = {**self.draft, "review_flags": ["neutral_vocabulary_gap"]}
        with self.assertRaisesRegex(ValueError, "review flags"):
            review.validate_decision({**self.value, "flags_reviewed": False}, flagged)

    def test_human_can_edit_the_actual_three_word_answers(self):
        result = review.validate_decision({**self.value, "chosen": "Joyous, Delight, Smiling"}, self.draft)
        self.assertEqual(result["chosen"], "Joyous, Delight, Smiling")
        self.assertEqual(self.draft["review_status"], "draft")

    @unittest.skipUnless(shutil.which("node"), "Node is required for the review-page logic check")
    def test_photo_first_page_logic_uses_fixtures_not_human_decisions(self):
        result = subprocess.run(["rtk", "node", str(Path(__file__).with_name("review_page.test.cjs"))],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_pending_and_needs_revision_do_not_export(self):
        with self.assertRaisesRegex(ValueError, "explicit human approval"):
            review.approved_records([self.row], [self.draft], {})
        decision = {**self.decision(), "status": "needs_revision"}
        with self.assertRaisesRegex(ValueError, "explicit human approval"):
            review.approved_records([self.row], [self.draft], {"fixture": decision})

    def test_all_drafts_required_and_test_records_forbidden(self):
        with self.assertRaisesRegex(ValueError, "drafts must exist"):
            review.approved_records([self.row], [], {})
        test_row = {**self.row, "image_id": "test", "split": "test"}
        test_draft = {**self.draft, "image_id": "test", "split": "test"}
        with self.assertRaises(ValueError):
            review.approved_records([self.row, test_row], [self.draft, test_draft], {})
        with self.assertRaisesRegex(ValueError, "held-out"):
            review.approved_records([self.row, test_row], [self.draft],
                                    {"fixture": self.decision(), "test": self.decision()})

    def test_changed_image_split_policy_receipt_or_status_invalidates_export(self):
        for change in ({"image_sha256": "other"}, {"split": "development"},
                       {"teacher_policy_sha256": "other"}, {"teacher_receipt": "other"},
                       {"review_status": "approved"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                review.approved_records([self.row], [{**self.draft, **change}],
                                        {"fixture": self.decision()})

    def test_export_requires_real_decisions_and_preserves_exact_prompt(self):
        result = review.approved_records([self.row], [self.draft], {"fixture": self.decision()})
        self.assertEqual(result[0]["review_status"], "approved")
        self.assertEqual(result[0]["prompt"], study.IAT_PROMPT)
        self.assertEqual(result[0]["teacher_receipt"], self.draft["teacher_receipt"])
        self.assertIn("human reviewed", result[0]["target_provenance"])

    def test_duplicate_drafts_do_not_export(self):
        with self.assertRaises(ValueError):
            review.approved_records([self.row], [self.draft, copy.deepcopy(self.draft)],
                                    {"fixture": self.decision()})

    def test_direct_inspection_keeps_provenance_abstentions_and_approval_separate(self):
        assessment = {"image_id": "fixture", "image_sha256": "fixture-hash",
                      "expression": "neutral", "chosen": None, "rejected": None,
                      "visible_evidence": "Relaxed closed mouth.", "rejected_reason": "No neutral vocabulary.",
                      "confidence": 0.2, "review_flags": [], "review_status": "approved", "reviewer": "agent"}
        direct = review.direct_draft(self.draft, assessment)
        self.assertEqual(direct["review_status"], "draft")
        self.assertEqual(direct["reviewer"], "")
        self.assertEqual(direct["chosen"], "")
        self.assertTrue(direct["target_unresolved"])
        self.assertIn("neutral_vocabulary_gap", direct["review_flags"])
        decision = review.validate_decision({**self.value, "status": "needs_revision", "chosen": "", "rejected": ""}, direct)
        self.assertEqual(decision["status"], "needs_revision")
        with self.assertRaises(ValueError):
            review.validate_decision({**self.value, "chosen": "", "rejected": ""}, direct)
        with self.assertRaises(ValueError):
            review.direct_draft(self.draft, {**assessment, "image_sha256": "other"})
        with self.assertRaises(ValueError):
            review.validate_decision(self.value, {**self.draft, "direct_review_pending": True})
        with self.assertRaisesRegex(ValueError, "no teacher fallback"):
            review.approved_records([self.row], [self.draft], {"fixture": self.decision()}, require_direct=True)

    def test_direct_proposals_invalidate_approvals_when_the_assessment_changes(self):
        assessment = {"image_id": "fixture", "image_sha256": "fixture-hash", "expression": "positive",
                      "chosen": ["Smiling", "Happy", "Cheerful"], "rejected": ["Sadness", "Angry", "Negative"],
                      "visible_evidence": "Raised mouth corners.", "rejected_reason": "No frown visible.",
                      "confidence": 0.8, "review_flags": []}
        direct = review.direct_draft(self.draft, assessment)
        decision = review.validate_decision(self.value, direct)
        changed = review.direct_draft(self.draft, {**assessment, "chosen": ["Smiling", "Joyful", "Glad"]})
        with self.assertRaisesRegex(ValueError, "frozen image/split/policy"):
            review.approved_records([self.row], [changed], {"fixture": decision}, require_direct=True)

    def test_non_object_request_is_rejected_without_saving(self):
        import io
        from types import SimpleNamespace
        store = SimpleNamespace(save=lambda value: self.fail("Must not save malformed requests"))
        handler = object.__new__(review.make_handler(store))
        handler.server = SimpleNamespace(server_address=("127.0.0.1", 8879))
        handler.headers = {"Origin": "http://127.0.0.1:8879", "Host": "127.0.0.1:8879", "Content-Length": "4"}
        handler.rfile, handler.path = io.BytesIO(b"null"), "/api/decision"
        results = []
        handler.send = lambda value, status=200, mime=None: results.append((value, status))
        handler.do_POST()
        self.assertEqual(results[0][1], 400)

    def test_stale_approval_is_not_presented_as_current_and_is_not_erased(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decisions.json"
            original = {"fixture": self.decision()}
            study.save_json(path, original)
            store = object.__new__(review.ReviewStore)
            store.drafts = lambda: [{**self.draft, "draft_revision_sha256": "new-direct-version"}]
            with patch.object(review, "DECISIONS", path):
                current = store.decisions()
            self.assertEqual(current["fixture"]["status"], "needs_revision")
            self.assertEqual(json.loads(path.read_text())["fixture"]["status"], "approved")

    def test_review_file_is_separate_and_existing_approved_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            drafts = root / "draft_targets.json"
            decisions = root / "human_review_decisions.json"
            approved = root / "approved.json"
            study.save_json(drafts, [self.draft])
            store = object.__new__(review.ReviewStore)
            store.rows, store.images = [self.row], {"fixture": self.row}
            store.config = {"targets_path": str(approved)}
            import threading
            store.lock = threading.Lock()
            with patch.object(drafting, "PRIVATE", root), patch.object(review, "DECISIONS", decisions):
                store.save(self.value)
                self.assertFalse(approved.exists())
                self.assertEqual(json.loads(drafts.read_text())[0]["review_status"], "draft")
                store.export()
                before = approved.read_bytes()
                with self.assertRaisesRegex(ValueError, "already exists"):
                    store.export()
                self.assertEqual(approved.read_bytes(), before)

    def test_authorized_ai_approval_is_separate_revocable_and_never_exports_a_subset(self):
        import threading
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            asset = root / self.row["image_path"]
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"FICTIONAL HASH FIXTURE, NOT A PHOTO")
            row = {**self.row, "image_sha256": study.sha(asset)}
            other = {**row, "image_id": "unresolved", "image_path": "images/fairface/unresolved.png"}
            (root / other["image_path"]).write_bytes(asset.read_bytes())
            assessment = {"image_id": "fixture", "image_sha256": row["image_sha256"], "expression": "positive",
                          "chosen": ["Happy", "Joyful", "Smiling"], "rejected": ["Angry", "Pain", "Sadness"],
                          "visible_evidence": "Fixture only.", "rejected_reason": "Fixture only.",
                          "confidence": 0.8, "review_flags": ["emotion_words_are_tentative"]}
            resolved = review.direct_draft({**self.draft, **row}, assessment)
            unresolved = review.direct_draft({**self.draft, **other},
                                             {**assessment, "image_id": "unresolved", "expression": "uncertain",
                                              "chosen": None, "rejected": None})
            store = object.__new__(review.ReviewStore)
            store.primary_direct, store.lock = True, threading.Lock()
            store.rows, store.images = [row, other], {r["image_id"]: r for r in (row, other)}
            store.config = {"asset_root": str(root), "targets_path": str(root / "approved.json")}
            store.drafts = lambda: [resolved, unresolved]
            human_path, ai_path = root / "human.json", root / "ai.json"
            with patch.object(review, "DECISIONS", human_path), patch.object(review, "AI_DECISIONS", ai_path), \
                    patch.object(review.curated, "ai_approval_authorization", return_value="a" * 64):
                result = store.approve_direct_proposals()
                self.assertEqual((result["ai_approved_records"], result["unresolved_targets"]), (1, 1))
                self.assertFalse(human_path.exists())
                before = ai_path.read_bytes()
                store.approve_direct_proposals()
                self.assertEqual(ai_path.read_bytes(), before)  # Idempotent approval timestamp.
                value = store.decisions()["fixture"]
                exported = review.approved_records([row], [resolved], {"fixture": value}, True, "a" * 64)
                self.assertEqual(exported[0]["reviewer_kind"], "ai")
                self.assertIn("no independent human review", exported[0]["target_provenance"])
                self.assertNotIn("human reviewed", exported[0]["target_provenance"])
                with self.assertRaisesRegex(ValueError, "Every image requires"):
                    store.export()
                self.assertFalse((root / "approved.json").exists())
                with patch.object(review.curated, "ai_approval_authorization", return_value=None):
                    self.assertEqual(store.decisions()["fixture"]["status"], "needs_revision")
                self.assertEqual(ai_path.read_bytes(), before)  # Revocation does not erase evidence.
                with self.assertRaisesRegex(ValueError, "current explicit user authorization"):
                    review.validate_decision(value, resolved)
                with self.assertRaisesRegex(ValueError, "manual review form"):
                    store.save(value)
                store.drafts = lambda: [{**resolved, "draft_revision_sha256": "changed"}, unresolved]
                self.assertEqual(store.decisions()["fixture"]["status"], "needs_revision")
                store.drafts = lambda: [resolved, unresolved]
                store.save({**self.value, "draft_revision_sha256": resolved["draft_revision_sha256"]})
                self.assertEqual(store.decisions()["fixture"]["reviewer_kind"], "human")
                self.assertTrue(all(v["reviewer_kind"] == "human" for v in json.loads(human_path.read_text()).values()))
                self.assertEqual(ai_path.read_bytes(), before)  # Human override leaves separate AI evidence intact.

    def test_ai_approval_cannot_change_words_use_teacher_fallback_or_fill_blanks(self):
        assessment = {"image_id": "fixture", "image_sha256": "fixture-hash", "expression": "positive",
                      "chosen": ["Happy", "Joyful", "Smiling"], "rejected": ["Angry", "Pain", "Sadness"],
                      "visible_evidence": "Fixture only.", "rejected_reason": "Fixture only.",
                      "confidence": 0.8, "review_flags": []}
        draft = review.direct_draft(self.draft, assessment)
        value = {**self.value, "reviewer_kind": "ai", "approval_authorization_sha256": "a" * 64}
        self.assertEqual(review.validate_decision(value, draft, "a" * 64)["reviewer_kind"], "ai")
        for change in ({"target_unresolved": True}, {"draft_source": "prime_haiku_draft"}, {"chosen": "Glad, Happy, Joyful"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "resolved, unchanged"):
                review.validate_decision(value, {**draft, **change}, "a" * 64)


if __name__ == "__main__":
    unittest.main()
