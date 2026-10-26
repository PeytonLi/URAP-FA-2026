"""Offline contracts: honest candidates, split isolation and non-permutation variety."""
from __future__ import annotations

import copy
import csv
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "2026-10-20/labeling"), str(REPO / "2026-10-03")]
import candidates_v3 as cv
import audit_v3
import curated
import export_v3 as ev
import policy
import policy_v3 as pv
import native_train


def row(iid, race="Black", split="train", sid="pair"):
    return {"image_id": iid, "source_race": race, "split": split,
            "match_set_id": sid, "image_sha256": iid * 64,
            "pixel_sha256": "pixels", "dataset": "fixture",
            "expression_class": "positive"}


def value(words=("Happy", "Joyful", "Smiling", "Glad")):
    return {"candidates": [
        {"word": word, "confidence": 0.9, "visible_evidence": "Raised mouth corners."}
        for word in words], "rejected": ["Angry", "Annoy", "Sadness"],
        "rejected_reason": "Tension words do not fit the open smile.",
        "review_flags": [], "review_notes": []}


def record(image, tag="A", words=pv.ELIGIBLE):
    return {"image_id": image["image_id"], "image_sha256": image["image_sha256"],
            "pixel_sha256": image["pixel_sha256"], "split": image["split"],
            "tag": tag, "model": cv.TEACHERS[tag], "evaluated_words": list(words),
            "policy_id": pv.POLICY_ID,
            "policy_sha256": pv.POLICY_SHA, "schema_sha256": pv.SCHEMA_SHA,
            "request_id": "request" + tag, "raw_response": {"id": "api" + tag},
            "status": "valid", "value": value()}


class CandidateTests(unittest.TestCase):
    def test_bitdeer_uses_documented_request_fields_and_separate_teacher_profile(self):
        image = row("a")
        old_records = [record(image, tag) for tag in ("A", "B")]
        try:
            cv.configure("bitdeer")
            self.assertFalse(cv.approved_candidates([image], old_records))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "fixture.png"
                path.write_bytes(b"fixture-only-not-sent")
                with patch.object(cv.common, "image_file", return_value=path):
                    payload = cv.request_payload(image, "A", list(pv.ELIGIBLE))
                    focused = cv.request_payload(image, "A", ["Smiling"])
            self.assertEqual(payload["model"], "deepseek-ai/DeepSeek-V4.1-Flash-1017")
            self.assertEqual(payload["max_tokens"], cv.MAX_TOKENS)
            self.assertNotIn("max_completion_tokens", payload)
            self.assertNotIn("reasoning_effort", payload)
            self.assertEqual(cv.KEY_ENV, "URAP_BITDEER_API_KEY")
            self.assertNotEqual(cv.TEACHERS["A"], cv.TEACHERS["B"])
            word_schema = focused["response_format"]["json_schema"]["schema"][
                "properties"]["candidates"]["items"]["properties"]["word"]
            self.assertEqual(word_schema["enum"], ["Smiling"])
            self.assertEqual(len(pv.RESPONSE_SCHEMA["properties"]["candidates"][
                "items"]["properties"]["word"]["enum"]), 50)
            self.assertIn("THREE DIFFERENT", focused["messages"][1]["content"][1]["text"])
            records = [{**record(image, tag), "teacher_profile": "bitdeer"} for tag in ("A", "B")]
            self.assertIn("a", cv.approved_candidates([image], records))
        finally:
            cv.configure("openai")

    def test_empty_is_valid_not_forced(self):
        self.assertEqual(pv.validate(value(()))["candidates"], [])

    def test_forbidden_and_unknown_and_duplicate_fail(self):
        for words in (("Evil",), ("Neutral",), ("Happy", "Happy")):
            with self.assertRaises(ValueError):
                pv.validate(value(words))

    def test_nan_confidence_fails(self):
        v = value()
        v["candidates"][0]["confidence"] = float("nan")
        with self.assertRaises(ValueError):
            pv.validate(v)

    def test_focus_never_adds_unrequested_words(self):
        with self.assertRaisesRegex(ValueError, "requested"):
            pv.validate(value(), evaluated=["Happy"])

    def test_both_teachers_required(self):
        image = row("a")
        self.assertFalse(cv.approved_candidates([image], [record(image)]))
        approved = cv.approved_candidates([image], [record(image, t) for t in ("A", "B")])
        self.assertEqual(set(approved["a"]["words"]), {"Happy", "Joyful", "Smiling", "Glad"})

    def test_short_candidate_sets_count_for_coverage_not_training_readiness(self):
        image = row("a")
        records = [record(image, t) for t in ("A", "B")]
        for r in records:
            r["value"] = value(("Smiling",))
        supported = cv.approved_candidates([image], records)
        self.assertEqual(supported["a"]["words"], ["Smiling"])
        self.assertFalse(ev.select_variants([image], supported)[0])
        audit = audit_v3.audit([image], records)
        self.assertEqual(audit["support_by_split"]["train"]["union_count"], 1)
        self.assertEqual(audit["support_by_split"]["train"][
            "images_with_at_least_three_supported_words"], 0)
        self.assertFalse(audit["variants"]["2"]["has_minimum_complete_blocks"])

    def test_low_confidence_and_flags_veto(self):
        image = row("a")
        records = [record(image, t) for t in ("A", "B")]
        records[0]["value"]["review_flags"] = ["ambiguous_expression"]
        self.assertFalse(cv.approved_candidates([image], records))
        records[0]["value"]["review_flags"] = []
        for c in records[0]["value"]["candidates"]:
            c["confidence"] = 0.7
        self.assertFalse(cv.approved_candidates([image], records))

    def test_general_notes_do_not_veto_supported_words(self):
        image = row("a")
        records = [record(image, t) for t in ("A", "B")]
        records[0]["value"]["review_notes"] = ["Confidence is uncalibrated."]
        self.assertIn("a", cv.approved_candidates([image], records))
        records[0]["value"]["review_flags"] = ["unreadable_image"]
        self.assertFalse(cv.approved_candidates([image], records))

    def test_blocking_flags_must_be_explicit_and_notes_bounded(self):
        v = value()
        v["review_flags"] = ["Confidence is uncalibrated."]
        with self.assertRaisesRegex(ValueError, "review_flags"):
            pv.validate(v)
        v["review_flags"] = []
        v["review_notes"] = ["x" * 601]
        with self.assertRaisesRegex(ValueError, "review_notes"):
            pv.validate(v)

    def test_prior_protocol_is_preserved_but_never_reinterpreted(self):
        image = row("a")
        old = record(image)
        old.pop("policy_id")
        current = record(image, "B")
        self.assertEqual(cv.active_policy_records([old, current]), [current])
        self.assertFalse(cv.approved_candidates([image], cv.active_policy_records([old, current])))

    def test_latest_curriculum_omission_is_veto(self):
        image = row("a")
        records = [record(image, t) for t in ("A", "B")]
        followup = record(image, "A", words=["Glad"])
        followup["value"] = value(())
        approved = cv.approved_candidates([image], records + [followup])
        self.assertNotIn("Glad", approved["a"]["words"])

    def test_split_hash_and_policy_drift_fail(self):
        image = row("a")
        for field in ("split", "image_sha256", "pixel_sha256", "policy_sha256"):
            bad = record(image)
            bad[field] = "changed"
            with self.assertRaises(ValueError):
                cv.approved_candidates([image], [bad])

    def test_curriculum_ignores_dev_support_and_test(self):
        rows = [row("a"), row("b", "White"), row("c", split="development"),
                row("d", split="test")]
        plan = cv.coverage_plan(rows, {"c": {"words": list(pv.ELIGIBLE)}}, per_word=2)
        self.assertEqual(set(plan), {"a", "b"})
        self.assertEqual(set(plan["a"]), set(pv.ELIGIBLE))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row("a"), row("b", "White")]
        self.approved = {r["image_id"]: {"words": ["Happy", "Joyful", "Smiling", "Glad", "Cheer"]}
                         for r in self.rows}

    def test_four_genuinely_different_triples(self):
        selected, dropped = ev.select_variants(self.rows, self.approved)
        self.assertFalse(dropped)
        for iid, triples in selected.items():
            self.assertEqual(len({frozenset(t) for t in triples}), 4)
            self.assertTrue(all(set(t) <= set(self.approved[iid]["words"]) for t in triples))

    def test_selection_deterministic_including_input_order(self):
        a = ev.select_variants(self.rows, self.approved)
        b = ev.select_variants(list(reversed(self.rows)), self.approved)
        self.assertEqual(a, b)

    def test_unpaired_or_incomplete_drops_whole_block(self):
        selected, dropped = ev.select_variants(self.rows, {"a": self.approved["a"]})
        self.assertFalse(selected)
        self.assertEqual(len(dropped), 1)

    def test_only_one_possible_triple_drops_whole_block(self):
        for proposal in self.approved.values():
            proposal["words"] = ["Happy", "Joyful", "Smiling"]
        self.assertFalse(ev.select_variants(self.rows, self.approved)[0])

    def test_exact_positive_count_matched_per_round(self):
        self.approved["a"]["words"] += ["Angry", "Annoy", "Sadness", "Pain"]
        self.approved["b"]["words"] += ["Angry", "Negative", "Grief", "Pain"]
        selected, _ = ev.select_variants(self.rows, self.approved)
        pos = ev.POSITIVE_IAT_SET
        self.assertEqual([len(set(t) & pos) for t in selected["a"]],
                         [len(set(t) & pos) for t in selected["b"]])

    def test_development_does_not_change_training_export(self):
        train, _ = ev.select_variants(self.rows, self.approved)
        rows = self.rows + [row("c", split="development", sid="dev"),
                            row("d", "White", "development", sid="dev")]
        approved = {**self.approved, "c": self.approved["a"], "d": self.approved["b"]}
        together, _ = ev.select_variants(rows, approved)
        self.assertEqual(train, {k: together[k] for k in train})

    def test_entropy_has_fixed_61_word_denominator(self):
        self.assertAlmostEqual(ev.entropy({"a": 1, "b": 1}),
                               1 / __import__("math").log2(61))


class VariantContractTests(unittest.TestCase):
    def target(self):
        image = row("a")
        support = cv.approved_candidates([image], [record(image, t) for t in ("A", "B")])["a"]["provenance"]
        prefs = [{"chosen": ", ".join(words), "rejected": "Angry, Annoy, Sadness"} for words in (
            ("Happy", "Joyful", "Smiling"), ("Happy", "Glad", "Smiling"))]
        return {**prefs[0], "preference_variants": prefs, "approved_candidates": support}

    def test_valid_two_variants(self):
        self.assertEqual(curated.variant_failures(self.target()), [])

    def test_permutation_only_is_not_variety(self):
        target = self.target()
        target["preference_variants"][1]["chosen"] = "Smiling, Happy, Joyful"
        self.assertIn("permutation_only_variant", curated.variant_failures(target))

    def test_no_teacher_evidence_cannot_be_approved(self):
        target = self.target()
        del target["approved_candidates"]["Glad"]
        self.assertIn("missing_independent_word_support", curated.variant_failures(target))

    def test_same_teacher_twice_is_not_independent(self):
        target = copy.deepcopy(self.target())
        target["approved_candidates"]["Happy"][1]["teacher"] = cv.TEACHERS["A"]
        self.assertIn("missing_independent_word_support", curated.variant_failures(target))

    def test_forbidden_policy_remains_consistent(self):
        self.assertEqual(curated.FORBIDDEN_CHOSEN, policy.FORBIDDEN_CHOSEN)
        self.assertEqual(len(pv.ELIGIBLE), 50)

    def test_study_training_rejects_sft_kto_and_other_models(self):
        for model, method in (("gemma3_12b", "sft"), ("gemma3_12b", "kto"),
                              ("gemma4", "orpo"), ("qwen3vl", "orpo")):
            with self.assertRaisesRegex(ValueError, "ORPO-only"):
                native_train.train(model, method, 60)


class BudgetTests(unittest.TestCase):
    def contexts(self, stack, root):
        stack.enter_context(patch.object(cv, "STORE", root))
        stack.enter_context(patch.object(cv.common, "labelable_rows", return_value=[row("a"), row("b", "White")]))
        stack.enter_context(patch.object(cv.common, "read_csv", return_value=[]))
        stack.enter_context(patch.object(cv.common, "verify_image_file", return_value=True))
        stack.enter_context(patch.object(cv, "request_payload", return_value={}))

    def test_budget_refuses_before_call(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.contexts(stack, Path(directory))
            send = stack.enter_context(patch.object(cv, "send"))
            report = cv.run("pilot", workers=1, cap_usd=0.001)
            self.assertTrue(report["budget_exhausted"])
            send.assert_not_called()

    def test_provider_quota_fails_fast_and_does_not_charge_tokens(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.contexts(stack, Path(directory))
            error = urllib.error.HTTPError(
                "https://api.openai.com", 429, "quota", {}, io.BytesIO(json.dumps({
                    "error": {"code": "credit_balance_exhausted"}}).encode()))
            send = stack.enter_context(patch.object(cv, "send", side_effect=error))
            report = cv.run("pilot", workers=1, cap_usd=1)
            self.assertEqual(send.call_count, 1)
            self.assertEqual(report["failed"], 1)
            self.assertAlmostEqual(report["accounted_usd"], 0)
            self.assertEqual(report["provider_blocked"]["code"], "credit_balance_exhausted")

    def test_network_failure_stops_new_calls_and_keeps_reservation(self):
        from contextlib import ExitStack
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            self.contexts(stack, Path(directory))
            send = stack.enter_context(patch.object(cv, "send", side_effect=TimeoutError))
            report = cv.run("pilot", workers=1, cap_usd=1)
            self.assertEqual(send.call_count, 1)
            self.assertEqual(report["failed"], 1)
            self.assertGreater(report["accounted_usd"], 0)
            self.assertEqual(report["provider_blocked"]["error_type"], "TimeoutError")


class ExportIntegrationTests(unittest.TestCase):
    def test_export_roundtrip_into_trainer_without_duplicate_image_ids(self):
        from contextlib import ExitStack
        import study
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            work = root / "2026-10-20"
            store = work / "data/private/target_review/candidates_v3"
            config_dir = root / "2026-10-03/config"
            config_dir.mkdir(parents=True)
            store.mkdir(parents=True)
            (work / "config").mkdir()
            permission = json.loads((REPO / "2026-10-03/config/prime_budget.json").read_text())
            (config_dir / "prime_budget.json").write_text(json.dumps(permission))
            rows = []
            for split, sid in (("train", "train"), ("development", "dev1"),
                               ("development", "dev2"), ("development", "dev3")):
                for race in ("Black", "White"):
                    for gender in ("Female", "Male"):
                        iid = f"{sid}_{race}_{gender}"
                        rows.append({
                            **row(iid, race, split, sid=sid),
                            "dataset": "fairface", "source_gender": gender,
                            "age_band": "30-39", "screening_status": "eligible",
                            "pixel_sha256": iid + "pixels",
                            "image_path": f"images/fairface/{iid}.png"})
            manifest = root / "2026-09-29/data/selected_images.csv"
            manifest.parent.mkdir(parents=True)
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            benchmark = root / "2026-09-11/data/iat12/manifest.csv"
            benchmark.parent.mkdir(parents=True)
            benchmark.write_text("sha256\nbenchmark_only\n")
            responses = [record(image, tag) for image in rows for tag in ("A", "B")]
            (store / "responses.jsonl").write_text("\n".join(map(json.dumps, responses)) + "\n")
            for name, value in (("REPO", root), ("WORKDIR", work), ("PRIVATE", store.parent),
                                ("MANIFEST", manifest)):
                stack.enter_context(patch.object(cv.common, name, value))
            stack.enter_context(patch.object(cv, "STORE", store))
            stack.enter_context(patch.object(study, "REPO", root))
            stack.enter_context(patch.object(study, "ROOT", config_dir.parent))
            config, audit = ev.export()
            with patch.object(sys.modules["builtins"], "print"):
                expanded = curated.training_rows(config)
            self.assertEqual(len(expanded), 64)
            self.assertEqual(len({r["image_id"] for r in expanded}), 16)
            self.assertEqual(config["training_methods"], ["orpo"])
            self.assertEqual(audit["splits"]["train"]["examples"], 16)
            self.assertFalse(audit["test_used_for_selection"])
            # Removing one approved target fails closed rather than dropping an image.
            path = root / config["targets_path"]
            targets = json.loads(path.read_text())
            path.write_text(json.dumps(targets[1:]))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                curated.training_rows(config)


if __name__ == "__main__":
    unittest.main()
