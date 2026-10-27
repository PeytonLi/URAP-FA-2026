import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prime_native import extras_only_plan


class ExtrasOnlyPlanTest(unittest.TestCase):
    def test_tuned_only_needs_recovered_checkpoint(self):
        plan = extras_only_plan(["orpo"], {("gemma3_12b", "orpo"): None})
        self.assertEqual(plan, {"mode": "extras_only", "extra_methods": ["orpo"]})
        with self.assertRaises(ValueError):
            extras_only_plan(["orpo"], {})

    def test_base_needs_no_checkpoint(self):
        self.assertEqual(extras_only_plan(["base"], {})["extra_methods"], ["base"])

    def test_rejects_unknown_empty_or_duplicate_methods(self):
        for methods in ([], ["sft"], ["kto"], ["orpo", "orpo"]):
            with self.assertRaises(ValueError):
                extras_only_plan(methods, {("gemma3_12b", "orpo"): None})


if __name__ == "__main__":
    unittest.main()
