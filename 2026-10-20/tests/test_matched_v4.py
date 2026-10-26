"""Matched-answer design: identical block answers, even coverage, mixed valence."""
from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "2026-10-20/labeling"), str(REPO / "2026-10-03")]
import curated
import matched_v4 as mv
from multimodel_bias.constants import POSITIVE_IAT_SET


def rows(blocks=12, split="train", lean="negative"):
    result = []
    for b in range(blocks):
        for race in ("Black", "Black", "White", "White"):
            iid = f"{split}{b}_{race}_{len(result)}"
            result.append({"image_id": iid, "split": split, "match_set_id": f"{split}{b}",
                           "source_race": race, "expression_class": lean})
    return result


class MatchedDesignTests(unittest.TestCase):
    def test_every_triple_mixes_valence_and_follows_lean(self):
        plan = mv.design(rows(lean="negative") + rows(split="development", lean="positive"))
        for (split, _), answers in plan.items():
            schedule = mv.SCHEDULES["negative" if split == "train" else "positive"]
            self.assertEqual([len(set(c) & POSITIVE_IAT_SET) for c, _ in answers], list(schedule))
            for chosen, rejected in answers:
                self.assertFalse(set(chosen) & set(rejected))
                self.assertIn(len(set(rejected) & POSITIVE_IAT_SET), (0, 3))
            self.assertEqual(len({frozenset(c) for c, _ in answers}), 4)

    def test_coverage_is_even_over_all_50_words(self):
        plan = mv.design(rows(blocks=40))
        counts = Counter(w for answers in plan.values() for c, _ in answers for w in c)
        self.assertEqual(set(counts), set(mv.POSITIVE) | set(mv.NEGATIVE))
        # 30 positive vs 20 negative eligible words: evenness is per pole.
        for pole in (mv.POSITIVE, mv.NEGATIVE):
            used = [counts[w] for w in pole]
            self.assertLessEqual(max(used) - min(used), 1)

    def test_deterministic_and_development_does_not_steer_train(self):
        train = rows()
        alone = mv.design(train)
        together = mv.design(train + rows(split="development"))
        self.assertEqual(alone, {k: together[k] for k in alone})
        self.assertEqual(alone, mv.design(list(reversed(train))))

    def test_unbalanced_or_mixed_lean_block_fails(self):
        bad = rows(blocks=1)
        bad[0]["source_race"] = "White"
        with self.assertRaisesRegex(ValueError, "Race-unbalanced"):
            mv.design(bad)
        bad = rows(blocks=1)
        bad[0]["expression_class"] = "positive"
        with self.assertRaisesRegex(ValueError, "expression"):
            mv.design(bad)

    def test_curated_rejects_overlap_but_skips_teacher_evidence(self):
        target = {"chosen": "Glad, Pain, Grief", "rejected": "Angry, Annoy, Scorn",
                  "preference_variants": [
                      {"chosen": "Glad, Pain, Grief", "rejected": "Angry, Annoy, Scorn"},
                      {"chosen": "Love, Pain, Ugly", "rejected": "Angry, Annoy, Scorn"}]}
        self.assertEqual(curated.variant_failures(target, require_word_support=False), [])
        target["preference_variants"][1]["rejected"] = "Love, Annoy, Scorn"
        self.assertIn("chosen_rejected_overlap",
                      curated.variant_failures(target, require_word_support=False))


if __name__ == "__main__":
    unittest.main()
