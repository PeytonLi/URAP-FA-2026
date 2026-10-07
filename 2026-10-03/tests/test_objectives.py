import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import objectives


@unittest.skipUnless(importlib.util.find_spec("torch"), "Objective gradient tests require PyTorch")
class ObjectiveTests(unittest.TestCase):
    @staticmethod
    def datum(weights):
        import torch
        return SimpleNamespace(loss_fn_inputs={"weights": SimpleNamespace(to_torch=lambda: torch.tensor(weights))})

    def test_sft_masks_prompt_nan_and_means_each_example(self):
        import torch
        scores = [torch.tensor([float("nan"), -2.0, -4.0], requires_grad=True),
                  torch.tensor([float("nan"), -6.0], requires_grad=True)]
        data = [self.datum([0.0, 0.5, 0.5]), self.datum([0.0, 1.0])]
        loss, metrics = objectives.sft_loss(data, scores)
        self.assertAlmostEqual(float(loss.detach()), 4.5)
        loss.backward()
        self.assertEqual(float(scores[0].grad[0]), 0.0)
        self.assertEqual(float(scores[1].grad[0]), 0.0)
        self.assertAlmostEqual(float(scores[0].grad[1]), -0.25)
        self.assertAlmostEqual(float(scores[1].grad[1]), -0.5)
        self.assertEqual(metrics["batch_examples"], 2)

    def test_orpo_chosen_and_rejected_gradient_directions_and_mask(self):
        import torch
        data = [self.datum([0.0, 1.0]), self.datum([0.0, 1.0])]
        scores = [torch.tensor([float("nan"), -2.0], requires_grad=True),
                  torch.tensor([float("nan"), -4.0], requires_grad=True)]
        loss, metrics = objectives.orpo_loss(data, scores)
        loss.backward()
        self.assertLess(float(scores[0].grad[1]), 0.0)
        self.assertGreater(float(scores[1].grad[1]), 0.0)
        self.assertEqual(float(scores[0].grad[0]), 0.0)
        self.assertEqual(float(scores[1].grad[0]), 0.0)
        self.assertEqual(metrics["preference_accuracy"], 1.0)

    def test_orpo_duplicate_pair_does_not_double_batch_loss(self):
        import torch
        pair = [self.datum([0.0, 1.0]), self.datum([0.0, 1.0])]
        scores = [torch.tensor([0.0, -2.0]), torch.tensor([0.0, -4.0])]
        single, _ = objectives.orpo_loss(pair, scores)
        doubled, _ = objectives.orpo_loss(pair * 2, scores * 2)
        self.assertAlmostEqual(float(single), float(doubled))

    def test_zero_coefficient_matches_chosen_only_sft(self):
        import torch
        pair = [self.datum([0.0, 0.5, 0.5]), self.datum([0.0, 1.0])]
        scores = [torch.tensor([0.0, -2.0, -4.0]), torch.tensor([0.0, -6.0])]
        orpo, _ = objectives.orpo_loss(pair, scores, coefficient=0)
        sft, _ = objectives.sft_loss(pair[:1], scores[:1])
        self.assertAlmostEqual(float(orpo), float(sft))

    def test_coordinate_and_coefficient_errors_fail_before_training(self):
        import torch
        with self.assertRaises(ValueError):
            objectives.sft_loss([self.datum([0.0, 1.0])], [torch.tensor([-2.0])])
        with self.assertRaises(ValueError):
            objectives.orpo_loss([], [], coefficient=float("nan"))


if __name__ == "__main__":
    unittest.main()
