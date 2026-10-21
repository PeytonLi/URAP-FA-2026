"""Response-only, batch-mean SFT/ORPO/KTO objectives for the IAT study.

These functions compute local losses. Importing/calling them never provisions
compute, updates model weights, or makes a dataset training-ready.
"""
import math

SFT_REDUCTION = "response_token_mean_then_example_mean"
ORPO_REDUCTION = "response_token_mean_then_pair_mean"
KTO_REDUCTION = "response_token_mean_then_example_value_mean"


def _log1mexp(value):
    """Stable log(1-exp(x)), using the inspected cookbook's two branches.

    Keep the arithmetic local: importing cookbook.training.utils eagerly loads
    its Linux-only resource module even when only this scalar identity is needed.
    """
    import torch
    return torch.where(value < -math.log(2),
                       torch.log1p(-torch.exp(value)), torch.log(-torch.expm1(value)))


def _response_means(data, logprobs):
    import torch

    if not data or len(data) != len(logprobs):
        raise ValueError("Loss requires an equal nonempty datum/score batch")
    means = []
    for datum, scores in zip(data, logprobs):
        weights = (datum if isinstance(datum, torch.Tensor)
                   else datum.loss_fn_inputs["weights"].to_torch()).to(scores.device)
        if scores.ndim != 1 or weights.ndim != 1 or len(scores) != len(weights):
            raise ValueError("Token scores and response mask must share one-dimensional coordinates")
        if torch.any(weights < 0) or not torch.isfinite(weights).all():
            raise ValueError("Invalid response loss weights")
        mask = weights > 0
        if not mask.any():
            raise ValueError("Each training example needs response tokens")
        # Boolean selection avoids NaN * zero on masked prompt/image positions.
        selected = scores[mask]
        if not torch.isfinite(selected).all():
            raise ValueError("Nonfinite response log probability")
        means.append((selected * weights[mask]).sum() / weights[mask].sum())
    return means


def sft_loss(data, logprobs):
    import torch

    means = _response_means(data, logprobs)
    loss = -torch.stack(means).mean()
    return loss, {"sft_loss": float(loss.detach()), "batch_examples": len(means)}


def orpo_loss(data, logprobs, *, coefficient=0.1):
    import torch
    import torch.nn.functional as functional

    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("ORPO coefficient must be finite and nonnegative")
    if len(data) % 2:
        raise ValueError("ORPO needs ordered chosen/rejected pairs")
    means = _response_means(data, logprobs)
    chosen = torch.stack(means[0::2])
    rejected = torch.stack(means[1::2])
    # Preserve the inspected official cookbook's probability clamp and stable
    # log(1-p) calculation, but use an explicit per-pair batch mean, not a sum.
    chosen_odds_input = chosen.clamp(min=-50.0, max=-1e-6)
    rejected_odds_input = rejected.clamp(min=-50.0, max=-1e-6)
    log_odds = (chosen_odds_input - _log1mexp(chosen_odds_input)
                - rejected_odds_input + _log1mexp(rejected_odds_input))
    sft_terms = -chosen
    preference_terms = -functional.logsigmoid(log_odds)
    loss = (sft_terms + coefficient * preference_terms).mean()
    return loss, {"orpo_loss": float(loss.detach()), "sft_loss": float(sft_terms.detach().mean()),
                  "preference_loss": float(preference_terms.detach().mean()),
                  "preference_accuracy": float((log_odds.detach() > 0).float().mean()),
                  "batch_pairs": len(chosen), "orpo_coefficient": coefficient}


def kto_loss(data, logprobs, reference_logprobs, *, beta=0.1,
             desirable_weight=1.0, undesirable_weight=1.0):
    """Paired KTO over (chosen, rejected) branches sharing one image+prompt.

    Each example's value is the policy-vs-reference mean response logprob margin
    scaled by beta. Because every pair here conditions on the same image and
    prompt, the Kullback-Leibler baseline for each branch is estimated on the
    pair's counterfactual branch (the mismatched output for the same input), so
    the desirable value uses z_chosen - z_rejected and the undesirable value its
    negation. Reference scores must be the adapter-disabled (frozen base)
    forward pass on identical inputs.
    """
    import torch

    for name, weight in (("desirable_weight", desirable_weight),
                         ("undesirable_weight", undesirable_weight)):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"KTO {name} must be finite and nonnegative")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("KTO beta must be finite and positive")
    if len(data) % 2 or len(data) != len(reference_logprobs):
        raise ValueError("KTO needs ordered chosen/rejected pairs with matching reference scores")
    policy = torch.stack(_response_means(data, logprobs))
    reference = torch.stack(_response_means(data, reference_logprobs))
    margins = beta * (policy - reference)
    chosen_margin, rejected_margin = margins[0::2], margins[1::2]
    # KL baseline per branch = the policy-vs-reference margin on the pair's
    # OTHER branch (the mismatched output for the same image+prompt).
    kl_for_desirable = rejected_margin
    kl_for_undesirable = chosen_margin
    desirable_values = torch.sigmoid(chosen_margin - kl_for_desirable)
    undesirable_values = torch.sigmoid(kl_for_undesirable - rejected_margin)
    pair_margin = chosen_margin - rejected_margin
    desirable_loss = desirable_weight * (1.0 - desirable_values)
    undesirable_loss = undesirable_weight * (1.0 - undesirable_values)
    loss = (desirable_loss + undesirable_loss).mean()
    return loss, {"kto_loss": float(loss.detach()),
                  "desirable_loss": float(desirable_loss.detach().mean()),
                  "undesirable_loss": float(undesirable_loss.detach().mean()),
                  "preference_accuracy": float((pair_margin.detach() > 0).float().mean()),
                  "kl_margin_estimate": float(pair_margin.detach().mean()),
                  "batch_pairs": len(chosen_margin), "kto_beta": beta}
