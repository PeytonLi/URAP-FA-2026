from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


def development_preference_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Conservative pilot gate for race-specific development-set failures."""

    groups = summary.get("groups", {})
    required = {
        f"{task}/{race}"
        for task in ("grounded_profile", "iat_3word")
        for race in ("Black", "White")
    }
    checks: dict[str, bool] = {
        "all_task_race_groups_present": required <= set(groups),
        "overall_preference_accuracy_at_least_55pct": (
            float(summary.get("preference_accuracy", 0.0)) >= 0.55
        ),
        "overall_mean_margin_delta_positive": (
            float(summary.get("mean_margin_delta", float("-inf"))) > 0.0
        ),
    }
    if required <= set(groups):
        for task in ("grounded_profile", "iat_3word"):
            black = groups[f"{task}/Black"]
            white = groups[f"{task}/White"]
            checks[f"{task}_both_races_mean_margin_positive"] = (
                float(black["mean_margin_delta"]) > 0.0
                and float(white["mean_margin_delta"]) > 0.0
            )
            checks[f"{task}_preference_accuracy_gap_at_most_35pp"] = (
                abs(float(black["preference_accuracy"]) - float(white["preference_accuracy"]))
                <= 0.35
            )
            checks[f"{task}_minimum_group_accuracy_at_least_30pct"] = (
                min(float(black["preference_accuracy"]), float(white["preference_accuracy"]))
                >= 0.30
            )
    return {"pass": all(checks.values()), "checks": checks}


def effective_warmup_updates(planned_updates: int, requested_updates: int = 0) -> int:
    if planned_updates <= 0 or requested_updates < 0:
        raise ValueError("planned updates must be positive and requested warmup nonnegative")
    if requested_updates:
        return min(requested_updates, planned_updates)
    return max(1, min(10, round(planned_updates * 0.10)))


def dpo_loss_value(
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    beta: float,
) -> float:
    import math

    z = beta * (
        (policy_chosen - policy_rejected)
        - (reference_chosen - reference_rejected)
    )
    return math.log1p(math.exp(-abs(z))) + max(-z, 0.0)


def dpo_margin_gradient_coefficient(
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    beta: float,
) -> float:
    import math

    z = beta * (
        (policy_chosen - policy_rejected)
        - (reference_chosen - reference_rejected)
    )
    if z >= 0:
        sigmoid_negative_z = math.exp(-z) / (1.0 + math.exp(-z))
    else:
        sigmoid_negative_z = 1.0 / (1.0 + math.exp(z))
    return -beta * sigmoid_negative_z


def pair_seed(seed: int, pair_id: str, side: str) -> int:
    digest = hashlib.sha256(f"{pair_id}\0{side}".encode("utf-8")).digest()
    return (seed + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def set_torch_seed(torch: Any, seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_device(model: Any):
    return next(model.parameters()).device


def response_logit_positions(prompt_len: int, sequence_length: int) -> range:
    """Positions whose logits predict response tokens after causal shifting."""

    if prompt_len < 1 or sequence_length < prompt_len:
        raise ValueError("prompt and sequence lengths are inconsistent")
    return range(prompt_len - 1, sequence_length - 1)


def get_image_token(processor: Any) -> str:
    return getattr(processor, "boi_token", None) or "<image>"


def ensure_single_image_token(processor: Any, text: str) -> str:
    image_token = get_image_token(processor)
    without = str(text).replace(image_token, " ").replace("<start_of_image>", " ").strip()
    return image_token if not without else f"{image_token}\n{without}"


def format_chat(processor: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = [f"{message['role'].upper()}: {message['content']}" for message in messages]
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n".join(rendered)


def build_inputs_for_logps(
    processor: Any,
    image: Any,
    prompt_text: str,
    response_text: str,
) -> tuple[dict[str, Any], int]:
    prompt_only = ensure_single_image_token(
        processor,
        format_chat(
            processor,
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
        ),
    )
    full_text = ensure_single_image_token(
        processor,
        format_chat(
            processor,
            [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response_text},
            ],
            add_generation_prompt=False,
        ),
    )
    prompt_inputs = processor(images=[image], text=prompt_only, return_tensors="pt")
    full_inputs = processor(images=[image], text=full_text, return_tensors="pt")
    return full_inputs, int(prompt_inputs["input_ids"].shape[1])


def inputs_to_cpu(inputs: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    return {
        key: value.cpu() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def build_pair_inputs(processor: Any, pair: Mapping[str, Any], torch: Any):
    from PIL import Image

    with Image.open(Path(str(pair["image_path"]))) as source:
        image = source.convert("RGB").copy()
    chosen, chosen_prompt_len = build_inputs_for_logps(
        processor,
        image,
        str(pair["prompt"]),
        str(pair["chosen"]),
    )
    rejected, rejected_prompt_len = build_inputs_for_logps(
        processor,
        image,
        str(pair["prompt"]),
        str(pair["rejected"]),
    )
    return (
        inputs_to_cpu(chosen, torch),
        chosen_prompt_len,
        inputs_to_cpu(rejected, torch),
        rejected_prompt_len,
    )


def sequence_logp(
    model: Any,
    inputs: Mapping[str, Any],
    prompt_len: int,
    *,
    max_prompt_tokens: int,
    max_response_tokens: int,
    torch: Any,
    reduction: str = "sum",
):
    device = get_model_device(model)
    prepared = {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    sequence_length = prepared["input_ids"].shape[1]
    response_length = max(sequence_length - prompt_len, 0)
    has_pixels = (
        torch.is_tensor(prepared.get("pixel_values"))
        and prepared["pixel_values"].numel() > 0
    )
    prompt_keep = prompt_len if has_pixels else min(prompt_len, max_prompt_tokens)
    response_keep = min(response_length, max_response_tokens)
    if prompt_keep + response_keep < sequence_length:
        prompt_start = 0 if has_pixels else max(prompt_len - prompt_keep, 0)
        response_end = prompt_len + response_keep
        indices = torch.tensor(
            list(range(prompt_start, prompt_len)) + list(range(prompt_len, response_end)),
            device=device,
        )
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in prepared:
                prepared[key] = prepared[key].index_select(1, indices)
        if not has_pixels:
            prompt_len = prompt_keep

    positions = torch.tensor(
        list(response_logit_positions(prompt_len, prepared["input_ids"].shape[1])),
        dtype=torch.long,
        device=device,
    )
    if positions.numel() == 0:
        raise ValueError("response contains no scoreable tokens")
    outputs = model(**prepared, logits_to_keep=positions)
    response_labels = prepared["input_ids"].index_select(1, positions + 1)
    token_logps = torch.log_softmax(outputs.logits, dim=-1).gather(
        -1, response_labels.unsqueeze(-1)
    ).squeeze(-1)
    return reduce_response_logps(token_logps, reduction)


def reduce_response_logps(token_logps: Any, reduction: str):
    if reduction == "sum":
        return token_logps.sum(dim=-1)
    if reduction == "mean":
        return token_logps.mean(dim=-1)
    raise ValueError(f"unsupported response log-probability reduction: {reduction}")
