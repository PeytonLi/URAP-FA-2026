from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any


def summarize_usage(
    records: Iterable[Mapping[str, Any]],
    rates: Mapping[str, Mapping[str, Any]],
    *,
    starting_balance_usd: Decimal | None = None,
    reserve_fraction: Decimal = Decimal("0.10"),
) -> dict[str, Any]:
    if starting_balance_usd is not None and starting_balance_usd <= 0:
        raise ValueError("starting balance must be positive")
    if not Decimal("0") <= reserve_fraction < Decimal("1"):
        raise ValueError("reserve fraction must be in [0, 1)")

    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    )
    response_ids: set[str] = set()
    for record in records:
        response_id = str(record.get("response_id", "")).strip()
        if response_id and response_id in response_ids:
            continue
        if response_id:
            response_ids.add(response_id)
        model = str(record.get("model", "")).strip()
        if model not in rates:
            raise ValueError(f"no pricing configured for Claude model {model!r}")
        totals[model]["calls"] += 1
        totals[model]["input_tokens"] += int(record.get("input_tokens", 0) or 0)
        totals[model]["output_tokens"] += int(record.get("output_tokens", 0) or 0)

    cost = Decimal("0")
    by_model: dict[str, dict[str, Any]] = {}
    million = Decimal(1_000_000)
    for model, usage in sorted(totals.items()):
        model_rates = rates[model]
        model_cost = (
            Decimal(usage["input_tokens"]) * Decimal(str(model_rates["input"]))
            + Decimal(usage["output_tokens"]) * Decimal(str(model_rates["output"]))
        ) / million
        cost += model_cost
        by_model[model] = {**usage, "estimated_cost_usd": float(model_cost)}

    result = {
        "calls": sum(value["calls"] for value in totals.values()),
        "input_tokens": sum(value["input_tokens"] for value in totals.values()),
        "output_tokens": sum(value["output_tokens"] for value in totals.values()),
        "estimated_cost_usd": float(cost),
        "reserve_fraction": float(reserve_fraction),
        "by_model": by_model,
        "balance_caveat": (
            "Estimated from this run's returned token usage; unrelated account activity, "
            "taxes, credit adjustments, and provider-side rounding are not observable."
        ),
    }
    if starting_balance_usd is None:
        result.update(
            {
                "starting_balance_usd": None,
                "reserve_usd": None,
                "estimated_remaining_usd": None,
                "estimated_remaining_fraction": None,
                "pause_judging": None,
                "reserve_enforceable": False,
            }
        )
        return result

    reserve_usd = starting_balance_usd * reserve_fraction
    estimated_remaining = starting_balance_usd - cost
    result.update(
        {
            "starting_balance_usd": float(starting_balance_usd),
            "reserve_usd": float(reserve_usd),
            "estimated_remaining_usd": float(estimated_remaining),
            "estimated_remaining_fraction": float(estimated_remaining / starting_balance_usd),
            "pause_judging": estimated_remaining <= reserve_usd,
            "reserve_enforceable": True,
        }
    )
    return result
