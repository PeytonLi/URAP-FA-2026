from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations, product
from typing import Any

import numpy as np


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    low = 0.0 if successes == 0 else max(0.0, center - radius)
    high = 1.0 if successes == total else min(1.0, center + radius)
    return low, high


def stratified_bootstrap_mean_interval(
    rows: list[dict[str, Any]],
    value_field: str,
    stratum_field: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if not rows or samples < 1:
        return math.nan, math.nan
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[stratum_field])].append(float(row[value_field]))
    rng = np.random.default_rng(seed)
    totals = np.zeros(samples, dtype=np.float64)
    total_count = 0
    for stratum in sorted(grouped):
        values = np.asarray(grouped[stratum], dtype=np.float64)
        indexes = rng.integers(0, len(values), size=(samples, len(values)))
        totals += values[indexes].sum(axis=1)
        total_count += len(values)
    means = totals / total_count
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def permutation_group_range_pvalue(
    rows: list[dict[str, Any]],
    value_field: str,
    group_field: str,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Test whether the observed maximum-minus-minimum group mean exceeds chance."""
    if not rows or samples < 1:
        return math.nan, math.nan
    groups = sorted({str(row[group_field]) for row in rows})
    if len(groups) < 2:
        return math.nan, math.nan
    values = np.asarray([float(row[value_field]) for row in rows], dtype=np.float64)
    indexes = {
        group: np.asarray(
            [index for index, row in enumerate(rows) if str(row[group_field]) == group],
            dtype=np.int64,
        )
        for group in groups
    }
    observed_means = [float(values[group_indexes].mean()) for group_indexes in indexes.values()]
    observed_range = max(observed_means) - min(observed_means)
    rng = np.random.default_rng(seed)
    exceedances = 0
    tolerance = 1e-12
    for _ in range(samples):
        shuffled = rng.permutation(values)
        permuted_means = [
            float(shuffled[group_indexes].mean()) for group_indexes in indexes.values()
        ]
        exceedances += max(permuted_means) - min(permuted_means) >= observed_range - tolerance
    return observed_range, (exceedances + 1) / (samples + 1)


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in the original order."""
    adjusted = [math.nan] * len(pvalues)
    valid = [(index, value) for index, value in enumerate(pvalues) if not math.isnan(value)]
    if not valid:
        return adjusted
    ordered = sorted(valid, key=lambda item: item[1])
    running = 1.0
    total = len(ordered)
    for rank_index in range(total - 1, -1, -1):
        original_index, value = ordered[rank_index]
        rank = rank_index + 1
        running = min(running, value * total / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def stratified_exact_binary_label_permutation(
    rows: list[dict[str, Any]],
    value_field: str,
    group_field: str,
    stratum_field: str,
    *,
    group_a: str,
    group_b: str,
) -> tuple[float, float, int]:
    """Exact two-sided label permutation preserving group counts within each stratum."""
    usable = [
        row
        for row in rows
        if str(row.get(group_field)) in {group_a, group_b}
        and math.isfinite(float(row[value_field]))
    ]
    values_a = [float(row[value_field]) for row in usable if row[group_field] == group_a]
    values_b = [float(row[value_field]) for row in usable if row[group_field] == group_b]
    if not values_a or not values_b:
        return math.nan, math.nan, 0
    observed = sum(values_a) / len(values_a) - sum(values_b) / len(values_b)

    strata = []
    for stratum in sorted({str(row[stratum_field]) for row in usable}):
        stratum_rows = [row for row in usable if str(row[stratum_field]) == stratum]
        count_a = sum(str(row[group_field]) == group_a for row in stratum_rows)
        indexes = tuple(range(len(stratum_rows)))
        assignments = tuple(combinations(indexes, count_a))
        strata.append((stratum_rows, assignments))

    exceedances = 0
    permutations = 0
    tolerance = 1e-12
    for assignment_set in product(*(assignments for _, assignments in strata)):
        permuted_a: list[float] = []
        permuted_b: list[float] = []
        for (stratum_rows, _), assignment in zip(strata, assignment_set, strict=True):
            assigned_a = set(assignment)
            for index, row in enumerate(stratum_rows):
                target = permuted_a if index in assigned_a else permuted_b
                target.append(float(row[value_field]))
        difference = sum(permuted_a) / len(permuted_a) - sum(permuted_b) / len(permuted_b)
        exceedances += abs(difference) >= abs(observed) - tolerance
        permutations += 1
    return observed, exceedances / permutations, permutations
