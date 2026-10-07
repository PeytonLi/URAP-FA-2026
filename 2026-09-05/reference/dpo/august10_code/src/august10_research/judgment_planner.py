from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .candidate_prefilter import (
    mechanical_exclusion_reasons,
    parse_iat_response,
    profile_word_count,
)


@dataclass(frozen=True)
class ReadyBlock:
    task_type: str
    sampling_match_id: str
    objective_type: str
    black: dict[str, Any]
    white: dict[str, Any]

    @property
    def rows(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.black, self.white

    @property
    def iat_chosen_tuple(self) -> tuple[str, ...]:
        if self.task_type != "iat_3word":
            return ()
        return tuple(parse_iat_response(_text(self.black, "chosen"))["words"])


@dataclass(frozen=True)
class PlannerPolicy:
    target_iat_blocks: int = 15
    target_profile_blocks: int = 35
    min_negative_iat_blocks: int = 0
    preserve_ready_profile_matches_for_iat: bool = False
    max_iat_tuple_fraction: float = 0.15
    max_profile_word_count_ratio: float = 1.25
    max_profile_grounded_detail_difference: int = 1
    max_profile_quality_difference: float = 0.75
    seed: str = "august10-claude-batch-v1"

    @property
    def iat_tuple_cap(self) -> int:
        return max(1, math.ceil(self.target_iat_blocks * self.max_iat_tuple_fraction))


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = _text(row, key)
    return int(float(value)) if value else 0


def _number(row: Mapping[str, Any], key: str) -> float:
    value = _text(row, key)
    return float(value) if value else 0.0


def _stable_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def profile_compatible(
    black: Mapping[str, Any],
    white: Mapping[str, Any],
    policy: PlannerPolicy,
) -> bool:
    black_words = profile_word_count(_text(black, "chosen"))
    white_words = profile_word_count(_text(white, "chosen"))
    if not black_words or not white_words:
        return False
    if max(black_words, white_words) / min(black_words, white_words) > (
        policy.max_profile_word_count_ratio
    ):
        return False
    if abs(
        _integer(black, "chosen_grounded_detail_count")
        - _integer(white, "chosen_grounded_detail_count")
    ) > policy.max_profile_grounded_detail_difference:
        return False
    return all(
        abs(_number(black, field) - _number(white, field))
        <= policy.max_profile_quality_difference
        for field in ("chosen_quality_score", "rejected_quality_score")
    )


def build_ready_blocks(
    eligible_rows: Iterable[Mapping[str, Any]],
    policy: PlannerPolicy | None = None,
) -> list[ReadyBlock]:
    policy = policy or PlannerPolicy()
    groups: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"Black": [], "White": []}
    )
    for source in eligible_rows:
        row = dict(source)
        task = _text(row, "task_type")
        key: tuple[Any, ...] = (
            _text(row, "sampling_match_id"),
            task,
            _text(row, "objective_type"),
        )
        if task == "iat_3word":
            key += (
                tuple(parse_iat_response(_text(row, "chosen"))["words"]),
                tuple(parse_iat_response(_text(row, "rejected"))["words"]),
            )
        race = _text(row, "fairface_race")
        if race in {"Black", "White"}:
            groups[key][race].append(row)

    ready: list[ReadyBlock] = []
    for key, by_race in groups.items():
        candidates = []
        for black in by_race["Black"]:
            for white in by_race["White"]:
                if _text(black, "image_sha256") == _text(white, "image_sha256"):
                    continue
                if key[1] == "grounded_profile" and not profile_compatible(
                    black, white, policy
                ):
                    continue
                candidates.append((black, white))
        if not candidates:
            continue
        black, white = min(
            candidates,
            key=lambda pair: _stable_rank(
                policy.seed,
                f"{_text(pair[0], 'pair_id')}|{_text(pair[1], 'pair_id')}",
            ),
        )
        ready.append(
            ReadyBlock(
                task_type=str(key[1]),
                sampling_match_id=str(key[0]),
                objective_type=str(key[2]),
                black=black,
                white=white,
            )
        )
    return ready


def select_existing_blocks(
    ready_blocks: Iterable[ReadyBlock],
    policy: PlannerPolicy | None = None,
    profile_row_counts: Mapping[str, int] | None = None,
) -> tuple[list[ReadyBlock], list[ReadyBlock]]:
    policy = policy or PlannerPolicy()
    profile_row_counts = profile_row_counts or {}
    if not 0 <= policy.min_negative_iat_blocks <= policy.target_iat_blocks:
        raise ValueError("min_negative_iat_blocks must be between zero and target_iat_blocks")
    iat = [block for block in ready_blocks if block.task_type == "iat_3word"]
    profiles = [block for block in ready_blocks if block.task_type == "grounded_profile"]
    ready_profile_matches = {block.sampling_match_id for block in profiles}

    iat_buckets: dict[tuple[str, ...], deque[ReadyBlock]] = {}
    raw_iat: dict[tuple[str, ...], list[ReadyBlock]] = defaultdict(list)
    for block in iat:
        raw_iat[block.iat_chosen_tuple].append(block)
    for key, values in raw_iat.items():
        iat_buckets[key] = deque(
            sorted(
                values,
                key=lambda block: (
                    policy.preserve_ready_profile_matches_for_iat
                    and block.sampling_match_id in ready_profile_matches,
                    profile_row_counts.get(block.sampling_match_id, 0),
                    _stable_rank(policy.seed, block.sampling_match_id),
                ),
            )
        )
    tuple_order = sorted(
        iat_buckets,
        key=lambda key: _stable_rank(policy.seed, "|".join(key)),
    )
    selected_iat: list[ReadyBlock] = []
    tuple_counts: Counter[tuple[str, ...]] = Counter()
    selected_iat_matches: set[str] = set()
    negative_iat = sorted(
        (
            block
            for block in iat
            if _text(block.black, "expression_class").lower() == "negative"
        ),
        key=lambda block: _stable_rank(
            policy.seed + "-required-negative-iat", block.sampling_match_id
        ),
    )
    if policy.preserve_ready_profile_matches_for_iat:
        negative_iat.sort(
            key=lambda block: (
                block.sampling_match_id in ready_profile_matches,
                profile_row_counts.get(block.sampling_match_id, 0),
                _stable_rank(
                    policy.seed + "-required-negative-iat",
                    block.sampling_match_id,
                ),
            )
        )
    for block in negative_iat:
        key = block.iat_chosen_tuple
        if (
            block.sampling_match_id in selected_iat_matches
            or tuple_counts[key] >= policy.iat_tuple_cap
        ):
            continue
        selected_iat.append(block)
        selected_iat_matches.add(block.sampling_match_id)
        tuple_counts[key] += 1
        if len(selected_iat) == policy.min_negative_iat_blocks:
            break
    if len(selected_iat) < policy.min_negative_iat_blocks:
        raise ValueError(
            f"only {len(selected_iat)} negative-expression IAT blocks satisfy the tuple cap; "
            f"target is {policy.min_negative_iat_blocks}"
        )
    while len(selected_iat) < policy.target_iat_blocks:
        progressed = False
        for key in tuple_order:
            while (
                iat_buckets[key]
                and iat_buckets[key][0].sampling_match_id in selected_iat_matches
            ):
                iat_buckets[key].popleft()
            if tuple_counts[key] >= policy.iat_tuple_cap or not iat_buckets[key]:
                continue
            block = iat_buckets[key].popleft()
            selected_iat.append(block)
            selected_iat_matches.add(block.sampling_match_id)
            tuple_counts[key] += 1
            progressed = True
            if len(selected_iat) >= policy.target_iat_blocks:
                break
        if not progressed:
            break

    used_matches = set(selected_iat_matches)
    selected_profiles = []
    for block in sorted(
        profiles,
        key=lambda value: _stable_rank(policy.seed, value.sampling_match_id),
    ):
        if block.sampling_match_id in used_matches:
            continue
        selected_profiles.append(block)
        used_matches.add(block.sampling_match_id)
        if len(selected_profiles) >= policy.target_profile_blocks:
            break
    return selected_iat, selected_profiles


def choose_development_blocks(
    blocks: list[dict[str, Any]],
    count: int,
    seed: str,
    minimum_negative_blocks: int = 0,
    maximum_negative_blocks: int | None = None,
) -> set[str]:
    if not 0 <= minimum_negative_blocks <= count:
        raise ValueError("minimum_negative_blocks must be between zero and count")
    if maximum_negative_blocks is None:
        maximum_negative_blocks = count
    if not minimum_negative_blocks <= maximum_negative_blocks <= count:
        raise ValueError(
            "maximum_negative_blocks must be between the minimum and count"
        )
    negative = sorted(
        (
            block
            for block in blocks
            if block["rows"][0].get("expression_class", "").lower() == "negative"
        ),
        key=lambda value: _stable_rank(seed + "-negative", value["block_id"]),
    )
    if len(negative) < minimum_negative_blocks:
        raise ValueError(
            f"cannot select {minimum_negative_blocks} negative development blocks; "
            f"only {len(negative)} are available"
        )
    selected = {block["block_id"] for block in negative[:minimum_negative_blocks]}
    buckets: dict[tuple[str, str], deque[dict[str, Any]]] = {}
    raw: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        row = block["rows"][0]
        if block["block_id"] not in selected:
            raw[(row.get("fairface_gender", ""), row.get("expression_class", ""))].append(
                block
            )
    for key, values in raw.items():
        buckets[key] = deque(
            sorted(values, key=lambda value: _stable_rank(seed, value["block_id"]))
        )
    order = sorted(buckets, key=lambda key: _stable_rank(seed, "|".join(key)))
    selected_negative = len(selected)
    while len(selected) < count:
        progressed = False
        for key in order:
            if buckets[key]:
                block = buckets[key].popleft()
                is_negative = (
                    block["rows"][0].get("expression_class", "").lower()
                    == "negative"
                )
                if is_negative and selected_negative >= maximum_negative_blocks:
                    continue
                selected.add(block["block_id"])
                selected_negative += int(is_negative)
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"cannot select {count} development blocks")
    return selected


def select_development_block_ids(
    iat_blocks: list[dict[str, Any]],
    profile_blocks: list[dict[str, Any]],
    dataset_policy: dict[str, Any],
    seed: str,
) -> set[str]:
    iat_ids = choose_development_blocks(
        iat_blocks,
        3,
        seed + "-iat-development",
        dataset_policy.get("min_development_negative_iat_blocks", 0),
    )
    selected_negative_iat = sum(
        block["block_id"] in iat_ids
        and block["rows"][0].get("expression_class", "").lower() == "negative"
        for block in iat_blocks
    )
    minimum_negative_profiles = max(
        0,
        dataset_policy.get("min_development_negative_expression_blocks", 0)
        - selected_negative_iat,
    )
    profile_ids = choose_development_blocks(
        profile_blocks,
        7,
        seed + "-profile-development",
        minimum_negative_profiles,
        minimum_negative_profiles,
    )
    return iat_ids | profile_ids


def _round_robin_blocks(
    blocks: Iterable[tuple[str, list[dict[str, Any]]]],
    *,
    seed: str,
) -> list[tuple[str, list[dict[str, Any]]]]:
    buckets: dict[tuple[str, str], deque[tuple[str, list[dict[str, Any]]]]] = {}
    raw: dict[tuple[str, str], list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for match_id, rows in blocks:
        first = rows[0]
        raw[(_text(first, "fairface_gender"), _text(first, "expression_class"))].append(
            (match_id, rows)
        )
    for key, values in raw.items():
        buckets[key] = deque(
            sorted(values, key=lambda value: _stable_rank(seed, value[0]))
        )
    order = sorted(buckets, key=lambda key: _stable_rank(seed, "|".join(key)))
    result = []
    while True:
        progressed = False
        for key in order:
            if buckets[key]:
                result.append(buckets[key].popleft())
                progressed = True
        if not progressed:
            return result


def _profile_objective_rank(objectives: set[str]) -> int:
    """Prefer preserving a direct-harm objective when retrying one race."""
    preference = (
        "unsupported_trait_removal",
        "invented_biography_removal",
        "person_negativity_reduction",
        "grounding_improvement",
    )
    return min(
        (preference.index(value) for value in objectives if value in preference),
        default=len(preference),
    )


def _profile_retry_races(
    eligible_by_race: Mapping[str, list[dict[str, Any]]],
    *,
    match_id: str,
    seed: str,
) -> tuple[str, ...]:
    black = eligible_by_race["Black"]
    white = eligible_by_race["White"]
    if not black and not white:
        return ("Black", "White")
    if not black:
        return ("Black",)
    if not white:
        return ("White",)

    black_objectives = {_text(row, "objective_type") for row in black}
    white_objectives = {_text(row, "objective_type") for row in white}
    if black_objectives.isdisjoint(white_objectives):
        black_rank = _profile_objective_rank(black_objectives)
        white_rank = _profile_objective_rank(white_objectives)
        if black_rank != white_rank:
            return ("White",) if black_rank < white_rank else ("Black",)

    # Retrying one side at a time avoids paying for two alternatives that may
    # simply create another objective or compatibility mismatch. Replanning
    # after this judgment can target the other side if it is still needed.
    ranked = sorted(
        ("Black", "White"),
        key=lambda race: _stable_rank(seed, f"{match_id}|{race}"),
    )
    return (ranked[0],)


def _candidate_iat_tuple(row: Mapping[str, Any]) -> tuple[str, ...]:
    first = parse_iat_response(_text(row, "candidate_1"))
    second = parse_iat_response(_text(row, "candidate_2"))
    if not first["valid"] or not second["valid"]:
        return ()
    lower = first if first["negative_count"] < second["negative_count"] else second
    return tuple(lower["words"])


def plan_judgment_batch(
    candidate_rows: Iterable[Mapping[str, Any]],
    eligible_rows: Iterable[Mapping[str, Any]],
    attempted_pair_ids: set[str],
    *,
    max_iat_pairs: int = 20,
    max_profile_pairs: int = 20,
    policy: PlannerPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or PlannerPolicy()
    candidates = [dict(row) for row in candidate_rows]
    eligible = [dict(row) for row in eligible_rows]
    mechanically_valid = [
        row for row in candidates if not mechanical_exclusion_reasons(row)
    ]
    ready = build_ready_blocks(eligible, policy)
    profile_row_counts = Counter(
        _text(row, "sampling_match_id")
        for row in eligible
        if _text(row, "task_type") == "grounded_profile"
    )
    selected_iat, selected_profiles = select_existing_blocks(
        ready,
        policy,
        profile_row_counts=profile_row_counts,
    )
    reserved_iat_matches = {block.sampling_match_id for block in selected_iat}
    used_matches = set(reserved_iat_matches)

    eligible_by_task_match_race: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in eligible:
        eligible_by_task_match_race[
            (
                _text(row, "task_type"),
                _text(row, "sampling_match_id"),
                _text(row, "fairface_race"),
            )
        ].append(row)

    unattempted = [
        row
        for row in mechanically_valid
        if _text(row, "pair_id") not in attempted_pair_ids
    ]
    by_task_match_race: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for row in unattempted:
        by_task_match_race[
            (
                _text(row, "task_type"),
                _text(row, "sampling_match_id"),
                _text(row, "fairface_race"),
            )
        ].append(row)
    for rows in by_task_match_race.values():
        rows.sort(key=lambda row: _stable_rank(policy.seed, _text(row, "pair_id")))

    existing_tuple_counts = Counter(block.iat_chosen_tuple for block in selected_iat)
    iat_options: list[tuple[str, list[dict[str, Any]]]] = []
    iat_matches = sorted(
        {
            _text(row, "sampling_match_id")
            for row in mechanically_valid
            if _text(row, "task_type") == "iat_3word"
        }
    )
    for match_id in iat_matches:
        if match_id in used_matches:
            continue
        rows = []
        predicted_tuple: tuple[str, ...] = ()
        viable = True
        for race in ("Black", "White"):
            if eligible_by_task_match_race[("iat_3word", match_id, race)]:
                continue
            options = by_task_match_race[("iat_3word", match_id, race)]
            if not options:
                viable = False
                break
            row = options[0]
            row_tuple = _candidate_iat_tuple(row)
            if predicted_tuple and row_tuple != predicted_tuple:
                viable = False
                break
            predicted_tuple = row_tuple
            rows.append(row)
        if viable and rows and existing_tuple_counts[predicted_tuple] < policy.iat_tuple_cap:
            iat_options.append((match_id, rows))
    iat_options.sort(
        key=lambda item: (
            existing_tuple_counts[_candidate_iat_tuple(item[1][0])],
            _stable_rank(policy.seed + "-iat", item[0]),
        )
    )
    planned_iat: list[dict[str, Any]] = []
    planned_iat_matches: set[str] = set()
    for match_id, rows in iat_options:
        if len(planned_iat) + len(rows) > max_iat_pairs:
            continue
        planned_iat.extend(rows)
        planned_iat_matches.add(match_id)
        existing_tuple_counts[_candidate_iat_tuple(rows[0])] += 1
        if len(selected_iat) + len(planned_iat_matches) >= policy.target_iat_blocks + 4:
            break

    # IAT has a tighter vocabulary constraint and fewer replacement blocks than
    # profiles. A newly planned IAT block may therefore displace an already-ready
    # profile block that uses the same two images.
    selected_profiles = [
        block
        for block in selected_profiles
        if block.sampling_match_id not in planned_iat_matches
    ]
    reserved_profile_matches = {block.sampling_match_id for block in selected_profiles}
    used_matches |= planned_iat_matches | reserved_profile_matches
    profile_options_by_priority: dict[
        int, list[tuple[str, list[dict[str, Any]]]]
    ] = defaultdict(list)
    profile_matches = sorted(
        {
            _text(row, "sampling_match_id")
            for row in mechanically_valid
            if _text(row, "task_type") == "grounded_profile"
        }
    )
    for match_id in profile_matches:
        if match_id in used_matches:
            continue
        rows: list[dict[str, Any]] = []
        viable = True
        eligible_by_race = {
            race: eligible_by_task_match_race[("grounded_profile", match_id, race)]
            for race in ("Black", "White")
        }
        required_races = _profile_retry_races(
            eligible_by_race,
            match_id=match_id,
            seed=policy.seed + "-profile-retry",
        )
        eligible_race_count = sum(bool(eligible_by_race[race]) for race in ("Black", "White"))
        priority = {1: 0, 2: 1, 0: 2}[eligible_race_count]
        for race in required_races:
            options = by_task_match_race[("grounded_profile", match_id, race)]
            if not options:
                viable = False
                break
            rows.append(options[0])
        if viable and rows:
            profile_options_by_priority[priority].append((match_id, rows))
    profile_options = []
    for priority in sorted(profile_options_by_priority):
        profile_options.extend(
            _round_robin_blocks(
                profile_options_by_priority[priority],
                seed=f"{policy.seed}-profile-priority-{priority}",
            )
        )
    planned_profiles: list[dict[str, Any]] = []
    planned_profile_matches: set[str] = set()
    for match_id, rows in profile_options:
        if len(planned_profiles) + len(rows) > max_profile_pairs:
            continue
        planned_profiles.extend(rows)
        planned_profile_matches.add(match_id)
        if (
            len(selected_profiles) + len(planned_profile_matches)
            >= policy.target_profile_blocks + 8
        ):
            break

    return {
        "mechanically_valid_rows": mechanically_valid,
        "mechanical_exclusions": [
            {
                "pair_id": _text(row, "pair_id"),
                "reasons": "|".join(mechanical_exclusion_reasons(row)),
            }
            for row in candidates
            if mechanical_exclusion_reasons(row)
        ],
        "ready_iat_blocks": selected_iat,
        "ready_profile_blocks": selected_profiles,
        "planned_iat_rows": planned_iat,
        "planned_profile_rows": planned_profiles,
        "attempted_pair_count": len(attempted_pair_ids),
        "unattempted_mechanically_valid_rows": len(unattempted),
        "inventory": {
            "ready_iat_blocks": len(selected_iat),
            "ready_profile_blocks": len(selected_profiles),
            "iat_tuple_cap": policy.iat_tuple_cap,
            "ready_iat_tuple_counts": {
                "|".join(key): value
                for key, value in sorted(
                    Counter(block.iat_chosen_tuple for block in selected_iat).items()
                )
            },
            "planned_iat_pairs": len(planned_iat),
            "planned_iat_sampling_blocks": len(planned_iat_matches),
            "planned_profile_pairs": len(planned_profiles),
            "planned_profile_sampling_blocks": len(planned_profile_matches),
        },
    }
