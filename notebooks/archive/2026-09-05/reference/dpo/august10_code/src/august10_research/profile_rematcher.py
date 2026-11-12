from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .dataset_gate import PROFILE_HARM_FIELDS
from .judgment_planner import PlannerPolicy, profile_compatible

AGE_BUCKET_ORDER = {
    "0-2": 0,
    "3-9": 1,
    "10-19": 2,
    "20-29": 3,
    "30-39": 4,
    "40-49": 5,
    "50-59": 6,
    "60-69": 7,
}


@dataclass(frozen=True)
class ProfileRematchPolicy:
    target_blocks: int = 35
    min_negative_expression_blocks: int = 0
    max_age_bucket_difference: int = 1
    max_expression_intensity_difference: int = 1
    min_expression_confidence: float = 0.70
    require_same_gender: bool = True
    require_same_expression_class: bool = True
    require_same_expression_fine: bool = True
    require_same_response_provenance: bool = True
    seed: str = "august10-profile-rematch-v1"


@dataclass(frozen=True)
class ProfileMatch:
    black: dict[str, Any]
    white: dict[str, Any]
    direct_harm_reduction: bool
    age_bucket_difference: int
    expression_intensity_difference: int
    minimum_expression_confidence: float

    @property
    def match_id(self) -> str:
        digest = hashlib.sha256(
            f"{self.black['pair_id']}|{self.white['pair_id']}".encode()
        ).hexdigest()[:16]
        return f"aug10-profile-rematch-{digest}"


@dataclass
class _FlowEdge:
    to: int
    reverse_index: int
    capacity: int
    cost: int


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


def _profile_direct(row: Mapping[str, Any]) -> bool:
    chosen_harm = sum(_integer(row, f"chosen_{field}") for field in PROFILE_HARM_FIELDS)
    rejected_harm = sum(_integer(row, f"rejected_{field}") for field in PROFILE_HARM_FIELDS)
    return chosen_harm == 0 and chosen_harm < rejected_harm


def _age_difference(black: Mapping[str, Any], white: Mapping[str, Any]) -> int | None:
    black_age = AGE_BUCKET_ORDER.get(_text(black, "age_bucket"))
    white_age = AGE_BUCKET_ORDER.get(_text(white, "age_bucket"))
    if black_age is None or white_age is None:
        return None
    return abs(black_age - white_age)


def _candidate_match(
    black: Mapping[str, Any],
    white: Mapping[str, Any],
    policy: ProfileRematchPolicy,
) -> ProfileMatch | None:
    if _text(black, "objective_type") != _text(white, "objective_type"):
        return None
    if policy.require_same_gender and _text(black, "fairface_gender") != _text(
        white, "fairface_gender"
    ):
        return None
    if policy.require_same_expression_class and _text(
        black, "expression_class"
    ) != _text(white, "expression_class"):
        return None
    if policy.require_same_expression_fine and _text(black, "expression_fine") != _text(
        white, "expression_fine"
    ):
        return None
    if policy.require_same_response_provenance and any(
        _text(black, field) != _text(white, field)
        for field in ("chosen_response_provenance", "rejected_response_provenance")
    ):
        return None
    age_difference = _age_difference(black, white)
    if age_difference is None or age_difference > policy.max_age_bucket_difference:
        return None
    intensity_difference = abs(
        _integer(black, "expression_intensity")
        - _integer(white, "expression_intensity")
    )
    if intensity_difference > policy.max_expression_intensity_difference:
        return None
    minimum_confidence = min(
        _number(black, "expression_confidence"),
        _number(white, "expression_confidence"),
    )
    if minimum_confidence < policy.min_expression_confidence:
        return None
    if not profile_compatible(black, white, PlannerPolicy()):
        return None
    if _text(black, "image_sha256") == _text(white, "image_sha256"):
        return None
    if _text(black, "chosen").casefold() == _text(white, "chosen").casefold():
        return None
    return ProfileMatch(
        black=dict(black),
        white=dict(white),
        direct_harm_reduction=_profile_direct(black) and _profile_direct(white),
        age_bucket_difference=age_difference,
        expression_intensity_difference=intensity_difference,
        minimum_expression_confidence=minimum_confidence,
    )


def _match_preference(match: ProfileMatch, seed: str) -> tuple[Any, ...]:
    return (
        not match.direct_harm_reduction,
        match.age_bucket_difference,
        match.expression_intensity_difference,
        -match.minimum_expression_confidence,
        _stable_rank(seed, f"{match.black['pair_id']}|{match.white['pair_id']}"),
    )


def _flow_cost(match: ProfileMatch, seed: str) -> int:
    # A single additional non-direct block must cost more than all secondary
    # matching preferences across the complete 35-block pilot.
    non_direct_cost = 1_000_000_000 if not match.direct_harm_reduction else 0
    confidence_penalty = round((1.0 - match.minimum_expression_confidence) * 10_000)
    stable_tie = int(
        _stable_rank(seed, f"{match.black['pair_id']}|{match.white['pair_id']}")[:6],
        16,
    ) % 1_000
    return (
        non_direct_cost
        + match.age_bucket_difference * 1_000_000
        + match.expression_intensity_difference * 100_000
        + confidence_penalty * 10
        + stable_tie
    )


def _add_edge(
    graph: list[list[_FlowEdge]], source: int, target: int, capacity: int, cost: int
) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity, cost)
    reverse = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _minimum_cost_matching(
    matches_by_images: Mapping[tuple[str, str], ProfileMatch],
    target_blocks: int,
    seed: str,
) -> list[ProfileMatch]:
    if target_blocks == 0:
        return []
    black_images = sorted(
        {black for black, _ in matches_by_images},
        key=lambda value: _stable_rank(seed, value),
    )
    white_images = sorted(
        {white for _, white in matches_by_images},
        key=lambda value: _stable_rank(seed, value),
    )
    node_count = 2 + len(black_images) + len(white_images)
    source = node_count - 2
    sink = node_count - 1
    graph: list[list[_FlowEdge]] = [[] for _ in range(node_count)]
    black_node = {value: index for index, value in enumerate(black_images)}
    white_node = {
        value: len(black_images) + index for index, value in enumerate(white_images)
    }
    for node in black_node.values():
        _add_edge(graph, source, node, 1, 0)
    for node in white_node.values():
        _add_edge(graph, node, sink, 1, 0)
    match_edges: list[tuple[_FlowEdge, ProfileMatch]] = []
    for image_pair, match in sorted(
        matches_by_images.items(),
        key=lambda item: (
            _flow_cost(item[1], seed),
            _stable_rank(seed, "|".join(item[0])),
        ),
    ):
        edge = _add_edge(
            graph,
            black_node[image_pair[0]],
            white_node[image_pair[1]],
            1,
            _flow_cost(match, seed),
        )
        match_edges.append((edge, match))

    flow = 0
    infinity = 10**30
    while flow < target_blocks:
        distance = [infinity] * node_count
        previous: list[tuple[int, int] | None] = [None] * node_count
        distance[source] = 0
        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distance[node] == infinity:
                    continue
                for edge_index, edge in enumerate(edges):
                    candidate = distance[node] + edge.cost
                    if edge.capacity and candidate < distance[edge.to]:
                        distance[edge.to] = candidate
                        previous[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if previous[sink] is None:
            break
        node = sink
        while node != source:
            prior_node, edge_index = previous[node] or (-1, -1)
            edge = graph[prior_node][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse_index].capacity += 1
            node = prior_node
        flow += 1

    if flow < target_blocks:
        raise ValueError(
            f"only {flow} compatible profile blocks are available; "
            f"target is {target_blocks}"
        )
    selected = [match for edge, match in match_edges if edge.capacity == 0]
    if len(selected) != target_blocks:
        raise RuntimeError("profile flow result did not materialize the requested block count")
    return selected


def rematch_profiles(
    eligible_rows: Iterable[Mapping[str, Any]],
    reserved_image_hashes: set[str] | None = None,
    policy: ProfileRematchPolicy | None = None,
) -> list[ProfileMatch]:
    policy = policy or ProfileRematchPolicy()
    if not 0 <= policy.min_negative_expression_blocks <= policy.target_blocks:
        raise ValueError(
            "min_negative_expression_blocks must be between zero and target_blocks"
        )
    reserved_image_hashes = reserved_image_hashes or set()
    deduplicated = {
        _text(row, "pair_id"): dict(row)
        for row in eligible_rows
        if _text(row, "pair_id")
        and _text(row, "task_type") == "grounded_profile"
        and _text(row, "fairface_race") in {"Black", "White"}
        and _text(row, "image_sha256") not in reserved_image_hashes
    }
    by_race_image: dict[str, dict[str, list[dict[str, Any]]]] = {
        "Black": defaultdict(list),
        "White": defaultdict(list),
    }
    for row in deduplicated.values():
        by_race_image[_text(row, "fairface_race")][
            _text(row, "image_sha256")
        ].append(row)

    black_images = sorted(
        by_race_image["Black"], key=lambda value: _stable_rank(policy.seed, value)
    )
    white_images = sorted(
        by_race_image["White"], key=lambda value: _stable_rank(policy.seed, value)
    )
    matches_by_images: dict[tuple[str, str], ProfileMatch] = {}
    for black_image in black_images:
        for white_image in white_images:
            candidates = [
                match
                for black in by_race_image["Black"][black_image]
                for white in by_race_image["White"][white_image]
                if (match := _candidate_match(black, white, policy)) is not None
            ]
            if candidates:
                matches_by_images[(black_image, white_image)] = min(
                    candidates, key=lambda match: _match_preference(match, policy.seed)
                )

    negative_candidates = {
        image_pair: match
        for image_pair, match in matches_by_images.items()
        if _text(match.black, "expression_class").lower() == "negative"
    }
    selected = _minimum_cost_matching(
        negative_candidates,
        policy.min_negative_expression_blocks,
        policy.seed + "-required-negative",
    )
    selected_images = {
        _text(row, "image_sha256")
        for match in selected
        for row in (match.black, match.white)
    }
    remaining_candidates = {
        image_pair: match
        for image_pair, match in matches_by_images.items()
        if not (set(image_pair) & selected_images)
    }
    selected.extend(
        _minimum_cost_matching(
            remaining_candidates,
            policy.target_blocks - len(selected),
            policy.seed + "-remaining",
        )
    )
    return sorted(selected, key=lambda match: _stable_rank(policy.seed, match.match_id))
