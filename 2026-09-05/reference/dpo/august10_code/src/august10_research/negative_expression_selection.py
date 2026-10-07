from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

AGE_BUCKET_ORDER = {
    "0-2": 0,
    "3-9": 1,
    "10-19": 2,
    "20-29": 3,
    "30-39": 4,
    "40-49": 5,
    "50-59": 6,
    "60-69": 7,
    "70+": 8,
}


@dataclass(frozen=True)
class NegativeSelectionPolicy:
    target_blocks: int = 9
    minimum_confidence: float = 0.70
    max_age_bucket_difference: int = 1
    max_expression_intensity_difference: int = 1
    seed: str = "august10-negative-expression-expansion-v1"


@dataclass(frozen=True)
class NegativeExpressionMatch:
    black: dict[str, Any]
    white: dict[str, Any]
    age_bucket_difference: int
    expression_intensity_difference: int
    minimum_confidence: float

    @property
    def sampling_match_id(self) -> str:
        digest = hashlib.sha256(
            f"{self.black['image_sha256']}|{self.white['image_sha256']}".encode()
        ).hexdigest()[:16]
        return f"aug10-negative-{digest}"


@dataclass
class _FlowEdge:
    to: int
    reverse_index: int
    capacity: int
    cost: int


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _number(row: Mapping[str, Any], key: str) -> float:
    value = _text(row, key)
    return float(value) if value else 0.0


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = _text(row, key)
    return int(float(value)) if value else 0


def _stable_rank(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _candidate_match(
    black: Mapping[str, Any],
    white: Mapping[str, Any],
    policy: NegativeSelectionPolicy,
) -> NegativeExpressionMatch | None:
    if _text(black, "fairface_gender") != _text(white, "fairface_gender"):
        return None
    if _text(black, "expression_fine") != _text(white, "expression_fine"):
        return None
    black_age = AGE_BUCKET_ORDER.get(_text(black, "age_bucket"))
    white_age = AGE_BUCKET_ORDER.get(_text(white, "age_bucket"))
    if black_age is None or white_age is None:
        return None
    age_difference = abs(black_age - white_age)
    if age_difference > policy.max_age_bucket_difference:
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
    if minimum_confidence < policy.minimum_confidence:
        return None
    return NegativeExpressionMatch(
        black=dict(black),
        white=dict(white),
        age_bucket_difference=age_difference,
        expression_intensity_difference=intensity_difference,
        minimum_confidence=minimum_confidence,
    )


def _match_cost(match: NegativeExpressionMatch, seed: str) -> int:
    confidence_penalty = round((1.0 - match.minimum_confidence) * 10_000)
    tie = int(
        _stable_rank(
            seed,
            f"{match.black['image_sha256']}|{match.white['image_sha256']}",
        )[:6],
        16,
    ) % 1_000
    return (
        match.age_bucket_difference * 1_000_000
        + match.expression_intensity_difference * 100_000
        + confidence_penalty * 10
        + tie
    )


def _add_edge(
    graph: list[list[_FlowEdge]], source: int, target: int, capacity: int, cost: int
) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), capacity, cost)
    reverse = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def select_negative_expression_pairs(
    annotated_rows: Iterable[Mapping[str, Any]],
    excluded_image_hashes: set[str],
    policy: NegativeSelectionPolicy | None = None,
) -> list[NegativeExpressionMatch]:
    policy = policy or NegativeSelectionPolicy()
    eligible = [
        dict(row)
        for row in annotated_rows
        if _text(row, "expression_class") == "negative"
        and _text(row, "fairface_race") in {"Black", "White"}
        and _number(row, "expression_confidence") >= policy.minimum_confidence
        and _text(row, "image_sha256") not in excluded_image_hashes
    ]
    by_race = {
        race: sorted(
            [row for row in eligible if _text(row, "fairface_race") == race],
            key=lambda row: _stable_rank(policy.seed, _text(row, "image_sha256")),
        )
        for race in ("Black", "White")
    }
    black_rows = by_race["Black"]
    white_rows = by_race["White"]
    candidates: dict[tuple[str, str], NegativeExpressionMatch] = {}
    for black in black_rows:
        for white in white_rows:
            match = _candidate_match(black, white, policy)
            if match is not None:
                candidates[
                    (_text(black, "image_sha256"), _text(white, "image_sha256"))
                ] = match

    node_count = len(black_rows) + len(white_rows) + 2
    source = node_count - 2
    sink = node_count - 1
    graph: list[list[_FlowEdge]] = [[] for _ in range(node_count)]
    black_node = {
        _text(row, "image_sha256"): index for index, row in enumerate(black_rows)
    }
    white_node = {
        _text(row, "image_sha256"): len(black_rows) + index
        for index, row in enumerate(white_rows)
    }
    for node in black_node.values():
        _add_edge(graph, source, node, 1, 0)
    for node in white_node.values():
        _add_edge(graph, node, sink, 1, 0)
    match_edges: list[tuple[_FlowEdge, NegativeExpressionMatch]] = []
    for image_pair, match in sorted(
        candidates.items(),
        key=lambda item: (
            _match_cost(item[1], policy.seed),
            _stable_rank(policy.seed, "|".join(item[0])),
        ),
    ):
        edge = _add_edge(
            graph,
            black_node[image_pair[0]],
            white_node[image_pair[1]],
            1,
            _match_cost(match, policy.seed),
        )
        match_edges.append((edge, match))

    flow = 0
    infinity = 10**30
    while flow < policy.target_blocks:
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
            prior, edge_index = previous[node] or (-1, -1)
            edge = graph[prior][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse_index].capacity += 1
            node = prior
        flow += 1
    if flow < policy.target_blocks:
        raise ValueError(
            f"only {flow} negative-expression blocks are available; "
            f"target is {policy.target_blocks}"
        )
    selected = [match for edge, match in match_edges if edge.capacity == 0]
    if len(selected) != policy.target_blocks:
        raise RuntimeError("negative-expression flow result has the wrong size")
    return sorted(
        selected,
        key=lambda match: _stable_rank(policy.seed, match.sampling_match_id),
    )
