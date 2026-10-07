from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .constants import TASK_IAT, TASK_PROFILE
from .dataset_audit import DatasetAudit, DatasetPolicy, audit_dataset
from .pair_policy import (
    PairPolicy,
    parse_iat_response,
    profile_word_count,
    tokens,
    validate_pair,
)

PROFILE_EXACT_MATCH_FIELDS = (
    "chosen_visible_expression_support",
    "rejected_visible_expression_support",
    "chosen_person_negative_count",
    "rejected_person_negative_count",
    "chosen_unwarranted_negative_count",
    "rejected_unwarranted_negative_count",
    "chosen_unsupported_trait_count",
    "rejected_unsupported_trait_count",
    "chosen_identity_leakage_count",
    "rejected_identity_leakage_count",
    "chosen_invented_biography_count",
    "rejected_invented_biography_count",
)


@dataclass(frozen=True)
class BalancePolicy:
    target_rows: int = 1000
    target_iat_fraction: float = 0.30
    development_fraction: float = 0.20
    max_iat_fixed_substitution_fraction: float = 0.20
    max_iat_chosen_tuple_fraction: float = 0.15
    iat_tuple_balance_min_blocks: int = 10
    max_profile_objective_fraction: float = 0.40
    max_profile_word_count_ratio: float = 1.25
    max_profile_grounded_detail_difference: int = 1
    max_profile_quality_difference: float = 0.75
    max_pairs_per_image: int = 1
    max_profile_chosen_reuse: int = 1
    seed: str = "august3-v1"

    def __post_init__(self) -> None:
        if self.target_rows <= 0 or self.target_rows % 2:
            raise ValueError("target_rows must be a positive even number")
        for name, value in (
            ("target_iat_fraction", self.target_iat_fraction),
            ("development_fraction", self.development_fraction),
            ("max_iat_fixed_substitution_fraction", self.max_iat_fixed_substitution_fraction),
            ("max_iat_chosen_tuple_fraction", self.max_iat_chosen_tuple_fraction),
            ("max_profile_objective_fraction", self.max_profile_objective_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.max_pairs_per_image <= 0:
            raise ValueError("max_pairs_per_image must be positive")
        if self.max_iat_chosen_tuple_fraction <= 0.0:
            raise ValueError("max_iat_chosen_tuple_fraction must be positive")
        if self.max_profile_chosen_reuse <= 0:
            raise ValueError("max_profile_chosen_reuse must be positive")
        if self.iat_tuple_balance_min_blocks <= 0:
            raise ValueError("iat_tuple_balance_min_blocks must be positive")

    @property
    def target_blocks(self) -> int:
        return self.target_rows // 2

    @property
    def target_iat_blocks(self) -> int:
        return round(self.target_blocks * self.target_iat_fraction)

    @property
    def target_profile_blocks(self) -> int:
        return self.target_blocks - self.target_iat_blocks

    @property
    def iat_chosen_tuple_block_cap(self) -> int:
        if self.target_iat_blocks < self.iat_tuple_balance_min_blocks:
            return self.target_iat_blocks
        return max(1, math.ceil(self.target_iat_blocks * self.max_iat_chosen_tuple_fraction))


@dataclass(frozen=True)
class CandidateBlock:
    block_id: str
    black: dict[str, Any]
    white: dict[str, Any]

    @property
    def rows(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.black, self.white

    @property
    def task_type(self) -> str:
        return _text(self.black, "task_type")

    @property
    def objective_type(self) -> str:
        return _text(self.black, "objective_type")

    @property
    def image_hashes(self) -> tuple[str, str]:
        return _text(self.black, "image_sha256"), _text(self.white, "image_sha256")


@dataclass
class BalanceResult:
    rows: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    invalid_pairs: dict[str, list[str]] = field(default_factory=dict)
    audit: DatasetAudit | None = None

    @property
    def complete(self) -> bool:
        return bool(self.audit and self.audit.valid and not self.report.get("unfilled_blocks"))


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
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _base_signature(row: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        _text(row, "task_type"),
        _text(row, "fairface_gender"),
        _text(row, "age_bucket"),
        _text(row, "expression_class").lower(),
        _text(row, "expression_fine").lower(),
        _text(row, "expression_intensity"),
        _text(row, "expression_pair_equivalent").lower(),
        _text(row, "expression_pair_min_confidence"),
        _text(row, "expression_pair_max_cue_difference"),
        _text(row, "expression_pair_audit_version"),
        _text(row, "objective_type"),
    )


def _exact_match_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    base = _base_signature(row)
    if _text(row, "task_type") == TASK_IAT:
        chosen = parse_iat_response(_text(row, "chosen"))
        rejected = parse_iat_response(_text(row, "rejected"))
        return base + (
            chosen["word_tuple"],
            rejected["word_tuple"],
            _integer(row, "chosen_unwarranted_negative_count"),
            _integer(row, "rejected_unwarranted_negative_count"),
            _integer(row, "chosen_visible_expression_support"),
            _integer(row, "rejected_visible_expression_support"),
        )
    return base + tuple(_integer(row, field_name) for field_name in PROFILE_EXACT_MATCH_FIELDS)


def _profile_compatible(
    black: Mapping[str, Any],
    white: Mapping[str, Any],
    policy: BalancePolicy,
) -> bool:
    black_words = profile_word_count(_text(black, "chosen"))
    white_words = profile_word_count(_text(white, "chosen"))
    if not black_words or not white_words:
        return False
    if max(black_words, white_words) / min(black_words, white_words) > policy.max_profile_word_count_ratio:
        return False
    if abs(
        _integer(black, "chosen_grounded_detail_count")
        - _integer(white, "chosen_grounded_detail_count")
    ) > policy.max_profile_grounded_detail_difference:
        return False
    for prefix in ("chosen", "rejected"):
        if abs(
            _number(black, f"{prefix}_quality_score")
            - _number(white, f"{prefix}_quality_score")
        ) > policy.max_profile_quality_difference:
            return False
    return True


def _match_cost(black: Mapping[str, Any], white: Mapping[str, Any]) -> tuple[float, ...]:
    if _text(black, "task_type") == TASK_IAT:
        return (0.0,)
    black_words = profile_word_count(_text(black, "chosen"))
    white_words = profile_word_count(_text(white, "chosen"))
    return (
        abs(math.log(black_words / white_words)),
        abs(
            _integer(black, "chosen_grounded_detail_count")
            - _integer(white, "chosen_grounded_detail_count")
        ),
        abs(_number(black, "chosen_quality_score") - _number(white, "chosen_quality_score")),
        abs(_number(black, "rejected_quality_score") - _number(white, "rejected_quality_score")),
    )


def _maximum_matching(
    black_rows: list[dict[str, Any]],
    white_rows: list[dict[str, Any]],
    policy: BalancePolicy,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    black_rows = sorted(black_rows, key=lambda row: _stable_rank(policy.seed, _text(row, "pair_id")))
    white_rows = sorted(white_rows, key=lambda row: _stable_rank(policy.seed, _text(row, "pair_id")))
    edges: list[list[int]] = []
    for black in black_rows:
        candidates = []
        for index, white in enumerate(white_rows):
            if _text(black, "image_sha256") == _text(white, "image_sha256"):
                continue
            if _text(black, "task_type") == TASK_PROFILE and not _profile_compatible(
                black, white, policy
            ):
                continue
            candidates.append(index)
        candidates.sort(
            key=lambda index: (
                _match_cost(black, white_rows[index]),
                _stable_rank(policy.seed, _text(white_rows[index], "pair_id")),
            )
        )
        edges.append(candidates)

    white_to_black: dict[int, int] = {}

    def augment(black_index: int, seen: set[int]) -> bool:
        for white_index in edges[black_index]:
            if white_index in seen:
                continue
            seen.add(white_index)
            previous = white_to_black.get(white_index)
            if previous is None or augment(previous, seen):
                white_to_black[white_index] = black_index
                return True
        return False

    for black_index in range(len(black_rows)):
        augment(black_index, set())

    result = [
        (black_rows[black_index], white_rows[white_index])
        for white_index, black_index in white_to_black.items()
    ]
    return sorted(
        result,
        key=lambda pair: _stable_rank(
            policy.seed, f"{_text(pair[0], 'pair_id')}|{_text(pair[1], 'pair_id')}"
        ),
    )


def build_candidate_blocks(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy: BalancePolicy | None = None,
    pair_policy: PairPolicy | None = None,
    forbidden_eval_hashes: set[str] | None = None,
) -> tuple[list[CandidateBlock], dict[str, list[str]], dict[str, Any]]:
    policy = policy or BalancePolicy()
    pair_policy = pair_policy or PairPolicy()
    forbidden_eval_hashes = forbidden_eval_hashes or set()
    materialized = [dict(row) for row in rows]
    pair_counts = Counter(_text(row, "pair_id") for row in materialized)
    invalid: dict[str, list[str]] = {}
    valid_rows: list[dict[str, Any]] = []

    for index, row in enumerate(materialized):
        pair_id = _text(row, "pair_id") or f"<row-{index}>"
        issues: list[str] = []
        if pair_counts[_text(row, "pair_id")] > 1:
            issues.append("duplicate_pair_id")
        if _text(row, "image_sha256") in forbidden_eval_hashes:
            issues.append("evaluation_image_forbidden")
        validation = validate_pair(row, pair_policy)
        issues.extend(validation.issues)
        if issues:
            invalid[pair_id] = sorted(set(issues))
        else:
            valid_rows.append(row)

    groups: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"Black": [], "White": []}
    )
    for row in valid_rows:
        groups[_exact_match_signature(row)][_text(row, "fairface_race")].append(row)

    blocks: list[CandidateBlock] = []
    unmatched = Counter()
    for signature, race_rows in groups.items():
        matched = _maximum_matching(race_rows["Black"], race_rows["White"], policy)
        unmatched["Black"] += len(race_rows["Black"]) - len(matched)
        unmatched["White"] += len(race_rows["White"]) - len(matched)
        for black, white in matched:
            raw_id = f"{_text(black, 'pair_id')}|{_text(white, 'pair_id')}"
            block_id = "aug3-block-" + hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]
            blocks.append(CandidateBlock(block_id=block_id, black=black, white=white))

    inventory = {
        "input_rows": len(materialized),
        "valid_rows": len(valid_rows),
        "invalid_rows": len(invalid),
        "candidate_blocks": len(blocks),
        "candidate_blocks_by_task": dict(Counter(block.task_type for block in blocks)),
        "candidate_blocks_by_objective": dict(Counter(block.objective_type for block in blocks)),
        "unmatched_valid_rows_by_race": dict(unmatched),
    }
    return blocks, invalid, inventory


def _diversity_key(block: CandidateBlock) -> tuple[str, ...]:
    row = block.black
    key = (
        block.objective_type,
        _text(row, "fairface_gender"),
        _text(row, "age_bucket"),
        _text(row, "expression_class").lower(),
        _text(row, "expression_fine").lower(),
    )
    if block.task_type == TASK_IAT:
        key += (parse_iat_response(_text(row, "chosen"))["word_tuple"],)
    return key


def _select_round_robin(
    blocks: Iterable[CandidateBlock],
    target: int,
    *,
    policy: BalancePolicy,
    image_counts: Counter[str],
    objective_cap: int | None = None,
    profile_chosen_counts: Counter[str] | None = None,
    iat_chosen_counts: Counter[str] | None = None,
    iat_chosen_cap: int | None = None,
) -> list[CandidateBlock]:
    grouped: dict[tuple[str, ...], deque[CandidateBlock]] = {}
    raw_groups: dict[tuple[str, ...], list[CandidateBlock]] = defaultdict(list)
    for block in blocks:
        raw_groups[_diversity_key(block)].append(block)
    for key, values in raw_groups.items():
        grouped[key] = deque(
            sorted(values, key=lambda block: _stable_rank(policy.seed, block.block_id))
        )
    group_order = sorted(
        grouped,
        key=lambda key: _stable_rank(policy.seed, "|".join(key)),
    )
    selected: list[CandidateBlock] = []
    objective_counts: Counter[str] = Counter()
    while len(selected) < target:
        progress = False
        for key in group_order:
            queue = grouped[key]
            while queue:
                block = queue.popleft()
                if any(
                    image_counts[digest] >= policy.max_pairs_per_image
                    for digest in block.image_hashes
                ):
                    continue
                if objective_cap is not None and objective_counts[block.objective_type] >= objective_cap:
                    continue
                iat_chosen_tuple = ""
                if iat_chosen_counts is not None:
                    iat_chosen_tuple = parse_iat_response(
                        _text(block.black, "chosen")
                    )["word_tuple"]
                    if (
                        iat_chosen_cap is not None
                        and iat_chosen_counts[iat_chosen_tuple] >= iat_chosen_cap
                    ):
                        continue
                block_profile_signatures: Counter[str] = Counter()
                if profile_chosen_counts is not None:
                    block_profile_signatures.update(
                        " ".join(tokens(_text(row, "chosen"))) for row in block.rows
                    )
                    if any(
                        profile_chosen_counts[signature] + count
                        > policy.max_profile_chosen_reuse
                        for signature, count in block_profile_signatures.items()
                    ):
                        continue
                selected.append(block)
                objective_counts[block.objective_type] += 1
                image_counts.update(block.image_hashes)
                if profile_chosen_counts is not None:
                    profile_chosen_counts.update(block_profile_signatures)
                if iat_chosen_counts is not None:
                    iat_chosen_counts[iat_chosen_tuple] += 1
                progress = True
                break
            if len(selected) >= target:
                break
        if not progress:
            break
    return selected


def _assign_splits(blocks: list[CandidateBlock], policy: BalancePolicy) -> dict[str, str]:
    development_target = round(len(blocks) * policy.development_fraction)
    assignments = {block.block_id: "train" for block in blocks}
    if development_target == 0:
        return assignments

    by_task: dict[str, list[CandidateBlock]] = defaultdict(list)
    for block in blocks:
        by_task[block.task_type].append(block)

    task_floors = {
        task: math.floor(len(values) * policy.development_fraction)
        for task, values in by_task.items()
    }
    if development_target >= len(by_task):
        for task in by_task:
            task_floors[task] = max(1, task_floors[task])
    assigned = sum(task_floors.values())
    if assigned > development_target:
        removable = sorted(
            task_floors,
            key=lambda task: (
                len(by_task[task]) * policy.development_fraction - task_floors[task],
                _stable_rank(policy.seed, task),
            ),
        )
        for task in removable:
            minimum = 1 if development_target >= len(by_task) else 0
            while assigned > development_target and task_floors[task] > minimum:
                task_floors[task] -= 1
                assigned -= 1
    task_remainders = sorted(
        by_task,
        key=lambda task: (
            -(len(by_task[task]) * policy.development_fraction - task_floors[task]),
            _stable_rank(policy.seed, task),
        ),
    )
    while assigned < development_target:
        progress = False
        for task in task_remainders:
            if assigned >= development_target:
                break
            if task_floors[task] < len(by_task[task]):
                task_floors[task] += 1
                assigned += 1
                progress = True
        if not progress:
            raise RuntimeError("could not allocate the requested development blocks")

    for task, task_blocks in sorted(by_task.items()):
        strata: dict[tuple[str, ...], list[CandidateBlock]] = defaultdict(list)
        for block in task_blocks:
            row = block.black
            strata[
                (
                    block.objective_type,
                    _text(row, "fairface_gender"),
                    _text(row, "expression_class").lower(),
                )
            ].append(block)
        target = task_floors[task]
        floors = {
            key: math.floor(len(values) * target / len(task_blocks))
            for key, values in strata.items()
        }
        remaining = target - sum(floors.values())
        remainders = sorted(
            strata,
            key=lambda key: (
                -(len(strata[key]) * target / len(task_blocks) - floors[key]),
                _stable_rank(policy.seed, f"{task}|{'|'.join(key)}"),
            ),
        )
        for key in remainders[:remaining]:
            floors[key] += 1
        for key, values in strata.items():
            ordered = sorted(
                values,
                key=lambda block: _stable_rank(
                    policy.seed + "-development", block.block_id
                ),
            )
            for block in ordered[: floors[key]]:
                assignments[block.block_id] = "development"
    return assignments


def balance_pairs(
    rows: Iterable[Mapping[str, Any]],
    *,
    policy: BalancePolicy | None = None,
    pair_policy: PairPolicy | None = None,
    dataset_policy: DatasetPolicy | None = None,
    forbidden_eval_hashes: set[str] | None = None,
) -> BalanceResult:
    policy = policy or BalancePolicy()
    pair_policy = pair_policy or PairPolicy()
    blocks, invalid, inventory = build_candidate_blocks(
        rows,
        policy=policy,
        pair_policy=pair_policy,
        forbidden_eval_hashes=forbidden_eval_hashes,
    )
    image_counts: Counter[str] = Counter()

    direct_iat = [
        block
        for block in blocks
        if block.task_type == TASK_IAT and block.objective_type == "negative_count_reduction"
    ]
    iat_chosen_counts: Counter[str] = Counter()
    selected_iat = _select_round_robin(
        direct_iat,
        policy.target_iat_blocks,
        policy=policy,
        image_counts=image_counts,
        iat_chosen_counts=iat_chosen_counts,
        iat_chosen_cap=policy.iat_chosen_tuple_block_cap,
    )
    iat_shortfall = policy.target_iat_blocks - len(selected_iat)
    fixed_cap = math.floor(
        policy.target_iat_blocks * policy.max_iat_fixed_substitution_fraction
    )
    if iat_shortfall and fixed_cap:
        fixed_iat = [
            block
            for block in blocks
            if block.task_type == TASK_IAT
            and block.objective_type == "unsupported_negative_substitution"
        ]
        selected_iat.extend(
            _select_round_robin(
                fixed_iat,
                min(iat_shortfall, fixed_cap),
                policy=policy,
                image_counts=image_counts,
                iat_chosen_counts=iat_chosen_counts,
                iat_chosen_cap=policy.iat_chosen_tuple_block_cap,
            )
        )

    profile_blocks = [block for block in blocks if block.task_type == TASK_PROFILE]
    profile_chosen_counts: Counter[str] = Counter()
    profile_objective_cap = max(
        1,
        math.floor(
            policy.target_profile_blocks * policy.max_profile_objective_fraction
        ),
    )
    selected_profiles = _select_round_robin(
        profile_blocks,
        policy.target_profile_blocks,
        policy=policy,
        image_counts=image_counts,
        objective_cap=profile_objective_cap,
        profile_chosen_counts=profile_chosen_counts,
    )
    selected = selected_iat + selected_profiles
    assignments = _assign_splits(selected, policy)

    final_rows: list[dict[str, Any]] = []
    for block in selected:
        for source_row in block.rows:
            row = dict(source_row)
            row["source_split"] = _text(source_row, "split")
            row["split"] = assignments[block.block_id]
            row["match_block_id"] = block.block_id
            final_rows.append(row)
    final_rows.sort(key=lambda row: (_text(row, "match_block_id"), _text(row, "fairface_race")))

    effective_dataset_policy = dataset_policy or DatasetPolicy(
        min_rows=policy.target_rows,
        min_unique_images=math.ceil(policy.target_rows / policy.max_pairs_per_image),
        min_iat_fraction=max(0.0, policy.target_iat_fraction - 0.001),
        max_iat_fraction=min(1.0, policy.target_iat_fraction + 0.001),
        min_iat_negative_count_reduction_fraction=(
            1.0 - policy.max_iat_fixed_substitution_fraction
        ),
        max_iat_fixed_substitution_fraction=policy.max_iat_fixed_substitution_fraction,
        max_iat_chosen_tuple_fraction=policy.max_iat_chosen_tuple_fraction,
        iat_tuple_balance_min_rows=policy.iat_tuple_balance_min_blocks * 2,
        max_controlled_iat_fraction_of_all_rows=policy.target_iat_fraction,
        max_profile_block_word_count_ratio=policy.max_profile_word_count_ratio,
        max_profile_block_grounded_detail_difference=policy.max_profile_grounded_detail_difference,
        max_profile_block_quality_difference=policy.max_profile_quality_difference,
        max_profile_objective_fraction=policy.max_profile_objective_fraction,
        max_profile_chosen_reuse=policy.max_profile_chosen_reuse,
    )
    audit = audit_dataset(
        final_rows,
        pair_policy=pair_policy,
        dataset_policy=effective_dataset_policy,
        forbidden_eval_hashes=forbidden_eval_hashes,
    )
    selected_counts = Counter(block.task_type for block in selected)
    selected_objectives = Counter(block.objective_type for block in selected)
    report = {
        **inventory,
        "target_rows": policy.target_rows,
        "target_blocks": policy.target_blocks,
        "target_iat_blocks": policy.target_iat_blocks,
        "target_profile_blocks": policy.target_profile_blocks,
        "selected_rows": len(final_rows),
        "selected_blocks": len(selected),
        "selected_blocks_by_task": dict(selected_counts),
        "selected_blocks_by_objective": dict(selected_objectives),
        "iat_chosen_tuple_block_cap": policy.iat_chosen_tuple_block_cap,
        "iat_chosen_tuple_block_counts": dict(sorted(iat_chosen_counts.items())),
        "selected_unique_images": len({_text(row, "image_sha256") for row in final_rows}),
        "split_rows": dict(Counter(_text(row, "split") for row in final_rows)),
        "split_blocks_by_task": {
            f"{split}/{task}": len(
                {
                    _text(row, "match_block_id")
                    for row in final_rows
                    if _text(row, "split") == split and _text(row, "task_type") == task
                }
            )
            for split in ("train", "development")
            for task in (TASK_IAT, TASK_PROFILE)
        },
        "unfilled_blocks": {
            TASK_IAT: max(0, policy.target_iat_blocks - len(selected_iat)),
            TASK_PROFILE: max(0, policy.target_profile_blocks - len(selected_profiles)),
        },
        "audit_valid": audit.valid,
        "audit_issues": audit.issues,
    }
    report["unfilled_blocks"] = {
        key: value for key, value in report["unfilled_blocks"].items() if value
    }
    return BalanceResult(rows=final_rows, report=report, invalid_pairs=invalid, audit=audit)
