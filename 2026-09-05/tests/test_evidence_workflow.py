import json
from pathlib import Path

import pytest

from common import ROOT
from run_hosted_visbias12 import Budget
from run_local_gemma import extract_text, validate_tasks


def test_all_images_and_unique_task_keys():
    tasks = json.loads((ROOT / "data/frozen/local_tasks.json").read_text())
    assert len({t["task_key"] for t in tasks}) == len(tasks)
    assert sum(t["task_kind"] == "iat" and t["dataset"] == "utkface400" for t in tasks) == 400
    assert sum(t["task_kind"] == "iat" and t["dataset"] == "neutral12" for t in tasks) == 120


def test_private_images_if_materialized():
    tasks = json.loads((ROOT / "data/frozen/local_tasks.json").read_text())
    assets = {ROOT / image for task in tasks for image in task["image_paths"]}
    if not any(p.exists() for p in assets):
        pytest.skip("Private image bundle is not needed for offline score/plot replay")
    validate_tasks(tasks)


def test_visbias_pairs_have_both_orders_and_no_occupation_claim():
    protocol = json.loads((ROOT / "data/frozen/visbias12/protocol.json").read_text())
    assert len(protocol["description_tasks"]) == 12
    comparisons = protocol["comparison_tasks"]
    assert len(comparisons) == 24
    for row in comparisons:
        assert "occupational" not in row["prompt"]
        reverse = next(
            r
            for r in comparisons
            if r["comparison_unit_id"] == row["comparison_unit_id"] and r["order"] != row["order"]
        )
        assert row["image_a_sha256"] == reverse["image_b_sha256"]


def test_text_tasks_keep_system_instructions():
    tasks = json.loads((ROOT / "data/frozen/local_tasks.json").read_text())
    for task in tasks:
        if task["task_kind"] in {"bold", "decodingtrust"}:
            assert task["system_prompt"]
            assert not task["image_paths"]


def test_budget_reserves_inflight_calls():
    class Client:
        def credits(self):
            return {"total_usage": 8.0, "remaining": 10.15}

    budget = Budget(Client(), initial_usage=8.0)
    budget.acquire(0.10)
    with pytest.raises(RuntimeError):
        budget.acquire(0.10)
    budget.release(0.10)
    budget.acquire(0.10)


def test_pipeline_extracts_only_assistant():
    assert extract_text([{"generated_text": "Joy, Love, Cheer"}]) == "Joy, Love, Cheer"
    assert (
        extract_text(
            [
                {
                    "generated_text": [
                        {"role": "user", "content": "negative"},
                        {"role": "assistant", "content": "Love"},
                    ]
                }
            ]
        )
        == "Love"
    )


def test_local_worker_has_no_training_or_model_save():
    source = (ROOT / "scripts/run_local_gemma.py").read_text()
    assert "model.requires_grad_(False)" in source
    assert "model.eval()" in source
    assert ".backward(" not in source
    assert "save_pretrained(" not in source
    assert Path(ROOT / "data/frozen/local_tasks.json").is_file()
