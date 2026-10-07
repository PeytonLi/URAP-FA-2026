from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from multimodel_bias.artifacts import sanitize_for_artifact
from multimodel_bias.bold import RACE_CATEGORIES
from multimodel_bias.bold import load_race_tasks as load_bold_race_tasks
from multimodel_bias.bold import load_tasks as load_bold_tasks
from multimodel_bias.constants import (
    IAT_WORDS,
    NEGATIVE_IAT_WORDS,
    POSITIVE_IAT_WORDS,
)
from multimodel_bias.decodingtrust import RACE_DEMOGRAPHIC_GROUPS, classify_response
from multimodel_bias.decodingtrust import (
    load_race_tasks as load_decodingtrust_race_tasks,
)
from multimodel_bias.decodingtrust import (
    load_tasks as load_decodingtrust_tasks,
)
from multimodel_bias.iat import (
    ModelSpec,
    dedupe_records,
    deterministic_seed,
    is_exact_three_word_format,
    load_model_specs,
    load_stimuli,
    parse_three_words,
    validated_model_snapshot,
)
from multimodel_bias.openrouter import OpenRouterClient, OpenRouterError
from multimodel_bias.text_runner import (
    TextBenchmarkRunner,
    TextRunConfig,
    TextTask,
    _task_id,
    freeze_selected_prompts,
)
from multimodel_bias.uncertainty import (
    benjamini_hochberg,
    permutation_group_range_pvalue,
    stratified_bootstrap_mean_interval,
    stratified_exact_binary_label_permutation,
    wilson_interval,
)


def test_historical_and_api_prompts_differ_only_by_image_transport() -> None:
    from multimodel_bias.constants import IAT_PROMPT

    historical = f"<start_of_image> {IAT_PROMPT}"
    assert historical.removeprefix("<start_of_image> ") == IAT_PROMPT


def test_historical_lexicon_sizes() -> None:
    assert len(POSITIVE_IAT_WORDS) == 30
    assert len(NEGATIVE_IAT_WORDS) == 31
    assert len(IAT_WORDS) == 61
    assert len(set(IAT_WORDS)) == 61


def test_pipeline_compatible_parser() -> None:
    assert parse_three_words("Happy, Delight, Joyous") == ["Happy", "Delight", "Joyous"]
    assert parse_three_words("The selections are pain; GRIEF; and lovely.") == [
        "Pain",
        "Grief",
        "Lovely",
    ]
    assert parse_three_words("unknown words only") == []


def test_exact_format_is_separate_from_parse_validity() -> None:
    words = parse_three_words("Happy, Delight, Joyous")
    assert is_exact_three_word_format("Happy, Delight, Joyous", words)
    assert not is_exact_three_word_format("The words are Happy, Delight, Joyous", words)


def test_subset_resume_preserves_full_text_run_model_contract(tmp_path: Path) -> None:
    models = [
        ModelSpec(
            key=f"model_{index}",
            display_name=f"Model {index}",
            model_id=f"vendor/model-{index}",
            api_revision_hint=f"vendor/model-{index}-revision",
            release_date="2026-01-01",
            weight_access="open-weight",
            provider_endpoint=f"provider/route-{index}",
            max_concurrency=1,
            supports_temperature=True,
            supports_top_p=True,
            reasoning_effort=None,
            max_tokens=16,
        )
        for index in range(2)
    ]
    task = TextTask(key="task", prompt="Prompt", system_prompt="", metadata={}, max_tokens=8)
    config = TextRunConfig(
        benchmark_version="test-v1",
        repeats=1,
        temperature=1.0,
        top_p=1.0,
        base_seed=1,
        max_workers=1,
        minimum_request_interval_seconds=0,
        max_run_cost_usd=1,
        minimum_account_balance_usd=0,
        provider={"allow_fallbacks": False},
    )

    class FakeClient:
        timeout_seconds = 1
        max_retries = 0

        def credits(self) -> dict[str, float]:
            return {"total_credits": 10.0, "total_usage": 0.0, "remaining": 10.0}

        def key_metadata(self) -> dict[str, object]:
            return {}

        def model_catalog(self) -> list[dict[str, object]]:
            return [
                {"id": model.model_id, "canonical_slug": model.api_revision_hint}
                for model in models
            ]

        def get(self, path: str) -> dict[str, object]:
            model = next(model for model in models if model.model_id in path)
            return {
                "data": {
                    "endpoints": [
                        {"tag": model.provider_endpoint, "status": 0, "provider_name": "Test"}
                    ]
                }
            }

        def chat_completion(self, _payload: dict[str, object]) -> dict[str, object]:
            raise AssertionError("A completed subset resume must not make a paid call")

    previous_manifest = {"models": [models[0].__dict__]}
    (tmp_path / "run_manifest.json").write_text(json.dumps(previous_manifest), encoding="utf-8")
    existing = [
        {
            "task_id": _task_id(config.benchmark_version, model, task, 1),
            "status": "completed",
            "model_key": model.key,
            "iteration": 1,
            "task_key": task.key,
            "cost_usd": 0,
        }
        for model in models
    ]
    (tmp_path / "responses.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in existing),
        encoding="utf-8",
    )
    runner = TextBenchmarkRunner(
        FakeClient(),  # type: ignore[arg-type]
        "resume_test",
        config,
        [models[0]],
        [task],
        tmp_path,
        source_manifest={},
        model_universe=models,
    )
    records, manifest = runner.run()
    assert len(records) == 2
    assert [model["key"] for model in manifest["models"]] == ["model_0", "model_1"]
    assert manifest["planned_records"] == 2


def test_frozen_prompts_reject_task_set_change_after_responses(tmp_path: Path) -> None:
    path = tmp_path / "selected_prompts.csv"
    freeze_selected_prompts(path, [{"task_key": "first", "prompt": "First"}])
    (tmp_path / "responses.jsonl").write_text("{}\n", encoding="utf-8")
    try:
        freeze_selected_prompts(path, [{"task_key": "second", "prompt": "Second"}])
    except ValueError as exc:
        assert "Use a new run ID" in str(exc)
    else:
        raise AssertionError("Expected a changed prompt set to be rejected")


def test_frozen_prompts_allow_exact_resume(tmp_path: Path) -> None:
    path = tmp_path / "selected_prompts.csv"
    prompts = [{"task_key": "first", "prompt": "First"}]
    freeze_selected_prompts(path, prompts)
    (tmp_path / "responses.csv").write_text("task_key\nfirst\n", encoding="utf-8")
    freeze_selected_prompts(path, prompts)
    assert path.read_text(encoding="utf-8") == "prompt,task_key\nFirst,first\n"


def test_frozen_stimuli_hashes_and_balance() -> None:
    stimuli = load_stimuli()
    assert len(stimuli) == 12
    assert sum(stimulus.race == "Black" for stimulus in stimuli) == 6
    assert sum(stimulus.race == "White" for stimulus in stimuli) == 6
    assert sum(stimulus.gender == "Female" for stimulus in stimuli) == 6
    assert sum(stimulus.gender == "Male" for stimulus in stimuli) == 6
    assert len({stimulus.sha256 for stimulus in stimuli}) == 12


def test_default_eight_model_contracts() -> None:
    _, models = load_model_specs()
    assert len(models) == 8
    assert len({model.key for model in models}) == 8
    assert len({model.model_id for model in models}) == 8
    assert {model.weight_access for model in models} == {"open-weight", "closed-source"}


def test_august24_explicit_model_contracts() -> None:
    path = Path(__file__).parents[1] / "config" / "models_explicit_aug24.json"
    _, models = load_model_specs(path)
    assert len(models) == 8
    assert len({model.key for model in models}) == 8
    assert len({model.model_id for model in models}) == 8
    assert "google/gemma-3-12b-it" not in {model.model_id for model in models}


def test_model_catalog_alias_drift_is_rejected() -> None:
    _, models = load_model_specs()
    client = OpenRouterClient("not-a-real-key")
    client.model_catalog = lambda: [  # type: ignore[method-assign]
        {"id": model.model_id, "canonical_slug": "changed/revision"} for model in models
    ]
    try:
        validated_model_snapshot(client, models)
    except ValueError as exc:
        assert "alias drift" in str(exc)
    else:
        raise AssertionError("Expected alias drift to stop the run")


def test_model_endpoint_drift_is_rejected() -> None:
    _, models = load_model_specs()
    model = models[0]
    client = OpenRouterClient("not-a-real-key")
    client.model_catalog = lambda: [  # type: ignore[method-assign]
        {
            "id": model.model_id,
            "canonical_slug": model.api_revision_hint,
        }
    ]
    client.get = lambda _path: {  # type: ignore[method-assign]
        "data": {"endpoints": [{"tag": "different/route", "status": 0}]}
    }
    try:
        validated_model_snapshot(client, [model])
    except ValueError as exc:
        assert "endpoint drift" in str(exc)
    else:
        raise AssertionError("Expected endpoint drift to stop the run")


def test_model_snapshot_records_selected_endpoint() -> None:
    _, models = load_model_specs()
    model = models[0]
    client = OpenRouterClient("not-a-real-key")
    client.model_catalog = lambda: [  # type: ignore[method-assign]
        {
            "id": model.model_id,
            "canonical_slug": model.api_revision_hint,
        }
    ]
    client.get = lambda _path: {  # type: ignore[method-assign]
        "data": {
            "endpoints": [
                {
                    "tag": model.provider_endpoint,
                    "provider_name": "Pinned Provider",
                    "quantization": "bf16",
                    "status": 0,
                }
            ]
        }
    }
    snapshot = validated_model_snapshot(client, [model])
    assert snapshot[0]["selected_endpoint"] == {
        "name": None,
        "tag": model.provider_endpoint,
        "provider_name": "Pinned Provider",
        "quantization": "bf16",
        "context_length": None,
        "max_prompt_tokens": None,
        "max_completion_tokens": None,
        "pricing": None,
        "supported_parameters": None,
        "status": 0,
    }


def test_seed_schedule_is_reproducible() -> None:
    assert deterministic_seed(17082026, 1, 1) == 17083027
    assert deterministic_seed(17082026, 2, 1) != deterministic_seed(17082026, 1, 1)


def test_embedded_provider_error_is_retried() -> None:
    client = OpenRouterClient("not-a-real-key", max_retries=1)
    responses = iter(
        [
            {"error": {"code": 429, "message": "temporarily rate-limited"}},
            {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"cost": 0.0},
            },
        ]
    )
    client._request = lambda *_args, **_kwargs: next(responses)  # type: ignore[method-assign]
    result = client.chat_completion({"model": "test"})
    assert result["choices"][0]["message"]["content"] == "OK"
    assert result["_client"]["retry_count"] == 1


def test_non_retryable_embedded_provider_error_fails() -> None:
    client = OpenRouterClient("not-a-real-key", max_retries=0)
    client._request = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "error": {"code": 400, "message": "invalid parameter"}
    }
    try:
        client.chat_completion({"model": "test"})
    except OpenRouterError as exc:
        assert exc.failure.status_code == 400
    else:
        raise AssertionError("Expected OpenRouterError")


def test_decodingtrust_classifier_contract() -> None:
    assert classify_response("This is false. I disagree.") == "DISAGREE"
    assert classify_response("That is correct. I agree.") == "AGREE"
    assert classify_response("As an AI, I cannot provide a response.") == "NO_RESPONSE"
    # This counterintuitive edge case is retained from DecodingTrust's classifier.
    assert classify_response("As an AI, I cannot agree or disagree.") == "DISAGREE"
    assert classify_response("I agree. I disagree.") == "NO_RESPONSE"


def test_wilson_interval_contract() -> None:
    low, high = wilson_interval(0, 0)
    assert math.isnan(low) and math.isnan(high)
    low, high = wilson_interval(0, 16)
    assert low == 0.0
    assert 0.19 < high < 0.20
    low, high = wilson_interval(0, 25)
    assert low == 0.0
    assert 0.13 < high < 0.14
    low, high = wilson_interval(16, 16)
    assert 0.80 < low < 0.81
    assert high == 1.0


def test_stratified_bootstrap_interval_contract() -> None:
    rows = [
        {"domain": "one", "value": 0},
        {"domain": "one", "value": 0},
        {"domain": "two", "value": 1},
        {"domain": "two", "value": 1},
    ]
    low, high = stratified_bootstrap_mean_interval(
        rows,
        "value",
        "domain",
        samples=1_000,
        seed=42,
    )
    assert low == high == 0.5


def test_permutation_and_multiple_testing_contracts() -> None:
    equal = [
        {"group": "a", "value": 0},
        {"group": "a", "value": 1},
        {"group": "b", "value": 0},
        {"group": "b", "value": 1},
    ]
    observed, p_value = permutation_group_range_pvalue(
        equal,
        "value",
        "group",
        samples=1_000,
        seed=42,
    )
    assert observed == 0
    assert p_value == 1
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.04, 0.04]


def test_exact_stratified_binary_permutation_contract() -> None:
    rows = [
        {"group": "a", "stratum": "x", "value": 1.0},
        {"group": "b", "stratum": "x", "value": 0.0},
        {"group": "a", "stratum": "y", "value": 1.0},
        {"group": "b", "stratum": "y", "value": 0.0},
    ]
    observed, p_value, permutations = stratified_exact_binary_label_permutation(
        rows,
        "value",
        "group",
        "stratum",
        group_a="a",
        group_b="b",
    )
    assert observed == 1.0
    assert p_value == 0.5
    assert permutations == 4


def test_decodingtrust_smoke_selection_is_balanced() -> None:
    tasks = load_decodingtrust_tasks(16, 17082026)
    assert len(tasks) == 48
    assert len({task.key for task in tasks}) == 48
    assert Counter(task.metadata["regime"] for task in tasks) == {
        "benign": 16,
        "untargeted": 16,
        "targeted": 16,
    }
    assert set(Counter(task.metadata["topic"] for task in tasks).values()) == {3}
    assert set(Counter(task.metadata["demographic_group"] for task in tasks).values()) == {2}


def test_decodingtrust_race_selection_is_matched() -> None:
    tasks = load_decodingtrust_race_tasks(4, 17082026)
    assert len(tasks) == 48
    assert Counter(task.metadata["demographic_group"] for task in tasks) == {
        group: 12 for group in RACE_DEMOGRAPHIC_GROUPS
    }
    assert set(Counter(task.metadata["regime"] for task in tasks).values()) == {16}
    assert set(Counter(task.metadata["topic"] for task in tasks).values()) == {12}


def test_bold_smoke_selection_is_balanced_and_reproducible() -> None:
    first = load_bold_tasks(25, 17082026)
    second = load_bold_tasks(25, 17082026)
    assert [task.key for task in first] == [task.key for task in second]
    assert len({task.key for task in first}) == 25
    assert set(Counter(task.metadata["domain"] for task in first).values()) == {5}
    assert {task.max_tokens for task in first} == {1024}
    assert all("Continue the user-provided text" in task.system_prompt for task in first)


def test_bold_race_selection_is_balanced() -> None:
    tasks = load_bold_race_tasks(10, 17082026)
    assert len(tasks) == 40
    assert Counter(task.metadata["category"] for task in tasks) == {
        category: 10 for category in RACE_CATEGORIES
    }
    assert {task.metadata["domain"] for task in tasks} == {"race"}


def test_key_metadata_removes_account_identifiers() -> None:
    client = OpenRouterClient("not-a-real-key")
    client._request = lambda *_args, **_kwargs: {  # type: ignore[method-assign]
        "data": {
            "label": "secret label",
            "creator_user_id": "private user id",
            "hash": "private hash",
            "usage": 1.25,
        }
    }
    assert client.key_metadata() == {"usage": 1.25}


def test_artifact_sanitizer_removes_identifiers_and_image_payloads() -> None:
    value = {
        "user_id": "private",
        "key": "model_key_must_remain",
        "name": "provider name must remain",
        "key_metadata": {"usage": 1.0},
        "nested": {"creator_user_id": "private", "safe": "kept"},
        "image": "data:" + "image/jpeg;base64,not-real",
    }
    assert sanitize_for_artifact(value) == {
        "key": "model_key_must_remain",
        "name": "provider name must remain",
        "nested": {"safe": "kept"},
        "image": "[redacted embedded image payload]",
    }


def test_dedupe_preserves_first_success_over_late_error() -> None:
    records = [
        {"task_id": "one", "status": "invalid", "response": ""},
        {"task_id": "one", "status": "completed", "response": "first success"},
        {"task_id": "one", "status": "completed", "response": "later success"},
        {"task_id": "one", "status": "invalid", "response": "late error"},
    ]
    assert dedupe_records(records) == [
        {"task_id": "one", "status": "completed", "response": "first success"}
    ]
