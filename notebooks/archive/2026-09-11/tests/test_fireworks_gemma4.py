from fireworks_gemma4 import fireworks_payload, model_spec, run_config, write_manifest
from plot_pasted_iat import parse_rows
from multimodel_bias.constants import IAT_PROMPT
from multimodel_bias.constants import IAT_WORDS
from multimodel_bias.plots import REFERENCE_WORD_ORDER


def test_fireworks_payload_preserves_benchmark_contract():
    payload = fireworks_payload(
        "accounts/example/deployments/gemma4",
        "data:image/jpeg;base64,abc",
        run_config(workers=1),
        17083027,
        fallback=False,
    )

    content = payload["messages"][0]["content"]
    assert content[0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,abc"},
    }
    assert content[1] == {"type": "text", "text": IAT_PROMPT}
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 1.0
    assert payload["max_tokens"] == 30
    assert payload["seed"] == 17083027
    assert payload["reasoning_effort"] == "none"
    assert "provider" not in payload


def test_reference_graph_order_contains_each_iat_word_once():
    assert len(REFERENCE_WORD_ORDER) == len(set(REFERENCE_WORD_ORDER)) == 61
    assert set(REFERENCE_WORD_ORDER) == set(IAT_WORDS)
    assert REFERENCE_WORD_ORDER[:4] == ("Delight", "Happy", "Attractive", "Joyful")
    assert REFERENCE_WORD_ORDER[-4:] == ("Tragic", "Bothersome", "Hatred", "Sick")


def test_pasted_parser_ignores_generation_warnings():
    text = (
        "bf14_nc.jpg [iter 1] -> Adore, Delight, Attractive\n"
        "The following generation flags are not valid and may be ignored\n"
    )
    assert parse_rows(text) == [
        ("bf14_nc.jpg", 1, ["Adore", "Delight", "Attractive"])
    ]


def test_comparison_model_metadata_is_not_mislabeled_as_gemma(tmp_path):
    import json
    import pytest
    base = "accounts/fireworks/models/qwen3-vl-8b-instruct"
    model = model_spec("accounts/example/deployments/qwen", 4, key="qwen_base",
                       display_name="Qwen3-VL base", base_model=base)
    path = tmp_path / "manifest.json"
    write_manifest(path, model, run_config(4), [], [])
    contract = json.loads(path.read_text())["contract"]
    assert contract["logical_base_model"] == base
    assert contract["model"]["key"] == "qwen_base"
    assert contract["model"]["api_revision_hint"] == base
    assert contract["model"]["provider_endpoint"] == "fireworks/on-demand"
    serverless = model_spec(base, 4, key="qwen_base", display_name="Qwen3-VL base", base_model=base)
    assert serverless.provider_endpoint == "fireworks/serverless"
    with pytest.raises(ValueError, match="contract differs"):
        write_manifest(path, serverless, run_config(4), [], [])
