from __future__ import annotations

from pathlib import Path

from PIL import Image

from multimodel_bias.analysis import _image_level_rows, _model_metrics
from multimodel_bias.iat import ModelSpec, load_stimuli
from multimodel_bias.uncertainty import stratified_binary_label_permutation
from multimodel_bias.utkface import (
    build_matched_pools,
    parse_utkface_filename,
    validate_frozen_manifests,
    write_frozen_utkface_artifacts,
)
from multimodel_bias.visbias import (
    _effective_max_tokens,
    _parse_comparison,
    _seed,
    _task_contract_sha256,
)
from multimodel_bias.visbias_analysis import score_descriptions


def test_parse_utkface_filename_contract() -> None:
    parsed = parse_utkface_filename("32_1_1_20170116180211518.jpg.chip.jpg")
    assert parsed == {
        "age": 32,
        "gender_code": 1,
        "gender": "Female",
        "race_code": 1,
        "race": "Black",
    }
    assert parse_utkface_filename("61_1_20170109150557335.jpg.chip.jpg") is None


def test_custom_stimulus_paths_are_normalized(tmp_path: Path) -> None:
    image_dir = tmp_path / "private" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "face.jpg"
    Image.new("RGB", (12, 12), (10, 20, 30)).save(image_path)
    import hashlib

    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    manifest_dir = tmp_path / "frozen" / "inputs"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "manifest.csv"
    manifest.write_text(
        "image_id,filename,race,gender,width,height,sha256,source_relative_path\n"
        f"sample,face.jpg,Black,Female,12,12,{digest},../../private/images/face.jpg\n",
        encoding="utf-8",
    )
    stimuli = load_stimuli(manifest, expected_count=1)
    assert stimuli[0].path == image_path.resolve()


def test_iat_model_metrics_count_images_with_and_without_valid_outputs() -> None:
    metrics = _model_metrics(
        [
            {
                "image_id": "a",
                "race": "Black",
                "gender": "Female",
                "valid_three": True,
                "negative_count": 1,
                "word_count": 3,
                "words": "Pain|Joyful|Love",
                "cost_usd": 0,
            },
            {
                "image_id": "b",
                "race": "White",
                "gender": "Female",
                "valid_three": False,
                "negative_count": 2,
                "word_count": 2,
                "words": "Pain|Ugly",
                "cost_usd": 0,
            },
        ],
        bootstrap_samples=10,
        bootstrap_seed=31,
    )
    assert metrics["distinct_images"] == 2
    assert metrics["images_with_valid_outputs"] == 1
    assert metrics["gap_ci_low"] is None


def test_iat_image_inference_excludes_partial_invalid_outputs() -> None:
    rows = _image_level_rows(
        [
            {
                "image_id": "valid",
                "race": "Black",
                "gender": "Female",
                "valid_three": True,
                "negative_count": 1,
                "word_count": 3,
            },
            {
                "image_id": "partial",
                "race": "White",
                "gender": "Female",
                "valid_three": False,
                "negative_count": 2,
                "word_count": 2,
            },
        ]
    )
    assert rows == [
        {
            "image_id": "valid",
            "race": "Black",
            "gender": "Female",
            "negative_rate": 1 / 3,
        }
    ]


def _fake_selected_row(path: Path, race: str, gender: str, index: int) -> dict:
    Image.new("RGB", (12, 12), (index * 10, 20, 30)).save(path)
    return {
        "source_filename": path.name,
        "image_path": str(path),
        "age": 30 + index,
        "age_band": "30_44",
        "gender_code": 1 if gender == "Female" else 0,
        "gender": gender,
        "race_code": 1 if race == "Black" else 0,
        "race": race,
        "width": 12,
        "height": 12,
        "brightness": 0.5,
        "contrast": 0.2,
        "edge_energy": 0.1,
        "quality_score": 0.7,
        "quality_bin": "high",
        "fer_label": "neutral",
        "fer_confidence": 0.9,
        "expression_class": "neutral",
        "expression_model_id": "test/fer",
        "expression_model_revision": "abc123",
    }


def test_utkface_eval_and_finetuning_pools_are_disjoint(tmp_path: Path) -> None:
    rows = []
    index = 0
    for gender in ("Female", "Male"):
        for pair in range(2):
            for race in ("Black", "White"):
                index += 1
                path = tmp_path / f"{gender}_{pair}_{race}.jpg"
                rows.append(_fake_selected_row(path, race, gender, index))
    duplicate_path = tmp_path / "duplicate.jpg"
    duplicate_path.write_bytes(Path(rows[0]["image_path"]).read_bytes())
    duplicate_row = dict(rows[0])
    duplicate_row.update(
        {"source_filename": duplicate_path.name, "image_path": str(duplicate_path)}
    )
    rows.append(duplicate_row)
    selected, report = build_matched_pools(
        rows,
        evaluation_per_intersection=1,
        finetuning_per_intersection=1,
        min_expression_confidence=0.5,
        min_quality_score=0.3,
        seed=31,
    )
    assert report["pool_counts"] == {"evaluation": 4, "future_finetuning": 4}
    assert report["exact_duplicate_source_files_removed"] == 1
    output = tmp_path / "data" / "frozen" / "utkface"
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"archive")
    write_frozen_utkface_artifacts(
        selected,
        report,
        output,
        source_archive_sha256="test",
        source_revision="revision",
        expression_model_id="test/fer",
        expression_model_revision="abc123",
    )
    validation = validate_frozen_manifests(output)
    assert validation == {
        "evaluation_images": 4,
        "finetuning_images": 4,
        "candidate_tasks": 8,
        "evaluation_finetuning_overlap": 0,
    }


def test_large_stratified_permutation_uses_monte_carlo() -> None:
    rows = [
        {
            "value": float(index % 3),
            "race": "Black" if index % 2 else "White",
            "gender": "Female" if index % 4 < 2 else "Male",
        }
        for index in range(40)
    ]
    observed, p_value, permutations, method = stratified_binary_label_permutation(
        rows,
        "value",
        "race",
        "gender",
        group_a="Black",
        group_b="White",
        samples=500,
        seed=31,
        exact_limit=10,
    )
    assert method == "monte_carlo"
    assert permutations == 500
    assert -2 <= observed <= 2
    assert 0 < p_value <= 1


def test_visbias_comparison_parser() -> None:
    assert _parse_comparison("A") == "A"
    assert _parse_comparison('{"answer": "B"}') == "B"
    assert _parse_comparison("Cannot determine from these images.") == "CANNOT_DETERMINE"
    assert _parse_comparison("It depends.") == "UNCLASSIFIED"


def test_visbias_seed_is_independent_of_model_subset_order() -> None:
    assert _seed(31, "qwen", 7) == _seed(31, "qwen", 7)
    assert _seed(31, "qwen", 7) != _seed(31, "grok", 7)
    assert 0 <= _seed(31_082_026, "gemini36_flash", 252) < 2**31


def test_visbias_reasoning_model_keeps_its_minimum_total_token_budget() -> None:
    common = {
        "display_name": "Test",
        "model_id": "test/model",
        "api_revision_hint": "test/model-revision",
        "release_date": "2026-01-01",
        "weight_access": "open-weight",
        "provider_endpoint": "test",
        "max_concurrency": 1,
        "supports_temperature": True,
        "supports_top_p": True,
        "max_tokens": 1024,
    }
    reasoning = ModelSpec(key="reasoning", reasoning_effort="minimal", **common)
    direct = ModelSpec(key="direct", reasoning_effort="none", **common)
    assert _effective_max_tokens(reasoning, 220) == 1024
    assert _effective_max_tokens(direct, 220) == 220


def test_visbias_task_contract_detects_prompt_or_image_drift() -> None:
    base = {
        "task_key": "description|1",
        "task_kind": "description",
        "image_hashes": ["abc"],
        "prompt_sha256": "prompt-a",
    }
    assert _task_contract_sha256([base]) == _task_contract_sha256([dict(base)])
    assert _task_contract_sha256([base]) != _task_contract_sha256(
        [{**base, "prompt_sha256": "prompt-b"}]
    )


def test_visbias_description_scoring() -> None:
    rows = score_descriptions(
        [
            {
                "task_kind": "description",
                "race": "black",
                "response": "A cheerful person stands in a bright room.",
            }
        ],
        {"black": ["criminal"], "white": ["rich"]},
    )
    assert len(rows) == 1
    assert rows[0]["word_count"] == 8
    assert rows[0]["race_stereotype_hits"] == 0
    assert rows[0]["identity_mention"] == 0


def test_visbias_identity_scoring_ignores_colors_but_counts_person_identity() -> None:
    rows = score_descriptions(
        [
            {
                "task_kind": "description",
                "race": "black",
                "response": "A person wears a black suit and white shirt.",
            },
            {
                "task_kind": "description",
                "race": "black",
                "response": "The image shows a Black man in a suit.",
            },
        ],
        {"black": [], "white": []},
    )
    assert [row["identity_mention_count"] for row in rows] == [0, 1]
