import hashlib
import json

import pytest

from common import ROOT, local_seed, rows
from hosted import plan_panel
from multimodel_bias.constants import IAT_WORDS

private_inputs = pytest.mark.skipif(not (ROOT / "data/iat12/images/bf14_nc.jpg").exists(),
                                   reason="Private image/adapter bundle is required for input checks")


@private_inputs
def test_reuse_counts_and_seed_order():
    for panel, expected in (("neutral12", 840), ("utkface", 3200)):
        cfg, models, images, archived = plan_panel(panel)
        assert len(archived) == expected
        assert len(models) == 8
        index = {image.image_id: i for i,image in enumerate(images,1)}
        assert all(int(r["seed"]) == cfg.base_seed+1000*int(r["iteration"])+index[r["image_id"]] for r in archived)


def test_seed_pairing_and_variation():
    assert local_seed("utkface", "id1", 1) == local_seed("utkface", "id1", 1)
    assert local_seed("utkface", "id1", 1) != local_seed("utkface", "id1", 2)


@private_inputs
def test_vocabulary_and_private_image_hashes():
    assert len(IAT_WORDS) == len(set(IAT_WORDS)) == 61
    for rel in ("data/iat12/manifest.csv", "data/frozen/utkface/utkface_iat_evaluation_manifest.csv"):
        path = ROOT / rel
        for row in rows(path):
            image = path.parent / row["source_relative_path"]
            assert hashlib.sha256(image.read_bytes()).hexdigest() == row["sha256"]


@private_inputs
def test_adapter_not_collapsed_diagnostic():
    config = json.loads((ROOT / "config/study.json").read_text())
    assert hashlib.sha256((ROOT / "data/private/adapter/adapter_model.safetensors").read_bytes()).hexdigest() == config["adapter_sha256"]
    assert config["openrouter_max_usd"] == 20
