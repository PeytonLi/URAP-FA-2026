from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/reproduce_aug31.py"
SPEC = importlib.util.spec_from_file_location("reproduce_aug31", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_data_contract() -> None:
    checks = MODULE.validate_raw_data(ROOT)
    assert checks["utkface_rows"] == 2_800
    assert checks["utkface_valid_rows"] == 2_789
    assert checks["visbias_rows"] == 1_764
    assert checks["visbias_valid_rows"] == 1_762
    assert checks["utkface_cross_pool_hash_overlap"] == 0
    assert checks["future_dpo_tasks"] == 800


def test_all_nine_plots_regenerate(tmp_path: Path) -> None:
    result = MODULE.reproduce(ROOT, tmp_path)
    assert len(result["plots"]) == 9
    assert result["checks"]["archived_corrected_significant"] == 0
    assert result["checks"]["utkface_corrected_significant"] == 0
    assert result["checks"]["visbias_corrected_significant"] == 0
    for value in result["plots"]:
        path = Path(value)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert path.stat().st_size > 20_000
