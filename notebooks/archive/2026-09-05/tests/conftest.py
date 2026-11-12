from pathlib import Path

import pytest


def pytest_collection_modifyitems(items):
    root = Path(__file__).resolve().parents[1]
    # Public score/plot replay excludes licensed images, not their frozen manifest.
    if not any((root / "data/iat12/images").glob("*")):
        for item in items:
            if item.nodeid == "tests/test_iat_contract.py::test_frozen_stimuli_hashes_and_balance":
                item.add_marker(
                    pytest.mark.skip(
                        reason="Private neutral12 image bytes are absent; required only for fresh inference/hash verification"
                    )
                )
