"""Canonical repo paths. Python modules stay in scripts/."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DATASETS = REPO / "datasets"
CONFIGS = REPO / "configs"
DOCS = REPO / "docs"
NOTEBOOKS = REPO / "notebooks"
RESULTS = REPO / "results"
SCRIPTS = REPO / "scripts"
TESTS = REPO / "tests"

ANALYSIS = SCRIPTS / "analysis"
LABELING = SCRIPTS / "labeling" / "src"
TRAIN_EVAL = SCRIPTS / "train_eval"
MULTIMODEL_SRC = SCRIPTS / "archive" / "multimodel_bias_src"

CONFIG_V1 = CONFIGS / "v1" / "active_dataset_gated_draft.json"
CONFIG_V2 = CONFIGS / "v2" / "active_dataset_gated_v2_draft.json"
CONFIG_V4 = CONFIGS / "v4" / "active_dataset_gated_v4_draft.json"
PROMPT_SWEEP_CONFIGS = CONFIGS / "prompt_sweep"

DATASET_V1 = DATASETS / "v1" / "selected_images_gated.csv"
DATASET_V2 = DATASETS / "v2" / "selected_images_gated_v2.csv"
DATASET_V4 = DATASETS / "v4" / "selected_images_gated_v4.csv"
FAIRFACE_POOL = DATASETS / "shared" / "fairface_pool"
IAT12 = DATASETS / "shared" / "iat12"
PRIVATE_TARGETS = DATASETS / "private_targets"

RESULTS_BASE = RESULTS / "base" / "historical_baseline_v1"
RESULTS_V1 = RESULTS / "v1"
RESULTS_V2 = RESULTS / "v2"
RESULTS_V4_IAT = RESULTS / "v4" / "iat"
RESULTS_V4_STATS = RESULTS_V4_IAT / "iat_stats_v4"
RESULTS_V4_EXTRAS = RESULTS_V4_IAT / "extras_v4"
PROMPT_SWEEPS = RESULTS / "v4" / "prompt_sweeps"

TRAIN_EVAL_PRIVATE = TRAIN_EVAL / "_data" / "private"
TRAIN_EVAL_RUNTIME = TRAIN_EVAL / "_runtime"


def ensure_import_paths() -> Path:
    """Put analysis, train_eval, and multimodel packages on sys.path."""
    for path in (ANALYSIS, LABELING, TRAIN_EVAL, MULTIMODEL_SRC, SCRIPTS):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return REPO


PROMPT_SWEEP_PLANS = {
    "v1": PROMPT_SWEEP_CONFIGS / "iat_prompt_sweep_analysis.json",
    "A": PROMPT_SWEEP_CONFIGS / "iat_prompt_sweep_v2_analysis.json",
    "B": PROMPT_SWEEP_CONFIGS / "iat_prompt_sweep_v2_analysis.json",
    "C": PROMPT_SWEEP_CONFIGS / "iat_prompt_sweep_v3_analysis.json",
    "D": PROMPT_SWEEP_CONFIGS / "iat_prompt_sweep_v3_analysis.json",
}
