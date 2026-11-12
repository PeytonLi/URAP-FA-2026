"""Rebuild the authorized v4 ORPO recovered tree from the private HF backup.

The original run.json is unavailable. This writes a hub-restore receipt plus a
synthetic run.json sufficient for prompt-sweep resume, gated on the frozen
adapter SHA and pinned Hub revision.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import study

REPO_ID = "peytonali/urap-gemma3-12b-orpo-20261026T185926.436094-0000"
REVISION = "9d6a12b59f3dc64978cb652d139bd9ff62c321d7"
RUN_ID = "gemma3_12b_orpo_20261026T185926.436094_0000"
ADAPTER_SHA256 = "62b19d87bf77029ea565dc288d7ef509d086238c80b6fa1bccd0319e9a1e9c59"
ATTEMPT = "20261026T185926.436094_0000"


def restore(token: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise ValueError("HF token required")

    attempt = study.ROOT / "data/private/prime_native" / ATTEMPT
    recovered = attempt / "recovered"
    run_dir = recovered / "data/private/native_runs" / RUN_ID
    checkpoint = run_dir / "epoch_3"
    checkpoint.mkdir(parents=True, exist_ok=True)

    staging = attempt / "_hub_download"
    staging.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, revision=REVISION, local_dir=staging, token=token)

    summary = json.loads((staging / "training_summary.json").read_text(encoding="utf-8"))
    files = summary["source_checkpoint_files"]
    for name, digest in files.items():
        source = staging / name
        if not source.is_file():
            raise ValueError("Hub backup missing " + name)
        if study.sha(source) != digest:
            raise ValueError("Hub file checksum mismatch: " + name)
        (checkpoint / name).write_bytes(source.read_bytes())
        if study.sha(checkpoint / name) != digest:
            raise ValueError("Copied checkpoint checksum mismatch: " + name)
    if study.sha(checkpoint / "adapter_model.safetensors") != ADAPTER_SHA256:
        raise ValueError("Restored adapter is not the authorized v4 ORPO checkpoint")

    targets = study.REPO / study.dataset_config()["targets_path"]
    if not targets.is_file():
        raise ValueError("Local curated targets missing; regenerate before restore")
    targets_sha = study.sha(targets)

    receipt = {
        "status": "training_complete_evaluation_pending",
        "model": summary["model"],
        "weight_and_processor_revision": summary["weight_and_processor_revision"],
        "method": summary["method"],
        "seed": summary["seed"],
        "dataset_id": summary["dataset_id"],
        "manifest_sha256": summary["manifest_sha256"],
        "targets_sha256": targets_sha,
        "cohort": summary["cohort"],
        "targets_are_ground_truth": summary["targets_are_ground_truth"],
        "target_approval_kinds": summary["target_approval_kinds"],
        "pilot_class_weights": summary["pilot_class_weights"],
        "pilot_policy_sha256": summary["pilot_policy_sha256"],
        "training_images": summary["training_images"],
        "development_images": summary["development_images"],
        "test_images_used": summary["test_images_used"],
        "profile_targets_used": summary["profile_targets_used"],
        "optimizer_updates": summary["optimizer_updates"],
        "planned_optimizer_updates": summary["planned_optimizer_updates"],
        "completed_epochs": summary["completed_epochs"],
        "requested_epochs": summary["requested_epochs"],
        "logical_batch_size": summary["logical_batch_size"],
        "optimizer": summary["optimizer"],
        "learning_rate": summary["learning_rate"],
        "weight_decay": summary["weight_decay"],
        "orpo_coefficient": summary["orpo_coefficient"],
        "loss_reduction": summary["loss_reduction"],
        "loss_weighting": summary["loss_weighting"],
        "versions": summary["versions"],
        "development": summary["development"],
        "history": [],
        "checkpoints": [{"path": "epoch_3", "files": files}],
        "hub_restored": True,
        "hub_restore": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "adapter_sha256": ADAPTER_SHA256,
            "original_run_id": RUN_ID,
            "original_targets_sha256": summary["targets_sha256"],
            "local_targets_sha256": targets_sha,
            "note": ("Original run.json unavailable; adapter verified against frozen SHA "
                     "and Hub revision. Local targets were regenerated for packaging."),
        },
    }
    study.save_json(run_dir / "run.json", receipt)
    study.save_json(run_dir / "hub_backup.json", {
        "repo_id": REPO_ID,
        "url": "https://huggingface.co/" + REPO_ID,
        "private": True,
        "status": "verified",
        "last_verified_revision": REVISION,
        "commits": [{"checkpoint": "epoch_3", "revision": REVISION,
                     "files_sha256": {**files, "training_summary.json": study.sha(staging / "training_summary.json")},
                     "training_complete": True}],
    })
    study.save_json(run_dir / "hub_restore.json", receipt["hub_restore"])
    print(json.dumps({"recovered": str(recovered), "adapter_sha256": ADAPTER_SHA256,
                      "targets_sha256": targets_sha}, indent=2))
    return recovered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path)
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip() if args.token_file else None
    restore(token)


if __name__ == "__main__":
    main()
