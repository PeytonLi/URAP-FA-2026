"""Freeze the September 11 inputs without modifying historical experiments."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / "research-evidence-september5"


def main():
    records = []

    def copy(source, relative):
        target = ROOT / relative
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing input differs: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        records.append(dict(source=str(source), path=str(relative), sha256=digest))

    for p in (OLD / "src/multimodel_bias").glob("*.py"):
        copy(p, p.relative_to(OLD))
    for name in ("requirements.lock", "LICENSE"):
        copy(OLD / name, name)
    for p in (OLD / "locks").glob("*.json"):
        copy(p, Path("reference/locks") / p.name)
    for dataset in ("neutral12", "utkface"):
        copy(OLD / f"results/analysis/iat_{dataset}_responses.csv",
             f"reference/iat_{dataset}_responses.csv")
    for name in ("iat_utkface_aug31.json", "models_aug31.json"):
        copy(OLD / "config" / name, Path("reference") / name)
    copy(OLD / "reference/august24/results/iat12/run_manifest.json",
         "reference/neutral12_run_manifest.json")
    copy(OLD / "report/dpo_training_record.json", "reference/dpo_training_record.json")
    copy(OLD / "report/MAIN_RESULTS_TABLE.csv", "reference/september5_results.csv")
    for name in ("benchmark_matrix.csv", "METHODS.md", "TASK_STATUS.md"):
        copy(OLD / "report" / name, Path("reference") / name)
    for name in ("FINE_TUNING_HISTORY.md",):
        copy(OLD / name, Path("reference") / name)
    for relative in ("data/iat12/manifest.csv", "data/frozen/utkface/utkface_iat_evaluation_manifest.csv"):
        source = OLD / relative
        copy(source, relative)
        with source.open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            p = (source.parent / row["source_relative_path"]).resolve()
            if hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError("Image hash mismatch")
            copy(p, p.relative_to(OLD))
    checkpoint = ROOT.parent / "gemma3-fairface-debiasing-august10/outputs/august10/training_v4e_meanlogp_lr2e5_rank8_seed8102026/checkpoints/final"
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        copy(checkpoint / name, Path("data/private/adapter") / name)
    configs = {}
    for panel, seed, repeats, fallback, manifest in (
        ("neutral12", 17082026, 50, 30, "data/iat12/manifest.csv"),
        ("utkface", 31082026, 5, 128, "data/frozen/utkface/utkface_iat_evaluation_manifest.csv"),
    ):
        source = "neutral12_run_manifest.json" if panel == "neutral12" else "models_aug31.json"
        models = json.loads((ROOT / "reference" / source).read_text())["models"]
        if panel == "neutral12":
            next(m for m in models if m["key"] == "mimo_v25")["provider_endpoint"] = "deepinfra/fp8"
        configs[panel] = dict(base_seed=seed, repeats=repeats, fallback_max_tokens=fallback,
                             manifest=manifest, models=models)
    config = dict(study_id="iat-expansion-september11-v1", panels=configs,
                  openrouter_max_usd=20.0, openrouter_min_balance_usd=1.0,
                  runpod_max_usd=10.0, runpod_max_seconds=9500,
                  network_volume_id="zecocmq95n", data_center_id="US-NE-1",
                  gpu_type_id="NVIDIA H100 80GB HBM3", bootstrap_samples=100000,
                  seed=9112026, local_seed_contract="per-image-iteration-sha256-v1",
                  gemma_revision="96b6f1eccf38110c56df3a15bffe176da04bfd80",
                  adapter_sha256="aa869029c73588c2cb5ff1bf805466f78e995de3e888ac7ff7eb567f8d421053",
                  qwen_revision="66285546d2b821cf421d4f5eb2576359d3770cd3")
    (ROOT / "config").mkdir(exist_ok=True)
    target = ROOT / "config/study.json"
    if target.exists() and json.loads(target.read_text()) != config:
        raise ValueError("Configuration freeze differs")
    target.write_text(json.dumps(config, indent=2) + "\n")
    (ROOT / "reference/input_checksums.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Frozen {len(records)} files; images/adapter remain private")


if __name__ == "__main__":
    main()
