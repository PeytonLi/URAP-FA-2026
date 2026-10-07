#!/usr/bin/env python3
"""Record installed packages and tested environment without editable absolute paths."""

import importlib.metadata
import json
import platform
import sys

from common import ROOT, write_json


def main():
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"] not in {"research-evidence-september5", "pip", "setuptools"}
    }
    lock = ROOT / "locks"
    lock.mkdir(exist_ok=True)
    (lock / "analysis-macos-py311.txt").write_text(
        "\n".join(
            f"{name}=={version}"
            for name, version in sorted(packages.items(), key=lambda p: p[0].lower())
        )
        + "\n"
    )
    write_json(
        lock / "analysis-environment.json",
        dict(python=sys.version, platform=platform.platform(), packages=packages),
    )
    for src, dest in (
        ("results/gemma_local/manifest.json", "gemma-linux-environment.json"),
        ("results/gemma_bold/scorer_manifest.json", "bold-scorer-environment.json"),
    ):
        source = ROOT / src
        if source.exists():
            data = json.loads(source.read_text())
            write_json(
                lock / dest,
                {
                    k: data[k]
                    for k in (
                        "environment",
                        "packages",
                        "gpu",
                        "precision",
                        "attention_implementation",
                        "regard_model",
                        "regard_revision",
                        "detoxify_checkpoint_sha256",
                    )
                    if k in data
                },
            )
    print("Environment locks captured")


if __name__ == "__main__":
    main()
