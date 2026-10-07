"""Export the checked public artifact inventory without private images or weights."""
import argparse
import shutil
from pathlib import Path

from common import ROOT, json_read, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists():
        raise ValueError("Destination already exists; never overwrite a historical release")
    inventory = json_read(ROOT / "artifact_checksums.json")
    for name, digest in inventory.items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or sha(source) != digest:
            raise ValueError(f"Source checksum/path mismatch: {name}")
    destination.mkdir(parents=True)
    for name in [*inventory, "artifact_checksums.json", "results/integrity.json"]:
        source, target = ROOT / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha(source) != sha(target):
            raise ValueError(f"Export checksum mismatch: {name}")
    print(f"Exported {len(inventory)} checked artifacts to {destination}")


if __name__ == "__main__":
    main()
