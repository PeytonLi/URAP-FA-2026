"""Verify a supplied public or private release without third-party dependencies."""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    folder = parser.parse_args().folder.resolve()
    checked = 0
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        file = (folder / relative).resolve()
        if not file.is_relative_to(folder):
            raise ValueError("Checksum path escapes package")
        h = hashlib.sha256()
        with file.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        if h.hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
        checked += 1
    print(f"Verified {checked} file checksums")


if __name__ == "__main__":
    main()
