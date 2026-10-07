"""Check saved notebook image outputs for exact missing-image hashes only."""
import base64
import hashlib
import io
import json

from PIL import Image

import study


def main():
    manifest = json.loads((study.ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    wanted = {row["image_sha256"]: row for row in manifest["missing_images"]}
    asset_root = study.ROOT / "data/private/assets"
    report = {"checked_at": study.now(), "scope": "Existing repository notebook image outputs only",
              "match_rule": "Exact source PNG SHA-256; no approximate replacement", "notebooks": [],
              "candidate_images": 0, "matches": []}
    notebooks = [path for path in study.REPO.rglob("*.ipynb")
                 if "runtime" not in path.parts and ".git" not in path.parts and "node_modules" not in path.parts]
    for notebook in sorted(notebooks):
        data = json.loads(notebook.read_text(encoding="utf-8"))
        count = 0
        for cell_index, cell in enumerate(data.get("cells", [])):
            for output_index, output in enumerate(cell.get("outputs", [])):
                for mime in ("image/png", "image/jpeg"):
                    encoded = output.get("data", {}).get(mime)
                    if not encoded:
                        continue
                    raw = base64.b64decode("".join(encoded) if isinstance(encoded, list) else encoded)
                    variants = [("original_bytes", raw)]
                    with Image.open(io.BytesIO(raw)) as image:
                        buffer = io.BytesIO()
                        image.convert("RGB").save(buffer, format="PNG")
                        variants.append(("rgb_png", buffer.getvalue()))
                    count += 1
                    report["candidate_images"] += 1
                    for encoding, png in variants:
                        digest = hashlib.sha256(png).hexdigest()
                        if digest not in wanted:
                            continue
                        row = wanted[digest]
                        target = asset_root / row["image_path"]
                        if not target.resolve().is_relative_to(asset_root.resolve()):
                            raise ValueError("Unsafe source image path")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists() and study.sha(target) != digest:
                            raise ValueError("Refusing to overwrite a mismatched image")
                        if not target.exists():
                            target.write_bytes(png)
                        report["matches"].append({**row, "notebook": str(notebook.relative_to(study.REPO)),
                                                   "cell_index": cell_index, "output_index": output_index,
                                                   "encoding": encoding})
                        del wanted[digest]
        report["notebooks"].append({"path": str(notebook.relative_to(study.REPO)), "embedded_images": count})
    report["remaining_missing_hashes"] = sorted(wanted)
    study.save_json(study.ROOT / "results/embedded_image_recovery.json", report)
    print(f"Checked {len(notebooks)} notebooks / {report['candidate_images']} embedded images; "
          f"recovered {len(report['matches'])} exact missing source files")
    if report["matches"]:
        return study.prepare(asset_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
