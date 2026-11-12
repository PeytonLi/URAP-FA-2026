from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multimodel_bias.constants import IAT_WORDS, NEGATIVE_IAT_SET  # noqa: E402
from multimodel_bias.iat import ModelSpec  # noqa: E402
from multimodel_bias.plots import _plot_model_word_counts  # noqa: E402

ROW_RE = re.compile(r"^(\S+\.jpg) \[iter (\d+)\] -> (.+)$")


def parse_rows(text: str) -> list[tuple[str, int, list[str]]]:
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = ROW_RE.match(line.strip())
        if not match:
            continue
        words = [word.strip() for word in match.group(3).split(",")]
        if len(words) != 3 or any(word not in IAT_WORDS for word in words):
            raise ValueError(f"Invalid response on line {line_number}: {line}")
        rows.append((match.group(1), int(match.group(2)), words))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    rows = parse_rows(args.input.read_text(encoding="utf-8"))
    slots = [(filename, iteration) for filename, iteration, _ in rows]
    by_file = Counter(filename for filename, _, _ in rows)
    if len(rows) != 600 or len(set(slots)) != 600 or set(by_file.values()) != {50}:
        raise ValueError(f"Expected 12 files × 50 unique iterations; found {dict(by_file)}")

    counts = {"Black": Counter(), "White": Counter()}
    for filename, _, words in rows:
        if filename.lower().startswith("b"):
            race = "Black"
        elif filename.lower().startswith("w"):
            race = "White"
        else:
            raise ValueError(f"Cannot infer race prefix from {filename}")
        counts[race].update(words)

    word_rows = [
        {
            "model_key": "pasted_iat",
            "race": race,
            "word": word,
            "count": counts[race][word],
        }
        for race in ("Black", "White")
        for word in IAT_WORDS
    ]
    model = ModelSpec(
        key="pasted_iat",
        display_name="Pasted IAT data",
        model_id="pasted-data",
        api_revision_hint="",
        release_date="",
        weight_access="",
        provider_endpoint="",
        max_concurrency=1,
        supports_temperature=False,
        supports_top_p=False,
        reasoning_effort=None,
        max_tokens=0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated = Path(
        _plot_model_word_counts(
            model,
            {"images_with_valid_outputs": len(by_file)},
            word_rows,
            args.output.parent,
        )
    )
    generated.replace(args.output)
    print(
        f"Parsed {len(rows)} responses and {len(rows) * 3} words; "
        f"negative terms: Black={sum(counts['Black'][word] for word in NEGATIVE_IAT_SET)}, "
        f"White={sum(counts['White'][word] for word in NEGATIVE_IAT_SET)}"
    )


if __name__ == "__main__":
    main()
