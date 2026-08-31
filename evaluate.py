"""Score model-agnostic VLM prediction exports."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional


CHOICE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def normalize_choice(value: object) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    if len(text) == 1 and text.isalpha():
        return text
    matches = CHOICE.findall(text)
    return matches[-1].upper() if matches else None


def score(records: Iterable[dict]) -> dict:
    totals = defaultdict(lambda: {"correct": 0, "total": 0, "invalid": 0})
    for record in records:
        category = str(record.get("category", "uncategorized"))
        expected = normalize_choice(record.get("label"))
        predicted = normalize_choice(record.get("prediction"))
        if expected is None:
            raise ValueError(f"invalid label for id={record.get('id')!r}")
        for key in ("overall", category):
            totals[key]["total"] += 1
            totals[key]["correct"] += int(predicted == expected)
            totals[key]["invalid"] += int(predicted is None)

    report = {}
    for key, values in sorted(totals.items()):
        total = values["total"]
        report[key] = {
            **values,
            "accuracy": values["correct"] / total if total else 0.0,
            "invalid_rate": values["invalid"] / total if total else 0.0,
        }
    return report


def load_jsonl(path: Path) -> list:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    print(json.dumps(score(load_jsonl(args.predictions)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

