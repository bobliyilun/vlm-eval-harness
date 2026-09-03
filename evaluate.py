"""Score model-agnostic VLM prediction exports."""

import argparse
import json
import re
import string
from collections import defaultdict
from decimal import Decimal, InvalidOperation
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


def normalize_text(
    value: object, *, case_sensitive: bool = True, punctuation_sensitive: bool = True
) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not punctuation_sensitive:
        text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        text = " ".join(text.split())
    if not case_sensitive:
        text = text.casefold()
    return text or None


def normalize_number(value: object) -> Optional[Decimal]:
    """Return a finite numeric answer when *value* contains only a number."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def score(
    records: Iterable[dict], *, case_sensitive: bool = True,
    punctuation_sensitive: bool = True, numeric_tolerance: Optional[float] = None,
) -> dict:
    if numeric_tolerance is not None and numeric_tolerance < 0:
        raise ValueError("numeric_tolerance must be non-negative")
    tolerance = Decimal(str(numeric_tolerance)) if numeric_tolerance is not None else None
    totals = defaultdict(lambda: {"correct": 0, "total": 0, "invalid": 0})
    for record in records:
        category = str(record.get("category", "uncategorized"))
        expected = normalize_choice(record.get("label"))
        if expected is not None:
            predicted = normalize_choice(record.get("prediction"))
            correct = predicted == expected
        elif tolerance is not None and (expected := normalize_number(record.get("label"))) is not None:
            predicted = normalize_number(record.get("prediction"))
            correct = predicted is not None and abs(predicted - expected) <= tolerance
        else:
            expected = normalize_text(
                record.get("label"),
                case_sensitive=case_sensitive,
                punctuation_sensitive=punctuation_sensitive,
            )
            predicted = normalize_text(
                record.get("prediction"),
                case_sensitive=case_sensitive,
                punctuation_sensitive=punctuation_sensitive,
            )
            correct = predicted == expected
        if expected is None:
            raise ValueError(f"invalid label for id={record.get('id')!r}")
        for key in ("overall", category):
            totals[key]["total"] += 1
            totals[key]["correct"] += int(correct)
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
    parser.add_argument("--ignore-case", action="store_true")
    parser.add_argument("--ignore-punctuation", action="store_true")
    parser.add_argument(
        "--numeric-tolerance", type=float,
        help="accept direct numeric predictions within this absolute tolerance",
    )
    args = parser.parse_args()
    print(json.dumps(
        score(
            load_jsonl(args.predictions),
            case_sensitive=not args.ignore_case,
            punctuation_sensitive=not args.ignore_punctuation,
            numeric_tolerance=args.numeric_tolerance,
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
