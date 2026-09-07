"""Score model-agnostic VLM prediction exports."""

import argparse
import json
import re
import random
import string
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional


CHOICE = re.compile(r"\b([A-Z])\b", re.IGNORECASE)


def bootstrap_interval(values: list[int], samples: int, seed: int) -> list[float]:
    """Return a deterministic percentile bootstrap 95% interval for binary values."""
    if not values:
        return [0.0, 0.0]
    generator = random.Random(seed)
    size = len(values)
    estimates = sorted(
        sum(values[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    lower_index = int(0.025 * (samples - 1))
    upper_index = int(0.975 * (samples - 1))
    return [estimates[lower_index], estimates[upper_index]]


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


def normalize_confidence(value: object) -> Optional[float]:
    """Return a finite confidence probability between zero and one."""
    number = normalize_number(value)
    if number is None or not Decimal("0") <= number <= Decimal("1"):
        return None
    return float(number)


def normalize_prediction(
    value: object, expected: object, *, case_sensitive: bool, punctuation_sensitive: bool
) -> Optional[object]:
    if isinstance(expected, str) and len(expected) == 1 and expected.isalpha():
        return normalize_choice(value)
    if isinstance(expected, Decimal):
        return normalize_number(value)
    return normalize_text(
        value, case_sensitive=case_sensitive, punctuation_sensitive=punctuation_sensitive
    )


def score(
    records: Iterable[dict], *, case_sensitive: bool = True,
    punctuation_sensitive: bool = True, numeric_tolerance: Optional[float] = None,
    bootstrap_samples: Optional[int] = None, bootstrap_seed: int = 0,
) -> dict:
    if numeric_tolerance is not None and numeric_tolerance < 0:
        raise ValueError("numeric_tolerance must be non-negative")
    if bootstrap_samples is not None and bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    tolerance = Decimal(str(numeric_tolerance)) if numeric_tolerance is not None else None
    totals = defaultdict(
        lambda: {
            "correct": 0, "total": 0, "invalid": 0, "confidence": [], "top_k": [],
            "outcomes": [],
        }
    )
    for record in records:
        category = str(record.get("category", "uncategorized"))
        expected = normalize_choice(record.get("label"))
        if expected is None and tolerance is not None:
            expected = normalize_number(record.get("label"))
        if expected is None:
            expected = normalize_text(
                record.get("label"),
                case_sensitive=case_sensitive,
                punctuation_sensitive=punctuation_sensitive,
            )
        if expected is None:
            raise ValueError(f"invalid label for id={record.get('id')!r}")
        predicted = normalize_prediction(
            record.get("prediction"), expected,
            case_sensitive=case_sensitive, punctuation_sensitive=punctuation_sensitive,
        )
        correct = (
            predicted == expected
            if not isinstance(expected, Decimal)
            else predicted is not None and abs(predicted - expected) <= tolerance
        )
        top_k = None
        if "top_k" in record:
            if not isinstance(record["top_k"], list) or not record["top_k"]:
                raise ValueError(f"invalid top_k for id={record.get('id')!r}")
            candidates = [
                normalize_prediction(
                    value, expected,
                    case_sensitive=case_sensitive, punctuation_sensitive=punctuation_sensitive,
                )
                for value in record["top_k"]
            ]
            top_k = any(
                candidate == expected
                if not isinstance(expected, Decimal)
                else candidate is not None and abs(candidate - expected) <= tolerance
                for candidate in candidates
            )
        confidence = None
        if "confidence" in record:
            confidence = normalize_confidence(record["confidence"])
            if confidence is None:
                raise ValueError(f"invalid confidence for id={record.get('id')!r}")
        for key in ("overall", category):
            totals[key]["total"] += 1
            totals[key]["correct"] += int(correct)
            totals[key]["invalid"] += int(predicted is None)
            totals[key]["outcomes"].append((int(correct), int(predicted is None)))
            if confidence is not None:
                totals[key]["confidence"].append((confidence, correct))
            if top_k is not None:
                totals[key]["top_k"].append(top_k)

    report = {}
    for key, values in sorted(totals.items()):
        total = values["total"]
        confidences = values.pop("confidence")
        top_k = values.pop("top_k")
        outcomes = values.pop("outcomes")
        report[key] = {
            **values,
            "accuracy": values["correct"] / total if total else 0.0,
            "invalid_rate": values["invalid"] / total if total else 0.0,
        }
        if confidences:
            bins = defaultdict(lambda: [0, 0, 0.0])
            for confidence, correct in confidences:
                bin_values = bins[min(int(confidence * 10), 9)]
                bin_values[0] += 1
                bin_values[1] += int(correct)
                bin_values[2] += confidence
            report[key].update({
                "confidence_count": len(confidences),
                "mean_confidence": sum(value[0] for value in confidences) / len(confidences),
                "calibration_error": sum(
                    count / len(confidences) * abs(correct / count - confidence / count)
                    for count, correct, confidence in bins.values()
                ),
            })
        if top_k:
            report[key].update({
                "top_k_count": len(top_k),
                "top_k_accuracy": sum(top_k) / len(top_k),
            })
        if bootstrap_samples is not None:
            report[key].update({
                "bootstrap_samples": bootstrap_samples,
                "accuracy_confidence_interval": bootstrap_interval(
                    [correct for correct, _ in outcomes], bootstrap_samples, bootstrap_seed
                ),
                "invalid_rate_confidence_interval": bootstrap_interval(
                    [invalid for _, invalid in outcomes], bootstrap_samples, bootstrap_seed + 1
                ),
            })
    categories = [values for key, values in report.items() if key != "overall"]
    if categories:
        report["overall"].update({
            "macro_accuracy": sum(values["accuracy"] for values in categories) / len(categories),
            "macro_invalid_rate": sum(values["invalid_rate"] for values in categories) / len(categories),
        })
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
    parser.add_argument(
        "--bootstrap-samples", type=int,
        help="report deterministic 95% bootstrap intervals using this many resamples",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=0,
        help="random seed used for bootstrap resampling (default: 0)",
    )
    args = parser.parse_args()
    print(json.dumps(
        score(
            load_jsonl(args.predictions),
            case_sensitive=not args.ignore_case,
            punctuation_sensitive=not args.ignore_punctuation,
            numeric_tolerance=args.numeric_tolerance,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
