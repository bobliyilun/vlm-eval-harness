"""Score model-agnostic VLM prediction exports."""

import argparse
import json
import math
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


def paired_comparison(
    baseline: list[dict], candidate: list[dict], *, case_sensitive: bool = True,
    punctuation_sensitive: bool = True, numeric_tolerance: Optional[float] = None,
) -> dict:
    """Compare two prediction exports with an exact two-sided paired test."""
    baseline = list(require_unique_ids(baseline))
    candidate = list(require_unique_ids(candidate))
    baseline_by_id = {record.get("id"): record for record in baseline}
    candidate_by_id = {record.get("id"): record for record in candidate}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("paired comparison requires matching record ids")
    wins = losses = baseline_correct = candidate_correct = 0
    options = {
        "case_sensitive": case_sensitive,
        "punctuation_sensitive": punctuation_sensitive,
        "numeric_tolerance": numeric_tolerance,
    }
    for record_id in baseline_by_id:
        first, second = baseline_by_id[record_id], candidate_by_id[record_id]
        if first.get("label") != second.get("label"):
            raise ValueError(f"mismatched label for id={record_id!r}")
        first_correct = score([first], **options)["overall"]["correct"]
        second_correct = score([second], **options)["overall"]["correct"]
        baseline_correct += first_correct
        candidate_correct += second_correct
        wins += second_correct and not first_correct
        losses += first_correct and not second_correct
    discordant = wins + losses
    tail = sum(math.comb(discordant, value) for value in range(min(wins, losses) + 1))
    return {
        "total": len(baseline),
        "baseline_accuracy": baseline_correct / len(baseline) if baseline else 0.0,
        "candidate_accuracy": candidate_correct / len(candidate) if candidate else 0.0,
        "accuracy_difference": (
            (candidate_correct - baseline_correct) / len(baseline) if baseline else 0.0
        ),
        "candidate_only_correct": wins,
        "baseline_only_correct": losses,
        "exact_p_value": min(1.0, 2 * tail / 2**discordant) if discordant else 1.0,
    }


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


def normalize_nonnegative_number(value: object) -> Optional[float]:
    """Return a finite, non-negative measurement."""
    number = normalize_number(value)
    return float(number) if number is not None and number >= 0 else None


def normalize_token_count(value: object) -> Optional[int]:
    """Return a non-negative integral token count."""
    number = normalize_number(value)
    return int(number) if number is not None and number >= 0 and number == number.to_integral() else None


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


def require_unique_ids(records: Iterable[dict]) -> Iterable[dict]:
    """Yield schema-valid records after rejecting duplicate IDs."""
    seen = set()
    for record_number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"record {record_number}: expected a JSON object")
        for field in ("id", "label", "prediction"):
            if field not in record:
                raise ValueError(f"record {record_number}: missing required field {field!r}")
        record_id = record["id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"record {record_number}: id must be a non-empty string")
        if "category" in record and (
            not isinstance(record["category"], str) or not record["category"].strip()
        ):
            raise ValueError(f"record {record_number}: category must be a non-empty string")
        if "image_path" in record and (
            not isinstance(record["image_path"], str) or not record["image_path"].strip()
        ):
            raise ValueError(f"record {record_number}: image_path must be a non-empty string")
        if record_id in seen:
            raise ValueError(f"record {record_number}: duplicate id={record_id!r}")
        seen.add(record_id)
        yield record


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
            "outcomes": [], "latency_ms": [], "input_tokens": [], "output_tokens": [],
        }
    )
    for record in require_unique_ids(records):
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
        latency_ms = None
        if "latency_ms" in record:
            latency_ms = normalize_nonnegative_number(record["latency_ms"])
            if latency_ms is None:
                raise ValueError(f"invalid latency_ms for id={record.get('id')!r}")
        token_counts = {}
        for field in ("input_tokens", "output_tokens"):
            if field in record:
                token_counts[field] = normalize_token_count(record[field])
                if token_counts[field] is None:
                    raise ValueError(f"invalid {field} for id={record.get('id')!r}")
        for key in ("overall", category):
            totals[key]["total"] += 1
            totals[key]["correct"] += int(correct)
            totals[key]["invalid"] += int(predicted is None)
            totals[key]["outcomes"].append((int(correct), int(predicted is None)))
            if confidence is not None:
                totals[key]["confidence"].append((confidence, correct))
            if top_k is not None:
                totals[key]["top_k"].append(top_k)
            if latency_ms is not None:
                totals[key]["latency_ms"].append(latency_ms)
            for field, value in token_counts.items():
                totals[key][field].append(value)

    report = {}
    for key, values in sorted(totals.items()):
        total = values["total"]
        confidences = values.pop("confidence")
        top_k = values.pop("top_k")
        outcomes = values.pop("outcomes")
        latency_ms = values.pop("latency_ms")
        input_tokens = values.pop("input_tokens")
        output_tokens = values.pop("output_tokens")
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
        if latency_ms:
            report[key].update({
                "latency_count": len(latency_ms),
                "mean_latency_ms": sum(latency_ms) / len(latency_ms),
                "total_latency_ms": sum(latency_ms),
            })
        for field, counts in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
            if counts:
                report[key][field] = sum(counts)
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
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
                if isinstance(record, dict) and "image_path" in record:
                    image_path = record["image_path"]
                    if not isinstance(image_path, str) or not image_path.strip():
                        raise ValueError(
                            f"record {line_number}: image_path must be a non-empty string"
                        )
                    resolved = Path(image_path)
                    if not resolved.is_absolute():
                        resolved = path.parent / resolved
                    if not resolved.is_file():
                        raise ValueError(
                            f"record {line_number}: image_path does not name a file: {image_path!r}"
                        )
                records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--ignore-case", action="store_true")
    parser.add_argument("--ignore-punctuation", action="store_true")
    parser.add_argument(
        "--compare-with", type=Path,
        help="second JSONL export to compare against the positional baseline",
    )
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
    baseline = load_jsonl(args.predictions)
    options = {
        "case_sensitive": not args.ignore_case,
        "punctuation_sensitive": not args.ignore_punctuation,
        "numeric_tolerance": args.numeric_tolerance,
    }
    report = score(
        baseline,
        **options,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    if args.compare_with:
        candidate = load_jsonl(args.compare_with)
        report = {
            "baseline": report,
            "candidate": score(
                candidate,
                **options,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=args.bootstrap_seed,
            ),
            "paired_comparison": paired_comparison(baseline, candidate, **options),
        }
    print(json.dumps(
        report,
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
