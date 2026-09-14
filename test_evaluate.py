import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from evaluate import (
    bootstrap_interval,
    normalize_choice,
    normalize_confidence,
    normalize_number,
    normalize_text,
    paired_comparison,
    load_jsonl,
    score,
)
from inference import HTTPInferenceAdapter, OpenAICompatibleAdapter


class EvaluationTests(unittest.TestCase):
    def test_adversarial_answer_format_fixture(self):
        records = load_jsonl(Path(__file__).parent / "data/adversarial_answer_formats.jsonl")
        report = score(records)["choice-format"]
        self.assertEqual(report["total"], 7)
        self.assertEqual(report["correct"], 5)
        self.assertEqual(report["invalid"], 2)

    def test_http_adapter_posts_json_and_parses_prediction(self):
        adapter = HTTPInferenceAdapter(
            "http://example.test/infer",
            lambda record: {"question": record["prompt"]},
            lambda response: response["answer"],
        )
        response = MagicMock()
        response.read.return_value = b'{"answer": "B"}'
        transport = MagicMock()
        transport.__enter__.return_value = response
        with patch("inference.urlopen", return_value=transport) as urlopen:
            self.assertEqual(adapter.infer({"prompt": "Which letter?"}), "B")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://example.test/infer")
        self.assertEqual(json.loads(request.data), {"question": "Which letter?"})
        self.assertEqual(request.get_method(), "POST")

    def test_openai_compatible_adapter_posts_chat_completion(self):
        adapter = OpenAICompatibleAdapter(
            "http://example.test/v1/chat/completions", "demo-vlm", api_key="secret"
        )
        response = MagicMock()
        response.read.return_value = b'{"choices": [{"message": {"content": "B"}}]}'
        transport = MagicMock()
        transport.__enter__.return_value = response
        with patch("inference.urlopen", return_value=transport) as urlopen:
            self.assertEqual(adapter.infer({"prompt": "Which letter?"}), "B")
        request = urlopen.call_args.args[0]
        self.assertEqual(json.loads(request.data), {
            "model": "demo-vlm",
            "messages": [{"role": "user", "content": "Which letter?"}],
        })
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")

    def test_normalizes_wrapped_choices(self):
        self.assertEqual(normalize_choice("The answer is B."), "B")
        self.assertEqual(normalize_choice("c"), "C")
        self.assertIsNone(normalize_choice("unknown"))

    def test_normalizes_exact_match_text(self):
        self.assertEqual(normalize_text("  red bus  "), "red bus")
        self.assertIsNone(normalize_text("   "))

    def test_configures_text_case_and_punctuation_normalization(self):
        self.assertEqual(
            normalize_text(
                "Red, Bus!", case_sensitive=False, punctuation_sensitive=False
            ),
            "red bus",
        )

    def test_reports_overall_and_category_metrics(self):
        report = score([
            {"id": "1", "category": "ocr", "label": "A", "prediction": "A"},
            {"id": "2", "category": "ocr", "label": "B", "prediction": "no answer"},
            {"id": "3", "category": "reasoning", "label": "C", "prediction": "Answer: B"},
        ])
        self.assertAlmostEqual(report["overall"]["accuracy"], 1 / 3)
        self.assertEqual(report["overall"]["invalid"], 1)
        self.assertAlmostEqual(report["ocr"]["accuracy"], 0.5)

    def test_reports_macro_metrics_across_categories(self):
        report = score([
            {"id": "1", "category": "small", "label": "A", "prediction": "A"},
            {"id": "2", "category": "large", "label": "A", "prediction": "B"},
            {"id": "3", "category": "large", "label": "A", "prediction": "A"},
            {"id": "4", "category": "large", "label": "A", "prediction": ""},
        ])
        self.assertAlmostEqual(report["overall"]["macro_accuracy"], 2 / 3)
        self.assertAlmostEqual(report["overall"]["macro_invalid_rate"], 1 / 6)

    def test_scores_text_labels_by_exact_match(self):
        report = score([
            {"id": "1", "category": "text", "label": "red bus", "prediction": "red bus"},
            {"id": "2", "category": "text", "label": "red bus", "prediction": "Red bus"},
            {"id": "3", "category": "text", "label": "red bus", "prediction": ""},
        ])
        self.assertEqual(report["text"]["correct"], 1)
        self.assertEqual(report["text"]["invalid"], 1)
        self.assertAlmostEqual(report["text"]["accuracy"], 1 / 3)

    def test_scores_text_with_configured_normalization(self):
        report = score(
            [{"id": "1", "category": "text", "label": "Red bus!", "prediction": "red, bus"}],
            case_sensitive=False,
            punctuation_sensitive=False,
        )
        self.assertEqual(report["text"]["correct"], 1)

    def test_scores_numbers_with_absolute_tolerance(self):
        report = score([
            {"id": "1", "category": "count", "label": "2.5", "prediction": "2.54"},
            {"id": "2", "category": "count", "label": "2.5", "prediction": "2.56"},
            {"id": "3", "category": "count", "label": "2.5", "prediction": "about 2.5"},
        ], numeric_tolerance=0.05)
        self.assertEqual(report["count"]["correct"], 1)
        self.assertEqual(report["count"]["invalid"], 1)

    def test_numeric_tolerance_is_opt_in_and_validated(self):
        self.assertEqual(normalize_number("1e2"), 100)
        self.assertIsNone(normalize_number("NaN"))
        self.assertEqual(
            score([{"id": "1", "label": "2", "prediction": "2.0"}])["overall"]["correct"],
            0,
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            score([], numeric_tolerance=-0.1)

    def test_reports_confidence_and_calibration_error(self):
        report = score([
            {"id": "1", "label": "A", "prediction": "A", "confidence": 0.9},
            {"id": "2", "label": "A", "prediction": "B", "confidence": 0.8},
            {"id": "3", "label": "A", "prediction": "A"},
        ])
        self.assertEqual(report["overall"]["confidence_count"], 2)
        self.assertAlmostEqual(report["overall"]["mean_confidence"], 0.85)
        self.assertAlmostEqual(report["overall"]["calibration_error"], 0.45)

    def test_validates_confidence_probabilities(self):
        self.assertEqual(normalize_confidence("0.5"), 0.5)
        self.assertIsNone(normalize_confidence(1.1))
        with self.assertRaisesRegex(ValueError, "invalid confidence"):
            score([{"id": "1", "label": "A", "prediction": "A", "confidence": None}])

    def test_reports_latency_and_token_usage(self):
        report = score([
            {"id": "1", "category": "ocr", "label": "A", "prediction": "A",
             "latency_ms": 125.5, "input_tokens": 10, "output_tokens": 2},
            {"id": "2", "category": "ocr", "label": "A", "prediction": "A",
             "latency_ms": "74.5", "input_tokens": 8},
            {"id": "3", "label": "A", "prediction": "A", "output_tokens": 3},
        ])
        self.assertEqual(report["overall"]["latency_count"], 2)
        self.assertEqual(report["overall"]["total_latency_ms"], 200)
        self.assertEqual(report["overall"]["mean_latency_ms"], 100)
        self.assertEqual(report["overall"]["input_tokens"], 18)
        self.assertEqual(report["overall"]["output_tokens"], 5)
        with self.assertRaisesRegex(ValueError, "invalid latency_ms"):
            score([{"id": "1", "label": "A", "prediction": "A", "latency_ms": -1}])
        with self.assertRaisesRegex(ValueError, "invalid input_tokens"):
            score([{"id": "1", "label": "A", "prediction": "A", "input_tokens": 1.5}])

    def test_reports_top_k_hits_without_changing_top_one_accuracy(self):
        report = score([
            {"id": "1", "label": "A", "prediction": "B", "top_k": ["B", "A"]},
            {"id": "2", "label": "A", "prediction": "A", "top_k": ["B", "C"]},
            {"id": "3", "label": "A", "prediction": "A"},
        ])
        self.assertAlmostEqual(report["overall"]["accuracy"], 2 / 3)
        self.assertEqual(report["overall"]["top_k_count"], 2)
        self.assertAlmostEqual(report["overall"]["top_k_accuracy"], 0.5)

    def test_validates_top_k_predictions(self):
        with self.assertRaisesRegex(ValueError, "invalid top_k"):
            score([{"id": "1", "label": "A", "prediction": "A", "top_k": []}])

    def test_reports_deterministic_bootstrap_confidence_intervals(self):
        records = [
            {"id": "1", "category": "ocr", "label": "A", "prediction": "A"},
            {"id": "2", "category": "ocr", "label": "A", "prediction": "B"},
            {"id": "3", "category": "ocr", "label": "A", "prediction": ""},
        ]
        report = score(records, bootstrap_samples=200, bootstrap_seed=7)
        self.assertEqual(report["ocr"]["bootstrap_samples"], 200)
        self.assertEqual(report["ocr"]["accuracy_confidence_interval"], [0.0, 1.0])
        self.assertEqual(report["ocr"]["invalid_rate_confidence_interval"], [0.0, 1.0])
        self.assertEqual(bootstrap_interval([1, 0, 0], 200, 7), [0.0, 1.0])

    def test_validates_bootstrap_sample_count(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            score([], bootstrap_samples=0)

    def test_rejects_missing_and_duplicate_record_ids(self):
        record = {"id": "1", "label": "A", "prediction": "A"}
        with self.assertRaisesRegex(ValueError, "missing required field 'id'"):
            score([{"label": "A", "prediction": "A"}])
        with self.assertRaisesRegex(ValueError, "duplicate id='1'"):
            score([record, record])

    def test_reports_actionable_dataset_schema_errors(self):
        with self.assertRaisesRegex(ValueError, "record 1: expected a JSON object"):
            score(["not a record"])
        with self.assertRaisesRegex(ValueError, "record 1: missing required field 'prediction'"):
            score([{"id": "1", "label": "A"}])
        with self.assertRaisesRegex(ValueError, "record 1: id must be a non-empty string"):
            score([{"id": 1, "label": "A", "prediction": "A"}])
        with self.assertRaisesRegex(ValueError, "record 1: category must be a non-empty string"):
            score([{"id": "1", "category": "", "label": "A", "prediction": "A"}])
        with self.assertRaisesRegex(ValueError, "record 1: image_path must be a non-empty string"):
            score([{"id": "1", "image_path": "", "label": "A", "prediction": "A"}])

    def test_checks_local_image_paths_relative_to_jsonl(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").touch()
            export = root / "predictions.jsonl"
            export.write_text(
                '{"id": "1", "label": "A", "prediction": "A", "image_path": "image.png"}\n',
                encoding="utf-8",
            )
            self.assertEqual(load_jsonl(export)[0]["image_path"], "image.png")
            export.write_text(
                '{"id": "1", "label": "A", "prediction": "A", "image_path": "missing.png"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "record 1: image_path does not name a file"):
                load_jsonl(export)

    def test_compares_paired_predictions_with_exact_significance(self):
        baseline = [
            {"id": "1", "label": "A", "prediction": "B"},
            {"id": "2", "label": "A", "prediction": "A"},
            {"id": "3", "label": "A", "prediction": "B"},
        ]
        candidate = [
            {"id": "1", "label": "A", "prediction": "A"},
            {"id": "2", "label": "A", "prediction": "B"},
            {"id": "3", "label": "A", "prediction": "A"},
        ]
        report = paired_comparison(baseline, candidate)
        self.assertEqual(report["candidate_only_correct"], 2)
        self.assertEqual(report["baseline_only_correct"], 1)
        self.assertAlmostEqual(report["accuracy_difference"], 1 / 3)
        self.assertEqual(report["exact_p_value"], 1.0)

    def test_paired_comparison_requires_matching_unique_ids(self):
        record = {"id": "1", "label": "A", "prediction": "A"}
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            paired_comparison([record, record], [record, record])
        with self.assertRaisesRegex(ValueError, "missing required field 'id'"):
            paired_comparison([{"label": "A", "prediction": "A"}], [record])
        with self.assertRaisesRegex(ValueError, "matching"):
            paired_comparison([record], [{"id": "2", "label": "A", "prediction": "A"}])


if __name__ == "__main__":
    unittest.main()
