import unittest

from evaluate import (
    normalize_choice,
    normalize_confidence,
    normalize_number,
    normalize_text,
    score,
)


class EvaluationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
