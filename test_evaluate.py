import unittest

from evaluate import normalize_choice, normalize_text, score


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


if __name__ == "__main__":
    unittest.main()
