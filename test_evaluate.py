import unittest

from evaluate import normalize_choice, score


class EvaluationTests(unittest.TestCase):
    def test_normalizes_wrapped_choices(self):
        self.assertEqual(normalize_choice("The answer is B."), "B")
        self.assertEqual(normalize_choice("c"), "C")
        self.assertIsNone(normalize_choice("unknown"))

    def test_reports_overall_and_category_metrics(self):
        report = score([
            {"id": "1", "category": "ocr", "label": "A", "prediction": "A"},
            {"id": "2", "category": "ocr", "label": "B", "prediction": "no answer"},
            {"id": "3", "category": "reasoning", "label": "C", "prediction": "Answer: B"},
        ])
        self.assertAlmostEqual(report["overall"]["accuracy"], 1 / 3)
        self.assertEqual(report["overall"]["invalid"], 1)
        self.assertAlmostEqual(report["ocr"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()

