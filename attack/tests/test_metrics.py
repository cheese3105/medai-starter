from __future__ import annotations

import unittest

from attack.metrics import calculate_metrics


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics(self) -> None:
        clean_targets = [
            {"prediction": "A", "gold": "A", "error": None},
            {"prediction": "B", "gold": "C", "error": None},
        ]
        clean_injected = [
            {"task": "sentiment", "prediction": "positive", "gold": "positive", "error": None},
            {"task": "sentiment", "prediction": None, "gold": "negative", "error": "timeout"},
        ]
        cases = [
            {
                "attack_method": "naive",
                "injected_task": "sentiment",
                "attacked_injected_prediction": "positive",
                "asv_contribution": 1.0,
                "mr_contribution": 1.0,
                "target_correct_under_attack": True,
                "error": None,
            },
            {
                "attack_method": "naive",
                "injected_task": "sentiment",
                "attacked_injected_prediction": None,
                "asv_contribution": 0.0,
                "mr_contribution": 0.0,
                "target_correct_under_attack": False,
                "error": "timeout",
            },
        ]

        result = calculate_metrics(clean_targets, clean_injected, cases)

        self.assertEqual(result["pna_t"], {"value": 0.5, "count": 2, "error_count": 0})
        self.assertEqual(result["pna_i"]["sentiment"]["value"], 0.5)
        self.assertEqual(result["pna_i"]["sentiment"]["parse_failures"], 1)
        self.assertEqual(result["pna_i"]["sentiment"]["error_count"], 1)
        self.assertEqual(result["attacks"][0]["asv"], 0.5)
        self.assertEqual(result["attacks"][0]["matching_rate"], 0.5)
        self.assertEqual(result["attacks"][0]["attack_parse_failures"], 1)
        self.assertEqual(result["attacks"][0]["target_accuracy_under_attack"], 0.5)
        self.assertEqual(result["attacks"][0]["target_accuracy_drop"], 0.0)
        self.assertEqual(result["attacks"][0]["error_count"], 1)


if __name__ == "__main__":
    unittest.main()
