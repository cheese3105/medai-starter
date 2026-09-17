from __future__ import annotations

import unittest

from attack.metrics import calculate_metrics


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics(self) -> None:
        clean_targets = [
            {"prediction": "A", "gold": "A", "latency_ms": 100.0,
             "token_usage": {"input": 10, "output": 2}, "error": None},
            {"prediction": "B", "gold": "C", "latency_ms": 300.0,
             "token_usage": {"input": 20, "output": 4}, "error": None},
        ]
        clean_injected = [
            {"task": "sentiment", "prediction": "positive", "gold": "positive",
             "latency_ms": 50.0, "token_usage": {"input": 8, "output": 2}, "error": None},
            {"task": "sentiment", "prediction": None, "gold": "negative",
             "latency_ms": 150.0, "token_usage": {}, "error": "timeout"},
        ]
        cases = [
            {
                "attack_method": "naive",
                "injected_task": "sentiment",
                "attacked_injected_prediction": "positive",
                "asv_contribution": 1.0,
                "mr_contribution": 1.0,
                "target_correct_under_attack": True,
                "latency_ms": 200.0,
                "token_usage": {"input": 30, "output": 6},
                "error": None,
            },
            {
                "attack_method": "naive",
                "injected_task": "sentiment",
                "attacked_injected_prediction": None,
                "asv_contribution": 0.0,
                "mr_contribution": 0.0,
                "target_correct_under_attack": False,
                "latency_ms": 400.0,
                "token_usage": {},
                "error": "timeout",
            },
        ]

        result = calculate_metrics(clean_targets, clean_injected, cases)

        self.assertEqual(result["pna_t"]["value"], 0.5)
        self.assertEqual(result["pna_t"]["count"], 2)
        self.assertEqual(result["pna_t"]["error_count"], 0)
        self.assertEqual(result["pna_t"]["avg_latency_ms"], 200.0)
        self.assertEqual(result["pna_t"]["median_latency_ms"], 200.0)
        self.assertEqual(result["pna_t"]["p95_latency_ms"], 290.0)
        self.assertEqual(result["pna_t"]["total_input_tokens"], 30)
        self.assertEqual(result["pna_t"]["total_output_tokens"], 6)
        self.assertEqual(result["pna_t"]["avg_tokens_per_case"], 18.0)
        self.assertEqual(result["pna_t"]["token_measurement_count"], 2)
        self.assertEqual(result["pna_i"]["sentiment"]["value"], 0.5)
        self.assertEqual(result["pna_i"]["sentiment"]["parse_failures"], 1)
        self.assertEqual(result["pna_i"]["sentiment"]["error_count"], 1)
        self.assertEqual(result["pna_i"]["sentiment"]["avg_latency_ms"], 50.0)
        self.assertEqual(result["pna_i"]["sentiment"]["latency_measurement_count"], 1)
        self.assertEqual(result["pna_i"]["sentiment"]["token_measurement_count"], 1)
        self.assertEqual(result["attacks"][0]["asv"], 0.5)
        self.assertEqual(result["attacks"][0]["matching_rate"], 0.5)
        self.assertEqual(result["attacks"][0]["attack_parse_failures"], 1)
        self.assertEqual(result["attacks"][0]["target_accuracy_under_attack"], 0.5)
        self.assertEqual(result["attacks"][0]["target_accuracy_drop"], 0.0)
        self.assertEqual(result["attacks"][0]["error_count"], 1)
        self.assertEqual(result["attacks"][0]["avg_latency_ms"], 200.0)
        self.assertEqual(result["attacks"][0]["latency_measurement_count"], 1)
        self.assertEqual(result["attacks"][0]["avg_tokens_per_case"], 36.0)


if __name__ == "__main__":
    unittest.main()
