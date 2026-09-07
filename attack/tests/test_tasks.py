from __future__ import annotations

import unittest
from unittest.mock import patch

from attack.tasks import InjectedTask, TASK_SPECS, load_task


def _task(name: str, labels: tuple[str, ...]) -> InjectedTask:
    return InjectedTask(name, "classify", labels, (), "dataset", "validation")


class InjectedTaskTests(unittest.TestCase):
    def test_parses_json_and_plain_text(self) -> None:
        task = _task("sentiment", ("negative", "positive"))
        self.assertEqual(task.parse_label('{"label": "Positive"}'), "positive")
        self.assertEqual(task.parse_label("The label is NEGATIVE."), "negative")
        self.assertEqual(task.parse_label('```json\n{"answer": "positive"}\n```'), "positive")

    def test_rejects_ambiguous_or_partial_labels(self) -> None:
        task = _task("spam", ("ham", "spam"))
        self.assertIsNone(task.parse_label("It may be spam or ham."))
        self.assertIsNone(task.parse_label("The message is spammy."))
        self.assertIsNone(task.parse_label("No classification supplied."))
        self.assertIsNone(task.parse_label('{"answer":"A","explanation":"This says spam"}'))

    @patch("datasets.load_dataset")
    def test_remote_sentiment_rows_are_deterministic(self, load_dataset) -> None:
        load_dataset.return_value = [
            {"sentence": "negative text", "label": 0},
            {"sentence": "positive text", "label": 1},
            {"sentence": "another negative", "label": 0},
        ]
        first = load_task("sentiment", injected_limit=2, seed=42)
        second = load_task("sentiment", injected_limit=2, seed=42)
        self.assertEqual(first.examples, second.examples)
        self.assertEqual(len(first.examples), 2)
        self.assertEqual(first.dataset_name, "stanfordnlp/sst2")
        load_dataset.assert_called_with("stanfordnlp/sst2", split="validation")

    @patch("datasets.load_dataset")
    def test_remote_spam_labels_are_normalized(self, load_dataset) -> None:
        load_dataset.return_value = [
            {"sms": "hello", "label": 0},
            {"sms": "win now", "label": 1},
        ]
        task = load_task("spam", injected_limit=2, seed=1)
        self.assertEqual({example.gold_label for example in task.examples}, {"ham", "spam"})

    def test_unknown_task_and_bad_limit_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_task("unknown", 1, 42)
        with self.assertRaises(ValueError):
            load_task("sentiment", 0, 42)

    def test_remote_specs_use_approved_datasets(self) -> None:
        self.assertEqual(TASK_SPECS["sentiment"]["dataset_name"], "stanfordnlp/sst2")
        self.assertEqual(TASK_SPECS["spam"]["dataset_name"], "ucirvine/sms_spam")


if __name__ == "__main__":
    unittest.main()

