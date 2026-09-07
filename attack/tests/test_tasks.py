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

    def test_overlapping_labels_parse_without_false_ambiguity(self) -> None:
        duplicate = _task("duplicate", ("not equivalent", "equivalent"))
        nli = _task("nli", ("entailment", "not entailment"))
        self.assertEqual(duplicate.parse_label("not equivalent"), "not equivalent")
        self.assertEqual(nli.parse_label("not entailment"), "not entailment")
        self.assertIsNone(duplicate.parse_label("equivalent or not equivalent"))

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

    @patch("datasets.load_dataset")
    def test_remote_duplicate_rows_are_serialized(self, load_dataset) -> None:
        load_dataset.return_value = [
            {"sentence1": "A", "sentence2": "B", "label": 0},
            {"sentence1": "C", "sentence2": "D", "label": 1},
        ]
        task = load_task("duplicate", injected_limit=2, seed=1)
        self.assertIn("Sentence 1:", task.examples[0].text)
        self.assertIn("Sentence 2:", task.examples[0].text)
        self.assertEqual({example.gold_label for example in task.examples},
                         {"not equivalent", "equivalent"})
        load_dataset.assert_called_with("nyu-mll/glue", "mrpc", split="validation")

    @patch("datasets.load_dataset")
    def test_remote_hate_mapping_matches_paper(self, load_dataset) -> None:
        load_dataset.return_value = [
            {"tweet": "hate", "class": 0},
            {"tweet": "offensive", "class": 1},
            {"tweet": "neutral", "class": 2},
        ]
        task = load_task("hate", injected_limit=3, seed=1)
        labels = {example.text: example.gold_label for example in task.examples}
        self.assertEqual(labels, {"hate": "yes", "offensive": "yes", "neutral": "no"})

    @patch("datasets.load_dataset")
    def test_remote_nli_mapping_matches_rte(self, load_dataset) -> None:
        load_dataset.return_value = [
            {"sentence1": "A", "sentence2": "B", "label": 0},
            {"sentence1": "C", "sentence2": "D", "label": 1},
        ]
        task = load_task("nli", injected_limit=2, seed=1)
        self.assertEqual({example.gold_label for example in task.examples},
                         {"entailment", "not entailment"})
        load_dataset.assert_called_with("nyu-mll/glue", "rte", split="train")

    def test_unknown_task_and_bad_limit_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            load_task("unknown", 1, 42)
        with self.assertRaises(ValueError):
            load_task("sentiment", 0, 42)

    def test_remote_specs_use_approved_datasets(self) -> None:
        self.assertEqual(TASK_SPECS["sentiment"]["dataset_name"], "stanfordnlp/sst2")
        self.assertEqual(TASK_SPECS["spam"]["dataset_name"], "ucirvine/sms_spam")
        self.assertEqual(TASK_SPECS["duplicate"]["config_name"], "mrpc")
        self.assertEqual(TASK_SPECS["hate"]["dataset_name"],
                         "tdavidson/hate_speech_offensive")
        self.assertEqual(TASK_SPECS["nli"]["config_name"], "rte")


if __name__ == "__main__":
    unittest.main()
