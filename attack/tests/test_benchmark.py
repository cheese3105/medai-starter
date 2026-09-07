from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attack.benchmark import run_attack_benchmark
from attack.tasks import InjectedExample, InjectedTask
from core.config import RunConfig, StageConfig
from core.types import EpisodeInput, EpisodeResult


class _FakeRunner:
    def __init__(self, config, logger):
        self.config = config

    def run_episode(self, episode):
        if "Classify the sentiment" in episode.external_source:
            raw = '{"label":"positive"}'
        elif "Classify the input message" in episode.external_source:
            raw = '{"label":"spam"}'
        else:
            raw = '{"answer":"A"}'
        return EpisodeResult(
            question_id=episode.question_id,
            variant=self.config.variant,
            model=self.config.model,
            predicted_answer="A",
            gold_answer=episode.gold_answer,
            is_correct=episode.gold_answer == "A",
            reasoning_raw_response=raw,
        )


def _fake_injected(config, task, example):
    if example.id.endswith("_2"):
        raise TimeoutError("simulated timeout")
    return json.dumps({"label": example.gold_label})


class BenchmarkOrchestrationTests(unittest.TestCase):
    @patch("attack.benchmark.invoke_injected_task", side_effect=_fake_injected)
    @patch("attack.benchmark.load_targets")
    @patch("attack.benchmark.load_tasks")
    @patch("attack.benchmark.Runner", _FakeRunner)
    @patch("attack.benchmark.load_config")
    def test_writes_expected_outputs(
        self, load_config, load_tasks, load_targets, _invoke
    ) -> None:
        load_config.return_value = RunConfig(
            variant="fake-v0",
            model="fake-model",
            pipeline=[StageConfig(name="reasoning", prompt_template="{question}")],
        )
        load_targets.return_value = [
            EpisodeInput("target_1", "question", ["a", "b", "c", "d"], "A")
        ]
        load_tasks.return_value = [
            InjectedTask(
                "sentiment", "Classify the sentiment", ("negative", "positive"),
                (InjectedExample("sentiment_1", "good", "positive"),
                 InjectedExample("sentiment_2", "bad", "negative")),
                "stanfordnlp/sst2", "validation",
            ),
            InjectedTask(
                "spam", "Classify the input message", ("ham", "spam"),
                (InjectedExample("spam_1", "win", "spam"),
                 InjectedExample("spam_2", "hello", "ham")),
                "ucirvine/sms_spam", "train",
            ),
        ]

        with tempfile.TemporaryDirectory() as directory:
            result = run_attack_benchmark(
                target_config="configs/v0.yaml",
                attack_names=["naive", "combined"],
                task_names=["sentiment", "spam"],
                split="test",
                target_limit=1,
                injected_limit=2,
                sample_size=1,
                seed=42,
                output_dir=directory,
            )

            self.assertEqual(result["case_count"], 4)
            output = Path(directory)
            self.assertTrue((output / "cases.jsonl").is_file())
            self.assertTrue((output / "clean_targets.jsonl").is_file())
            self.assertTrue((output / "clean_injected.jsonl").is_file())
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "run_config.json").is_file())
            cases = (output / "cases.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(cases), 4)
            clean_injected = (output / "clean_injected.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(clean_injected), 4)
            self.assertEqual(result["metrics"]["pna_i"]["sentiment"]["count"], 2)
            self.assertEqual(result["metrics"]["pna_i"]["spam"]["count"], 2)
            self.assertEqual(result["metrics"]["pna_i"]["sentiment"]["error_count"], 1)
            self.assertEqual(result["metrics"]["pna_i"]["spam"]["error_count"], 1)
            self.assertIn("{external_source}", load_config.return_value.get_stage("reasoning").prompt_template)


if __name__ == "__main__":
    unittest.main()
