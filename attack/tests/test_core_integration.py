from __future__ import annotations

import unittest

from core.config import RunConfig, StageConfig
from core.logger import DebugLogger
from core.runner import Runner, _STAGE_CLASSES
from core.types import EpisodeInput, StageOutput


class _FakeReasoningStage:
    last_context = None

    def __init__(self, run_config, stage_config):
        pass

    def run(self, context):
        type(self).last_context = dict(context)
        return StageOutput(
            stage_name="reasoning",
            data={
                "answer": "A",
                "explanation": "ok",
                "confidence": 1.0,
                "raw_response": '{"answer":"A"}',
            },
        )


class CoreIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = dict(_STAGE_CLASSES)
        _STAGE_CLASSES.clear()
        _STAGE_CLASSES["reasoning"] = _FakeReasoningStage

    def tearDown(self) -> None:
        _STAGE_CLASSES.clear()
        _STAGE_CLASSES.update(self.original)

    def test_external_source_reaches_reasoning_and_raw_response_is_kept(self) -> None:
        config = RunConfig(
            variant="test",
            model="fake",
            pipeline=[StageConfig(name="reasoning", prompt_template="{external_source}")],
        )
        runner = Runner(config, DebugLogger())
        result = runner.run_episode(EpisodeInput(
            question_id="q1",
            question="question",
            choices=["one", "two", "three", "four"],
            gold_answer="A",
            external_source="external payload",
        ))

        self.assertEqual(_FakeReasoningStage.last_context["external_source"], "external payload")
        self.assertEqual(result.reasoning_raw_response, '{"answer":"A"}')
        self.assertTrue(result.is_correct)

    def test_external_source_defaults_to_empty(self) -> None:
        episode = EpisodeInput(question_id="q1", question="question")
        self.assertEqual(episode.external_source, "")


if __name__ == "__main__":
    unittest.main()

