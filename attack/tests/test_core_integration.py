from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.config import RunConfig, StageConfig
from core.logger import DebugLogger
from core.llm_client import build_llm, extract_token_usage
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
        self.assertEqual(result.total_token_usage, {})

    def test_external_source_defaults_to_empty(self) -> None:
        episode = EpisodeInput(question_id="q1", question="question")
        self.assertEqual(episode.external_source, "")

    def test_normalizes_usage_metadata_and_reports_missing_usage(self) -> None:
        response = SimpleNamespace(
            usage_metadata={"input_tokens": 12, "output_tokens": 3}
        )
        self.assertEqual(extract_token_usage(response), {"input": 12, "output": 3})
        legacy_response = SimpleNamespace(
            usage_metadata=None,
            response_metadata={
                "token_usage": {"prompt_tokens": 7, "completion_tokens": 2}
            },
        )
        self.assertEqual(extract_token_usage(legacy_response), {"input": 7, "output": 2})
        self.assertEqual(extract_token_usage(SimpleNamespace()), {})

    @patch("core.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENROUTER_PROVIDER": "deepinfra"})
    def test_openrouter_provider_is_pinned_without_fallbacks(self, chat_openai) -> None:
        build_llm(
            model="test-model",
            base_url="https://openrouter.ai/api/v1",
            api_key="test-key",
            enable_reasoning=False,
        )

        self.assertEqual(
            chat_openai.call_args.kwargs["extra_body"],
            {
                "reasoning": {"effort": "none"},
                "provider": {
                    "only": ["deepinfra"],
                    "allow_fallbacks": False,
                },
            },
        )

    @patch("core.llm_client.ChatOpenAI")
    @patch.dict("os.environ", {"OPENROUTER_PROVIDER": "deepinfra"})
    def test_provider_setting_is_not_sent_to_other_endpoints(self, chat_openai) -> None:
        build_llm(
            model="test-model",
            base_url="http://localhost:11434/v1",
            api_key="test-key",
            enable_reasoning=False,
        )

        self.assertEqual(
            chat_openai.call_args.kwargs["extra_body"],
            {"reasoning": {"effort": "none"}},
        )


if __name__ == "__main__":
    unittest.main()
