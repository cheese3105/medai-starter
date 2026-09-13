"""Reasoning stage — gọi LLM với prompt_template, parse JSON response."""

from __future__ import annotations

import json
import re
import time
from typing import Any

from core.config import RunConfig, StageConfig
from core.llm_client import build_llm
from core.types import StageOutput
from stages.base import BaseStage

LABELS = ["A", "B", "C", "D"]


def _extract_json(text: str) -> dict:
    """Trích JSON từ response (xử lý cả markdown fence)."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Không tìm thấy JSON: {text[:200]}")


class ReasoningStage(BaseStage):

    def __init__(self, run_config: RunConfig, stage_config: StageConfig):
        super().__init__(run_config, stage_config)
        enable_reasoning = stage_config.extra.get("reasoning", True)
        reasoning_effort = stage_config.extra.get("reasoning_effort", None)
        self.llm = build_llm(
            model=run_config.model,
            base_url=run_config.model_base_url,
            api_key=run_config.model_api_key,
            temperature=stage_config.temperature,
            seed=run_config.seed,
            enable_reasoning=enable_reasoning,
            reasoning_effort=reasoning_effort,
        )

    def run(self, context: dict[str, Any]) -> StageOutput:
        start = time.time()

        prompt = self.stage_config.prompt_template.format_map(_SafeDict(context))
        response = self.llm.invoke(prompt)
        latency_ms = (time.time() - start) * 1000

        token_usage = {"input": 0, "output": 0}
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage", {})
            token_usage["input"] = usage.get("prompt_tokens", 0)
            token_usage["output"] = usage.get("completion_tokens", 0)

        raw_text = response.content
        try:
            parsed = _extract_json(raw_text)
        except (ValueError, json.JSONDecodeError):
            parsed = {"answer": "INVALID", "explanation": raw_text, "confidence": 0.0}

        answer = parsed.get("answer", "INVALID")
        if context.get("choices") and answer not in LABELS:
            answer = "INVALID"

        return StageOutput(
            stage_name="reasoning",
            data={
                "answer": answer,
                "explanation": parsed.get("explanation", ""),
                "confidence": float(parsed.get("confidence", 0.0)),
                "raw_response": raw_text,
                "prompt": prompt,
            },
            latency_ms=latency_ms,
            token_usage=token_usage,
        )


class _SafeDict(dict):
    """Trả về chuỗi rỗng cho placeholder không tồn tại."""
    def __missing__(self, key: str) -> str:
        return ""
