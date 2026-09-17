"""Query Rewriter — viết lại search query khi verifier từ chối.

Chỉ chạy có điều kiện (verdict=unsupported), không phải stage tuần tự.
Nếu parse lỗi, trả nguyên query cũ — không crash episode.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from core.config import RunConfig, StageConfig
from core.llm_client import build_llm, extract_token_usage
from core.types import StageOutput
from stages.base import BaseStage


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Không tìm thấy JSON: {text[:200]}")


class QueryRewriterStage(BaseStage):

    def __init__(self, run_config: RunConfig, stage_config: StageConfig):
        super().__init__(run_config, stage_config)
        enable_reasoning = stage_config.extra.get("reasoning", False)
        reasoning_effort = stage_config.extra.get("reasoning_effort", None)
        self.llm = build_llm(
            model=run_config.query_rewriter_model,
            base_url=run_config.query_rewriter_base_url,
            api_key=run_config.query_rewriter_api_key,
            temperature=0.0,
            seed=run_config.seed,
            enable_reasoning=enable_reasoning,
            reasoning_effort=reasoning_effort,
        )

    def run(self, context: dict[str, Any]) -> StageOutput:
        start = time.time()
        prompt = self.stage_config.prompt_template.format_map(_SafeDict(context))
        response = self.llm.invoke(prompt)
        latency_ms = (time.time() - start) * 1000

        token_usage = extract_token_usage(response)

        raw_text = response.content
        try:
            parsed = _extract_json(raw_text)
            rewritten = parsed.get("rewritten_query", "").strip()
            reasoning = parsed.get("reasoning", "")
        except (ValueError, json.JSONDecodeError):
            rewritten = ""
            reasoning = ""

        # Fallback: nếu rewriter không trả query hợp lệ, giữ nguyên query cũ
        if not rewritten:
            rewritten = context.get("question", "")

        return StageOutput(
            stage_name="query_rewriter",
            data={
                "rewritten_query": rewritten,
                "reasoning": reasoning,
                "raw_response": raw_text,
                "prompt": prompt,
            },
            latency_ms=latency_ms,
            token_usage=token_usage,
        )


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""
