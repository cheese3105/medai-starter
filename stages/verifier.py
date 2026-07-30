"""Verifier stage — fact-check draft answer, trả verdict + suggested_query."""

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
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Không tìm thấy JSON: {text[:200]}")


class VerifierStage(BaseStage):

    def __init__(self, run_config: RunConfig, stage_config: StageConfig):
        super().__init__(run_config, stage_config)
        self.llm = build_llm(
            model=run_config.verifier_model,
            base_url=run_config.verifier_base_url,
            api_key=run_config.verifier_api_key,
            temperature=0.0,
            seed=run_config.seed,
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
            parsed = {
                "verdict": "supported",
                "final_answer": context.get("draft_answer", "INVALID"),
                "confidence": context.get("draft_confidence", 0.0),
            }

        verdict = parsed.get("verdict", "supported").lower().strip()
        if verdict not in ("supported", "unsupported"):
            verdict = "supported"

        final_answer = parsed.get("final_answer", context.get("draft_answer", ""))
        confidence = float(parsed.get("confidence", context.get("draft_confidence", 0.0)))

        if verdict == "unsupported":
            confidence = min(confidence, 0.3)
        if context.get("choices") and final_answer not in LABELS:
            final_answer = context.get("draft_answer", "INVALID")

        return StageOutput(
            stage_name="verifier",
            data={
                "verdict": verdict,
                "final_answer": final_answer,
                "confidence": confidence,
                "explanation": parsed.get("explanation", ""),
                "suggested_query": parsed.get("suggested_query", ""),
                "raw_response": raw_text,
                "prompt": prompt,
            },
            latency_ms=latency_ms,
            token_usage=token_usage,
        )


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""
