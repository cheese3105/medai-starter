import json
import re
import time
from typing import Optional

from llm_client import build_llm
from run_config import RunConfig
from state import AgentState

LABELS = ["A", "B", "C", "D"]


def _extract_json(text: str) -> dict:
    """LLM đôi khi bọc JSON trong ```json ... ``` -> bóc tách an toàn."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group(0) if match else text
    return json.loads(raw)


def _estimate_cost(token_usage: Optional[dict], pricing) -> Optional[float]:
    if not token_usage or pricing.input_per_1k is None or pricing.output_per_1k is None:
        return None
    input_tokens = token_usage.get("input_tokens", 0) or 0
    output_tokens = token_usage.get("output_tokens", 0) or 0
    return (input_tokens / 1000) * pricing.input_per_1k + (output_tokens / 1000) * pricing.output_per_1k


def make_reasoning_agent_node(run_config: RunConfig):
    """Factory tạo node Reasoning Agent, đóng gói run_config qua closure.

    Config (model/prompt_version/temperature/seed) KHÔNG đi qua AgentState -
    state chỉ mang dữ liệu của từng câu hỏi, config được "khoá" 1 lần lúc
    build graph. Nhờ vậy đổi biến thể (variant) chỉ cần đổi file YAML, không
    sửa logic node.
    """
    prompt_template = run_config.prompt_template

    def reasoning_agent_node(state: AgentState) -> AgentState:
        llm = build_llm(
            model=run_config.model,
            temperature=run_config.temperature,
            seed=run_config.seed,
        )

        choices_text = "\n".join(
            f"{label}. {choice}" for label, choice in zip(LABELS, state["choices"])
        )
        prompt = prompt_template.format(question=state["question"], choices=choices_text)

        start = time.perf_counter()
        response = llm.invoke(prompt)
        latency_ms = (time.perf_counter() - start) * 1000

        raw_output = response.content
        token_usage = getattr(response, "usage_metadata", None)

        try:
            parsed = _extract_json(raw_output)
            predicted = parsed.get("answer")
            if predicted not in LABELS:
                predicted = "INVALID"
        except (json.JSONDecodeError, AttributeError, TypeError):
            parsed = {}
            predicted = "INVALID"

        return {
            **state,
            "answer": predicted,
            "explanation": parsed.get("explanation"),
            "confidence": parsed.get("confidence"),
            "raw_output": raw_output,
            "latency_ms": latency_ms,
            "token_usage": token_usage,
            "estimated_cost": _estimate_cost(token_usage, run_config.pricing),
        }

    return reasoning_agent_node
