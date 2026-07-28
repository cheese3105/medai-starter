"""Verifier Agent (v2) - kiểm chứng đáp án của Reasoning Agent có được
evidence hỗ trợ không, rồi chốt answer/explanation/confidence cuối cùng.

Theo đúng pattern của agents/reasoning_agent.py và agents/retrieval_agent.py:
1 factory `make_verifier_agent_node(run_config)` nhận config qua closure,
trả về 1 node function thuần `state -> state`. Được add vào graph.py khi
`run_config.verifier.enabled = true`.

Nguyên tắc an toàn y khoa: ở mode mặc định "annotate_and_guard", khi
evidence KHÔNG hỗ trợ đáp án, Verifier không tự bịa ra đáp án khác - chỉ
hạ confidence và gắn cờ verifier_verdict="unsupported" để phục vụ error
analysis (thà báo "không chắc" còn hơn tự tin sai).

Verifier CHỈ ghi vào các field đã có sẵn trong AgentState từ đầu
(answer/explanation/confidence bị ghi đè bằng giá trị đã chốt,
verifier_verdict/verifier_support_score/verifier_notes là field mới) -
không đổi schema của predictions.jsonl, evaluate.py không cần sửa gì.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from llm_client import build_llm
from run_config import RunConfig
from state import AgentState

LABELS = ["A", "B", "C", "D"]

DEFAULT_VERIFIER_PROMPT = """You are a careful medical fact-checker. Your job is to
check whether a draft answer to a medical multiple-choice question is actually
supported by the retrieved evidence below - NOT to re-solve the question from
scratch.

Question:
{question}

Choices:
{choices}

Retrieved evidence:
{evidence}

Draft answer: {draft_answer}
Draft explanation: {draft_explanation}

Instructions:
- If the evidence clearly supports the draft answer, verdict="supported".
- If the evidence is irrelevant, missing, or too weak to confirm the draft
  answer, verdict="unsupported" - do NOT invent support that isn't there.
- If the evidence partially supports it (e.g. supports the general topic but
  not the specific choice), verdict="partial".
- Only change final_answer away from the draft answer if the evidence
  clearly and specifically points to a different choice. Otherwise keep
  final_answer equal to the draft answer.
- final_confidence should be LOWER than the draft confidence when verdict is
  "unsupported" or "partial", and can stay the same or increase slightly
  when verdict is "supported".

Return JSON only, no other text outside the JSON:
{{
  "verdict": "supported" or "partial" or "unsupported",
  "support_score": confidence from 0 to 1 that the evidence backs the answer,
  "final_answer": "A" or "B" or "C" or "D",
  "final_explanation": "short explanation citing which evidence was used (or noting the lack of it)",
  "final_confidence": confidence from 0 to 1,
  "notes": "one short sentence on why you made this decision"
}}
"""


def _extract_json(text: str) -> dict:
    """LLM đôi khi bọc JSON trong ```json ... ``` -> bóc tách an toàn."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group(0) if match else text
    return json.loads(raw)


def _format_evidence(retrieved_docs: Optional[list[Any]]) -> str:
    """Giống hệt helper trong reasoning_agent.py - lặp lại có chủ đích để
    verifier_agent.py độc lập, xoá được mà không ảnh hưởng agent khác."""
    if not retrieved_docs:
        return "No retrieved evidence."

    blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        if not isinstance(doc, dict) or doc.get("error"):
            continue
        title = doc.get("title") or "unknown source"
        source = doc.get("source") or "unknown corpus"
        text = doc.get("text", "")
        blocks.append(f"[{i}] title={title}; source={source}\n{text}")

    return "\n\n".join(blocks) if blocks else "No retrieved evidence."


def _add_tokens(u1: Optional[dict], u2: Optional[dict]) -> Optional[dict]:
    if not u1: return u2
    if not u2: return u1
    return {
        "input_tokens": u1.get("input_tokens", 0) + u2.get("input_tokens", 0),
        "output_tokens": u1.get("output_tokens", 0) + u2.get("output_tokens", 0),
        "total_tokens": u1.get("total_tokens", 0) + u2.get("total_tokens", 0),
    }


def _estimate_cost(token_usage: Optional[dict], pricing) -> Optional[float]:
    if not token_usage or pricing.input_per_1k is None or pricing.output_per_1k is None:
        return None
    input_tokens = token_usage.get("input_tokens", 0) or 0
    output_tokens = token_usage.get("output_tokens", 0) or 0
    return (input_tokens / 1000) * pricing.input_per_1k + (output_tokens / 1000) * pricing.output_per_1k


def make_verifier_agent_node(run_config: RunConfig):
    """Factory tạo node Verifier Agent, đóng gói run_config qua closure.

    verifier.model cho phép dùng 1 model khác (rẻ hơn/khác vendor) để kiểm
    chứng, tách khỏi model suy luận chính - None thì dùng chung
    run_config.model.
    """
    verifier_config = run_config.verifier
    prompt_template = verifier_config.prompt_template or DEFAULT_VERIFIER_PROMPT
    model = verifier_config.model or run_config.model

    def verifier_agent_node(state: AgentState) -> AgentState:
        import time
        start_time = time.perf_counter()
        verifier_tokens = None
        draft_answer = state.get("answer")

        if run_config.debug:
            print(f"\n--- [Verifier Agent] Handling Question ID: {state.get('question_id')} ---")
            print(f"[Verifier Agent] Input Question: {state['question']}")
            print(f"[Verifier Agent] Draft Answer: '{draft_answer}', Draft Confidence: {state.get('confidence')}")

        # Reasoning Agent đã fail (INVALID/None) -> không có gì để verify,
        # pass-through nguyên trạng, không cố "sửa" 1 câu trả lời không tồn tại.
        if draft_answer in (None, "INVALID"):
            if run_config.debug:
                print(f"[Verifier Agent] Skipped: Draft answer is invalid or missing.")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return {
                **state,
                "verifier_verdict": "error",
                "verifier_support_score": None,
                "verifier_notes": "Draft answer invalid - verifier skipped.",
                "verifier_latency_ms": duration_ms,
                "verifier_token_usage": None,
            }

        try:
            llm = build_llm(model=model, temperature=0.0, env_prefix="VERIFIER_MODEL")

            choices_text = "\n".join(
                f"{label}. {choice}" for label, choice in zip(LABELS, state["choices"])
            )
            prompt = prompt_template.format(
                question=state["question"],
                choices=choices_text,
                evidence=_format_evidence(state.get("retrieved_docs")),
                draft_answer=draft_answer,
                draft_explanation=state.get("explanation") or "",
            )

            response = llm.invoke(prompt)
            verifier_tokens = getattr(response, "usage_metadata", None)

            parsed = _extract_json(response.content)

            verdict = parsed.get("verdict", "unsupported")
            support_score = parsed.get("support_score")

            final_answer = parsed.get("final_answer")
            if final_answer not in LABELS:
                final_answer = draft_answer  # format lỗi -> giữ nguyên đáp án gốc

            final_explanation = parsed.get("final_explanation") or state.get("explanation")
            final_confidence = parsed.get("final_confidence")
            if final_confidence is None:
                final_confidence = state.get("confidence")

            # "annotate_and_guard": không tự đổi đáp án khi unsupported, chỉ
            # hạ confidence + gắn cờ - đúng nguyên tắc an toàn y khoa.
            if verifier_config.mode == "annotate_and_guard" and verdict == "unsupported":
                final_answer = draft_answer
                final_confidence = min(final_confidence if final_confidence is not None else 0.0, 0.3)

            if run_config.debug:
                print(f"[Verifier Agent] Output: Final Answer = '{final_answer}', Verdict = {verdict}, Confidence = {final_confidence}, Support Score = {support_score}")

            duration_ms = (time.perf_counter() - start_time) * 1000
            total_latency = (state.get("latency_ms") or 0.0) + duration_ms
            total_tokens = _add_tokens(state.get("token_usage"), verifier_tokens)

            return {
                **state,
                "answer": final_answer,
                "explanation": final_explanation,
                "confidence": final_confidence,
                "verifier_verdict": verdict,
                "verifier_support_score": support_score,
                "verifier_notes": parsed.get("notes"),
                "verifier_latency_ms": duration_ms,
                "verifier_token_usage": verifier_tokens,
                "latency_ms": total_latency,
                "token_usage": total_tokens,
                "estimated_cost": _estimate_cost(total_tokens, run_config.pricing),
            }

        except Exception as exc:
            # 1 câu lỗi verifier không được làm sập cả graph - giữ nguyên
            # đáp án của Reasoning Agent, chỉ đánh dấu verifier bị lỗi.
            if run_config.debug:
                print(f"[Verifier Agent] Verification error occurred: {exc}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            total_latency = (state.get("latency_ms") or 0.0) + duration_ms
            return {
                **state,
                "verifier_verdict": "error",
                "verifier_support_score": None,
                "verifier_notes": f"Verifier failed: {exc}",
                "verifier_latency_ms": duration_ms,
                "verifier_token_usage": None,
                "latency_ms": total_latency,
            }

    return verifier_agent_node
