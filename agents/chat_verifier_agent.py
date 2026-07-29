"""Verifier Agent for Chat mode - checks if answer is supported by evidence,
adds disclaimer if unsupported.

Differences from benchmark verifier_agent.py:
- Does not pick A/B/C/D final answer
- Adds disclaimer text when evidence doesn't support the answer
- Output is still free-form text for user
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from llm_client import build_llm
from run_config import RunConfig


CHAT_VERIFIER_PROMPT = """You are a medical fact-checker. Check whether the draft answer
below is supported by the retrieved evidence. Do NOT re-answer the question.

User's question:
{question}

Retrieved evidence:
{evidence}

Draft answer:
{draft_answer}

Instructions:
- If the evidence clearly supports the draft answer, verdict="supported".
- If the evidence is irrelevant or too weak, verdict="unsupported".
- If partially supported, verdict="partial".
- Do NOT change the answer content. Only assess support level.
- Suggest a brief disclaimer if the answer is not fully supported.

Return JSON only:
{{
  "verdict": "supported" or "partial" or "unsupported",
  "support_score": confidence from 0 to 1,
  "disclaimer": "brief disclaimer to append if verdict is not supported (empty string if supported)",
  "notes": "one sentence explanation"
}}
"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group(0) if match else text
    return json.loads(raw)


def _format_evidence(retrieved_docs: Optional[list[Any]]) -> str:
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


def make_chat_verifier_node(run_config: RunConfig):
    """Factory for chat-mode verifier node.

    Checks evidence support and appends disclaimer when needed.
    Does not change the answer content itself.
    """
    verifier_config = run_config.verifier
    prompt_template = getattr(run_config, 'chat_verifier_prompt', None) or CHAT_VERIFIER_PROMPT
    model = verifier_config.model or run_config.model

    def chat_verifier_node(state: dict) -> dict:
        start_time = time.perf_counter()
        draft_answer = state.get("answer")

        if run_config.debug:
            print(f"\n--- [Chat Verifier Agent] ---")
            print(f"[Chat Verifier] Checking answer support...")

        if not draft_answer:
            duration_ms = (time.perf_counter() - start_time) * 1000
            return {
                **state,
                "verifier_verdict": "error",
                "verifier_support_score": None,
                "verifier_notes": "No draft answer to verify.",
                "verifier_latency_ms": duration_ms,
                "verifier_token_usage": None,
            }

        try:
            llm = build_llm(model=model, temperature=0.0, env_prefix="VERIFIER_MODEL")

            prompt = prompt_template.format(
                question=state["question"],
                evidence=_format_evidence(state.get("retrieved_docs")),
                draft_answer=draft_answer,
            )

            response = llm.invoke(prompt)
            verifier_tokens = getattr(response, "usage_metadata", None)

            parsed = _extract_json(response.content)

            verdict = parsed.get("verdict", "unsupported")
            support_score = parsed.get("support_score")
            disclaimer = parsed.get("disclaimer", "")
            notes = parsed.get("notes", "")

            # Append disclaimer to answer if not supported
            final_answer = draft_answer
            if verdict in ("unsupported", "partial") and disclaimer:
                final_answer = f"{draft_answer}\n\nNote: {disclaimer}"

            # Adjust confidence based on verdict
            confidence = state.get("confidence")
            if confidence is not None:
                if verdict == "unsupported":
                    confidence = min(confidence, 0.3)
                elif verdict == "partial":
                    confidence = min(confidence, 0.6)

            if run_config.debug:
                print(f"[Chat Verifier] Verdict: {verdict}, Support Score: {support_score}")
                if disclaimer:
                    print(f"[Chat Verifier] Disclaimer added: {disclaimer[:80]}")

            duration_ms = (time.perf_counter() - start_time) * 1000
            total_latency = (state.get("latency_ms") or 0.0) + duration_ms
            total_tokens = _add_tokens(state.get("token_usage"), verifier_tokens)

            return {
                **state,
                "answer": final_answer,
                "confidence": confidence,
                "verifier_verdict": verdict,
                "verifier_support_score": support_score,
                "verifier_notes": notes,
                "verifier_latency_ms": duration_ms,
                "verifier_token_usage": verifier_tokens,
                "latency_ms": total_latency,
                "token_usage": total_tokens,
            }

        except Exception as exc:
            if run_config.debug:
                print(f"[Chat Verifier] Error: {exc}")
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

    return chat_verifier_node
