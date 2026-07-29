"""Reasoning Agent for Chat mode - generates free-form answers (not multiple choice).

Differences from benchmark reasoning_agent.py:
- Output: free-form English text (not JSON with A/B/C/D)
- Input: includes STM history and LTM context
- Internally outputs JSON for confidence tracking, but user sees plain text
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Optional

from llm_client import build_llm
from run_config import RunConfig


LABELS = ["A", "B", "C", "D"]

CHAT_REASONING_PROMPT = """You are a medical AI assistant. Answer the user's question
accurately and concisely based on the retrieved evidence and conversation context.

{conversation_context}

Retrieved evidence:
{evidence}

User's question:
{question}

Instructions:
- Use the retrieved evidence to support your answer when relevant.
- If the evidence is irrelevant or insufficient, answer from your medical knowledge.
- Be concise but thorough.
- Always recommend consulting a doctor for serious or persistent symptoms.
- Answer in English.

Return JSON only, no other text:
{{
  "answer": "your complete answer text here",
  "confidence": confidence from 0 to 1 that your answer is correct,
  "key_sources": "brief note on which evidence pieces you relied on (or 'general knowledge' if none)"
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


def make_chat_reasoning_node(run_config: RunConfig):
    """Factory for chat-mode reasoning node.

    Uses chat-specific prompt that outputs free-form text + confidence.
    Accepts conversation_context from STM/LTM via state.
    """
    prompt_template = getattr(run_config, 'chat_reasoning_prompt', None) or CHAT_REASONING_PROMPT

    def chat_reasoning_node(state: dict) -> dict:
        llm = build_llm(
            model=run_config.model,
            temperature=run_config.temperature,
            seed=run_config.seed,
        )

        conversation_context = state.get("conversation_context") or ""
        evidence_text = _format_evidence(state.get("retrieved_docs"))

        prompt = prompt_template.format(
            question=state["question"],
            evidence=evidence_text,
            conversation_context=conversation_context,
        )

        if run_config.debug:
            print(f"\n--- [Chat Reasoning Agent] ---")
            print(f"[Chat Reasoning] Question: {state['question']}")
            if conversation_context:
                print(f"[Chat Reasoning] Has conversation context ({len(conversation_context)} chars)")

        start = time.perf_counter()
        response = llm.invoke(prompt)
        latency_ms = (time.perf_counter() - start) * 1000

        raw_output = response.content
        token_usage = getattr(response, "usage_metadata", None)

        try:
            parsed = _extract_json(raw_output)
            answer = parsed.get("answer", raw_output)
            confidence = parsed.get("confidence")
            key_sources = parsed.get("key_sources")
        except (json.JSONDecodeError, AttributeError, TypeError):
            answer = raw_output
            confidence = None
            key_sources = None

        if run_config.debug:
            print(f"[Chat Reasoning] Confidence: {confidence}")
            print(f"[Chat Reasoning] Latency: {latency_ms:.0f}ms")

        total_latency = (state.get("latency_ms") or 0.0) + latency_ms
        total_tokens = _add_tokens(state.get("token_usage"), token_usage)

        return {
            **state,
            "answer": answer,
            "confidence": confidence,
            "explanation": key_sources,
            "raw_output": raw_output,
            "reasoning_latency_ms": latency_ms,
            "reasoning_token_usage": token_usage,
            "latency_ms": total_latency,
            "token_usage": total_tokens,
        }

    return chat_reasoning_node
