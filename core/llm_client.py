"""LLM client factory — tạo ChatOpenAI từ .env config."""

from __future__ import annotations

from typing import Optional

from langchain_openai import ChatOpenAI


def build_llm(
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    enable_reasoning: bool = True,
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = 2048,
) -> ChatOpenAI:
    kwargs = {"seed": seed} if seed is not None else {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    extra_body: dict[str, Any] = {}
    if not enable_reasoning:
        # Tắt hidden chain-of-thought cho các stage chỉ cần JSON ngắn (verifier, query_rewriter)
        extra_body["reasoning"] = {"effort": "none"}
    elif reasoning_effort is not None:
        extra_body["reasoning"] = {"effort": reasoning_effort}

    return ChatOpenAI(
        model=model, base_url=base_url, api_key=api_key,
        temperature=temperature,
        timeout=60.0 if enable_reasoning else 30.0,
        max_retries=2,
        extra_body=extra_body if extra_body else None,
        **kwargs,
    )
