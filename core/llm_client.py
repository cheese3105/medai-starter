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
) -> ChatOpenAI:
    kwargs = {"seed": seed} if seed is not None else {}
    return ChatOpenAI(
        model=model, base_url=base_url, api_key=api_key,
        temperature=temperature,
        timeout=10.0,
        max_retries=2,
        extra_body={"reasoning": {"effort": "none"}},
        **kwargs,
    )
