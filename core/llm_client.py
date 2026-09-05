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
        # Tắt hidden chain-of-thought (giảm latency ~50-60% cho các stage chỉ cần JSON ngắn).
        model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}},
        **kwargs,
    )
