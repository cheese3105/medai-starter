"""LLM client factory — tạo ChatOpenAI từ .env config."""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI


def extract_token_usage(response: Any) -> dict[str, int]:
    """Return normalized input/output tokens when the provider reports them."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and (
        "input_tokens" in usage or "output_tokens" in usage
    ):
        return {
            "input": int(usage.get("input_tokens") or 0),
            "output": int(usage.get("output_tokens") or 0),
        }

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        usage = metadata.get("token_usage")
        if isinstance(usage, dict) and (
            "prompt_tokens" in usage
            or "completion_tokens" in usage
            or "input_tokens" in usage
            or "output_tokens" in usage
        ):
            return {
                "input": int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
                "output": int(
                    usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                ),
            }

    return {}


def build_llm(
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    enable_reasoning: bool = True,
    reasoning_effort: Optional[str] = None,
    max_tokens: Optional[int] = 2048,
    provider: Optional[str] = None,
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

    # Pin OpenRouter requests when a provider is configured. `only` restricts
    # routing to that provider and disabling fallbacks prevents silent rerouting.
    selected_provider = provider or os.getenv("OPENROUTER_PROVIDER", "").strip()
    if selected_provider and "openrouter.ai" in base_url.lower():
        extra_body["provider"] = {
            "only": [selected_provider],
            "allow_fallbacks": False,
        }

    return ChatOpenAI(
        model=model, base_url=base_url, api_key=api_key,
        temperature=temperature,
        timeout=60.0 if enable_reasoning else 30.0,
        max_retries=2,
        extra_body=extra_body if extra_body else None,
        **kwargs,
    )
