"""LLM client factory — tạo ChatOpenAI từ .env config."""

from __future__ import annotations

from collections.abc import Callable
from email.utils import parsedate_to_datetime
import os
import time
from typing import Any, Optional

from langchain_openai import ChatOpenAI


RATE_LIMIT_MAX_RETRIES = 4
RATE_LIMIT_INITIAL_BACKOFF_SECONDS = 10.0
RATE_LIMIT_MAX_BACKOFF_SECONDS = 60.0


def _is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    return status_code == 429


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            delay = float(retry_after_ms) / 1000
            return delay if delay > 0 else None
        except (TypeError, ValueError):
            pass

    retry_after = headers.get("retry-after")
    if retry_after is None:
        return None
    try:
        delay = float(retry_after)
        return delay if delay > 0 else None
    except (TypeError, ValueError):
        pass

    try:
        retry_at = parsedate_to_datetime(str(retry_after))
        delay = retry_at.timestamp() - time.time()
        return delay if delay > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def invoke_with_rate_limit_backoff(
    llm: Any,
    prompt: str,
    *,
    _sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Invoke an LLM, silently retrying exhausted HTTP 429 responses.

    ChatOpenAI first performs its own short retries. This longer backoff applies
    only if those retries still end in a rate-limit error.
    """
    for retry_index in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            if not _is_rate_limit_error(exc) or retry_index == RATE_LIMIT_MAX_RETRIES:
                raise
            retry_after = _retry_after_seconds(exc)
            delay = retry_after if retry_after is not None else (
                RATE_LIMIT_INITIAL_BACKOFF_SECONDS * (2 ** retry_index)
            )
            _sleep(min(delay, RATE_LIMIT_MAX_BACKOFF_SECONDS))

    raise RuntimeError("unreachable")


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
