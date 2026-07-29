import os
from typing import Optional

from langchain_openai import ChatOpenAI


def build_llm(
    model: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    env_prefix: Optional[str] = None,
) -> ChatOpenAI:
    """Create LLM client via OpenAI-compatible endpoint.

    Args:
        model: Model name
        temperature: Sampling temperature
        seed: Optional seed for reproducibility
        env_prefix: Optional env var prefix for base_url/api_key.
            E.g. env_prefix="VERIFIER_MODEL" reads VERIFIER_MODEL_BASE_URL and
            VERIFIER_MODEL_API_KEY. Falls back to REASONING_MODEL_* if not set.
    """
    kwargs = {}
    if seed is not None:
        kwargs["seed"] = seed

    # Resolve base_url and api_key based on env_prefix
    if env_prefix:
        base_url = os.getenv(f"{env_prefix}_BASE_URL") or os.getenv("REASONING_MODEL_BASE_URL", "http://localhost:20128/v1")
        api_key = os.getenv(f"{env_prefix}_API_KEY") or os.getenv("REASONING_MODEL_API_KEY", "dummy")
    else:
        base_url = os.getenv("REASONING_MODEL_BASE_URL", "http://localhost:20128/v1")
        api_key = os.getenv("REASONING_MODEL_API_KEY", "dummy")

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
