import os
from typing import Optional

from langchain_openai import ChatOpenAI


def build_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    seed: Optional[int] = None,
) -> ChatOpenAI:
    """Tạo LLM client qua endpoint OpenAI-compatible.

    `model`/`seed` thường được truyền từ RunConfig (Mode 1 - benchmark) để
    không hard-code. Nếu không truyền (VD Mode 2 - chat), fallback về
    LLM_MODEL trong .env - phù hợp vì Mode 2 không cần chạy nhiều biến thể.

    base_url/api_key luôn đọc từ .env vì đây là thông tin hạ tầng/secret,
    không phải tham số thí nghiệm.
    """
    kwargs = {}
    if seed is not None:
        kwargs["seed"] = seed

    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("NINE_ROUTER_BASE_URL", "http://localhost:20128/v1"),
        api_key=os.getenv("NINE_ROUTER_API_KEY", "dummy"),
        temperature=temperature,
        **kwargs,
    )
