import os
from typing import Optional

from langchain_openai import ChatOpenAI


def build_llm(
    model: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
) -> ChatOpenAI:
    """Tạo LLM client qua endpoint OpenAI-compatible.

    `model`, `temperature`, và `seed` được truyền từ RunConfig hoặc cấu hình
    để không hard-code.

    base_url/api_key luôn đọc từ .env vì đây là thông tin hạ tầng/secret,
    không phải tham số thí nghiệm.
    """
    kwargs = {}
    if seed is not None:
        kwargs["seed"] = seed

    return ChatOpenAI(
        model=model,
        base_url=os.getenv("NINE_ROUTER_BASE_URL", "http://localhost:20128/v1"),
        api_key=os.getenv("NINE_ROUTER_API_KEY", "dummy"),
        temperature=temperature,
        **kwargs,
    )
