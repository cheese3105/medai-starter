"""Config chạy cho 1 lần benchmark (1 variant).

Toàn bộ tham số ảnh hưởng đến kết quả (model, prompt_version, temperature,
seed, retrieval, verifier) đọc từ file YAML - KHÔNG hard-code trong code.
Điều này giúp khoá config trước khi chạy test set chính thức, và log lại
run_config đầy đủ cho từng file predictions.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml


@dataclass
class RetrievalConfig:
    enabled: bool = False
    top_k: int = 5


@dataclass
class VerifierConfig:
    enabled: bool = False


@dataclass
class PricingConfig:
    """Để trống (null) nếu chưa rõ giá - estimated_cost sẽ trả về None."""
    input_per_1k: Optional[float] = None
    output_per_1k: Optional[float] = None


@dataclass
class RunConfig:
    variant: str
    model: str
    prompt_version: str        # chỉ là nhãn/tên để log, KHÔNG dùng để tra cứu nữa
    prompt_template: str        # nội dung prompt thật - sửa trực tiếp trong YAML
    temperature: float = 0.0
    seed: Optional[int] = None
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_run_config(path: str) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # model is now optional in YAML since it is defined in .env
    required = ["variant", "prompt_version", "prompt_template"]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"File config '{path}' thiếu trường bắt buộc: {missing}")

    model_name = os.getenv("REASONING_MODEL") or raw.get("model")
    if not model_name:
        raise ValueError(f"Model name is not defined. Please set REASONING_MODEL in .env or 'model' in config '{path}'")

    return RunConfig(
        variant=raw["variant"],
        model=model_name,
        prompt_version=raw["prompt_version"],
        prompt_template=raw["prompt_template"],
        temperature=raw.get("temperature", 0.0),
        seed=raw.get("seed"),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        verifier=VerifierConfig(**raw.get("verifier", {})),
        pricing=PricingConfig(**raw.get("pricing", {})),
    )
