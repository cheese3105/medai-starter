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
    # v2: Retrieval Agent tự đánh giá evidence đủ chưa và viết lại truy vấn
    # nếu chưa đủ. max_iterations=1 (mặc định) = tắt hoàn toàn, hành vi và
    # chi phí giống hệt v1 (không tốn thêm lời gọi LLM nào).
    max_iterations: int = 1
    sufficiency_threshold: float = 0.6
    # None -> dùng chung model + provider của Reasoning Agent (REASONING_MODEL_*).
    # Set tên model khác nếu muốn dùng model/provider riêng cho bước tự đánh
    # giá (đọc endpoint qua RETRIEVAL_CHECK_MODEL_BASE_URL/_API_KEY trong .env).
    check_model: Optional[str] = None
    prompt_template: Optional[str] = None


@dataclass
class VerifierConfig:
    enabled: bool = False
    # "annotate_and_guard": khi evidence không hỗ trợ đáp án, KHÔNG tự bịa
    # đáp án khác - chỉ hạ confidence + gắn cờ verifier_verdict để phục vụ
    # error analysis (nguyên tắc an toàn y khoa).
    # "annotate_only": chỉ gắn nhãn/điểm, không đổi confidence.
    mode: str = "annotate_and_guard"
    # None -> dùng chung model + provider của Reasoning Agent (REASONING_MODEL_*).
    # Set tên model khác (VD model rẻ hơn hoặc của provider khác) nếu muốn -
    # endpoint/API key đọc qua VERIFIER_MODEL_BASE_URL/VERIFIER_MODEL_API_KEY
    # trong .env (không set -> tự fallback dùng REASONING_MODEL_*).
    model: Optional[str] = None
    prompt_template: Optional[str] = None  # None -> dùng DEFAULT_VERIFIER_PROMPT


@dataclass
class FollowUpDetectionConfig:
    method: str = "keyword"  # "keyword" | "semantic"
    keywords: list = field(default_factory=lambda: [
        "nó", "cái đó", "ở trên", "trước đó", "điều đó",
        "it", "this", "that", "above", "previous", "earlier"
    ])


@dataclass
class ShortTermMemoryConfig:
    enabled: bool = False
    max_turns: Optional[int] = None  # None = unlimited, số = sliding window
    persistence: bool = True
    session_dir: str = "memory/sessions"
    auto_load_last_session: bool = True
    follow_up_detection: FollowUpDetectionConfig = field(default_factory=FollowUpDetectionConfig)


@dataclass
class QACacheConfig:
    collection_name: str = "qa_cache"
    chroma_dir: str = "data/chroma"
    min_confidence: float = 0.8
    min_verifier_verdict: str = "supported"
    similarity_threshold: float = 0.85
    embedding_model: Optional[str] = None
    max_cache_size: int = 1000
    cache_eviction_policy: str = "lru"


@dataclass
class AnalyticsConfig:
    enabled: bool = False
    db_path: str = "memory/analytics.db"
    log_all_questions: bool = True


@dataclass
class LongTermMemoryConfig:
    enabled: bool = False
    qa_cache: QACacheConfig = field(default_factory=QACacheConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)


@dataclass
class MemoryConfig:
    debug: bool = False
    short_term: ShortTermMemoryConfig = field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)


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
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    debug: bool = False

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

    retrieval_raw = raw.get("retrieval", {})
    env_check_model = os.getenv("RETRIEVAL_CHECK_MODEL")
    if env_check_model:
        retrieval_raw["check_model"] = env_check_model
    retrieval_config = RetrievalConfig(**retrieval_raw)

    verifier_raw = raw.get("verifier", {})
    env_verifier_model = os.getenv("VERIFIER_MODEL")
    if env_verifier_model:
        verifier_raw["model"] = env_verifier_model
    verifier_config = VerifierConfig(**verifier_raw)

    # Parse memory config (v3)
    memory_raw = raw.get("memory", {})

    # Short-term memory
    stm_raw = memory_raw.get("short_term", {})
    follow_up_raw = stm_raw.pop("follow_up_detection", {})
    follow_up_config = FollowUpDetectionConfig(**follow_up_raw)
    stm_config = ShortTermMemoryConfig(**stm_raw, follow_up_detection=follow_up_config)

    # Long-term memory
    ltm_raw = memory_raw.get("long_term", {})
    qa_cache_config = QACacheConfig(**ltm_raw.get("qa_cache", {}))
    analytics_config = AnalyticsConfig(**ltm_raw.get("analytics", {}))
    ltm_config = LongTermMemoryConfig(
        enabled=ltm_raw.get("enabled", False),
        qa_cache=qa_cache_config,
        analytics=analytics_config
    )

    memory_config = MemoryConfig(
        debug=memory_raw.get("debug", False),
        short_term=stm_config,
        long_term=ltm_config
    )

    return RunConfig(
        variant=raw["variant"],
        model=model_name,
        prompt_version=raw["prompt_version"],
        prompt_template=raw["prompt_template"],
        temperature=raw.get("temperature", 0.0),
        seed=raw.get("seed"),
        retrieval=retrieval_config,
        verifier=verifier_config,
        pricing=PricingConfig(**raw.get("pricing", {})),
        memory=memory_config,
        debug=raw.get("debug", False),
    )
