"""Config loader — đọc YAML (.env cho secret, YAML cho tham số thí nghiệm)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class StageConfig:
    name: str
    enabled: bool = True
    prompt_template: str = ""
    max_iterations: int = 3
    top_k: int = 5
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShortTermMemoryConfig:
    enabled: bool = False
    scope: str = "loop"      # "loop" | "session" | "both"
    max_items: int = 10
    max_turns: int = 6


@dataclass
class LongTermMemoryConfig:
    enabled: bool = False
    top_k: int = 3
    mode: str = "read_only"  # "read_only" | "read_write"
    collection_name: str = "user_memory"
    chroma_dir: Optional[str] = None
    user_id: str = "default_user"
    write_after_answer: bool = False
    write_prompt_template: str = ""


@dataclass
class MemoryConfig:
    short_term: ShortTermMemoryConfig = field(default_factory=ShortTermMemoryConfig)
    long_term: LongTermMemoryConfig = field(default_factory=LongTermMemoryConfig)


@dataclass
class PricingConfig:
    input_per_1k: Optional[float] = None
    output_per_1k: Optional[float] = None


@dataclass
class RunConfig:
    variant: str = "v0"
    debug: str = "off"  # "off" | "info" | "verbose"
    pipeline: list[StageConfig] = field(default_factory=list)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    seed: Optional[int] = None

    # Đọc từ .env
    model: str = ""
    model_base_url: str = ""
    model_api_key: str = ""
    verifier_model: str = ""
    verifier_base_url: str = ""
    verifier_api_key: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    query_rewriter_model: str = ""
    query_rewriter_base_url: str = ""
    query_rewriter_api_key: str = ""

    @property
    def is_debug(self) -> bool:
        return self.debug in ("info", "verbose")

    @property
    def is_verbose(self) -> bool:
        return self.debug == "verbose"

    def get_stage(self, name: str) -> Optional[StageConfig]:
        for s in self.pipeline:
            if s.name == name and s.enabled:
                return s
        return None

    @property
    def max_iterations(self) -> int:
        v = self.get_stage("verifier")
        return v.max_iterations if v else 1


VALID_STAGE_NAMES = {"retrieval", "reasoning", "verifier", "query_rewriter"}
VALID_STM_SCOPES = {"loop", "session", "both"}
VALID_LTM_MODES = {"read_only", "read_write"}


def _parse_stage(raw: dict) -> StageConfig:
    known_keys = {"name", "enabled", "prompt_template", "max_iterations", "top_k", "temperature"}
    extra = {k: v for k, v in raw.items() if k not in known_keys}
    return StageConfig(
        name=raw["name"],
        enabled=raw.get("enabled", True),
        prompt_template=raw.get("prompt_template", ""),
        max_iterations=raw.get("max_iterations", 3),
        top_k=raw.get("top_k", 5),
        temperature=raw.get("temperature", 0.0),
        extra=extra,
    )


def _parse_memory(raw: dict) -> MemoryConfig:
    st_raw = raw.get("short_term", {})
    lt_raw = raw.get("long_term", {})
    stm = ShortTermMemoryConfig(
        enabled=st_raw.get("enabled", False),
        scope=st_raw.get("scope", "loop"),
        max_items=st_raw.get("max_items", 10),
        max_turns=st_raw.get("max_turns", 6),
    )
    ltm = LongTermMemoryConfig(
        enabled=lt_raw.get("enabled", False),
        top_k=lt_raw.get("top_k", 3),
        mode=lt_raw.get("mode", "read_only"),
        collection_name=lt_raw.get("collection_name", "user_memory"),
        chroma_dir=lt_raw.get("chroma_dir"),
        user_id=lt_raw.get("user_id", "default_user"),
        write_after_answer=lt_raw.get("write_after_answer", False),
        write_prompt_template=lt_raw.get("write_prompt_template", ""),
    )
    return MemoryConfig(short_term=stm, long_term=ltm)


def load_config(config_path: str, env_path: str = ".env") -> RunConfig:
    """Load YAML + .env, validate, trả về RunConfig."""

    env_file = Path(env_path)
    if env_file.exists():
        load_dotenv(env_file)

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Pipeline
    stages = [_parse_stage(s) for s in raw.get("pipeline", [])]
    for s in stages:
        if s.name not in VALID_STAGE_NAMES:
            raise ValueError(f"Stage '{s.name}' không hợp lệ. Chấp nhận: {VALID_STAGE_NAMES}")
    if not any(s.name == "reasoning" for s in stages):
        raise ValueError("Pipeline bắt buộc phải có stage 'reasoning'.")

    # Memory
    mem = _parse_memory(raw.get("memory", {}))
    if mem.short_term.scope not in VALID_STM_SCOPES:
        raise ValueError(f"STM scope '{mem.short_term.scope}' không hợp lệ.")
    if mem.long_term.mode not in VALID_LTM_MODES:
        raise ValueError(f"LTM mode '{mem.long_term.mode}' không hợp lệ.")

    # Pricing
    pricing_raw = raw.get("pricing", {})
    pricing = PricingConfig(
        input_per_1k=pricing_raw.get("input_per_1k"),
        output_per_1k=pricing_raw.get("output_per_1k"),
    )

    # Model (bắt buộc)
    model = os.getenv("REASONING_MODEL", "")
    if not model:
        raise ValueError("REASONING_MODEL chưa được set trong .env.")

    # Debug level (tương thích ngược với bool cũ)
    raw_debug = raw.get("debug", False)
    if isinstance(raw_debug, bool):
        debug_level = "info" if raw_debug else "off"
    elif isinstance(raw_debug, str):
        debug_level = raw_debug.strip().lower()
        if debug_level not in ("off", "info", "verbose"):
            raise ValueError(f"debug='{raw_debug}' không hợp lệ. Chấp nhận: off/info/verbose")
    else:
        raise ValueError(f"debug phải là bool hoặc string, nhận được: {raw_debug!r}")

    return RunConfig(
        variant=raw.get("variant", "v0"),
        debug=debug_level,
        pipeline=stages,
        memory=mem,
        pricing=pricing,
        seed=raw.get("seed"),
        model=model,
        model_base_url=os.getenv("REASONING_MODEL_BASE_URL", "http://localhost:11434/v1"),
        model_api_key=os.getenv("REASONING_MODEL_API_KEY", "dummy"),
        verifier_model=os.getenv("VERIFIER_MODEL", model),
        verifier_base_url=os.getenv("VERIFIER_MODEL_BASE_URL",
                                     os.getenv("REASONING_MODEL_BASE_URL", "http://localhost:11434/v1")),
        verifier_api_key=os.getenv("VERIFIER_MODEL_API_KEY",
                                    os.getenv("REASONING_MODEL_API_KEY", "dummy")),
        embedding_model=os.getenv("EMBEDDING_MODEL", ""),
        embedding_base_url=os.getenv("EMBEDDING_MODEL_API_BASE",
                                      os.getenv("EMBEDDING_MODEL_BASE_URL", "")),
        embedding_api_key=os.getenv("EMBEDDING_MODEL_API_KEY", ""),
        query_rewriter_model=os.getenv("QUERY_REWRITER_MODEL", model),
        query_rewriter_base_url=os.getenv("QUERY_REWRITER_MODEL_BASE_URL",
                                           os.getenv("REASONING_MODEL_BASE_URL", "http://localhost:11434/v1")),
        query_rewriter_api_key=os.getenv("QUERY_REWRITER_MODEL_API_KEY",
                                          os.getenv("REASONING_MODEL_API_KEY", "dummy")),
    )
