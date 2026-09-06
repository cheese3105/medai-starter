"""Shared types dùng xuyên suốt hệ thống V0-V3.

Field nào version không dùng thì để None, không xoá khỏi schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EpisodeInput:
    """Input cho 1 câu hỏi (benchmark) hoặc 1 lượt chat."""

    question_id: str
    question: str
    choices: list[str] = field(default_factory=list)
    gold_answer: Optional[str] = None


@dataclass
class StageOutput:
    """Kết quả trả về từ 1 stage sau 1 lần chạy."""

    stage_name: str
    data: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)


@dataclass
class IterationRecord:
    """Ghi lại 1 vòng lặp của controller."""

    iteration: int
    stage_outputs: list[StageOutput] = field(default_factory=list)
    verdict: Optional[str] = None


@dataclass
class EpisodeResult:
    """Schema output cho log JSONL — cùng 1 schema cho mọi version."""

    # Nhóm A: mọi version
    question_id: str = ""
    variant: str = ""
    model: str = ""
    predicted_answer: str = ""
    explanation: str = ""
    confidence: float = 0.0
    gold_answer: Optional[str] = None
    is_correct: Optional[bool] = None
    total_latency_ms: float = 0.0
    total_token_usage: dict[str, int] = field(default_factory=dict)
    estimated_cost: Optional[float] = None

    # Nhóm B: từ V1 (retrieval)
    evidence_used: Optional[list[str]] = None
    retrieval_latency_ms: Optional[float] = None
    retrieval_token_usage: Optional[dict[str, int]] = None

    # Nhóm C: từ V2 (verifier + loop)
    iteration_count: Optional[int] = None
    verifier_verdict: Optional[str] = None
    verdict_history: Optional[list[str]] = None
    stopped_after_max_iterations: Optional[bool] = None
    reasoning_latency_ms: Optional[float] = None
    reasoning_token_usage: Optional[dict[str, int]] = None
    verifier_latency_ms: Optional[float] = None
    verifier_token_usage: Optional[dict[str, int]] = None
    query_rewrite_count: Optional[int] = None        # số lần QR chạy sau failed verification (retry only)
    rewriter_latency_ms: Optional[float] = None
    rewriter_token_usage: Optional[dict[str, int]] = None
    initial_query_generated: Optional[bool] = None  # True nếu QR chạy ở iter 1 (initial query formulation)

    # Nhóm D: từ V3 (memory)
    stm_scope_used: Optional[str] = None
    loop_history_length: Optional[int] = None
    ltm_facts_retrieved: Optional[list[str]] = None
    ltm_write_triggered: Optional[bool] = None
    user_id: Optional[str] = None
