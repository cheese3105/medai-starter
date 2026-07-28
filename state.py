from typing import TypedDict, List, Optional, Any


class AgentState(TypedDict):
    """State chảy xuyên suốt graph - dùng cho Mode 1 (benchmark, trắc nghiệm
    4 đáp án). KHÔNG chứa gold answer - tránh leakage vào pipeline sinh câu
    trả lời (gold answer được lưu riêng, ngoài graph, chỉ để join sau).
    """

    # Input
    question_id: Optional[str]
    question: str
    choices: List[str]

    # Output chính
    answer: Optional[str]          # "A"/"B"/"C"/"D" hoặc "INVALID" nếu parse lỗi
    explanation: Optional[str]
    confidence: Optional[float]
    raw_output: Optional[str]      # text thô từ LLM, phục vụ debug parser

    # Phục vụ bảng cost/latency
    latency_ms: Optional[float]
    token_usage: Optional[dict]
    estimated_cost: Optional[float]

    # --- v2: Chi tiết latency và token cho từng agent ---
    retrieval_latency_ms: Optional[float]
    retrieval_token_usage: Optional[dict]
    reasoning_latency_ms: Optional[float]
    reasoning_token_usage: Optional[dict]
    verifier_latency_ms: Optional[float]
    verifier_token_usage: Optional[dict]

    # Phục vụ error analysis - None khi chưa bật Retrieval/Verifier Agent,
    # tự động có giá trị khi 2 agent đó được thêm vào graph (không cần sửa
    # schema/benchmark.py lúc đó)
    retrieved_docs: Optional[List[Any]]
    agent_trace: Optional[List[Any]]

    # --- v2: Retrieval Agent tự đánh giá + truy vấn lại (self-retry) ---
    # None khi retrieval.max_iterations=1 (mặc định, hành vi giống hệt v1)
    query_history: Optional[List[str]]
    retrieval_iterations: Optional[int]
    retrieval_sufficiency: Optional[float]

    # --- v2: Verifier Agent - None khi verifier.enabled=false ---
    verifier_verdict: Optional[str]  # "supported" | "partial" | "unsupported" | "error"
    verifier_support_score: Optional[float]
    verifier_notes: Optional[str]
