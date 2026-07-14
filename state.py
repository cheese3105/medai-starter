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

    # Phục vụ error analysis - None khi chưa bật Retrieval/Verifier Agent,
    # tự động có giá trị khi 2 agent đó được thêm vào graph (không cần sửa
    # schema/benchmark.py lúc đó)
    retrieved_docs: Optional[List[Any]]
    agent_trace: Optional[List[Any]]
