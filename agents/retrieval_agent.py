"""Retrieval Agent (v1) - truy xuất evidence từ MedRAG textbooks (Chroma)
trước khi Reasoning Agent trả lời.

Theo đúng pattern của agents/reasoning_agent.py: 1 factory
`make_retrieval_agent_node(run_config)` nhận config qua closure, trả về 1
node function thuần `state -> state`. Được add vào graph.py khi
`run_config.retrieval.enabled = true`.

Chỉ ghi vào field `retrieved_docs` (đã có sẵn trong AgentState từ đầu) -
không đổi schema, không ảnh hưởng các variant không dùng retrieval (v0).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from retrieval.retriever import MedicalEvidenceRetriever
from run_config import RunConfig
from state import AgentState

# Cache theo (chroma_dir, collection) - tránh mở lại Chroma PersistentClient
# + load lại embedding config cho MỖI câu hỏi khi benchmark hàng nghìn câu.
_retriever_cache: dict[tuple[str, str], MedicalEvidenceRetriever] = {}


def _get_retriever(chroma_dir: str | None, collection_name: str | None) -> MedicalEvidenceRetriever:
    from retrieval.ingest_data import DEFAULT_CHROMA_DIR, DEFAULT_COLLECTION_NAME

    resolved_dir = Path(chroma_dir) if chroma_dir else DEFAULT_CHROMA_DIR
    resolved_collection = collection_name or DEFAULT_COLLECTION_NAME

    key = (str(resolved_dir), resolved_collection)
    if key not in _retriever_cache:
        _retriever_cache[key] = MedicalEvidenceRetriever(
            chroma_dir=resolved_dir,
            collection_name=resolved_collection,
        )
    return _retriever_cache[key]


def make_retrieval_agent_node(run_config: RunConfig):
    top_k = run_config.retrieval.top_k
    # chroma_dir/collection có thể để mặc định (None -> DEFAULT_* trong
    # ingest_data.py) hoặc đọc từ run_config nếu bạn mở rộng RetrievalConfig
    # thêm 2 field này sau này.
    chroma_dir = getattr(run_config.retrieval, "chroma_dir", None)
    collection_name = getattr(run_config.retrieval, "collection_name", None)

    def retrieval_agent_node(state: AgentState) -> AgentState:
        try:
            retriever = _get_retriever(chroma_dir, collection_name)
            evidence = retriever.retrieve(state["question"], top_k=top_k)
            retrieved_docs = [asdict(chunk) for chunk in evidence]
        except Exception as exc:
            # 1 câu lỗi retrieval không được làm sập cả graph - Reasoning
            # Agent vẫn chạy tiếp, chỉ là không có evidence (giống hệt
            # nguyên tắc "invalid response must be counted, not crash").
            retrieved_docs = [{"error": f"Retrieval failed: {exc}"}]

        return {**state, "retrieved_docs": retrieved_docs}

    return retrieval_agent_node
