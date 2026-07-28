"""Retrieval Agent (v1 + v2) - truy xuất evidence từ MedRAG textbooks (Chroma)
trước khi Reasoning Agent trả lời.

Theo đúng pattern của agents/reasoning_agent.py: 1 factory
`make_retrieval_agent_node(run_config)` nhận config qua closure, trả về 1
node function thuần `state -> state`. Được add vào graph.py khi
`run_config.retrieval.enabled = true`.

v2: agent tự đánh giá evidence đã "đủ" để trả lời chưa (bằng 1 lời gọi LLM
nhẹ), và nếu chưa đủ thì tự viết lại truy vấn để tìm thêm, lặp lại tối đa
`run_config.retrieval.max_iterations` lần. Khi max_iterations=1 (mặc định),
vòng self-check KHÔNG chạy - hành vi và chi phí giống hệt v1.

Chỉ ghi vào các field đã có sẵn/mới thêm trong AgentState
(retrieved_docs/query_history/retrieval_iterations/retrieval_sufficiency) -
không đổi schema, không ảnh hưởng các variant không dùng retrieval (v0).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm_client import build_llm
from retrieval.retriever import MedicalEvidenceRetriever
from run_config import RunConfig
from state import AgentState

LABELS = ["A", "B", "C", "D"]

# Cache theo (chroma_dir, collection) - tránh mở lại Chroma PersistentClient
# + load lại embedding config cho MỖI câu hỏi khi benchmark hàng nghìn câu.
_retriever_cache: dict[tuple[str, str], MedicalEvidenceRetriever] = {}

SUFFICIENCY_PROMPT = """You are grading whether the retrieved evidence below is
enough to answer a medical multiple-choice question. Be strict - only say
sufficient=true if the evidence actually lets someone pick the right choice.

Question:
{question}

Choices:
{choices}

Retrieved evidence so far:
{evidence}

Return JSON only, no other text:
{{
  "sufficient": true or false,
  "sufficiency_score": confidence from 0 to 1 that the evidence is enough,
  "next_query": "a reformulated, more specific search query to find the missing evidence (only used if sufficient=false)"
}}
"""


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


def _extract_json(text: str) -> dict:
    """LLM đôi khi bọc JSON trong ```json ... ``` -> bóc tách an toàn."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group(0) if match else text
    return json.loads(raw)


def _dedup_by_chunk_id(docs: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for doc in docs:
        key = doc.get("chunk_id", id(doc))
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _add_tokens(u1: Optional[dict], u2: Optional[dict]) -> Optional[dict]:
    if not u1: return u2
    if not u2: return u1
    return {
        "input_tokens": u1.get("input_tokens", 0) + u2.get("input_tokens", 0),
        "output_tokens": u1.get("output_tokens", 0) + u2.get("output_tokens", 0),
        "total_tokens": u1.get("total_tokens", 0) + u2.get("total_tokens", 0),
    }


def make_retrieval_agent_node(run_config: RunConfig):
    top_k = run_config.retrieval.top_k
    max_iterations = max(1, run_config.retrieval.max_iterations)
    sufficiency_threshold = run_config.retrieval.sufficiency_threshold
    # chroma_dir/collection có thể để mặc định (None -> DEFAULT_* trong
    # ingest_data.py) hoặc đọc từ run_config nếu bạn mở rộng RetrievalConfig
    # thêm 2 field này sau này.
    chroma_dir = getattr(run_config.retrieval, "chroma_dir", None)
    collection_name = getattr(run_config.retrieval, "collection_name", None)
    # None -> dùng chung model + provider (endpoint/API key) với Reasoning
    # Agent; set run_config.retrieval.check_model để dùng model/provider
    # riêng (đọc qua RETRIEVAL_CHECK_MODEL_BASE_URL/_API_KEY trong .env).
    check_model = run_config.retrieval.check_model or run_config.model
    prompt_template = run_config.retrieval.prompt_template or SUFFICIENCY_PROMPT

    def retrieval_agent_node(state: AgentState) -> AgentState:
        import time
        start_time = time.perf_counter()
        retrieval_tokens = None
        query_history: list[str] = []
        all_docs: list[dict] = []
        sufficiency_score = None

        if run_config.debug:
            print(f"\n--- [Retrieval Agent] Handling Question ID: {state.get('question_id')} ---")
            print(f"[Retrieval Agent] Input Question: {state['question']}")

        try:
            retriever = _get_retriever(chroma_dir, collection_name)
            query = state["question"]

            for iteration in range(1, max_iterations + 1):
                query_history.append(query)
                if run_config.debug:
                    print(f"[Retrieval Agent] Loop {iteration}: Searching with query: '{query}'")
                
                evidence = retriever.retrieve(query, top_k=top_k)
                all_docs = _dedup_by_chunk_id(all_docs + [asdict(chunk) for chunk in evidence])

                if iteration >= max_iterations:
                    break  # v1 (max_iterations=1) luôn break ở đây - không self-check

                choices_text = "\n".join(
                    f"{label}. {choice}" for label, choice in zip(LABELS, state["choices"])
                )
                evidence_text = "\n\n".join(
                    doc.get("text", "") for doc in all_docs if not doc.get("error")
                ) or "No evidence retrieved yet."

                llm = build_llm(model=check_model, temperature=0.0, env_prefix="RETRIEVAL_CHECK_MODEL")
                check_prompt = prompt_template.format(
                    question=state["question"],
                    choices=choices_text,
                    evidence=evidence_text,
                )
                response = llm.invoke(check_prompt)
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    retrieval_tokens = _add_tokens(retrieval_tokens, usage)

                check = _extract_json(response.content)
                sufficiency_score = check.get("sufficiency_score")
                is_sufficient = check.get("sufficient", True)

                if run_config.debug:
                    print(f"[Retrieval Agent] Loop {iteration}: Sufficiency Score = {sufficiency_score}, Sufficient = {is_sufficient}")

                enough = is_sufficient or (
                    sufficiency_score is not None and sufficiency_score >= sufficiency_threshold
                )
                if enough:
                    break

                next_query = check.get("next_query")
                if not next_query or next_query in query_history:
                    break  # tránh lặp vô hạn khi LLM trả lại cùng 1 query

                if run_config.debug:
                    print(f"[Retrieval Agent] Loop {iteration}: Reformulated query for next iteration: '{next_query}'")
                query = next_query

            # Gộp nhiều vòng xong, giữ top_k evidence tốt nhất (score = khoảng
            # cách/distance từ Chroma - thấp hơn = giống hơn).
            scored = [d for d in all_docs if not d.get("error") and d.get("score") is not None]
            unscored = [d for d in all_docs if d.get("error") or d.get("score") is None]
            scored.sort(key=lambda d: d["score"])
            retrieved_docs = (scored[:top_k] + unscored) if (scored or unscored) else all_docs

            if run_config.debug:
                print(f"[Retrieval Agent] Final selected docs count: {len(retrieved_docs)}")

        except Exception as exc:
            # 1 câu lỗi retrieval không được làm sập cả graph - Reasoning
            # Agent vẫn chạy tiếp, chỉ là không có evidence.
            retrieved_docs = [{"error": f"Retrieval failed: {exc}"}]
            if run_config.debug:
                print(f"[Retrieval Agent] Retrieval error occurred: {exc}")

        duration_ms = (time.perf_counter() - start_time) * 1000
        total_latency = (state.get("latency_ms") or 0.0) + duration_ms
        total_tokens = _add_tokens(state.get("token_usage"), retrieval_tokens)

        return {
            **state,
            "retrieved_docs": retrieved_docs,
            "query_history": query_history,
            "retrieval_iterations": len(query_history),
            "retrieval_sufficiency": sufficiency_score,
            "retrieval_latency_ms": duration_ms,
            "retrieval_token_usage": retrieval_tokens,
            "latency_ms": total_latency,
            "token_usage": total_tokens,
        }

    return retrieval_agent_node
