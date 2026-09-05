"""Retrieval stage — query Chroma vector store lấy evidence."""

from __future__ import annotations

import time
from typing import Any

from core.config import RunConfig, StageConfig
from core.types import StageOutput
from stages.base import BaseStage


class RetrievalStage(BaseStage):

    def __init__(self, run_config: RunConfig, stage_config: StageConfig):
        super().__init__(run_config, stage_config)
        self.top_k = stage_config.top_k
        chroma_dir = stage_config.extra.get("chroma_dir", "data/chroma")
        collection_name = stage_config.extra.get("collection_name", "medrag_textbooks")

        from retrieval.retriever import MedicalRetriever
        self.retriever = MedicalRetriever(
            embedding_model=run_config.embedding_model,
            embedding_base_url=run_config.embedding_base_url,
            embedding_api_key=run_config.embedding_api_key,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
        )

    def run(self, context: dict[str, Any]) -> StageOutput:
        start = time.time()
        # Ưu tiên retrieval_query (do QR sinh ra) nếu có; fallback về question gốc.
        # Backward compatible: v0/v1/v2 không set retrieval_query → dùng question như cũ.
        query = context.get("retrieval_query") or context.get("question", "")
        results = self.retriever.query(query, top_k=self.top_k)
        latency_ms = (time.time() - start) * 1000

        evidence_texts, evidence_ids = [], []
        for i, doc in enumerate(results, 1):
            evidence_texts.append(f"[{i}] {doc.get('text', '')}")
            evidence_ids.append(doc.get("id", f"doc_{i}"))

        evidence_block = "\n\n".join(evidence_texts) if evidence_texts else ""

        return StageOutput(
            stage_name="retrieval",
            data={
                "evidence": evidence_block,
                "evidence_ids": evidence_ids,
                "num_results": len(results),
                "query": query,
                "prompt": query,
                "raw_response": evidence_block,
            },
            latency_ms=latency_ms,
            token_usage={},
        )
