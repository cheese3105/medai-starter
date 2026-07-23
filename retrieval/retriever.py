"""Reusable evidence retrieval over the persisted MedRAG Chroma index.

Port từ repo nghiên cứu RAG của thành viên trong nhóm
(Le-Kim-Hung-Security-for-MedQA-USMLE/src/retrieval/retriever.py).

Khác biệt so với bản gốc: bản gốc import `EvidenceChunk` từ
`src.core.schema` (schema riêng của repo đó). medai-starter chưa có schema
tương đương nên `EvidenceChunk` được định nghĩa lại ngay tại đây - giữ
đúng các field (text/title/source/chunk_id/score) để logic retrieve()
không đổi.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from retrieval.ingest_data import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    embed_batch,
    load_embedding_config,
)


@dataclass
class EvidenceChunk:
    text: str
    title: str | None = None
    source: str | None = None
    chunk_id: str | None = None
    score: float | None = None


class MedicalEvidenceRetriever:
    """Retrieve medical evidence chunks from the Chroma index."""

    def __init__(
        self,
        *,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.embedding_config = load_embedding_config()
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_collection(name=collection_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[EvidenceChunk]:
        query = query.strip()
        if not query:
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        collection_count = self.collection.count()
        if collection_count == 0:
            return []

        embeddings, _usage = embed_batch([query], config=self.embedding_config)
        result = self.collection.query(
            query_embeddings=[embeddings[0]],
            n_results=min(top_k, collection_count),
            include=["documents", "metadatas", "distances"],
        )
        return _to_evidence_chunks(result)


def _to_evidence_chunks(result: dict[str, Any]) -> list[EvidenceChunk]:
    ids = _first_query_values(result, "ids")
    documents = _first_query_values(result, "documents")
    metadatas = _first_query_values(result, "metadatas")
    distances = _first_query_values(result, "distances")

    chunks: list[EvidenceChunk] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] or {}
        chunks.append(
            EvidenceChunk(
                text=documents[index],
                title=_optional_metadata_text(metadata.get("title")),
                source=_optional_metadata_text(metadata.get("source")),
                chunk_id=str(chunk_id),
                score=float(distances[index]),
            )
        )
    return chunks


def _first_query_values(result: dict[str, Any], key: str) -> list[Any]:
    values = result.get(key) or [[]]
    return values[0]


def _optional_metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
