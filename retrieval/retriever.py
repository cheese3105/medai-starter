"""Medical evidence retriever — wrapper quanh Chroma.

Collection RAG được ingest bằng HTTP /embeddings endpoint trực tiếp
(không qua Chroma embedding_function), nên query cũng phải tự embed
cùng cách rồi truyền query_embeddings thay vì query_texts.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import chromadb

MAX_RETRIES = 3


def _embed_query(text: str, model: str, api_base: str, api_key: str) -> list[float]:
    """Gọi HTTP /embeddings endpoint, mirror cách ingest_data.py đã embed."""
    payload = json.dumps({
        "model": model,
        "input": [text],
        "encoding_format": "float",
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = f"{api_base.rstrip('/')}/embeddings"

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            embeddings = [item["embedding"] for item in body.get("data", [])]
            if not embeddings:
                raise RuntimeError("Embedding response rỗng")
            return embeddings[0]
        except urllib.error.HTTPError as exc:
            last_error = RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc

    raise RuntimeError(f"Embed thất bại sau {MAX_RETRIES} lần: {last_error}") from last_error


class MedicalRetriever:
    """Query Chroma collection đã tồn tại sẵn."""

    def __init__(
        self,
        chroma_dir: str = "data/chroma",
        collection_name: str = "medrag_textbooks",
        embedding_model: str = "",
        embedding_base_url: str = "",
        embedding_api_key: str = "dummy",
    ):
        self.embedding_model = embedding_model
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key = embedding_api_key

        if not (embedding_model and embedding_base_url and embedding_api_key):
            raise ValueError(
                "Thiếu EMBEDDING_MODEL / EMBEDDING_MODEL_API_BASE / EMBEDDING_MODEL_API_KEY "
                "trong .env — cần đủ 3 biến để embed câu hỏi khớp dimension với index."
            )

        self.client = chromadb.PersistentClient(path=chroma_dir)
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            raise ValueError(
                f"Không tìm thấy collection '{collection_name}' tại '{chroma_dir}'. "
                f"Lỗi gốc: {e}"
            ) from e

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query_text.strip():
            return []

        count = self.collection.count()
        if count == 0:
            return []

        query_vector = _embed_query(
            query_text,
            model=self.embedding_model,
            api_base=self.embedding_base_url,
            api_key=self.embedding_api_key,
        )

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]

        return [
            {"id": doc_id, "text": doc, "metadata": meta or {}, "distance": float(dist)}
            for doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances)
        ]
