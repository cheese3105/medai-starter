"""Long-term Memory - QA Cache using ChromaDB (separate collection from RAG DB).

Stores high-quality answers (confidence >= threshold AND verifier supported)
for reuse on similar future questions.

Uses same embedding model (bge-m3) and same PersistentClient path as RAG,
but different collection ("qa_cache" vs "medrag_textbooks") -> data fully isolated.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import chromadb

from retrieval.ingest_data import embed_batch, load_embedding_config, EmbeddingConfig


class LongTermMemory:
    """QA Cache backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str = "qa_cache",
        chroma_dir: str = "data/chroma",
        min_confidence: float = 0.8,
        min_verifier_verdict: str = "supported",
        similarity_threshold: float = 0.85,
        max_cache_size: int = 1000,
        cache_eviction_policy: str = "lru",
        debug: bool = False,
    ):
        self.collection_name = collection_name
        self.chroma_dir = Path(chroma_dir)
        self.min_confidence = min_confidence
        self.min_verifier_verdict = min_verifier_verdict
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.cache_eviction_policy = cache_eviction_policy
        self.debug = debug

        self.embedding_config = load_embedding_config()

        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
        )

        if self.debug:
            print(f"[LTM] Initialized: collection='{self.collection_name}', "
                  f"size={self.collection.count()}, chroma_dir='{self.chroma_dir}'")

    def search_similar_qa(self, question: str) -> Optional[Dict[str, Any]]:
        """Search for similar question in cache.

        Returns:
            Dict with cached answer data if cache hit, None if miss.
        """
        if self.collection.count() == 0:
            if self.debug:
                print("[LTM] Cache empty - skip search")
            return None

        embeddings, _ = embed_batch([question], config=self.embedding_config)
        query_embedding = embeddings[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            if self.debug:
                print("[LTM] No results from query")
            return None

        distance = results["distances"][0][0]
        similarity = 1.0 / (1.0 + distance)

        if self.debug:
            print(f"[LTM] Best match: distance={distance:.4f}, similarity={similarity:.4f}, "
                  f"threshold={self.similarity_threshold}")

        if similarity < self.similarity_threshold:
            if self.debug:
                print(f"[LTM] Cache MISS (similarity {similarity:.4f} < {self.similarity_threshold})")
            return None

        metadata = results["metadatas"][0][0]
        cached_question = results["documents"][0][0]
        doc_id = results["ids"][0][0]

        if self.debug:
            print(f"[LTM] Cache HIT! similarity={similarity:.4f}")

        self._update_last_used(doc_id, metadata)

        return {
            "question": cached_question,
            "answer": metadata.get("answer"),
            "explanation": metadata.get("explanation"),
            "confidence": metadata.get("confidence"),
            "verifier_verdict": metadata.get("verifier_verdict"),
            "similarity": similarity,
            "cached_at": metadata.get("cached_at"),
            "times_used": metadata.get("times_used", 0) + 1,
            "cache_id": doc_id,
        }

    def add_qa(
        self,
        question: str,
        answer: str,
        explanation: Optional[str] = None,
        confidence: Optional[float] = None,
        verifier_verdict: Optional[str] = None,
    ) -> bool:
        """Add Q&A to cache if it meets quality threshold.

        Returns True if cached, False if below threshold.
        """
        if not self._meets_quality_threshold(confidence, verifier_verdict):
            if self.debug:
                print(f"[LTM] Skip caching: confidence={confidence}, "
                      f"verdict={verifier_verdict} (below threshold)")
            return False

        existing = self.search_similar_qa(question)
        if existing and existing["similarity"] > 0.95:
            if self.debug:
                print(f"[LTM] Skip caching: near-duplicate exists (similarity={existing['similarity']:.4f})")
            return False

        if self.collection.count() >= self.max_cache_size:
            self._evict()

        embeddings, _ = embed_batch([question], config=self.embedding_config)

        doc_id = f"qa_{int(time.time())}_{hash(question) % 100000}"

        metadata = {
            "answer": answer or "",
            "explanation": (explanation or "")[:500],
            "confidence": confidence if confidence is not None else 0.0,
            "verifier_verdict": verifier_verdict or "",
            "cached_at": time.time(),
            "last_used": time.time(),
            "times_used": 0,
        }

        self.collection.add(
            ids=[doc_id],
            documents=[question],
            embeddings=[embeddings[0]],
            metadatas=[metadata],
        )

        if self.debug:
            print(f"[LTM] Cached: '{question[:60]}...' (confidence={confidence}, "
                  f"verdict={verifier_verdict}, total={self.collection.count()})")

        return True

    def _meets_quality_threshold(
        self, confidence: Optional[float], verifier_verdict: Optional[str]
    ) -> bool:
        if confidence is None or confidence < self.min_confidence:
            return False
        if self.min_verifier_verdict:
            if verifier_verdict != self.min_verifier_verdict:
                return False
        return True

    def _update_last_used(self, doc_id: str, metadata: Dict) -> None:
        updated_metadata = {**metadata}
        updated_metadata["last_used"] = time.time()
        updated_metadata["times_used"] = metadata.get("times_used", 0) + 1
        self.collection.update(ids=[doc_id], metadatas=[updated_metadata])

    def _evict(self) -> None:
        if self.collection.count() == 0:
            return

        all_entries = self.collection.get(include=["metadatas"], limit=self.collection.count())
        if not all_entries["ids"]:
            return

        oldest_idx = 0
        oldest_time = float("inf")
        time_key = "last_used" if self.cache_eviction_policy == "lru" else "cached_at"

        for i, meta in enumerate(all_entries["metadatas"]):
            t = meta.get(time_key, 0)
            if t < oldest_time:
                oldest_time = t
                oldest_idx = i

        evict_id = all_entries["ids"][oldest_idx]
        self.collection.delete(ids=[evict_id])

        if self.debug:
            print(f"[LTM] Evicted: {evict_id} (policy={self.cache_eviction_policy})")

    def get_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        if count == 0:
            return {"total_cached": 0, "collection_name": self.collection_name}

        all_entries = self.collection.get(include=["metadatas"], limit=count)
        confidences = []
        total_times_used = 0

        for meta in all_entries["metadatas"]:
            conf = meta.get("confidence")
            if conf is not None:
                confidences.append(conf)
            total_times_used += meta.get("times_used", 0)

        return {
            "total_cached": count,
            "collection_name": self.collection_name,
            "max_cache_size": self.max_cache_size,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
            "total_cache_hits": total_times_used,
            "similarity_threshold": self.similarity_threshold,
        }

    def clear(self) -> int:
        count = self.collection.count()
        if count > 0:
            all_ids = self.collection.get(limit=count)["ids"]
            self.collection.delete(ids=all_ids)
            if self.debug:
                print(f"[LTM] Cleared {count} entries")
        return count
