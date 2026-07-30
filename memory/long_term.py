"""Long-term memory — Chroma-backed persistent fact store."""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from core.logger import DebugLogger

EXTRACTION_PROMPT = """You are deciding whether a chat turn contains a durable
fact worth remembering about the user for FUTURE conversations (e.g.
allergies, chronic conditions, medications, past diagnoses, preferences).
Do NOT save small talk or one-off symptom questions.

User said: {user_message}
Assistant replied: {assistant_message}

Return JSON only:
{{
  "worth_remembering": true or false,
  "category": "allergy|chronic_condition|medication|preference|other",
  "fact": "short self-contained sentence (empty if not worth remembering)"
}}
"""


class LongTermMemory:
    def __init__(self, user_id: str = "default_user", chroma_dir: Optional[str] = None,
                 collection_name: str = "user_memory", embedding_model: str = "",
                 embedding_base_url: str = "", embedding_api_key: str = "dummy"):
        self.user_id = user_id
        self.client = chromadb.PersistentClient(path=chroma_dir or "data/chroma/user_memory")

        ef = None
        if embedding_model and embedding_base_url:
            ef = OpenAIEmbeddingFunction(
                model_name=embedding_model, api_key=embedding_api_key, api_base=embedding_base_url)

        try:
            self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=ef)
        except ValueError as e:
            if "embedding function conflict" in str(e).lower():
                self.collection = self.client.get_collection(name=collection_name)
            else:
                raise

    def write(self, fact: str, category: str = "other") -> None:
        fact = fact.strip()
        if not fact:
            return
        self.collection.upsert(
            ids=[str(uuid.uuid4())], documents=[fact],
            metadatas=[{"user_id": self.user_id, "category": category, "created_at": str(time.time())}])

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            return []
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_texts=[query], n_results=min(top_k, count),
            where={"user_id": self.user_id}, include=["documents", "metadatas", "distances"])
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        return [{"text": doc, "metadata": meta or {}, "distance": float(dist)}
                for doc, meta, dist in zip(documents, metadatas, distances)]


def maybe_extract_and_save(ltm: LongTermMemory, user_message: str, assistant_message: str,
                           model: str, base_url: str, api_key: str, logger: DebugLogger,
                           turn_id: str = "") -> None:
    """1 LLM call quyết định có đáng nhớ không, ghi nếu có."""
    try:
        from core.llm_client import build_llm
        llm = build_llm(model=model, base_url=base_url, api_key=api_key, temperature=0.0)
        prompt = EXTRACTION_PROMPT.format(user_message=user_message, assistant_message=assistant_message)
        response = llm.invoke(prompt)
        match = re.search(r"\{.*\}", response.content.strip(), re.DOTALL)
        if not match:
            return
        parsed = json.loads(match.group(0))
        if parsed.get("worth_remembering") and parsed.get("fact"):
            fact = str(parsed["fact"]).strip()
            category = str(parsed.get("category", "other"))
            if fact:
                ltm.write(fact, category=category)
                logger.memory_info(turn_id, f"Đã lưu LTM [{category}]: {fact}")
    except Exception:
        pass
