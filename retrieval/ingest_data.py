"""Ingest the MedRAG textbooks corpus into a persistent Chroma database.

Port từ repo nghiên cứu RAG của thành viên trong nhóm
(Le-Kim-Hung-Security-for-MedQA-USMLE/src/retrieval/ingest_data.py).
Không có thay đổi về logic - chỉ bỏ tiền tố `src.` để khớp cấu trúc
medai-starter. File này không phụ thuộc LlamaIndex hay bất kỳ SDK LLM nào,
chỉ dùng `chromadb` + `urllib` thuần, nên tương thích thẳng với stack
LangChain/ChatOpenAI hiện tại của medai-starter.

Chạy 1 lần (offline, có thể mất vài giờ) để build index:
    python -m retrieval.ingest_data

Nếu bạn đã có sẵn thư mục data/chroma/... (ví dụ lấy từ file data.zip mà
thành viên nghiên cứu RAG đã build sẵn), có thể bỏ qua bước này hoàn toàn.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import chromadb
from dotenv import load_dotenv


DEFAULT_INPUT_PATH = Path("data/corpus/medrag_textbooks/textbooks.jsonl")
DEFAULT_CHROMA_DIR = Path("data/chroma/medrag_textbooks_bge_m3_openrouter")
DEFAULT_COLLECTION_NAME = "medrag_textbooks"
DEFAULT_BATCH_SIZE = 64
MAX_RETRIES = 3


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    api_base: str
    api_key: str


@dataclass(frozen=True)
class CorpusRecord:
    id: str
    title: str
    text: str
    row_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed MedRAG textbook chunks and persist them in Chroma."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSONL corpus. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help=f"Persistent Chroma directory. Defaults to {DEFAULT_CHROMA_DIR}.",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name. Defaults to {DEFAULT_COLLECTION_NAME}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding/upsert batch size. Defaults to {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests.",
    )
    return parser.parse_args()


def load_embedding_config() -> EmbeddingConfig:
    load_dotenv()

    model = os.getenv("EMBEDDING_MODEL", "").strip()
    api_base = os.getenv("EMBEDDING_MODEL_API_BASE", "").strip().rstrip("/")
    api_key = os.getenv("EMBEDDING_MODEL_API_KEY", "").strip()

    missing = [
        name
        for name, value in {
            "EMBEDDING_MODEL": model,
            "EMBEDDING_MODEL_API_BASE": api_base,
            "EMBEDDING_MODEL_API_KEY": api_key,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required environment variable(s): {', '.join(missing)}")

    return EmbeddingConfig(model=model, api_base=api_base, api_key=api_key)


def iter_corpus_records(path: Path, *, limit: int | None = None) -> Iterable[CorpusRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    count = 0
    with path.open("r", encoding="utf-8") as file:
        for row_index, line in enumerate(file):
            if limit is not None and count >= limit:
                break

            stripped = line.strip()
            if not stripped:
                continue

            payload = json.loads(stripped)
            record_id = clean_text(payload.get("id")) or f"row_{row_index}"
            title = clean_text(payload.get("title"))
            text = clean_text(payload.get("contents")) or clean_text(payload.get("content"))
            if not text:
                continue

            count += 1
            yield CorpusRecord(
                id=record_id,
                title=title,
                text=text,
                row_index=row_index,
            )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def embed_batch(
    texts: list[str],
    *,
    config: EmbeddingConfig,
) -> tuple[list[list[float]], dict[str, Any]]:
    payload = {
        "model": config.model,
        "input": texts,
        "encoding_format": "float",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        request = urllib.request.Request(
            f"{config.api_base}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            embeddings = [item["embedding"] for item in response_payload.get("data", [])]
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                )
            return embeddings, response_payload.get("usage") or {}
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Embedding request failed after {attempt} attempts: "
                    f"HTTP {exc.code}: {body}"
                ) from exc
            wait_before_retry(attempt)
        except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Embedding request failed after {attempt} attempts: {exc}"
                ) from exc
            wait_before_retry(attempt)

    raise RuntimeError("Embedding request failed unexpectedly")


def wait_before_retry(attempt: int) -> None:
    time.sleep(2**attempt)


def upsert_batch(
    collection: Any,
    records: list[CorpusRecord],
    embeddings: list[list[float]],
) -> None:
    collection.upsert(
        ids=[record.id for record in records],
        documents=[record.text for record in records],
        embeddings=embeddings,
        metadatas=[
            {
                "title": record.title,
                "source": "MedRAG/textbooks",
                "row_index": record.row_index,
            }
            for record in records
        ],
    )


def write_index_metadata(
    *,
    chroma_dir: Path,
    collection_name: str,
    input_path: Path,
    embedding_model: str,
    embedding_dimension: int | None,
    ingested_count: int,
    collection_count: int,
    usage_totals: dict[str, float],
) -> None:
    metadata = {
        "collection": collection_name,
        "source_dataset": "MedRAG/textbooks",
        "source_file": str(input_path),
        "embedding_provider": "openrouter",
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "ingested_count_this_run": ingested_count,
        "collection_count": collection_count,
        "usage_totals": usage_totals,
        "updated_at_unix": int(time.time()),
    }
    chroma_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = chroma_dir / "index_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_usage(usage_totals: dict[str, float], usage: dict[str, Any]) -> None:
    for key in ("prompt_tokens", "total_tokens", "cost"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            usage_totals[key] = usage_totals.get(key, 0.0) + float(value)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    embedding_config = load_embedding_config()
    args.chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(args.chroma_dir))
    collection = client.get_or_create_collection(name=args.collection)

    total = 0
    embedding_dimension: int | None = None
    usage_totals: dict[str, float] = {}
    batch: list[CorpusRecord] = []

    for record in iter_corpus_records(args.input, limit=args.limit):
        batch.append(record)
        if len(batch) >= args.batch_size:
            embeddings, usage = embed_batch(
                [item.text for item in batch],
                config=embedding_config,
            )
            embedding_dimension = embedding_dimension or len(embeddings[0])
            upsert_batch(collection, batch, embeddings)
            add_usage(usage_totals, usage)
            total += len(batch)
            print(f"ingested={total} collection_count={collection.count()}")
            batch.clear()

    if batch:
        embeddings, usage = embed_batch(
            [item.text for item in batch],
            config=embedding_config,
        )
        embedding_dimension = embedding_dimension or len(embeddings[0])
        upsert_batch(collection, batch, embeddings)
        add_usage(usage_totals, usage)
        total += len(batch)
        print(f"ingested={total} collection_count={collection.count()}")

    write_index_metadata(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        input_path=args.input,
        embedding_model=embedding_config.model,
        embedding_dimension=embedding_dimension,
        ingested_count=total,
        collection_count=collection.count(),
        usage_totals=usage_totals,
    )

    print("Ingestion complete")
    print(f"chroma_dir={args.chroma_dir}")
    print(f"collection={args.collection}")
    print(f"ingested_count_this_run={total}")
    print(f"collection_count={collection.count()}")
    if embedding_dimension is not None:
        print(f"embedding_dimension={embedding_dimension}")
    if usage_totals:
        print(f"usage_totals={json.dumps(usage_totals, sort_keys=True)}")


if __name__ == "__main__":
    main()
