from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Any


# ------------------ Result ------------------
@dataclass
class IngestionStats:
    files_processed: int
    chunks_added: int


# ------------------ Read File ------------------
def read_text_file(path: Path) -> str:
    raw_bytes = path.read_bytes()
    return raw_bytes.decode("utf-8", errors="ignore")


# ------------------ Chunking ------------------
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(cleaned), step):
        end = start + chunk_size
        chunk = cleaned[start:end]

        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break

    return chunks


# ------------------ IDs ------------------
def build_chunk_ids(path: Path, count: int) -> list[str]:
    source = str(path.resolve())
    source_hash = hashlib.sha1(source.encode()).hexdigest()[:12]
    return [f"{path.stem}-{source_hash}-{i}" for i in range(count)]


# ------------------ Chroma ------------------
def create_collection(
    persist_dir="vector_db",
    collection_name="knowledge_base",
    embedding_model="nomic-embed-text",
) -> Any:
    import chromadb
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

    client = chromadb.PersistentClient(path=persist_dir)

    embedding_fn = OllamaEmbeddingFunction(
        model_name=embedding_model,
        url="http://localhost:11434/api/embeddings",
    )

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )


# ------------------ MAIN INGESTION ------------------
def ingest_documents(
    file_paths: Sequence[str] | Iterable[str],
    persist_dir="vector_db",
    collection_name="knowledge_base",
    embedding_model="nomic-embed-text",
) -> IngestionStats:

    collection = create_collection(persist_dir, collection_name, embedding_model)

    total_chunks = 0
    files_processed = 0

    for file_path in file_paths:
        path = Path(file_path)

        if not path.exists():
            continue

        print(f"\n📄 Processing: {path}")

        text = read_text_file(path)

        # 🔥 CHUNKING HAPPENS HERE
        chunks = chunk_text(text)

        print(f"👉 Chunks created: {len(chunks)}")

        if not chunks:
            continue

        ids = build_chunk_ids(path, len(chunks))
        metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metadatas
        )

        files_processed += 1
        total_chunks += len(chunks)

    return IngestionStats(files_processed, total_chunks)