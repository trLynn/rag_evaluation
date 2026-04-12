"""Utilities for ingesting local text files into a Chroma vector database."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from typing import Any


@dataclass
class IngestionStats:
    """Simple summary returned by the ingestion pipeline."""

    files_processed: int
    chunks_added: int


def read_text_file(path: Path) -> str:
    """Read text content from a file path."""
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks to improve semantic retrieval."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(cleaned), step):
        end = start + chunk_size
        segment = cleaned[start:end]
        if segment:
            chunks.append(segment)
        if end >= len(cleaned):
            break
    return chunks


def create_collection(
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
) -> Any:
    """Create/get a persistent Chroma collection configured for Ollama embeddings."""
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


def ingest_documents(
    file_paths: Sequence[str] | Iterable[str],
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> IngestionStats:
    """Ingest a list of text files into the vector DB."""
    collection = create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    total_chunks = 0
    files_processed = 0

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        text = read_text_file(path)
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if not chunks:
            continue

        ids = [f"{path.stem}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]

        collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)

        files_processed += 1
        total_chunks += len(chunks)

    return IngestionStats(files_processed=files_processed, chunks_added=total_chunks)
