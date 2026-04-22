from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class IngestionStats:
    files_processed: int
    chunks_added: int


def read_text_file(path: Path) -> str:
    raw_bytes = path.read_bytes()
    return raw_bytes.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(cleaned), step):
        end = start + chunk_size
        chunk = cleaned[start:end]

        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned):
            break

    return chunks


def build_chunk_ids(path: Path, count: int) -> list[str]:
    source = str(path.resolve())
    source_hash = hashlib.sha1(source.encode()).hexdigest()[:12]
    return [f"{path.stem}-{source_hash}-{i}" for i in range(count)]


def _resolve_ollama_host(ollama_base_url: str | None) -> str:
    return ollama_base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _ollama_troubleshooting_hint(embedding_model: str) -> str:
    return (
        "If `ollama serve` reports 'address already in use' on port 11434, "
        "Ollama is likely already running, so you should not start another server. "
        f"Instead verify the model exists with: `ollama pull {embedding_model}`."
    )


def _is_likely_dimension_mismatch_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "dimension" in text or "embedding size" in text or "incompatible" in text


def create_collection(
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
    ollama_base_url: str | None = None,
) -> Any:
    import chromadb
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

    client = chromadb.PersistentClient(path=persist_dir)
    host = _resolve_ollama_host(ollama_base_url)

    embedding_fn = OllamaEmbeddingFunction(
        model_name=embedding_model,
        # Chroma expects a base Ollama host.
        url=host,
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
    ollama_base_url: str | None = None,
) -> IngestionStats:
    collection = create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
    )
    host = _resolve_ollama_host(ollama_base_url)

    # Fail fast with a clear setup error before processing files/chunks.
    try:
        import ollama

        ollama.Client(host=host).embed(model=embedding_model, input=["health-check"])
    except Exception as exc:
        raise RuntimeError(
            "Ollama embedding preflight failed. "
            f"Host: {host}. Model: {embedding_model}. "
            "Ensure Ollama is reachable and the model exists "
            f"(`ollama pull {embedding_model}`). "
            f"{_ollama_troubleshooting_hint(embedding_model)}"
        ) from exc

    total_chunks = 0
    files_processed = 0

    for file_path in file_paths:
        path = Path(file_path)

        if not path.exists():
            continue

        print(f"\nProcessing: {path}")

        text = read_text_file(path)
        chunks = chunk_text(text)

        print(f"Chunks created: {len(chunks)}")

        if not chunks:
            continue

        ids = build_chunk_ids(path, len(chunks))
        metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]

        try:
            collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )
        except Exception as exc:
            # Common recovery path: existing collection was created with a
            # different embedding dimension/model.
            if _is_likely_dimension_mismatch_error(exc):
                import chromadb

                client = chromadb.PersistentClient(path=persist_dir)
                client.delete_collection(name=collection_name)
                collection = create_collection(
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    ollama_base_url=ollama_base_url,
                )
                collection.upsert(
                    ids=ids,
                    documents=chunks,
                    metadatas=metadatas,
                )
                continue

            raise RuntimeError(
                "Failed to upsert document chunks into Chroma. "
                f"Host: {host}. Model: {embedding_model}. "
                "Verify Ollama is reachable and the embedding model is available "
                f"(`ollama pull {embedding_model}`). "
                f"{_ollama_troubleshooting_hint(embedding_model)} "
                f"Original error: {exc!s}"
            ) from exc

        files_processed += 1
        total_chunks += len(chunks)

    return IngestionStats(files_processed, total_chunks)
