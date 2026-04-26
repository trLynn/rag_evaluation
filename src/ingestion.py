from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Sequence

@dataclass
class IngestionStats:
    files_processed: int
    chunks_added: int


@dataclass
class ChunkedFile:
    path: Path
    chunks: list[str]
    content_hash: str


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

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    return [chunk.strip() for chunk in splitter.split_text(cleaned) if chunk.strip()]


def build_chunk_ids(path: Path, count: int) -> list[str]:
    source = str(path.resolve())
    source_hash = hashlib.sha1(source.encode()).hexdigest()[:12]
    return [f"{path.stem}-{source_hash}-{i}" for i in range(count)]


def build_content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _chunk_file(path: Path) -> ChunkedFile:
    text = read_text_file(path)
    return ChunkedFile(
        path=path,
        chunks=chunk_text(text),
        content_hash=build_content_hash(text),
    )

def chunk_documents_in_background(
    file_paths: Sequence[str] | Iterable[str],
    max_workers: int = 4,
) -> list[ChunkedFile]:
    if max_workers <= 0:
        raise ValueError("max_workers must be > 0")

    existing_paths = [Path(file_path) for file_path in file_paths if Path(file_path).exists()]
    if not existing_paths:
        return []

    chunked_files: list[ChunkedFile] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_chunk_file, path): path for path in existing_paths}
        for future in as_completed(futures):
            chunked_files.append(future.result())

    # Keep deterministic order for predictable ingestion and logging.
    chunked_files.sort(key=lambda item: str(item.path))
    return chunked_files

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


def _is_timeout_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "timeout" in text or "timed out" in text or "readtimeout" in text


def _upsert_batch_with_backoff(
    collection: Any,
    batch_ids: list[str],
    batch_chunks: list[str],
    batch_metadatas: list[dict[str, Any]],
    min_batch_size: int = 1,
    max_timeout_retries: int = 3,
    base_retry_sleep_seconds: float = 1.0,
) -> None:
    if min_batch_size <= 0:
        raise ValueError("min_batch_size must be > 0")
    if max_timeout_retries < 0:
        raise ValueError("max_timeout_retries must be >= 0")

    last_timeout_exc: Exception | None = None
    for attempt in range(max_timeout_retries + 1):
        try:
            collection.upsert(
                ids=batch_ids,
                documents=batch_chunks,
                metadatas=batch_metadatas,
            )
            return
        except Exception as exc:
            if not _is_timeout_error(exc):
                raise
            last_timeout_exc = exc
            if attempt < max_timeout_retries:
                sleep_seconds = base_retry_sleep_seconds * (2**attempt)
                time.sleep(sleep_seconds)

    if len(batch_chunks) <= min_batch_size:
        raise RuntimeError(
            "Timed out while upserting even at minimum batch size "
            f"({len(batch_chunks)})."
        ) from last_timeout_exc

    midpoint = len(batch_chunks) // 2
    _upsert_batch_with_backoff(
        collection=collection,
        batch_ids=batch_ids[:midpoint],
        batch_chunks=batch_chunks[:midpoint],
        batch_metadatas=batch_metadatas[:midpoint],
        min_batch_size=min_batch_size,
        max_timeout_retries=max_timeout_retries,
        base_retry_sleep_seconds=base_retry_sleep_seconds,
    )
    _upsert_batch_with_backoff(
        collection=collection,
        batch_ids=batch_ids[midpoint:],
        batch_chunks=batch_chunks[midpoint:],
        batch_metadatas=batch_metadatas[midpoint:],
        min_batch_size=min_batch_size,
        max_timeout_retries=max_timeout_retries,
        base_retry_sleep_seconds=base_retry_sleep_seconds,
    )

def _create_chroma_client(persist_dir: str) -> Any:
    import chromadb
    from chromadb.config import Settings

    try:
        return chromadb.PersistentClient(path=persist_dir)
    except Exception as exc:
        error_text = str(exc).lower()
        if "default_tenant" in error_text or "rustbindingsapi" in error_text:
            return chromadb.Client(
                Settings(
                    is_persistent=True,
                    persist_directory=persist_dir,
                    chroma_api_impl="chromadb.api.segment.SegmentAPI",
                )
            )
        raise

def create_collection(
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
    ollama_base_url: str | None = None,
    ollama_timeout: int = 180,
)  -> Any:
    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

    client = _create_chroma_client(persist_dir=persist_dir)
    host = _resolve_ollama_host(ollama_base_url)

    embedding_fn = OllamaEmbeddingFunction(
        model_name=embedding_model,
        # Chroma expects a base Ollama host.
        url=host,
        timeout=ollama_timeout,
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
    batch_size: int = 100,
    ollama_timeout: int = 180,
    # ADD THESE TWO LINES BELOW:
    chunk_workers: int = 1, 
    max_timeout_retries: int = 3 
) -> IngestionStats:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    collection = create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        ollama_timeout=ollama_timeout,
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

    chunked_files = chunk_documents_in_background(file_paths, max_workers=chunk_workers)

    for chunked_file in chunked_files:
        path = chunked_file.path
        chunks = chunked_file.chunks
        content_hash = chunked_file.content_hash

        if not path.exists():
            continue

        print(f"\nProcessing: {path}")

        print(f"Chunks created: {len(chunks)}")

        if not chunks:
            continue

        try:
            existing = collection.get(
                where={"source": str(path)},
                include=["metadatas"],
            )
            existing_metadatas = existing.get("metadatas", []) if existing else []
            existing_chunks = len(existing.get("ids", [])) if existing else 0
            existing_hash = (
                existing_metadatas[0].get("content_hash")
                if existing_metadatas and existing_metadatas[0]
                else None
            )
            if existing_hash == content_hash and existing_chunks == len(chunks):
                print("⏭️ Skipping unchanged file (already indexed)")
                continue
        except Exception:
            # If metadata lookup fails, continue with normal upsert behavior.
            pass

        ids = build_chunk_ids(path, len(chunks))
        metadatas = [
            {"source": str(path), "chunk": i, "content_hash": content_hash}
            for i in range(len(chunks))
        ]

        try:
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i : i + batch_size]
                batch_ids = ids[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]

                collection.upsert(
                    ids=batch_ids,
                    documents=batch_chunks,
                    metadatas=batch_metadatas,
                )
        except Exception as exc:
            # Common recovery path: existing collection was created with a
            # different embedding dimension/model.
            if _is_likely_dimension_mismatch_error(exc):
                client = _create_chroma_client(persist_dir=persist_dir)
                client.delete_collection(name=collection_name)
                collection = create_collection(
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    embedding_model=embedding_model,
                    ollama_base_url=ollama_base_url,
                    ollama_timeout=ollama_timeout,
                )
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i : i + batch_size]
                    batch_ids = ids[i : i + batch_size]
                    batch_metadatas = metadatas[i : i + batch_size]
                    collection.upsert(
                        ids=batch_ids,
                        documents=batch_chunks,
                        metadatas=batch_metadatas,
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
