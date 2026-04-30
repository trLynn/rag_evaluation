from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable, Sequence

# A data class to keep track of the final results of the ingestion process
@dataclass
class IngestionStats:
    files_processed: int # Total number of files successfully processed
    chunks_added: int    # Total number of text chunks added to the database


# A data class to temporarily store information about a file after it's been chunked
@dataclass
class ChunkedFile:
    path: Path          # The file's location
    chunks: list[str]   # The list of text segments (chunks)
    content_hash: str   # A unique fingerprint (hash) of the content to check if it has changed


# Function to extract raw text from PDF or text files
def read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError(
                "PDF support requires `pypdf`. Install with: `pip install pypdf`."
            ) from exc

        # If it's a PDF, extract text page by page and join it together
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages).strip()

    # If it's a standard text file, just read and decode it
    raw_bytes = path.read_bytes()
    return raw_bytes.decode("utf-8", errors="ignore")


# Function to split long text into smaller, overlapping segments (chunks)
def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap  >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []

    # Uses Langchain's text splitter to intelligently cut text at paragraphs/sentences 
    # so we don't break the meaning of the sentences.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    return [chunk.strip() for chunk in splitter.split_text(cleaned) if chunk.strip()]


# Function to generate a unique ID for every single chunk
def build_chunk_ids(path: Path, count: int) -> list[str]:
    source = str(path.resolve())
    source_hash = hashlib.sha1(source.encode()).hexdigest()[:12]
    return [f"{path.stem}-{source_hash}-{i}" for i in range(count)]


# Function to create a unique fingerprint (Hash) of the text to track changes
def build_content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# Internal helper function to read and chunk a single file
def _chunk_file(path: Path) -> ChunkedFile:
    text = read_text_file(path)
    return ChunkedFile(
        path=path,
        chunks=chunk_text(text),
        content_hash=build_content_hash(text),
    )

# Function to process multiple files at the same time (Concurrency) to save time
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
    # ThreadPoolExecutor allows the program to read and chunk multiple files simultaneously
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_chunk_file, path): path for path in existing_paths}
        for future in as_completed(futures):
            chunked_files.append(future.result())

    # Sort the files back into their original order so logs remain predictable
    chunked_files.sort(key=lambda item: str(item.path))
    return chunked_files

# Determines the URL where the Ollama server is running
def _resolve_ollama_host(ollama_base_url: str | None) -> str:
    return ollama_base_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")


# Provides a helpful error message if the Ollama server fails to respond
def _ollama_troubleshooting_hint(embedding_model: str) -> str:
    return (
        "If `ollama serve` reports 'address already in use' on port 11434, "
        "Ollama is likely already running, so you should not start another server. "
        f"Instead verify the model exists with: `ollama pull {embedding_model}`."
    )


# Checks if an error is caused because the embedding model was changed (Dimension mismatch)
def _is_likely_dimension_mismatch_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "dimension" in text or "embedding size" in text or "incompatible" in text


# Checks if an error is just a server timeout
def _is_timeout_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "timeout" in text or "timed out" in text or "readtimeout" in text


# A robust function to insert data into the database. If it fails due to timeout, 
# it waits a bit and tries again (Exponential Backoff strategy).
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
    # Loop to retry insertion if it fails
    for attempt in range(max_timeout_retries + 1):
        try:
            collection.upsert(
                ids=batch_ids,
                documents=batch_chunks,
                metadatas=batch_metadatas,
            )
            return # Success! Exit the function.
        except Exception as exc:
            if not _is_timeout_error(exc):
                raise
            last_timeout_exc = exc
            # If timeout, wait a calculated amount of time before retrying
            if attempt < max_timeout_retries:
                sleep_seconds = base_retry_sleep_seconds * (2**attempt)
                time.sleep(sleep_seconds)

    if len(batch_chunks) <= min_batch_size:
        raise RuntimeError(
            "Timed out while upserting even at minimum batch size "
            f"({len(batch_chunks)})."
        ) from last_timeout_exc

    # If all retries fail, split the batch in half and try again (Recursion)
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

# Connects to the local Vector Database (ChromaDB)
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

# Retrieves an existing database collection (table) or creates a new one
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

    # Defines the embedding function (how text is converted to numbers) using Ollama
    embedding_fn = OllamaEmbeddingFunction(
        model_name=embedding_model,
        url=host,
        timeout=ollama_timeout,
    )

    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )


# **********************************************
# Main Function: The orchestrator that puts files into the database
# **********************************************
def ingest_documents(
    file_paths: Sequence[str] | Iterable[str],
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
    ollama_base_url: str | None = None,
    batch_size: int = 100,
    ollama_timeout: int = 180,
    chunk_workers: int = 1, 
    max_timeout_retries: int = 3 
) -> IngestionStats:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    # 1. Create or load the vector database collection
    collection = create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        ollama_timeout=ollama_timeout,
    )
    host = _resolve_ollama_host(ollama_base_url)

    # 2. Test if the Ollama server and the model are working before starting heavy work (Fail Fast)
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

    # 3. Read and split all the provided files into chunks in the background
    chunked_files = chunk_documents_in_background(file_paths, max_workers=chunk_workers)

    # 4. Loop through each file that has been chunked
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

        # 5. Check if we already processed this exact file before (Smart Caching).
        # If the file hasn't changed (same hash), skip it to save a lot of time.
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
            pass

        # 6. Prepare the IDs and Metadatas (tags/labels) for the database
        ids = build_chunk_ids(path, len(chunks))
        metadatas = [
            {"source": str(path), "chunk": i, "content_hash": content_hash}
            for i in range(len(chunks))
        ]

        # 7. Insert (Upsert) the chunks into the database in smaller batches
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
            # 8. If insertion fails because we changed the embedding model (e.g., from 384 dimensions to 768),
            # delete the old database collection entirely and recreate it with the new settings.
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
                # Re-insert the batch into the fresh database
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

            # If it's a completely different error, stop the program and show the error message.
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

    # Return the final count of files and chunks successfully processed
    return IngestionStats(files_processed, total_chunks)