from pathlib import Path
from typing import List


# ✅ 1. Read file safely (handles binary noise)
def read_text_file(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        content = f.read()

    # Ignore invalid bytes
    return content.decode("utf-8", errors="ignore")


# ✅ 2. Split text into chunks
def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[str]:

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


# ✅ 3. Build unique chunk IDs
def build_chunk_ids(file_path: Path, num_chunks: int) -> List[str]:
    return [
        f"{file_path.as_posix()}_{i}"
        for i in range(num_chunks)
    ]


# ✅ 4. Load files dynamically from folder
def load_supported_files(directory: Path) -> List[Path]:
    return list(directory.glob("*.txt")) + list(directory.glob("*.md"))


# ✅ 5. Full ingestion pipeline
def ingest_directory(directory: Path):
    files = load_supported_files(directory)

    all_data = []

    for file_path in files:
        text = read_text_file(file_path)
        chunks = chunk_text(text)
        ids = build_chunk_ids(file_path, len(chunks))

        all_data.append({
            "file": file_path,
            "chunks": chunks,
            "ids": ids
        })

    return all_data