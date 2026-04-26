"""
add.py

Run ingestion (chunking + embedding + storing) separately from the app.
"""

from pathlib import Path
import time

from src.ingestion import ingest_documents


DOCS_DIR = Path("docs")
PERSIST_DIR = "vector_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "nomic-embed-text"


def get_all_files():
    """Load all files from docs folder."""
    if not DOCS_DIR.exists():
        print("❌ docs/ folder not found")
        return []

    files = [str(p) for p in DOCS_DIR.glob("*") if p.is_file()]

    if not files:
        print("⚠️ No files found in docs/")
    else:
        print(f"📂 Found {len(files)} files")

    return files


def run_ingestion(chunk_workers: int = 4):
    """Run full ingestion pipeline."""
    file_paths = get_all_files()

    if not file_paths:
        return

    print("\n🚀 Starting ingestion process...\n")

    start_time = time.time()

    stats = ingest_documents(
        file_paths=file_paths,
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        chunk_workers=chunk_workers,
    )

    end_time = time.time()

    print("\n✅ Ingestion Complete!")
    print(f"📄 Files processed: {stats.files_processed}")
    print(f"🧩 Chunks created: {stats.chunks_added}")
    print(f"⏱ Time taken: {round(end_time - start_time, 2)} seconds\n")


# ------------------ OPTIONAL: WATCH MODE ------------------
def watch_and_update(interval=10, chunk_workers: int = 4):
    """
    Watch docs folder and auto-ingest when files change.
    """
    print("👀 Watching docs/ for changes... (Ctrl+C to stop)")

    seen_files = set()

    while True:
        current_files = set(get_all_files())

        if current_files != seen_files:
            print("\n🔄 Change detected! Re-ingesting...\n")
            run_ingestion(chunk_workers=chunk_workers)
            seen_files = current_files

        time.sleep(interval)


# ------------------ ENTRY ------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Run in watch mode")
    parser.add_argument(
        "--chunk-workers",
        type=int,
        default=4,
        help="Number of background workers to use while chunking files",
    )

    args = parser.parse_args()

    if args.watch:
        watch_and_update(chunk_workers=args.chunk_workers)
    else:
        run_ingestion(chunk_workers=args.chunk_workers)