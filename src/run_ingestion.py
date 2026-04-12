from pathlib import Path
from src.ingestion import ingest_documents

def main():
    docs_dir = Path("docs")

    file_paths = [str(p) for p in docs_dir.glob("*") if p.is_file()]

    if not file_paths:
        print("❌ No files found in docs/")
        return

    print(f"📂 Found {len(file_paths)} files")

    result = ingest_documents(file_paths)

    print("\n✅ Done!")
    print(result)


if __name__ == "__main__":
    main()