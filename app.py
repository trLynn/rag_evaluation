"""CLI ingestion entrypoint + Streamlit UI entrypoint."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.ingestion import ingest_documents
from src.retrieval import answer_question
import streamlit as st
import json
import os



DOCS_DIR = Path("docs")
PERSIST_DIR = "vector_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"


def get_all_files() -> list[str]:
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


def run_ingestion(
    file_paths: list[str] | None = None,
    chunk_workers: int = 4,
    batch_size: int = 100,
    ollama_timeout: int = 180,
    max_timeout_retries: int = 3,
) -> None:
    """Run full ingestion pipeline."""
    if file_paths is None:
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
        batch_size=batch_size,
        ollama_timeout=ollama_timeout,
        max_timeout_retries=max_timeout_retries,
    )

    end_time = time.time()
    print("\n✅ Ingestion Complete!")
    print(f"📄 Files processed: {stats.files_processed}")
    print(f"🧩 Chunks created: {stats.chunks_added}")
    print(f"⏱ Time taken: {round(end_time - start_time, 2)} seconds\n")


def watch_and_update(
    interval: int = 10,
    chunk_workers: int = 4,
    batch_size: int = 100,
    ollama_timeout: int = 180,
    max_timeout_retries: int = 3,
) -> None:
    """Watch docs folder and auto-ingest when files change."""
    print("👀 Watching docs/ for changes... (Ctrl+C to stop)")
    seen_files: set[str] = set()

    while True:
        current_files = set(get_all_files())
        if current_files != seen_files:
            print("\n🔄 Change detected! Re-ingesting...\n")
            run_ingestion(
                file_paths=list(current_files),
                chunk_workers=chunk_workers,
                batch_size=batch_size,
                ollama_timeout=ollama_timeout,
                max_timeout_retries=max_timeout_retries,
            )
            seen_files = current_files
        time.sleep(interval)


def _is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Chat", layout="wide")
    st.title("Chat")
    st.caption("Ask anything about your indexed knowledge base.")

    llm_model = LLM_MODEL
    embedding_model = EMBEDDING_MODEL
    top_k = 3

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("Clear chat"):
            st.session_state["messages"] = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_prompt = st.chat_input("Ask a question about your indexed documents...")
    if user_prompt:
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = answer_question(
                    question=user_prompt,
                    llm_model=llm_model,
                    top_k=top_k,
                    persist_dir=PERSIST_DIR,
                    collection_name=COLLECTION_NAME,
                    embedding_model=embedding_model,
                )
            if result.get("model_used"):
                st.caption(f"Model used: `{result['model_used']}`")
            st.markdown(result["answer"])

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": result["answer"],
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Run in watch mode")
    parser.add_argument(
        "--chunk-workers",
        type=int,
        default=4,
        help="Number of background workers to use while chunking files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Documents per embedding/upsert batch",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=180,
        help="Timeout in seconds for Ollama embedding requests",
    )
    parser.add_argument(
        "--max-timeout-retries",
        type=int,
        default=3,
        help="Number of retries before splitting a timed-out upsert batch",
    )
    args = parser.parse_args()

    if args.watch:
        watch_and_update(
            chunk_workers=args.chunk_workers,
            batch_size=args.batch_size,
            ollama_timeout=args.ollama_timeout,
            max_timeout_retries=args.max_timeout_retries,
        )
    else:
        run_ingestion(
            chunk_workers=args.chunk_workers,
            batch_size=args.batch_size,
            ollama_timeout=args.ollama_timeout,
            max_timeout_retries=args.max_timeout_retries,
        )

# 1. Function to save questions and answers into a JSON file
def log_chat_history(question, ai_response, log_file="chat_logs.json"):
    log_entry = {
        "question": question,
        "ai_response": ai_response,
        "expected_substring": ""  # You will fill this manually later
    }
    
    # If the file exists, read it; otherwise, create a new list
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []
        
    logs.append(log_entry)
    
    # Write back into the file
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)


# --- Your original Streamlit UI code ---
st.title("📄 Document Chatbot")

if prompt := st.chat_input("Ask a question..."):
    # Display the user’s question (your original code handles this)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Ask the AI for an answer
            response = answer_question(prompt, model_name="phi3")
            st.markdown(response)
            
            # 2. Save the question and answer into the JSON file
            log_chat_history(question=prompt, ai_response=response)


if _is_running_in_streamlit():
    run_streamlit_app()
elif __name__ == "__main__":
    main()