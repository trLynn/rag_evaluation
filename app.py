"""CLI ingestion entrypoint + Streamlit UI entrypoint (LangGraph + LangSmith Enabled)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import streamlit as st

from src.ingestion import ingest_documents
from src.graph_retrieval import AdaptiveRAGGraphEngine
from src.retrieval import answer_question

DOCS_DIR = Path("docs")
PERSIST_DIR = "vector_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"
CHAT_LOG_FILE = Path("chat_logs.json")


def setup_langsmith_tracing():
    """Ensure LangSmith environment variables are set for tracing."""
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if "LANGCHAIN_PROJECT" not in os.environ:
        os.environ["LANGCHAIN_PROJECT"] = "CraftGPT-Evaluation"


def get_local_ollama_models() -> list[str]:
    """Return locally available Ollama model names."""
    try:
        import ollama

        models = [
            item.get("model")
            for item in ollama.list().get("models", [])
            if item.get("model")
        ]
    except Exception:
        models = []

    if LLM_MODEL not in models:
        models.insert(0, LLM_MODEL)

    return list(dict.fromkeys(models))


def _init_chat_state(default_model: str) -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "selected_llm_model" not in st.session_state:
        st.session_state["selected_llm_model"] = default_model


def _render_model_toolbar(local_models: list[str]) -> str:
    """Render a compact model control row near chat input."""
    toolbar_col, model_col = st.columns([4, 2])
    with toolbar_col:
        st.caption(
            f"Active model: `{st.session_state['selected_llm_model']}` | Tracing: **LangSmith Active**"
        )
    with model_col:
        st.selectbox(
            "Model",
            options=local_models,
            index=local_models.index(st.session_state["selected_llm_model"]),
            key="selected_llm_model",
            help="Switch local Ollama model for the next message.",
            label_visibility="collapsed",
        )
    return st.session_state["selected_llm_model"]


def get_all_files() -> list[str]:
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


def log_chat_history(
    question: str,
    ai_response: str,
    log_file: Path = CHAT_LOG_FILE,
) -> None:
    log_entry = {
        "question": question,
        "ai_response": ai_response,
        "expected_substring": "",
    }

    logs: list[dict[str, str]] = []
    if log_file.exists():
        try:
            stored_logs = json.loads(log_file.read_text(encoding="utf-8"))
            if isinstance(stored_logs, list):
                logs = stored_logs
        except (json.JSONDecodeError, OSError):
            logs = []

    logs.append(log_entry)
    log_file.write_text(
        json.dumps(logs, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )


def run_streamlit_app() -> None:
    setup_langsmith_tracing()
    st.set_page_config(page_title="CraftGPT", layout="wide", page_icon="✦")

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');
            html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
            .stApp, [data-testid="stAppViewContainer"] { background-color: #F7F5F0; }
            [data-testid="stHeader"] { background: transparent; }
            #MainMenu, footer, [data-testid="stToolbar"] { display: none !important; }
            .hero-wrap { max-width: 680px; margin: clamp(3rem, 14vh, 9rem) auto 0 auto; text-align: center; padding: 0 1.25rem; }
            .hero-eyebrow { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; font-weight: 500; letter-spacing: 0.12em; text-transform: uppercase; color: #9A8F7E; margin-bottom: 1.25rem; }
            .hero-dot { width: 5px; height: 5px; border-radius: 50%; background: #C8B99A; display: inline-block; }
            .hero-title { font-family: 'DM Serif Display', Georgia, serif; font-size: clamp(2.2rem, 5vw, 3rem); font-weight: 400; color: #1C1A16; line-height: 1.15; margin-bottom: 0.85rem; letter-spacing: -0.02em; }
            .hero-title em { font-style: italic; color: #7C6A52; }
            .hero-sub { font-size: 1rem; font-weight: 300; color: #6B6358; line-height: 1.65; max-width: 440px; margin: 0 auto; }
            .chat-divider { max-width: 680px; margin: 2.5rem auto 0 auto; border: none; border-top: 1px solid #E3DDD5; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Render Hero only if chat history is empty
    if not st.session_state.get("messages"):
        st.markdown(
            """
            <div class="hero-wrap">
                <div class="hero-eyebrow">
                    <span class="hero-dot"></span> CraftGPT &nbsp;·&nbsp; LangGraph + LangSmith <span class="hero-dot"></span>
                </div>
                <div class="hero-title">Ask anything about<br><em>your documents</em></div>
                <div class="hero-sub">Adaptive document intelligence powered by state-graph agent loops.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    local_models = get_local_ollama_models()
    _init_chat_state(default_model=local_models[0])
    if st.session_state["selected_llm_model"] not in local_models:
        local_models.insert(0, st.session_state["selected_llm_model"])

    # Render chat history
    if st.session_state["messages"]:
        st.markdown('<hr class="chat-divider">', unsafe_allow_html=True)
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "docs" in message and message["docs"]:
                    with st.expander("🔍 LangGraph Trace Details"):
                        st.write(f"**Final Query Used:** `{message.get('rephrased')}`")
                        st.write("**Retrieved Documents:**")
                        for idx, doc in enumerate(message["docs"]):
                            st.info(f"**Chunk {idx+1}:** {doc}")

    selected_model = _render_model_toolbar(local_models=local_models)
    prompt = st.chat_input("Ask anything about your documents…")

    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        docs = []
        rephrased = prompt
        try:
            with st.spinner("Executing LangGraph State Machine…"):
                engine = AdaptiveRAGGraphEngine(
                    persist_dir=PERSIST_DIR,
                    collection_name=COLLECTION_NAME,
                    embedding_model=EMBEDDING_MODEL,
                    llm_model=selected_model,
                )
                graph_result = engine.run(prompt)
                answer_text = graph_result.get("answer")
                model_used = graph_result.get("model_used", selected_model)
                docs = graph_result.get("documents", [])
                rephrased = graph_result.get("final_query", prompt)

                # Fall back to base retrieval chain if graph generation returned empty
                if not answer_text:
                    fallback = answer_question(
                        question=prompt,
                        llm_model=selected_model,
                        top_k=3,
                        persist_dir=PERSIST_DIR,
                        collection_name=COLLECTION_NAME,
                        embedding_model=EMBEDDING_MODEL,
                    )
                    answer_text = fallback.get("answer", "I couldn't find an answer in the indexed documents.")
                    model_used = fallback.get("model_used", selected_model)

        except Exception as graph_error:
            # Fallback execution in case LangGraph encounters an error
            try:
                fallback = answer_question(
                    question=prompt,
                    llm_model=selected_model,
                    top_k=3,
                    persist_dir=PERSIST_DIR,
                    collection_name=COLLECTION_NAME,
                    embedding_model=EMBEDDING_MODEL,
                )
                answer_text = fallback.get("answer", "I couldn't find an answer in the indexed documents.")
                model_used = fallback.get("model_used", selected_model)
            except Exception as error:
                answer_text = (
                    "I couldn't process that request. Please confirm Ollama is running "
                    "and your documents have been indexed."
                )
                model_used = selected_model
                st.error(f"Request failed: {error}")

        st.markdown(answer_text)
        if docs:
            with st.expander("🔍 LangGraph Trace Details"):
                st.write(f"**Final Query Used:** `{rephrased}`")
                st.write("**Retrieved Documents:**")
                for idx, doc in enumerate(docs):
                    st.info(f"**Chunk {idx+1}:** {doc}")

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer_text,
        "docs": docs,
        "rephrased": rephrased,
    })

    try:
        log_chat_history(
            question=prompt,
            ai_response=f"[model: {model_used}] {answer_text}",
        )
    except OSError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Run in watch mode")
    parser.add_argument("--chunk-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--ollama-timeout", type=int, default=180)
    parser.add_argument("--max-timeout-retries", type=int, default=3)
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


if _is_running_in_streamlit():
    run_streamlit_app()
elif __name__ == "__main__":
    main()