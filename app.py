"""CLI ingestion entrypoint + Streamlit UI entrypoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import streamlit as st

from src.ingestion import ingest_documents
from src.retrieval import answer_question

DOCS_DIR = Path("docs")
PERSIST_DIR = "vector_db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1"


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

    # Keep order stable while removing duplicates
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
        st.caption(f"Active model: `{st.session_state['selected_llm_model']}`")
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
CHAT_LOG_FILE = Path("chat_logs.json")


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


def log_chat_history(question: str, ai_response: str, log_file: Path = CHAT_LOG_FILE) -> None:
    log_entry = {
        "question": question,
        "ai_response": ai_response,
        "expected_substring": "",
    }

    logs: list[dict[str, str]] = []
    if log_file.exists():
        logs = json.loads(log_file.read_text(encoding="utf-8"))

    logs.append(log_entry)
    log_file.write_text(json.dumps(logs, indent=4, ensure_ascii=False), encoding="utf-8")


def run_streamlit_app() -> None:
    st.set_page_config(page_title="CraftGPT", layout="wide", page_icon="✦")

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

            /* ── Base reset ── */
            html, body, [class*="css"] {
                font-family: 'DM Sans', sans-serif;
            }

            /* ── Page background ── */
            .stApp {
                background-color: #F7F5F0;
            }
            [data-testid="stAppViewContainer"] {
                background-color: #F7F5F0;
            }
            [data-testid="stHeader"] {
                background: transparent;
            }

            /* ── Hide Streamlit chrome ── */
            #MainMenu, footer, [data-testid="stToolbar"] {
                display: none !important;
            }

            /* ── Hero section ── */
            .hero-wrap {
                max-width: 680px;
                margin: clamp(3rem, 14vh, 9rem) auto 0 auto;
                text-align: center;
                padding: 0 1.25rem;
            }
            .hero-eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                font-size: 11px;
                font-weight: 500;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #9A8F7E;
                margin-bottom: 1.25rem;
            }
            .hero-dot {
                width: 5px;
                height: 5px;
                border-radius: 50%;
                background: #C8B99A;
                display: inline-block;
            }
            .hero-title {
                font-family: 'DM Serif Display', Georgia, serif;
                font-size: clamp(2.2rem, 5vw, 3rem);
                font-weight: 400;
                color: #1C1A16;
                line-height: 1.15;
                margin-bottom: 0.85rem;
                letter-spacing: -0.02em;
            }
            .hero-title em {
                font-style: italic;
                color: #7C6A52;
            }
            .hero-sub {
                font-size: 1rem;
                font-weight: 300;
                color: #6B6358;
                line-height: 1.65;
                max-width: 440px;
                margin: 0 auto 0 auto;
            }

            /* ── Example pills ── */
            .examples-wrap {
                max-width: 680px;
                margin: 1.75rem auto 0 auto;
                padding: 0 1.25rem;
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                justify-content: center;
            }
            .pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 0.42rem 0.85rem;
                border: 1px solid #DDD8CF;
                border-radius: 999px;
                color: #5C5549;
                font-size: 0.82rem;
                font-weight: 400;
                background: #FDFCF9;
                transition: background 0.15s, border-color 0.15s;
                cursor: default;
            }
            .pill:hover {
                background: #F2EDE4;
                border-color: #C8B99A;
            }
            .pill-icon {
                font-size: 13px;
                opacity: 0.7;
            }

            /* ── Chat input override ── */
            [data-testid="stChatInput"] {
                max-width: 680px;
                margin: 2rem auto 0 auto;
                background: #FDFCF9;
                border-radius: 14px !important;
                border: 1px solid #DDD8CF !important;
                box-shadow: 0 2px 12px rgba(0,0,0,0.05) !important;
                padding: 0.25rem 0.5rem !important;
            }
            [data-testid="stChatInput"]:focus-within {
                border-color: #A69580 !important;
                box-shadow: 0 0 0 3px rgba(166,149,128,0.15), 0 2px 12px rgba(0,0,0,0.05) !important;
            }
            [data-testid="stChatInput"] textarea {
                font-family: 'DM Sans', sans-serif !important;
                font-size: 0.95rem !important;
                color: #1C1A16 !important;
                background: transparent !important;
            }
            [data-testid="stChatInput"] textarea::placeholder {
                color: #A09489 !important;
            }

            /* ── Chat messages ── */
            [data-testid="stChatMessageContainer"] {
                max-width: 680px;
                margin: 0 auto;
                padding: 0.75rem 1.25rem;
            }
            [data-testid="stChatMessage"] {
                background: transparent !important;
                border: none !important;
                padding: 0.65rem 0 !important;
            }

            /* User bubble */
            [data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
                background: #EDE8DF;
                border-radius: 14px 14px 4px 14px;
                padding: 0.7rem 1rem;
                color: #1C1A16;
                font-size: 0.93rem;
                line-height: 1.6;
                display: inline-block;
                max-width: 88%;
                float: right;
            }

            /* Assistant bubble */
            div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown p {
                background: #FDFCF9;
                border: 1px solid #E3DDD5;
                border-radius: 4px 14px 14px 14px;
                padding: 0.7rem 1rem;
                color: #2A2720;
                font-size: 0.93rem;
                line-height: 1.7;
            }

            /* Avatar icons */
            [data-testid="chatAvatarIcon-user"] {
                background: #C8B99A !important;
                color: #3D3428 !important;
            }
            [data-testid="chatAvatarIcon-assistant"] {
                background: #1C1A16 !important;
                color: #F7F5F0 !important;
            }

            /* ── Spinner ── */
            [data-testid="stSpinner"] > div {
                color: #7C6A52;
                font-size: 0.85rem;
            }

            /* ── Divider between hero and chat ── */
            .chat-divider {
                max-width: 680px;
                margin: 2.5rem auto 0 auto;
                border: none;
                border-top: 1px solid #E3DDD5;
            }

            /* ── Responsive ── */
            @media (max-width: 640px) {
                .hero-wrap, .examples-wrap {
                    text-align: left;
                    justify-content: flex-start;
                }
                .hero-title {
                    font-size: 1.9rem;
                }
                .pill {
                    font-size: 0.8rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Hero ──────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-eyebrow">
                <span class="hero-dot"></span>
                CraftGPT &nbsp;·&nbsp; Document Intelligence
                <span class="hero-dot"></span>
            </div>
            <div class="hero-title">Ask anything about<br><em>your documents</em></div>
            <div class="hero-sub">
                Search, summarise, and extract insights from your indexed knowledge base — instantly.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Example pills ──────────────────────────────────────
    st.markdown(
        """
        <div class="examples-wrap">
            <span class="pill"><span class="pill-icon">◎</span> Summarise the uploaded policy doc</span>
            <span class="pill"><span class="pill-icon">◎</span> Find where cancellation terms are defined</span>
            <span class="pill"><span class="pill-icon">◎</span> List key numbers from the report</span>
            <span class="pill"><span class="pill-icon">◎</span> Compare two sections</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    local_models = get_local_ollama_models()
    _init_chat_state(default_model=local_models[0])
    if st.session_state["selected_llm_model"] not in local_models:
        local_models.insert(0, st.session_state["selected_llm_model"])

    # ── Model toolbar + chat input ────────────────────────
    selected_model = _render_model_toolbar(local_models=local_models)
    prompt = st.chat_input("Ask anything about your documents…")

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.spinner("Thinking…"):
            result = answer_question(
                question=prompt,
                llm_model=selected_model,
                top_k=3,
                persist_dir=PERSIST_DIR,
                collection_name=COLLECTION_NAME,
                embedding_model=EMBEDDING_MODEL,
            )
        answer_text = result.get("answer", "")
        model_used = result.get("model_used", selected_model)
        st.session_state["messages"].append({"role": "assistant", "content": answer_text})
        log_chat_history(
            question=prompt,
            ai_response=f"[model: {model_used}] {answer_text}",
        )

    # ── Render conversation ────────────────────────────────
    if st.session_state["messages"]:
        st.markdown('<hr class="chat-divider">', unsafe_allow_html=True)
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


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