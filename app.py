"""Streamlit interface for the Ollama + Chroma RAG chatbot."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ingestion import ingest_documents
from src.retrieval import answer_question


st.set_page_config(page_title="RAG Chatbot", page_icon="✨", layout="wide")

st.title("✨ Ollama RAG Chatbot")
st.caption("Ingest local .txt files and chat with your knowledge base.")

with st.sidebar:
    st.header("Settings")
    persist_dir = st.text_input("Persist directory", value="vector_db")
    collection_name = st.text_input("Collection", value="knowledge_base")
    embedding_model = st.text_input("Embedding model", value="nomic-embed-text")
    llm_model = st.text_input("LLM model", value="llama3.1")
    top_k = st.number_input("Top-k retrieval", min_value=1, max_value=10, value=3, step=1)

st.subheader("1) Ingest documents")
uploaded_files = st.file_uploader(
    "Upload one or more UTF-8 text files",
    type=["txt"],
    accept_multiple_files=True,
)

if st.button("Ingest uploaded files", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one .txt file.")
    else:
        docs_dir = Path("docs")
        docs_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[str] = []
        for uploaded_file in uploaded_files:
            file_path = docs_dir / uploaded_file.name
            file_path.write_bytes(uploaded_file.getbuffer())
            saved_paths.append(str(file_path))

        with st.spinner("Ingesting files into Chroma..."):
            stats = ingest_documents(
                file_paths=saved_paths,
                persist_dir=persist_dir,
                collection_name=collection_name,
                embedding_model=embedding_model,
            )

        st.success(
            f"Ingestion complete: files={stats.files_processed}, chunks={stats.chunks_added}"
        )

st.divider()
st.subheader("2) Ask a question")
question = st.text_input("Question", placeholder="What is the capital of France?")

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            result = answer_question(
                question=question,
                llm_model=llm_model,
                top_k=int(top_k),
                persist_dir=persist_dir,
                collection_name=collection_name,
                embedding_model=embedding_model,
            )

        st.markdown("### Answer")
        st.write(result["answer"])

        with st.expander("Retrieved context"):
            if result["context"]:
                for i, chunk in enumerate(result["context"], start=1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(chunk)
            else:
                st.write("No context found.")
