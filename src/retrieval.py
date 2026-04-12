"""Retrieval and generation helpers for an Ollama-backed RAG chatbot."""

from __future__ import annotations

from typing import Any

from src.ingestion import create_collection


SYSTEM_PROMPT = """You are a helpful assistant. Use the provided context to answer the question.
If the answer is not present in the context, say you do not know."""


def retrieve_context(
    question: str,
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
    top_k: int = 3,
) -> list[str]:
    """Return top-k relevant chunks for a question."""
    collection = create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    result = collection.query(query_texts=[question], n_results=top_k)
    return result.get("documents", [[]])[0]


def answer_question(
    question: str,
    llm_model: str = "llama3.1",
    top_k: int = 3,
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
) -> dict[str, Any]:
    """Run a full RAG pass: retrieve context then ask Ollama chat model."""
    context_chunks = retrieve_context(
        question=question,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    context_text = "\n\n".join(context_chunks) if context_chunks else "No context found."

    import ollama

    response = ollama.chat(
        model=llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}",
            },
        ],
    )

    answer = response["message"]["content"]
    return {
        "answer": answer,
        "context": context_chunks,
    }