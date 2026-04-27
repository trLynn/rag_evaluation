"""Retrieval and generation helpers for an Ollama-backed RAG chatbot."""

from __future__ import annotations

import re
from typing import Any

from src.ingestion import create_collection


SYSTEM_PROMPT = """You are a helpful assistant. Use the provided context to answer the question.
If the answer is not present in the context, say you do not know."""

HIGH_QUALITY_MODEL_CANDIDATES = [
    "llama3.1:8b",
    "qwen2.5:7b",
    "gemma2:9b",
    "llama3.1",
]

LOW_MEMORY_MODEL_CANDIDATES = [
    "llama3.2:1b",
    "qwen2.5:0.5b",
    "phi3:mini",
    "tinyllama",
]


def _available_ollama_models(ollama_module: Any) -> set[str]:
    available_models: set[str] = set()
    for item in ollama_module.list().get("models", []):
        model_name = item.get("model")
        if model_name:
            available_models.add(model_name)
    return available_models


def _keyword_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(w) > 2}


def _rerank_chunks(question: str, docs: list[str], limit: int) -> list[str]:
    if len(docs) <= limit:
        return docs

    query_terms = _keyword_set(question)
    scored: list[tuple[int, str]] = []
    for doc in docs:
        doc_terms = _keyword_set(doc)
        overlap = len(query_terms.intersection(doc_terms))
        scored.append((overlap, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


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
    fetch_k = max(top_k * 3, top_k)
    result = collection.query(query_texts=[question], n_results=fetch_k)
    docs = result.get("documents", [[]])[0]
    return _rerank_chunks(question=question, docs=docs, limit=top_k)


def answer_question(
    question: str,
    llm_model: str = "llama3:latest",
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}",
        },
    ]

    selected_model = llm_model
    try:
        available_models = _available_ollama_models(ollama)
        if llm_model == "llama3.1":
            preferred_high_model = next(
                (
                    candidate
                    for candidate in HIGH_QUALITY_MODEL_CANDIDATES
                    if candidate in available_models
                ),
                llm_model,
            )
            selected_model = preferred_high_model
    except Exception:
        # If model listing fails, fall back to the user-requested model.
        selected_model = llm_model

    try:
        response = ollama.chat(model=selected_model, messages=messages)
    except Exception as exc:
        if "requires more system memory" not in str(exc).lower():
            raise

        # Automatic low-memory fallback for constrained environments.
        try:
            available_models = _available_ollama_models(ollama)
        except Exception:
            available_models = set()

        fallback_model = next(
            (
                candidate
                for candidate in LOW_MEMORY_MODEL_CANDIDATES
                if candidate in available_models
            ),
            None,
        )
        if fallback_model is None:
            raise RuntimeError(
                "The selected Ollama model does not fit into available RAM, and no "
                "known low-memory fallback model was found locally. "
                "Try: `ollama pull llama3.2:1b` and run again."
            ) from exc

        selected_model = fallback_model
        response = ollama.chat(model=selected_model, messages=messages)

    answer = response["message"]["content"]
    return {
        "answer": answer,
        "context": context_chunks,
        "model_used": selected_model,
    }