"""Retrieval and generation helpers for an Ollama-backed RAG chatbot."""

from __future__ import annotations

import re
from typing import Any
import ollama

from src.ingestion import create_collection

# The strict instruction given to the AI so it doesn't make things up (hallucinate)
SYSTEM_PROMPT = """You are a friendly and helpful student advisor. 
When answering, do not just list data or JSON fields. 
Speak in full, natural human sentences as if you are talking to a friend. 
Summarize the information clearly and only use technical details if they are necessary for the answer."""

# A list of larger, smarter AI models to prefer if the user's computer has them installed
HIGH_QUALITY_MODEL_CANDIDATES = [
    "llama3.1:8b",
    "qwen2.5:7b",
    "gemma2:9b",
    "llama3.1",
]

# A list of smaller, lightweight AI models to fall back on if the computer runs out of memory (RAM)
LOW_MEMORY_MODEL_CANDIDATES = [
    "llama3.2:1b",
    "qwen2.5:0.5b",
    "phi3:mini",
    "tinyllama",
]


# Helper function to check which AI models are currently downloaded/available on the user's computer
def _available_ollama_models(ollama_module: Any) -> set[str]:
    available_models: set[str] = set()
    # Asks Ollama for a list of downloaded models
    for item in ollama_module.list().get("models", []):
        model_name = item.get("model")
        if model_name:
            available_models.add(model_name)
    return available_models


# Helper function to extract meaningful words (keywords) from a sentence 
# It converts text to lowercase, removes punctuation, and ignores tiny words (length <= 2)
def _keyword_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(w) > 2}


# Helper function to re-order the retrieved documents so the most relevant ones are at the top
def _rerank_chunks(question: str, docs: list[str], limit: int) -> list[str]:
    if len(docs) <= limit:
        return docs

    # Get keywords from the user's question
    query_terms = _keyword_set(question)
    scored: list[tuple[int, str]] = []
    
    # Loop through all retrieved documents to see how many keywords match (overlap)
    for doc in docs:
        doc_terms = _keyword_set(doc)
        overlap = len(query_terms.intersection(doc_terms))
        scored.append((overlap, doc))

    # Sort the documents from highest keyword match to lowest
    scored.sort(key=lambda item: item[0], reverse=True)
    
    # Return only the top 'limit' number of documents
    return [doc for _, doc in scored[:limit]]


# The main search function: Finds information in the database relevant to the question
def retrieve_context(
    question: str,
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
    top_k: int = 3,
    collection=None,  # accept cached instance
) -> list[str]:
    """Query the vector database and return the most relevant context chunks."""
    if collection is None:
        collection = create_collection(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )

    # Clamp fetch_k to the actual number of stored documents to avoid ChromaDB errors
    doc_count = collection.count()
    fetch_k = min(top_k * 3, doc_count) if doc_count > 0 else top_k

    result = collection.query(query_texts=[question], n_results=fetch_k)
    docs = result.get("documents", [[]])[0]
    return _rerank_chunks(question=question, docs=docs, limit=top_k)

# The main answering function: Searches for info, picks a model, and generates the final answer
def answer_question(
    question: str,
    llm_model: str = "llama3:latest",
    top_k: int = 3,
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
    collection=None,  # accept a pre-built collection to avoid re-loading from disk
) -> dict[str, Any]:
    """Perform RAG steps and request an answer from the AI model."""

    # 1. Fetch available models ONCE and reuse throughout this function
    available_models = _available_ollama_models(ollama)

    # 2. Select the best available model upfront — no conditional guard needed
    selected_model = next(
        (candidate for candidate in HIGH_QUALITY_MODEL_CANDIDATES if candidate in available_models),
        llm_model,  # fall back to caller-supplied model if nothing better is found
    )

    # 3. Retrieve relevant context chunks (reuse cached collection if provided)
    context_chunks = retrieve_context(
        question=question,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        collection=collection,  # pass through so retrieve_context skips create_collection
    )

    context_text = "\n\n".join(context_chunks) if context_chunks else "No context found."

    # 4. Build prompt messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}",
        },
    ]

    # 5. Call the model, with automatic fallback to a smaller one if RAM is exhausted
    try:
        response = ollama.chat(model=selected_model, messages=messages)
    except Exception as exc:
        if "requires more system memory" not in str(exc).lower():
            raise  # unrelated error — let it propagate as-is

        # Re-use the already-fetched available_models set — no second network call
        fallback_model = next(
            (candidate for candidate in LOW_MEMORY_MODEL_CANDIDATES if candidate in available_models),
            None,
        )
        if fallback_model is None:
            raise RuntimeError(
                f"Model '{selected_model}' requires more RAM than is available, "
                f"and no low-memory fallback was found among {LOW_MEMORY_MODEL_CANDIDATES}."
            ) from exc

        selected_model = fallback_model
        response = ollama.chat(model=selected_model, messages=messages)

    return {
        "answer": response["message"]["content"],
        "context": context_chunks,
        "model_used": selected_model,
    }