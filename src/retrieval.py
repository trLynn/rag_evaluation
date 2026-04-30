"""Retrieval and generation helpers for an Ollama-backed RAG chatbot."""

from __future__ import annotations

import re
from typing import Any

from src.ingestion import create_collection

# The strict instruction given to the AI so it doesn't make things up (hallucinate)
SYSTEM_PROMPT = """You are a helpful assistant. Use the provided context to answer the question.
If the answer is not present in the context, say you do not know."""

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
    top_k: int = 3, # Number of final documents we want
) -> list[str]:
    """Return top-k relevant chunks for a question."""
    
    # Connect to the Vector Database
    collection = create_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    
    # Strategy: Fetch extra documents first (e.g., if we want 3, fetch 9) 
    # so we have a larger pool to re-rank and choose the absolute best from.
    fetch_k = max(top_k * 3, top_k)
    result = collection.query(query_texts=[question], n_results=fetch_k)
    docs = result.get("documents", [[]])[0]
    
    # Re-rank the fetched documents and return the very best ones
    return _rerank_chunks(question=question, docs=docs, limit=top_k)


# The main answering function: Searches for info, picks a model, and generates the final answer
def answer_question(
    question: str,
    llm_model: str = "llama3:latest",
    top_k: int = 3,
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
) -> dict[str, Any]:
    """Run a full RAG pass: retrieve context then ask Ollama chat model."""
    
    # Step 1: Retrieve relevant information (context) from the database
    context_chunks = retrieve_context(
        question=question,
        top_k=top_k,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    # Combine the separate chunks of text into one big string
    context_text = "\n\n".join(context_chunks) if context_chunks else "No context found."

    import ollama

    # Step 2: Prepare the prompt to send to the AI
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, # Tell it the rules
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}", # Provide info & question
        },
    ]

    selected_model = llm_model
    
    # Step 3: Smart Model Selection
    try:
        available_models = _available_ollama_models(ollama)
        # If the user asked for "llama3.1", try to automatically upgrade them 
        # to the highest quality model they actually have downloaded.
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
        # If checking fails, just use the model they originally asked for
        selected_model = llm_model

    # Step 4: Ask the AI for the answer
    try:
        response = ollama.chat(model=selected_model, messages=messages)
    except Exception as exc:
        # If the error is NOT about running out of memory, stop the program and show the error
        if "requires more system memory" not in str(exc).lower():
            raise

        # Step 5: Fallback Mechanism (If computer runs out of memory/RAM)
        try:
            available_models = _available_ollama_models(ollama)
        except Exception:
            available_models = set()

        # Find a smaller model from our LOW_MEMORY list that is available on the computer
        fallback_model = next(
            (
                candidate
                for candidate in LOW_MEMORY_MODEL_CANDIDATES
                if candidate in available_models
            ),
            None,
        )
        
        # If no small models are downloaded, tell the user to download one
        if fallback_model is None:
            raise RuntimeError(
                "The selected Ollama model does not fit into available RAM, and no "
                "known low-memory fallback model was found locally. "
                "Try: `ollama pull llama3.2:1b` and run again."
            ) from exc

        # Try generating the answer again using the smaller, fallback model
        selected_model = fallback_model
        response = ollama.chat(model=selected_model, messages=messages)

    # Return the AI's final answer, the documents it used, and the name of the model it ended up using
    answer = response["message"]["content"]
    return {
        "answer": answer,
        "context": context_chunks,
        "model_used": selected_model,
    }