"""Minimal evaluation helpers for RAG quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.retrieval import retrieve_context


@dataclass
class EvalExample:
    question: str
    expected_substring: str


@dataclass
class EvalResult:
    total: int
    hits: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


def evaluate_retrieval(
    examples: Sequence[EvalExample],
    top_k: int = 3,
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
) -> EvalResult:
    """Simple hit-rate evaluation: did expected text appear in retrieved chunks?"""
    hits = 0
    for ex in examples:
        docs = retrieve_context(
            question=ex.question,
            top_k=top_k,
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        corpus = "\n".join(docs).lower()
        if ex.expected_substring.lower() in corpus:
            hits += 1

    return EvalResult(total=len(examples), hits=hits)