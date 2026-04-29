"""Minimal evaluation helpers for RAG quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Imports the retrieval logic to test against the vector database
from src.retrieval import retrieve_context


@dataclass
class EvalExample:
    """
    Represents a single test case for evaluation.
    
    Attributes:
        question: The query to send to the RAG system.
        expected_substring: A specific string that MUST exist in the 
                            retrieved documents for the test to pass.
    """
    question: str
    expected_substring: str


@dataclass
class EvalResult:
    """
    Stores the final results of an evaluation run.
    """
    total: int
    hits: int

    @property
    def hit_rate(self) -> float:
        """
        Calculates the percentage of successful retrievals (hits / total).
        Returns 0.0 if the total is zero to avoid division errors.
        """
        return self.hits / self.total if self.total else 0.0


def evaluate_retrieval(
    examples: Sequence[EvalExample],
    top_k: int = 3,
    persist_dir: str = "vector_db",
    collection_name: str = "knowledge_base",
    embedding_model: str = "nomic-embed-text",
) -> EvalResult:
    """
    Performs a 'Hit-Rate' evaluation to check the quality of the retriever.
    
    This function determines if the retriever found the 'right' information 
    by checking if a ground-truth substring exists within the top-k 
    retrieved document chunks.
    """
    hits = 0
    
    # Iterate through every test case in the provided list
    for ex in examples:
        # Step 1: Search the vector database for the top-k chunks
        docs = retrieve_context(
            question=ex.question,
            top_k=top_k,
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        
        # Step 2: Merge all retrieved chunks into one searchable string (lowercase)
        corpus = "\n".join(docs).lower()
        
        # Step 3: Check if the 'expected' answer text is present in the chunks
        if ex.expected_substring.lower() in corpus:
            hits += 1

    # Return the summary of how many questions were answered correctly
    return EvalResult(total=len(examples), hits=hits)