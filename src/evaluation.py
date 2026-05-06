from __future__ import annotations
import re
import csv
from dataclasses import dataclass
from typing import Sequence
from src.retrieval import retrieve_context, answer_question

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
        return (self.hits / self.total * 100) if self.total else 0.0

def professional_clean(text: str) -> str:
    """
    CLEANS THE 'DOUBLE SPACE' ISSUE:
    Turns 'Mae  Fah  Luang' into 'mae fah luang'.
    This is the fix for your 0% Hit Rate.
    """
    # Replace all tabs, newlines, and multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def evaluate_retrieval_to_csv(
    examples: Sequence[EvalExample],
    output_file: str = "evaluation_results.csv",
    top_k: int = 5
) -> EvalResult:
    hits = 0
    results_for_csv = []

    print(f"\n🚀 Starting Professional Evaluation (Top-K: {top_k})")

    for i, ex in enumerate(examples):
        # 1. Get RAG Response
        # We fetch the context and the AI's answer
        rag_data = answer_question(question=ex.question, top_k=top_k)
        
        context_list = rag_data['context']
        full_context_text = "\n".join(context_list)
        ai_answer = rag_data['answer']

        # 2. Comparison Logic (The Fix)
        # We clean BOTH the database text and your expected answer
        clean_context = professional_clean(full_context_text)
        clean_expected = professional_clean(ex.expected_substring)

        # Check if the answer is inside the context
        is_hit = clean_expected in clean_context
        if is_hit:
            hits += 1

        status = "✅ HIT" if is_hit else "❌ MISS"
        print(f"[{i+1}/{len(examples)}] {status} | Q: {ex.question[:40]}...")

        # 3. Store for CSV
        results_for_csv.append({
            "Test_Case": i + 1,
            "Question": ex.question,
            "Expected_Information": ex.expected_substring,
            "Match_Status": "HIT" if is_hit else "MISS",
            "AI_Final_Answer": ai_answer.replace("\n", " "),
            "Retrieved_Context_Preview": full_context_text.replace("\n", " ")[:300]
        })

    # 4. Write to CSV
    keys = results_for_csv[0].keys()
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results_data := results_for_csv)

    return EvalResult(total=len(examples), hits=hits)