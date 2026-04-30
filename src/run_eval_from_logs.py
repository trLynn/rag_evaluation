import argparse
import json

from src.evaluation import EvalExample, evaluate_retrieval


def load_eval_examples(log_file: str, expected_field: str = "expected_substring") -> list[EvalExample]:
    """Load evaluable examples from a JSON chat log file."""
    with open(log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    examples: list[EvalExample] = []
    for entry in logs:
        expected_value = str(entry.get(expected_field, "")).strip()
        question = str(entry.get("question", "")).strip()
        if question and expected_value:
            examples.append(EvalExample(question=question, expected_substring=expected_value))
    return examples


def run_evaluation_from_json(log_file: str = "chat_logs.json", expected_field: str = "expected_substring", top_k: int = 5):
    print(f"Reading logs from {log_file}...")
    test_cases = load_eval_examples(log_file=log_file, expected_field=expected_field)

    if not test_cases:
        print(
            f"❌ No questions available for evaluation. Please fill in non-empty '{expected_field}' values in '{log_file}' first."
        )
        return

    print(f"Found {len(test_cases)} questions ready for evaluation.")
    retrieval_result = evaluate_retrieval(examples=test_cases, top_k=top_k)

    print("\n=== FINAL RESULTS ===")
    print(f"Total Evaluated: {retrieval_result.total}")
    print(f"Hits: {retrieval_result.hits}")
    print(f"Hit Rate: {retrieval_result.hit_rate * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate chat logs with the project RAG evaluation module.")
    parser.add_argument("--log-file", default="chat_logs.json", help="Path to chat log JSON file")
    parser.add_argument(
        "--expected-field",
        default="expected_substring",
        help="Field name in each log entry containing the expected answer substring",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k contexts to retrieve during evaluation")
    args = parser.parse_args()
    run_evaluation_from_json(log_file=args.log_file, expected_field=args.expected_field, top_k=args.top_k)