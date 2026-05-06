import argparse
import json
import os
from src.evaluation import EvalExample, evaluate_retrieval_to_csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--file", default="chat_logs.json")
    args = parser.parse_args()

    # Load Logs
    if not os.path.exists(args.file):
        print(f"❌ Error: {args.file} not found.")
        return

    with open(args.file, "r", encoding="utf-8") as f:
        logs = json.load(f)

    examples = [
        EvalExample(question=entry["question"], expected_substring=entry["expected_substring"])
        for entry in logs if "question" in entry and "expected_substring" in entry
    ]

    # Run Eval
    result = evaluate_retrieval_to_csv(examples, top_k=args.top_k)

    print("\n" + "="*40)
    print("📊 THESIS EVALUATION REPORT")
    print("="*40)
    print(f"Total Questions : {result.total}")
    print(f"Successful Hits : {result.hits}")
    print(f"Final Hit Rate  : {result.hit_rate:.2f}%")
    print(f"CSV Report Saved: evaluation_results.csv")
    print("="*40)

if __name__ == "__main__":
    main()