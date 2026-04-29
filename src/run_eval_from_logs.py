import json
from src.evaluation import evaluate_retrieval, EvalExample

def run_evaluation_from_json(log_file="chat_logs.json"):
    print(f"Reading logs from {log_file}...")
    
    # Open the JSON log file and load the content
    with open(log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)
        
    test_cases = []
    
    for entry in logs:
        # Only evaluate questions where you have manually filled in 'expected_substring'
        if entry.get("expected_substring") != "":
            test_cases.append(EvalExample(
                question=entry["question"],
                expected_substring=entry["expected_substring"]
            ))
            
    if not test_cases:
        print("❌ No questions available for evaluation. Please fill in 'expected_substring' values in 'chat_logs.json' first.")
        return

    print(f"Found {len(test_cases)} questions ready for evaluation.")
    
    # Call the evaluation function from src/evaluation.py
    result = evaluate_retrieval(examples=test_cases, top_k=5)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Total Evaluated: {result.total}")
    print(f"Hit Rate: {result.hit_rate * 100}%")
    print(f"Average Latency: {result.avg_latency:.4f} seconds")

if __name__ == "__main__":
    run_evaluation_from_json()
