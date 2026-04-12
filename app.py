"""CLI chatbot application using Ollama + Chroma RAG."""

from __future__ import annotations

import argparse

from src.evaluation import EvalExample, evaluate_retrieval
from src.ingestion import ingest_documents
from src.retrieval import answer_question


def cmd_ingest(args: argparse.Namespace) -> None:
    stats = ingest_documents(
        file_paths=args.files,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Ingestion complete: files={stats.files_processed}, chunks={stats.chunks_added}")


def cmd_chat(args: argparse.Namespace) -> None:
    print("Type your question (or 'exit' to quit)")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            return
        result = answer_question(
            question=question,
            llm_model=args.model,
            top_k=args.top_k,
            persist_dir=args.persist_dir,
            collection_name=args.collection,
            embedding_model=args.embedding_model,
        )
        print(f"Assistant: {result['answer']}")


def cmd_eval(args: argparse.Namespace) -> None:
    examples = []
    for item in args.pairs:
        if "::" not in item:
            raise ValueError("Each pair must be formatted as 'question::expected_substring'")
        q, expected = item.split("::", 1)
        examples.append(EvalExample(question=q.strip(), expected_substring=expected.strip()))

    result = evaluate_retrieval(
        examples=examples,
        top_k=args.top_k,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    print(f"Evaluation: {result.hits}/{result.total} hits (hit_rate={result.hit_rate:.2%})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ollama RAG chatbot + evaluation")
    parser.add_argument("--persist-dir", default="vector_db")
    parser.add_argument("--collection", default="knowledge_base")
    parser.add_argument("--embedding-model", default="nomic-embed-text")

    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Ingest text files")
    ingest.add_argument("files", nargs="+")
    ingest.add_argument("--chunk-size", type=int, default=500)
    ingest.add_argument("--chunk-overlap", type=int, default=80)
    ingest.set_defaults(func=cmd_ingest)

    chat = sub.add_parser("chat", help="Start interactive chat")
    chat.add_argument("--model", default="llama3.1")
    chat.add_argument("--top-k", type=int, default=3)
    chat.set_defaults(func=cmd_chat)

    evaluate = sub.add_parser("eval", help="Evaluate retrieval hit-rate")
    evaluate.add_argument(
        "pairs",
        nargs="+",
        help="One or more 'question::expected_substring' items",
    )
    evaluate.add_argument("--top-k", type=int, default=3)
    evaluate.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()