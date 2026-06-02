"""English query → similar YAML chunks (no LLM generation yet)."""

import argparse
import sys

from config import TOP_K_DEFAULT
from vector_rag import semantic_retrieval


def print_results(query: str, top_k: int = TOP_K_DEFAULT) -> None:
    results = semantic_retrieval(query, top_k=top_k)
    if not results:
        print("No matching chunks found.")
        return
    for i, doc in enumerate(results, 1):
        print(f"\n Semantic Result {i}:\n{doc.page_content}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve similar schema chunks for an English question."
    )
    parser.add_argument("query", nargs="*", help="Your question in English")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    args = parser.parse_args()

    query = " ".join(args.query).strip()
    if not query:
        query = input("Question: ").strip()
    if not query:
        print("Provide a question, e.g.: python retrieve.py \"primary cancer by month\"")
        sys.exit(1)

    print_results(query, top_k=args.top_k)


if __name__ == "__main__":
    main()
