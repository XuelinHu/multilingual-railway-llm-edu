from __future__ import annotations

import argparse

from railway_rag.agent.formatter import format_answer
from railway_rag.agent.tools import classify_query, infer_record_types, search_regulation, search_term_dictionary
from railway_rag.config import load_config
from railway_rag.retrieval.vector_store import VectorStore
from railway_rag.safety.risk import risk_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions against the multilingual railway RAG index.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--query", required=True, help="Question string.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k.")
    args = parser.parse_args()

    config = load_config(args.config)
    vector_store = VectorStore.load(config["paths"]["vector_index"])
    top_k = args.top_k or int(config["retrieval"].get("top_k", 5))
    query_type = classify_query(args.query)
    record_types = infer_record_types(query_type)
    risk_result = risk_check(args.query)

    if record_types == ["glossary_term"]:
        hits = search_term_dictionary(vector_store, args.query, top_k=max(top_k, 8))
    elif query_type == "paper":
        hits = []
    else:
        hits = search_regulation(vector_store, args.query, top_k=top_k)

    answer = format_answer(query_type, args.query, hits, risk_result)

    print("Answer:")
    print(answer["answer"])
    print("\nCitations:")
    if not answer["citations"]:
        print("  (none)")
        return

    for citation in answer["citations"]:
        print(
            f"  [{citation['index']}] {citation['source_label']} "
            f"(record_id={citation['record_id']}, score={citation['score']})"
        )


if __name__ == "__main__":
    main()
