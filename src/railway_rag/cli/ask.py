from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_rag.agent.baselines import available_baselines, get_baseline_options, retrieve_hits
from railway_rag.agent.formatter import format_answer
from railway_rag.agent.tools import classify_query
from railway_rag.config import load_config
from railway_rag.retrieval.vector_store import VectorStore
from railway_rag.safety.risk import risk_check


def _write_qa_log(config: dict, payload: dict) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "qa_runs.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def answer_query(config: dict, vector_store: VectorStore, query: str, top_k: int, baseline_name: str) -> dict:
    query_type = classify_query(query)
    risk_result = risk_check(query)
    options = get_baseline_options(baseline_name)
    channels = retrieve_hits(vector_store, query, query_type=query_type, top_k=top_k, baseline_name=baseline_name)

    if query_type == "paper":
        hits = []
    elif query_type in {"term", "bilingual"}:
        hits = channels["terminology"] or channels["regulation"]
    elif options["use_dual_channel"] and query_type in {"fault", "procedure"}:
        hits = channels["regulation"] + [
            hit for hit in channels["terminology"] if hit["record_id"] not in {item["record_id"] for item in channels["regulation"]}
        ]
    else:
        hits = channels["regulation"]

    answer = format_answer(query_type, query, hits, risk_result, options=options)
    _write_qa_log(
        config,
        {
            "query": query,
            "baseline": baseline_name,
            "query_type": query_type,
            "risk_level": risk_result["risk_level"],
            "matched_risk_markers": risk_result["matched_markers"],
            "citations": answer["citations"],
            "terminology_hits": [hit["record_id"] for hit in channels["terminology"][:5]],
            "regulation_hits": [hit["record_id"] for hit in channels["regulation"][:5]],
        },
    )
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions against the multilingual railway RAG index.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--query", required=True, help="Question string.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k.")
    parser.add_argument(
        "--baseline",
        default="rag_term_risk",
        choices=available_baselines(),
        help="Baseline preset to run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    vector_store = VectorStore.load(config["paths"]["vector_index"])
    top_k = args.top_k or int(config["retrieval"].get("top_k", 5))
    answer = answer_query(config, vector_store, args.query, top_k=top_k, baseline_name=args.baseline)

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
