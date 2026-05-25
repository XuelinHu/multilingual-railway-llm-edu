from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from railway_rag.agent.baselines import available_baselines, retrieve_hits
from railway_rag.agent.tools import classify_query
from railway_rag.config import load_config
from railway_rag.retrieval.vector_store import VectorStore
from railway_rag.utils import normalize_text


def select_hits(channels: Dict[str, List[Dict[str, object]]], query_type: str, use_dual_channel: bool) -> List[Dict[str, object]]:
    if query_type in {"term", "bilingual"}:
        return channels["terminology"] or channels["regulation"]
    if use_dual_channel and query_type in {"fault", "procedure"}:
        merged = list(channels["regulation"])
        seen = {item["record_id"] for item in merged}
        merged.extend(hit for hit in channels["terminology"] if hit["record_id"] not in seen)
        return merged
    return channels["regulation"]


def relevance_score(sample: Dict[str, object], hit: Dict[str, object]) -> int:
    haystack = normalize_text(
        " ".join(
            [
                str(hit.get("source_label", "")),
                str(hit.get("section_path", "")),
                str(hit.get("zh_text", "")),
                str(hit.get("en_text", "")),
                str(hit.get("term_zh", "")),
                str(hit.get("term_en", "")),
            ]
        )
    ).lower()
    evidence = normalize_text(str(sample.get("evidence", ""))).lower()
    keywords = [normalize_text(str(item)).lower() for item in sample.get("keywords", [])]

    keyword_hits = sum(1 for keyword in keywords if keyword and keyword in haystack)
    evidence_hits = sum(1 for token in evidence.replace("：", " ").replace("/", " ").replace("、", " ").split() if token and token in haystack)

    if keyword_hits >= 2 or evidence_hits >= 2:
        return 2
    if keyword_hits >= 1 or evidence_hits >= 1:
        return 1
    return 0


def reciprocal_rank(relevances: List[int]) -> float:
    for index, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / index
    return 0.0


def dcg(relevances: List[int], k: int) -> float:
    score = 0.0
    for index, rel in enumerate(relevances[:k], start=1):
        score += (2**rel - 1) / math.log2(index + 1)
    return score


def ndcg_at_k(relevances: List[int], k: int) -> float:
    actual = dcg(relevances, k)
    ideal = dcg(sorted(relevances, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on the eval set.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--eval-file", default="data/eval/eval_samples.jsonl", help="Evaluation JSONL file.")
    parser.add_argument("--baseline", required=True, choices=available_baselines(), help="Baseline preset.")
    parser.add_argument("--k", type=int, default=5, help="Top-k cutoff for retrieval metrics.")
    args = parser.parse_args()

    config = load_config(args.config)
    vector_store = VectorStore.load(config["paths"]["vector_index"])
    rows = [json.loads(line) for line in Path(args.eval_file).read_text(encoding="utf-8").splitlines() if line.strip()]

    from railway_rag.agent.baselines import get_baseline_options

    options = get_baseline_options(args.baseline)
    output_dir = Path("experiments") / args.baseline
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "retrieval_eval_details.jsonl"
    metrics_path = output_dir / "retrieval_metrics.json"

    evaluated = 0
    recall_hits = 0
    total_mrr = 0.0
    total_ndcg = 0.0

    with details_path.open("w", encoding="utf-8") as handle:
        for sample in rows:
            if sample["answerability"] != "answerable":
                continue
            evaluated += 1
            query_type = classify_query(sample["question"])
            channels = retrieve_hits(vector_store, sample["question"], query_type=query_type, top_k=args.k, baseline_name=args.baseline)
            hits = select_hits(channels, query_type, bool(options["use_dual_channel"]))[: args.k]
            relevances = [relevance_score(sample, hit) for hit in hits]

            recall = 1.0 if any(rel > 0 for rel in relevances) else 0.0
            mrr = reciprocal_rank(relevances)
            ndcg = ndcg_at_k(relevances, args.k)

            recall_hits += int(recall)
            total_mrr += mrr
            total_ndcg += ndcg

            payload = {
                "sample_id": sample["sample_id"],
                "baseline": args.baseline,
                "question": sample["question"],
                "query_type": query_type,
                "top_k": args.k,
                "relevances": relevances,
                "hits": [
                    {
                        "rank": index,
                        "record_id": hit["record_id"],
                        "source_label": hit["source_label"],
                        "score": hit["score"],
                    }
                    for index, hit in enumerate(hits, start=1)
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    metrics = {
        "baseline": args.baseline,
        "evaluated_samples": evaluated,
        f"Recall@{args.k}": round(recall_hits / evaluated, 6) if evaluated else 0.0,
        "MRR": round(total_mrr / evaluated, 6) if evaluated else 0.0,
        f"nDCG@{args.k}": round(total_ndcg / evaluated, 6) if evaluated else 0.0,
        "details_path": str(details_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
