from __future__ import annotations

import argparse

from railway_rag.config import load_config
from railway_rag.qa.answering import build_answer
from railway_rag.retrieval.vector_store import VectorStore
from railway_rag.utils import has_cjk, normalize_text


def infer_record_types(query: str) -> list[str] | None:
    lowered = query.lower()
    translation_markers = [
        "英文",
        "中文",
        "怎么说",
        "翻译",
        "对应",
        "what does",
        "translate",
        "in chinese",
        "in english",
    ]
    if any(marker in lowered for marker in translation_markers):
        return ["glossary_term"]
    return ["regulation_clause"]


def extract_translation_candidate(query: str) -> str:
    normalized = normalize_text(query)
    if has_cjk(normalized):
        candidate = normalized
        for marker in ["英文怎么说", "中文怎么说", "英文", "中文", "怎么说", "翻译", "对应"]:
            candidate = candidate.replace(marker, " ")
        return normalize_text(candidate).strip(" ?？")

    lowered = normalized.lower()
    for marker in [
        "what does",
        "what is",
        "mean in chinese",
        "mean in english",
        "in chinese",
        "in english",
        "translate",
        "meaning of",
    ]:
        lowered = lowered.replace(marker, " ")
    return normalize_text(lowered).strip(" ?？")


def exact_glossary_hits(vector_store: VectorStore, query: str) -> list[dict]:
    candidate = extract_translation_candidate(query)
    if not candidate:
        return []

    candidate_norm = candidate.lower()
    matches = []
    for record in vector_store.records:
        if record.get("record_type") != "glossary_term":
            continue
        zh_term = normalize_text(str(record.get("term_zh", "")))
        en_term = normalize_text(str(record.get("term_en", "")))
        zh_norm = zh_term.lower()
        en_norm = en_term.lower()

        exact = candidate_norm == zh_norm or candidate_norm == en_norm
        contains = candidate_norm in zh_norm or candidate_norm in en_norm or zh_norm in candidate_norm or en_norm in candidate_norm
        if not (exact or contains):
            continue

        score = 10.0 if exact else 8.0
        if record.get("category_zh") and record.get("category_zh") != "通用词汇":
            score += 0.5
        score -= (len(zh_term) + len(en_term)) * 0.01
        hit = dict(record)
        hit["score"] = round(score, 6)
        matches.append(hit)

    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches[:10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions against the multilingual railway RAG index.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--query", required=True, help="Question string.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k.")
    args = parser.parse_args()

    config = load_config(args.config)
    vector_store = VectorStore.load(config["paths"]["vector_index"])
    top_k = args.top_k or int(config["retrieval"].get("top_k", 5))
    record_types = infer_record_types(args.query)
    hits = vector_store.search(args.query, top_k=max(top_k * 8, 30), record_types=record_types)
    if record_types == ["glossary_term"]:
        exact_hits = exact_glossary_hits(vector_store, args.query)
        seen_ids = {hit["record_id"] for hit in exact_hits}
        hits = exact_hits + [hit for hit in hits if hit["record_id"] not in seen_ids]
    answer = build_answer(args.query, hits)

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
