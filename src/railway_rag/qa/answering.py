from __future__ import annotations

from typing import Dict, List

from railway_rag.utils import has_cjk, normalize_text, simple_query_terms, split_sentences


def _rerank_hits(query: str, hits: List[Dict[str, object]]) -> List[Dict[str, object]]:
    query_terms = simple_query_terms(query)
    if not query_terms:
        return hits

    ranked = []
    for hit in hits:
        haystack = normalize_text(
            " ".join(
                [
                    str(hit.get("section_path", "")),
                    str(hit.get("zh_text", "")),
                    str(hit.get("en_text", "")),
                    str(hit.get("term_zh", "")),
                    str(hit.get("term_en", "")),
                ]
            )
        ).lower()
        unique_overlap = len({term for term in query_terms if term and term in haystack})
        total_overlap = sum(haystack.count(term) for term in query_terms if term)
        length_bonus = min(len(haystack), 200) / 200.0
        combined = unique_overlap * 2.0 + total_overlap * 0.2 + float(hit.get("score", 0.0)) + length_bonus
        ranked.append((combined, hit))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked]


def _best_snippet(query: str, text: str) -> str:
    sentences = split_sentences(text) or [text]
    query_terms = simple_query_terms(query)
    if not query_terms:
        return sentences[0]

    scored = []
    for sentence in sentences:
        sentence_norm = normalize_text(sentence).lower()
        score = sum(sentence_norm.count(term) for term in query_terms)
        scored.append((score, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _select_glossary_hit(query: str, hits: List[Dict[str, object]]) -> Dict[str, object] | None:
    query_norm = normalize_text(query).lower()
    glossary_hits = [hit for hit in hits if hit.get("record_type") == "glossary_term"]
    if not glossary_hits:
        return None

    ranked = []
    for hit in glossary_hits:
        zh_term = normalize_text(str(hit.get("term_zh", "")))
        en_term = normalize_text(str(hit.get("term_en", "")))
        exact_boost = 0
        if zh_term and zh_term.lower() in query_norm:
            exact_boost += 5
        if en_term and en_term.lower() in query_norm:
            exact_boost += 5
        category_bonus = 0.5 if hit.get("category_zh") and hit.get("category_zh") != "通用词汇" else 0.0
        compact_penalty = (len(zh_term) + len(en_term)) * 0.01
        score = exact_boost + category_bonus + float(hit.get("score", 0.0)) - compact_penalty
        ranked.append((score, hit))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def build_answer(query: str, hits: List[Dict[str, object]]) -> Dict[str, object]:
    if not hits:
        return {
            "answer": "未检索到相关知识片段，请尝试更具体的关键词。",
            "citations": [],
        }

    hits = _rerank_hits(query, hits)
    query_is_zh = has_cjk(query)
    top_hit = hits[0]

    glossary_hit = _select_glossary_hit(query, hits)
    if glossary_hit is not None and (
        top_hit["record_type"] == "glossary_term" or "怎么说" in query or "what does" in query.lower()
    ):
        top_hit = glossary_hit
        if query_is_zh:
            answer = f"术语“{top_hit['term_zh']}”对应英文为“{top_hit['term_en']}”。"
        else:
            answer = f'The term "{top_hit["term_en"]}" corresponds to "{top_hit["term_zh"]}".'
        if top_hit.get("category_zh"):
            answer = f"{answer} 术语类别：{top_hit['category_zh']}。"
        citations = [
            {
                "index": 1,
                "record_id": top_hit["record_id"],
                "source_label": top_hit["source_label"],
                "score": top_hit["score"],
            }
        ]
        return {"answer": answer, "citations": citations}

    snippets: List[str] = []
    citations: List[Dict[str, object]] = []
    for offset, hit in enumerate(hits[:3], start=1):
        preferred_text = hit["zh_text"] if query_is_zh else (hit["en_text"] or hit["zh_text"])
        fallback_text = hit["en_text"] if query_is_zh else hit["zh_text"]
        chosen_text = preferred_text or fallback_text
        snippet = _best_snippet(query, chosen_text)
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        citations.append(
            {
                "index": offset,
                "record_id": hit["record_id"],
                "source_label": hit["source_label"],
                "score": hit["score"],
            }
        )

    if query_is_zh:
        answer = "根据检索结果，相关内容如下：" + " ".join(snippets)
    else:
        answer = "Based on the retrieved knowledge, the most relevant content is: " + " ".join(snippets)

    return {"answer": answer, "citations": citations}
