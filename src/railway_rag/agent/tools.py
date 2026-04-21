from __future__ import annotations

import re
from typing import Dict, List, Tuple

from railway_rag.retrieval.vector_store import VectorStore
from railway_rag.utils import has_cjk, normalize_text


def classify_query(query: str) -> str:
    normalized = normalize_text(query)
    lowered = normalized.lower()

    paper_markers = ["论文", "实验", "消融", "评测", "baseline", "ablation", "metric", "contribution"]
    term_markers = [
        "英文",
        "中文",
        "怎么说",
        "翻译",
        "术语",
        "全称",
        "缩写",
        "对应",
        "what does",
        "translate",
        "in chinese",
        "in english",
    ]
    fault_markers = ["故障", "异常", "告警", "排查", "原因", "先查", "diagnose", "alarm"]
    bilingual_markers = ["双语", "bilingual", "english answer", "英文回答", "中英"]
    procedure_markers = ["流程", "步骤", "巡检", "检修", "如何组织", "怎么做"]

    if any(marker in lowered for marker in paper_markers):
        return "paper"
    if any(marker in lowered for marker in bilingual_markers):
        return "bilingual"
    if any(marker in lowered for marker in term_markers):
        return "term"
    if any(marker in lowered for marker in fault_markers):
        return "fault"
    if any(marker in lowered for marker in procedure_markers):
        return "procedure"
    return "regulation"


def infer_record_types(query_type: str) -> List[str] | None:
    if query_type == "term":
        return ["glossary_term"]
    if query_type == "paper":
        return None
    return ["regulation_clause"]


def extract_translation_candidate(query: str) -> str:
    normalized = normalize_text(query)
    if has_cjk(normalized):
        candidate = normalized
        for marker in ["英文怎么说", "中文怎么说", "英文", "中文", "怎么说", "翻译", "术语", "全称", "缩写", "对应"]:
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
        "term",
        "abbreviation",
        "full form",
    ]:
        lowered = lowered.replace(marker, " ")
    return normalize_text(lowered).strip(" ?？")


def extract_term_candidates(query: str) -> List[str]:
    normalized = normalize_text(query)
    lowered = normalized.lower()
    for marker in ["是否对应", "是否等价", "是否规范", "英文是什么", "中文是什么", "怎么说", "翻译", "对应", "term", "translate"]:
        lowered = lowered.replace(marker, " ")
    splitter_patterns = [r"\s+and\s+", r"\s+or\s+", r"\s+vs\.?\s+", r"\s*[与和及/]\s*", r"\s*[,，;；]\s*"]
    candidates = [lowered]
    for pattern in splitter_patterns:
        next_candidates: List[str] = []
        for item in candidates:
            parts = re.split(pattern, item)
            next_candidates.extend(parts)
        candidates = next_candidates
    cleaned = []
    for candidate in candidates:
        normalized_candidate = normalize_text(candidate).strip(" ?？")
        if normalized_candidate and normalized_candidate not in cleaned:
            cleaned.append(normalized_candidate)
    return cleaned or [extract_translation_candidate(query)]


def _term_aliases(record: Dict[str, object]) -> List[str]:
    record_aliases = record.get("aliases") or []
    aliases = [
        normalize_text(str(record.get("term_zh", ""))),
        normalize_text(str(record.get("term_en", ""))),
        normalize_text(str(record.get("abbreviation", ""))),
        normalize_text(str(record.get("full_form", ""))),
        normalize_text(str(record.get("zh_text", ""))),
        normalize_text(str(record.get("en_text", ""))),
    ]
    aliases.extend(normalize_text(str(item)) for item in record_aliases)
    deduped: List[str] = []
    for item in aliases:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def search_term_dictionary(vector_store: VectorStore, query: str, top_k: int = 8) -> List[Dict[str, object]]:
    candidates = extract_term_candidates(query)
    candidate_norms = [candidate.lower() for candidate in candidates if candidate]
    matches: List[Dict[str, object]] = []

    for record in vector_store.records:
        if record.get("record_type") != "glossary_term":
            continue
        aliases = _term_aliases(record)
        alias_norms = [alias.lower() for alias in aliases]
        exact_matches = sum(1 for candidate_norm in candidate_norms for alias in alias_norms if candidate_norm and candidate_norm == alias)
        contains_matches = sum(
            1
            for candidate_norm in candidate_norms
            for alias in alias_norms
            if candidate_norm and (candidate_norm in alias or alias in candidate_norm)
        )
        exact = exact_matches > 0
        contains = contains_matches > 0
        if not (exact or contains):
            continue

        score = 10.0 if exact else 8.0
        score += exact_matches * 2.0 + contains_matches * 0.5
        if record.get("category_zh") and record.get("category_zh") != "通用词汇":
            score += 0.5
        score -= sum(len(alias) for alias in aliases[:2]) * 0.01
        hit = dict(record)
        hit["score"] = round(score, 6)
        matches.append(hit)

    if matches:
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:top_k]

    return vector_store.search(query, top_k=top_k, record_types=["glossary_term"])


def _extract_abbreviation(term_en: str) -> Tuple[str, str]:
    match = re.search(r"\(([^()]{2,16})\)", term_en)
    if match:
        return match.group(1).strip(), term_en[: match.start()].strip()
    tokens = re.findall(r"[A-Za-z]+", term_en)
    if len(tokens) >= 2:
        initialism = "".join(token[0].upper() for token in tokens if token)
        if len(initialism) >= 2:
            return initialism, term_en
    return "", term_en


def expand_query_with_terms(vector_store: VectorStore, query: str, top_k: int = 3) -> str:
    glossary_hits = search_term_dictionary(vector_store, query, top_k=top_k)
    expansions = [query]
    seen = {normalize_text(query).lower()}
    for hit in glossary_hits:
        for alias in _term_aliases(hit):
            alias_norm = alias.lower()
            if alias_norm in seen:
                continue
            seen.add(alias_norm)
            expansions.append(alias)
        abbreviation, full_form = _extract_abbreviation(str(hit.get("term_en", "")))
        for extra in [abbreviation, full_form]:
            extra = normalize_text(extra)
            if extra and extra.lower() not in seen:
                seen.add(extra.lower())
                expansions.append(extra)
    return " ".join(expansions)


def search_regulation(vector_store: VectorStore, query: str, top_k: int = 5) -> List[Dict[str, object]]:
    expanded_query = expand_query_with_terms(vector_store, query, top_k=3)
    raw_hits = vector_store.search(query, top_k=max(top_k * 4, 16), record_types=["regulation_clause"])
    expanded_hits = vector_store.search(expanded_query, top_k=max(top_k * 4, 16), record_types=["regulation_clause"])
    merged: List[Dict[str, object]] = []
    seen = set()
    for hit in expanded_hits + raw_hits:
        record_id = hit["record_id"]
        if record_id in seen:
            continue
        seen.add(record_id)
        hit["retrieval_trace"] = {
            "channel": "regulation",
            "matched_query": expanded_query if hit in expanded_hits else query,
            "risk_level": hit.get("risk_level", ""),
            "content_type": hit.get("content_type", ""),
            "evidence_priority": hit.get("evidence_priority", 0),
        }
        merged.append(hit)
    return merged


def dual_channel_retrieve(vector_store: VectorStore, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, object]]]:
    terminology_hits = search_term_dictionary(vector_store, query, top_k=max(3, top_k))
    regulation_hits = search_regulation(vector_store, query, top_k=top_k)
    return {
        "terminology": terminology_hits,
        "regulation": regulation_hits,
    }
