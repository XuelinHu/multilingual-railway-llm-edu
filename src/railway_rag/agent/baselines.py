from __future__ import annotations

from typing import Dict, List

from railway_rag.agent.tools import classify_query, search_regulation, search_term_dictionary
from railway_rag.retrieval.vector_store import VectorStore


BASELINE_PRESETS: Dict[str, Dict[str, object]] = {
    "retrieval_only": {
        "use_query_classifier": False,
        "use_dual_channel": False,
        "use_term_dictionary": False,
        "use_term_expansion": False,
        "use_rerank": False,
        "use_risk_calibration": False,
    },
    "rag_baseline": {
        "use_query_classifier": True,
        "use_dual_channel": False,
        "use_term_dictionary": True,
        "use_term_expansion": False,
        "use_rerank": False,
        "use_risk_calibration": False,
    },
    "rag_rerank": {
        "use_query_classifier": True,
        "use_dual_channel": False,
        "use_term_dictionary": True,
        "use_term_expansion": False,
        "use_rerank": True,
        "use_risk_calibration": False,
    },
    "rag_term": {
        "use_query_classifier": True,
        "use_dual_channel": True,
        "use_term_dictionary": True,
        "use_term_expansion": True,
        "use_rerank": True,
        "use_risk_calibration": False,
    },
    "rag_term_risk": {
        "use_query_classifier": True,
        "use_dual_channel": True,
        "use_term_dictionary": True,
        "use_term_expansion": True,
        "use_rerank": True,
        "use_risk_calibration": True,
    },
}


def get_baseline_options(name: str) -> Dict[str, object]:
    if name not in BASELINE_PRESETS:
        raise ValueError(f"Unsupported baseline: {name}")
    return dict(BASELINE_PRESETS[name])


def available_baselines() -> List[str]:
    return list(BASELINE_PRESETS.keys())


def retrieve_hits(
    vector_store: VectorStore,
    query: str,
    query_type: str,
    top_k: int,
    baseline_name: str,
) -> Dict[str, List[Dict[str, object]]]:
    options = get_baseline_options(baseline_name)
    channels: Dict[str, List[Dict[str, object]]] = {"terminology": [], "regulation": []}

    if not options["use_query_classifier"]:
        channels["regulation"] = vector_store.search(query, top_k=max(top_k * 4, 16), record_types=["regulation_clause"])
        return channels

    if options["use_term_dictionary"]:
        channels["terminology"] = search_term_dictionary(vector_store, query, top_k=max(3, top_k))

    if options["use_term_expansion"]:
        channels["regulation"] = search_regulation(vector_store, query, top_k=top_k)
    else:
        channels["regulation"] = vector_store.search(query, top_k=max(top_k * 4, 16), record_types=["regulation_clause"])

    if not options["use_dual_channel"]:
        channels["terminology"] = channels["terminology"] if query_type in {"term", "bilingual"} else []

    return channels

