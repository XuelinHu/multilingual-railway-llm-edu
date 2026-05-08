from __future__ import annotations

import re
from typing import Dict, List

from railway_rag.qa.answering import _best_snippet, _rerank_hits
from railway_rag.safety.risk import safety_notice
from railway_rag.utils import has_cjk, normalize_text


def _format_citations(hits: List[Dict[str, object]], limit: int = 3) -> List[Dict[str, object]]:
    citations: List[Dict[str, object]] = []
    for index, hit in enumerate(hits[:limit], start=1):
        citations.append(
            {
                "index": index,
                "record_id": hit["record_id"],
                "source_label": hit["source_label"],
                "score": hit["score"],
            }
        )
    return citations


def _format_trace(hit: Dict[str, object]) -> str:
    trace = hit.get("retrieval_trace") or {}
    parts = [
        f"channel={trace.get('channel', hit.get('knowledge_channel', 'unknown'))}",
        f"content_type={hit.get('content_type', '')}",
        f"risk_level={hit.get('risk_level', '')}",
        f"evidence_priority={hit.get('evidence_priority', 0)}",
    ]
    return ", ".join(part for part in parts if not part.endswith("="))


def _extract_abbreviation(term_en: str) -> tuple[str, str]:
    match = re.search(r"\(([^()]{2,16})\)", term_en)
    if match:
        return match.group(1).strip(), term_en[: match.start()].strip()
    tokens = re.findall(r"[A-Za-z]+", term_en)
    if len(tokens) >= 2:
        initialism = "".join(token[0].upper() for token in tokens if token)
        if len(initialism) >= 2:
            return initialism, term_en
    return "", term_en


def format_term_answer(query: str, hits: List[Dict[str, object]], risk_result: Dict[str, object]) -> Dict[str, object]:
    if not hits:
        return {"answer": "当前术语词典中未检索到充分依据。", "citations": []}

    hit = hits[0]
    abbreviation, full_form = _extract_abbreviation(str(hit.get("term_en", "")))
    notes = safety_notice(risk_result)
    answer = "\n".join(
        [
            "1. 中文术语",
            f"{hit.get('term_zh', '未检索到')}",
            "2. 英文术语",
            f"{hit.get('term_en', '未检索到')}",
            "3. 缩略语 / 全称",
            f"{abbreviation or '未标注'} / {full_form or '未标注'}",
            "4. 定义说明",
            f"该术语收录于《{hit.get('source_title', '')}》术语词典，可作为标准表达使用。",
            "5. 在铁路供电场景中的用途",
            f"术语类别：{hit.get('category_zh', '未分类')}。别名：{' / '.join(hit.get('aliases', [])[:4]) or '未标注'}。",
            "6. 风险提示",
            notes or "术语解释类问题风险较低，但正式文档翻译应保持全文一致。",
        ]
    )
    return {"answer": answer, "citations": _format_citations(hits, limit=1)}


def format_regulation_answer(query: str, hits: List[Dict[str, object]], risk_result: Dict[str, object]) -> Dict[str, object]:
    if not hits:
        answer = "\n".join(
            [
                "1. 问题结论",
                "当前知识库中未检索到充分依据。",
                "2. 规章依据",
                "暂无可用条文。",
                "3. 关键条款摘要",
                "请换用更具体的设备、流程或条款关键词。",
                "4. 适用条件 / 限制条件",
                "当前无法确认。",
                "5. 风险提示",
                safety_notice(risk_result) or "规章类问题应以正式制度文本为准。",
            ]
        )
        return {"answer": answer, "citations": []}

    ranked_hits = _rerank_hits(query, hits)
    top_hit = ranked_hits[0]
    combined_text = " ".join(str(hit.get("zh_text") or hit.get("en_text") or "") for hit in ranked_hits[:3])
    allow_markers = ["是否允许", "允许", "可否", "能否", "可以吗"]
    explicit_policy_markers = ["允许进行", "可以进行", "不得", "禁止", "严禁", "必须", "应当", "可以", "允许"]
    topic_markers = ["带电", "停电", "送电", "倒闸", "接地", "验电", "作业", "检修"]
    lacks_explicit_policy = any(marker in query for marker in allow_markers) and not any(
        marker in combined_text for marker in explicit_policy_markers
    )
    if any(marker in query for marker in allow_markers):
        topic_overlap = [marker for marker in topic_markers if marker in query and marker in combined_text]
        if not topic_overlap:
            lacks_explicit_policy = True
    evidence = []
    for hit in ranked_hits[:3]:
        snippet = _best_snippet(query, str(hit.get("zh_text") or hit.get("en_text") or ""))
        if snippet:
            evidence.append(f"- {snippet}")

    conclusion = _best_snippet(query, str(top_hit.get("zh_text") or top_hit.get("en_text") or ""))
    if lacks_explicit_policy:
        conclusion = "当前检索到的规章片段未直接给出“允许/禁止”的明确结论，不能据此判断该操作是否可执行。"

    answer = "\n".join(
        [
            "1. 问题结论",
            conclusion,
            "2. 规章依据",
            top_hit.get("source_label", "未标注来源"),
            "3. 条文摘要",
            "\n".join(evidence),
            "4. 适用条件 / 限制条件",
            normalize_text(str(top_hit.get("section_path", ""))) or "请结合具体章节场景判断。",
            "5. 限制条件",
            "若条文未直接覆盖具体作业条件，则不得将该回答视为现场操作授权。" if risk_result["is_high_risk"] else "应结合更具体的设备状态、章节语境和现场流程判断。",
            "6. 证据类型说明",
            f"直接证据：{top_hit.get('content_type', '未标注')}；检索轨迹：{_format_trace(top_hit)}。",
            "7. 推断说明",
            "除直接摘录条文外，其余归纳内容仅是对检索片段的整理，不应替代正式制度解释。",
            "8. 风险提示",
            safety_notice(risk_result) or "回答基于检索到的规章片段，未检索到的制度要求不应擅自补充。",
        ]
    )
    return {"answer": answer, "citations": _format_citations(ranked_hits)}


def format_fault_answer(query: str, hits: List[Dict[str, object]], risk_result: Dict[str, object]) -> Dict[str, object]:
    ranked_hits = _rerank_hits(query, hits)
    snippets = [
        _best_snippet(query, str(hit.get("zh_text") or hit.get("en_text") or "")) for hit in ranked_hits[:3]
    ]
    snippets = [snippet for snippet in snippets if snippet]
    answer = "\n".join(
        [
            "1. 可能原因",
            snippets[0] if snippets else "当前知识库中未检索到充分依据。",
            "2. 优先排查项",
            "\n".join(f"- {item}" for item in snippets[1:]) or "- 先核对相关规章、设备状态和告警上下文。",
            "3. 相关设备 / 环节",
            normalize_text(str(ranked_hits[0].get("section_path", ""))) if ranked_hits else "未检索到",
            "4. 是否有规章依据",
            f"有，主证据类型为{ranked_hits[0].get('content_type', '未标注')}。" if ranked_hits else "暂无。",
            "5. 推断说明",
            "可能原因与排查路径是基于检索片段整理的辅助分析，若条文未明确规定，应视为经验性推断。",
            "6. 现场处置提示",
            safety_notice(risk_result) or "故障分析仅作辅助判断，不替代现场组织措施。",
        ]
    )
    return {"answer": answer, "citations": _format_citations(ranked_hits)}


def format_bilingual_answer(query: str, hits: List[Dict[str, object]], risk_result: Dict[str, object]) -> Dict[str, object]:
    if not hits:
        return {"answer": "当前知识库中未检索到充分依据。", "citations": []}

    hit = hits[0]
    zh_answer = str(hit.get("zh_text") or hit.get("term_zh") or "")
    en_answer = str(hit.get("en_text") or hit.get("term_en") or "")
    notes = safety_notice(risk_result) or "英文表达优先采用术语词典中的标准写法。"
    answer = "\n".join(
        [
            "1. 中文答案",
            zh_answer,
            "2. English Answer",
            en_answer,
            "3. Key Terms",
            f"{hit.get('term_zh') or zh_answer} / {hit.get('term_en') or en_answer}",
            "4. Terminology Notes",
            notes,
        ]
    )
    return {"answer": answer, "citations": _format_citations(hits, limit=1)}


def format_paper_answer(query: str) -> Dict[str, object]:
    answer = "\n".join(
        [
            "1. 研究问题",
            "RQ1 术语增强是否提升中英术语理解与召回；RQ2 规章约束是否降低幻觉并提升可追溯性；RQ3 风险感知是否减少危险性错误建议；RQ4 3090 单卡下的轻量方案是否兼顾性能与成本。",
            "2. 方法框架",
            "采用双通道知识融合框架：术语词典通道负责术语归一化与扩展，规章通道负责条文证据检索，之后进行 evidence-aware rerank、风险校准和模板化输出。",
            "3. 创新点",
            "突出术语增强检索、规章证据优先级建模、风险感知保守回答、3090 单卡轻量落地，以及铁路供电 QA benchmark 构建。",
            "4. 实验设计",
            "围绕术语问答、规章问答、流程问答、故障分析、双语问答和高风险拒答构造训练集与评测集，并记录检索日志与错误案例。",
            "5. 对比基线",
            "Base LLM、Base+RAG、Base+RAG+reranker、Base+RAG+term enhancement、Base+RAG+term+LoRA、Base+RAG+term+LoRA+risk-aware calibration。",
            "6. 消融设置",
            "去掉术语增强、去掉 reranker、去掉规章优先检索、去掉风险控制、去掉 LoRA、混合知识库替代双通道知识库。",
            "7. 结果分析",
            "重点从准确性、可追溯性、安全性三条主线分析，并加入缩略语提问、中英混合提问、模糊问法和不可回答问题的鲁棒性分析。",
            "8. 局限性",
            "当前语料仍集中于单套规章与词汇表，现场案例、跨文档冲突处理和人工标注 benchmark 仍需扩展。",
        ]
    )
    return {"answer": answer, "citations": []}


def format_answer(query_type: str, query: str, hits: List[Dict[str, object]], risk_result: Dict[str, object]) -> Dict[str, object]:
    if query_type == "term":
        return format_term_answer(query, hits, risk_result)
    if query_type == "fault":
        return format_fault_answer(query, hits, risk_result)
    if query_type == "bilingual":
        return format_bilingual_answer(query, hits, risk_result)
    if query_type == "paper":
        return format_paper_answer(query)
    return format_regulation_answer(query, hits, risk_result)
