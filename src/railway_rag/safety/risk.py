from __future__ import annotations

from typing import Dict, List

from railway_rag.utils import normalize_text


HIGH_RISK_MARKERS = [
    "带电",
    "停电",
    "送电",
    "倒闸",
    "验电",
    "接地",
    "高压",
    "接触网",
    "牵引变电所",
    "应急处置",
    "抢修",
    "故障处理",
    "调度命令",
    "授权",
]

MEDIUM_RISK_MARKERS = [
    "检修",
    "维修",
    "巡视",
    "巡检",
    "试验",
    "操作",
    "异常",
    "告警",
    "故障",
    "排查",
    "步骤",
    "流程",
]


def risk_check(query: str) -> Dict[str, object]:
    normalized = normalize_text(query).lower()
    matched_high = [marker for marker in HIGH_RISK_MARKERS if marker.lower() in normalized]
    matched_medium = [marker for marker in MEDIUM_RISK_MARKERS if marker.lower() in normalized]

    if matched_high:
        risk_level = "high"
    elif len(matched_medium) >= 2:
        risk_level = "medium"
    elif matched_medium:
        risk_level = "low"
    else:
        risk_level = "none"

    return {
        "risk_level": risk_level,
        "is_high_risk": risk_level == "high",
        "matched_markers": matched_high + matched_medium,
    }


def safety_notice(risk_result: Dict[str, object]) -> str:
    if risk_result["risk_level"] == "high":
        return "该问题涉及高风险供电作业场景，回答仅供知识检索与培训参考，现场必须以正式规章、调度命令、授权流程和持证作业要求为准。"
    if risk_result["risk_level"] == "medium":
        return "该问题涉及运维流程或故障排查，建议结合现场规章和作业许可制度执行。"
    return ""

