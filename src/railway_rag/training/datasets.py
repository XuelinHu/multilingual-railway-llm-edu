from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: str | Path) -> List[Dict[str, object]]:
    file_path = Path(path)
    return [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def format_instruction_sample(sample: Dict[str, object]) -> str:
    instruction = str(sample.get("instruction", "")).strip()
    question = str(sample.get("question", "")).strip()
    answer = str(sample.get("answer", "")).strip()
    return (
        "<|system|>\n"
        "你是一个面向铁路牵引供电问答的高可信助手，需要优先保持术语规范、规章依据和风险提示。\n"
        "<|user|>\n"
        f"{instruction}\n\n问题：{question}\n"
        "<|assistant|>\n"
        f"{answer}"
    )


def build_text_dataset(path: str | Path) -> List[Dict[str, str]]:
    rows = load_jsonl(path)
    return [{"text": format_instruction_sample(row)} for row in rows]

