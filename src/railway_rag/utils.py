from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List


CJK_RE = re.compile(r"[\u3400-\u9fff]")
EN_RE = re.compile(r"[A-Za-z]")


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(data: object, path: str | Path) -> None:
    target = ensure_parent(path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    cleaned = text.replace("\u3000", " ").replace("\xa0", " ")
    cleaned = cleaned.replace("\t", " ")
    cleaned = re.sub(r"[·•]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def has_english(text: str) -> bool:
    return bool(EN_RE.search(text))


def classify_text(text: str) -> str:
    cjk_count = len(CJK_RE.findall(text))
    en_count = len(EN_RE.findall(text))
    if cjk_count and not en_count:
        return "zh"
    if en_count and not cjk_count:
        return "en"
    if cjk_count >= en_count:
        return "mixed_zh"
    if en_count > cjk_count:
        return "mixed_en"
    return "other"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[。！？!?;；.])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def simple_query_terms(text: str) -> List[str]:
    normalized = normalize_text(text).lower()
    if has_cjk(normalized):
        chars = [normalized[i : i + 2] for i in range(max(len(normalized) - 1, 0))]
        return [item for item in chars if item.strip()]
    return re.findall(r"[a-z0-9]+", normalized)

