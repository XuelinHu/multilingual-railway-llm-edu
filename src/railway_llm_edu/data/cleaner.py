from __future__ import annotations

import re

from railway_llm_edu.schemas import RawParagraph
from railway_llm_edu.utils import normalize_space

CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]+")
PAGE_NO_RE = re.compile(r"^\s*[-\u2014]?\s*\d+\s*[-\u2014]?\s*$")


def clean_text(text: str) -> str:
    text = CONTROL_RE.sub("", text)
    text = normalize_space(text)
    text = re.sub(r"\s+([,.;:!?\u3002\uff0c\uff1b\uff1a\uff01\uff1f])", r"\1", text)
    return text.strip()


def clean_paragraphs(paragraphs: list[RawParagraph], min_chars: int = 2) -> list[RawParagraph]:
    cleaned: list[RawParagraph] = []
    seen: set[tuple[str, str]] = set()
    for para in paragraphs:
        text = clean_text(para.text)
        if len(text) < min_chars or PAGE_NO_RE.match(text):
            continue
        key = (para.doc_id, text)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(para.model_copy(update={"text": text}))
    return cleaned
