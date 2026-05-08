from __future__ import annotations

import re

from railway_llm_edu.schemas import RawParagraph

ZH_RE = re.compile(r"[\u4e00-\u9fff]")
EN_RE = re.compile(r"[A-Za-z]")


def split_bilingual_lines(text: str) -> tuple[str | None, str | None]:
    parts = [p.strip() for p in re.split(r"\t+|\s{2,}", text) if p.strip()]
    zh_parts = [p for p in parts if ZH_RE.search(p)]
    en_parts = [p for p in parts if EN_RE.search(p) and not ZH_RE.search(p)]
    if zh_parts and en_parts:
        return " ".join(zh_parts), " ".join(en_parts)
    return None, None


def align_adjacent_paragraphs(paragraphs: list[RawParagraph]) -> list[dict]:
    """Build lightweight zh-en paragraph pairs using inline splits and adjacent lines."""
    aligned: list[dict] = []
    for idx, para in enumerate(paragraphs):
        zh, en = split_bilingual_lines(para.text)
        if zh and en:
            aligned.append(
                {
                    "id": f"align-{para.paragraph_id}",
                    "zh": zh,
                    "en": en,
                    "source_ids": [para.paragraph_id],
                    "source_path": para.source_path,
                }
            )
            continue
        if idx + 1 >= len(paragraphs):
            continue
        nxt = paragraphs[idx + 1]
        if para.doc_id != nxt.doc_id:
            continue
        if ZH_RE.search(para.text) and EN_RE.search(nxt.text) and not ZH_RE.search(nxt.text):
            aligned.append(
                {
                    "id": f"align-{para.paragraph_id}-{nxt.paragraph_id}",
                    "zh": para.text,
                    "en": nxt.text,
                    "source_ids": [para.paragraph_id, nxt.paragraph_id],
                    "source_path": para.source_path,
                }
            )
    return aligned
