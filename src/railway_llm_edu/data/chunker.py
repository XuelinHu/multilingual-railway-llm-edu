from __future__ import annotations

import re

from railway_llm_edu.schemas import RawParagraph, TextChunk

ARTICLE_RE = re.compile(
    r"^(\u7b2c[\u4e00-\u9fff\d]+[\u7ae0\u8282\u6761\u6b3e]|Article\s+\d+|\d+[\.\u3001])"
)
ZH_RE = re.compile(r"[\u4e00-\u9fff]")
EN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    has_zh = bool(ZH_RE.search(text))
    has_en = bool(EN_RE.search(text))
    if has_zh and has_en:
        return "zh_en"
    if has_zh:
        return "zh"
    if has_en:
        return "en"
    return "unknown"


def _article_no(text: str) -> str | None:
    match = ARTICLE_RE.match(text.strip())
    return match.group(1) if match else None


def build_chunks(
    paragraphs: list[RawParagraph],
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    bucket: list[RawParagraph] = []
    current_len = 0
    chunk_idx = 0

    def flush() -> None:
        nonlocal bucket, current_len, chunk_idx
        if not bucket:
            return
        text = "\n".join(p.text for p in bucket)
        first_article = next((_article_no(p.text) for p in bucket if _article_no(p.text)), None)
        chunks.append(
            TextChunk(
                chunk_id=f"{bucket[0].doc_id}-c{chunk_idx:05d}",
                doc_id=bucket[0].doc_id,
                source_path=bucket[0].source_path,
                text=text,
                language=detect_language(text),
                article_no=first_article,
                paragraph_ids=[p.paragraph_id for p in bucket],
            )
        )
        chunk_idx += 1
        if overlap_chars > 0:
            tail: list[RawParagraph] = []
            tail_len = 0
            for p in reversed(bucket):
                if tail_len + len(p.text) > overlap_chars:
                    break
                tail.insert(0, p)
                tail_len += len(p.text)
            bucket = tail
            current_len = tail_len
        else:
            bucket = []
            current_len = 0

    for para in paragraphs:
        starts_article = bool(_article_no(para.text))
        if bucket and (current_len + len(para.text) > max_chars or starts_article):
            flush()
        bucket.append(para)
        current_len += len(para.text)
    flush()
    return chunks
