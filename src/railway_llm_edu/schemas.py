from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal


class DumpMixin:
    def model_dump(self, mode: str = "python") -> dict:
        return asdict(self)

    def model_copy(self, update: dict | None = None):
        return replace(self, **(update or {}))


@dataclass
class RawParagraph(DumpMixin):
    doc_id: str
    source_path: str
    paragraph_id: str
    text: str
    style: str | None = None
    table_id: str | None = None
    row_id: int | None = None


@dataclass
class TextChunk(DumpMixin):
    chunk_id: str
    doc_id: str
    source_path: str
    text: str
    language: Literal["zh", "en", "zh_en", "unknown"] = "unknown"
    title: str | None = None
    article_no: str | None = None
    paragraph_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class TermEntry(DumpMixin):
    term_id: str
    zh: str
    en: str
    source_path: str
    aliases: list[str] = field(default_factory=list)
    note: str | None = None


@dataclass
class InstructionSample(DumpMixin):
    id: str
    messages: list[dict[str, str]]
    task_type: str
    source_ids: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalHit(DumpMixin):
    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)
