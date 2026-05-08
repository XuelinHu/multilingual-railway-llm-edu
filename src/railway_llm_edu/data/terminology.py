from __future__ import annotations

import hashlib
import re

from railway_llm_edu.schemas import RawParagraph, TermEntry

ZH_CHARS = r"\u4e00-\u9fff"
ZH_TERM_CHARS = ZH_CHARS + r"\u3001\uff08\uff09()/\u00b7\-\s"
EN_TERM_CHARS = r"A-Za-z0-9 ,()/\u00b7\-\s"
TAB_PAIR_RE = re.compile(rf"^(?P<zh>[{ZH_TERM_CHARS}]+)\t+(?P<en>[A-Za-z][{EN_TERM_CHARS}]+)$")
INLINE_PAIR_RE = re.compile(
    rf"(?P<zh>[{ZH_CHARS}][{ZH_TERM_CHARS}]{{1,30}})\s+(?P<en>[A-Za-z][{EN_TERM_CHARS}]{{2,80}})"
)


def _term_id(zh: str, en: str) -> str:
    return hashlib.md5(f"{zh}|{en}".encode("utf-8")).hexdigest()[:12]


def _clean_side(value: str) -> str:
    return " ".join(value.replace("\t", " ").split()).strip(" \uff1a:;\uff1b,\uff0c")


def extract_terms(paragraphs: list[RawParagraph]) -> list[TermEntry]:
    """Extract bilingual term pairs from glossary-like paragraphs and tables."""
    terms: dict[str, TermEntry] = {}
    for para in paragraphs:
        candidates: list[tuple[str, str]] = []
        if match := TAB_PAIR_RE.match(para.text):
            candidates.append((match.group("zh"), match.group("en")))
        elif "\t" in para.text:
            parts = [_clean_side(p) for p in para.text.split("\t") if _clean_side(p)]
            if len(parts) >= 2:
                candidates.append((parts[0], parts[1]))
        else:
            for match in INLINE_PAIR_RE.finditer(para.text):
                candidates.append((match.group("zh"), match.group("en")))

        for zh, en in candidates:
            zh, en = _clean_side(zh), _clean_side(en)
            if not zh or not en or len(zh) > 50 or len(en) > 120:
                continue
            term = TermEntry(
                term_id=_term_id(zh, en),
                zh=zh,
                en=en,
                source_path=para.source_path,
            )
            terms[term.term_id] = term
    return sorted(terms.values(), key=lambda x: (x.zh, x.en))
