from __future__ import annotations

import re
from typing import Dict, List, Tuple

from railway_rag.parsers.docx_reader import read_docx_blocks
from railway_rag.utils import classify_text, has_cjk, has_english, normalize_text


PART_ZH_RE = re.compile(r"^第[一二三四五六七八九十百零]+部分")
PART_EN_RE = re.compile(r"^Part\s+[IVXLC\d]+", re.IGNORECASE)
CHAPTER_ZH_RE = re.compile(r"^第[一二三四五六七八九十百零]+章")
CHAPTER_EN_RE = re.compile(r"^Chapter\s+[IVXLC\d]+", re.IGNORECASE)
APPENDIX_ZH_RE = re.compile(r"^附件\s*\d+")
APPENDIX_EN_RE = re.compile(r"^(Attachment|Appendix)\s*\d+", re.IGNORECASE)
NOISE_RE = re.compile(r"^[,.'`“”‘’．。()\-\sI]+$")
GLOSSARY_PAIR_RE = re.compile(
    r"([\u3400-\u9fff][\u3400-\u9fffA-Za-z0-9（）()/\-、，,．. ]*?)\s+([A-Za-z(][A-Za-z0-9 /;,\-().:+]*?)(?=\s+[\u3400-\u9fff]|$)"
)
INDEX_MARKER_RE = re.compile(r"^Z\s*\d+\s*汉英索引$")

GLOSSARY_CATEGORY_ANCHORS: List[Tuple[str, str]] = [
    ("通用词汇", "General vocabulary"),
    ("通 信", "Communication"),
    ("信 号", "Signaling"),
    ("机 车", "Locomotive"),
    ("车 辆", "Rolling stock"),
    ("牵引供电", "Traction power supply"),
    ("工务工程", "Permanent way"),
    ("运输与经济", "Transportation and economy"),
    ("相关科学", "Related sciences"),
]


def _clean_heading_suffix(text: str) -> str:
    text = re.sub(r"\.{3,}\s*\d+$", "", text)
    text = re.sub(r"\s+\d+$", "", text)
    return normalize_text(text)


def _is_regulation_heading(text: str) -> bool:
    return bool(
        PART_ZH_RE.match(text)
        or PART_EN_RE.match(text)
        or CHAPTER_ZH_RE.match(text)
        or CHAPTER_EN_RE.match(text)
        or APPENDIX_ZH_RE.match(text)
        or APPENDIX_EN_RE.match(text)
    )


def _is_likely_noise(text: str) -> bool:
    if not text or NOISE_RE.match(text):
        return True
    if text in {"目录", "目 录", "Purpose record"}:
        return True
    if "\t" in text and re.search(r"\d+$", text):
        return True
    return False


def _collect_regulation_lines(file_path: str) -> List[str]:
    blocks = read_docx_blocks(file_path)
    lines: List[str] = []
    for block in blocks:
        if block.block_type == "paragraph":
            cleaned = _clean_heading_suffix(block.text)
            if cleaned:
                lines.append(cleaned)
    return lines


def build_regulation_records(document_config: Dict[str, str]) -> List[Dict[str, str]]:
    lines = _collect_regulation_lines(document_config["file"])
    source_id = document_config["source_id"]
    source_title = document_config["title"]

    start_index = None
    for idx, line in enumerate(lines):
        if line == "第一部分 总 则":
            start_index = idx
            break
    if start_index is None:
        raise ValueError("Failed to locate regulation body start.")

    content_lines = [line for line in lines[start_index:] if not _is_likely_noise(line)]

    records: List[Dict[str, str]] = []
    current_part_zh = ""
    current_part_en = ""
    current_chapter_zh = ""
    current_chapter_en = ""
    current_appendix_zh = ""
    current_appendix_en = ""
    clause_counter = 0
    idx = 0

    while idx < len(content_lines):
        line = content_lines[idx]
        next_line = content_lines[idx + 1] if idx + 1 < len(content_lines) else ""

        if PART_ZH_RE.match(line):
            current_part_zh = line
            current_part_en = next_line if PART_EN_RE.match(next_line) else ""
            current_chapter_zh = ""
            current_chapter_en = ""
            current_appendix_zh = ""
            current_appendix_en = ""
            idx += 2 if current_part_en else 1
            continue

        if CHAPTER_ZH_RE.match(line):
            current_chapter_zh = line
            current_chapter_en = next_line if CHAPTER_EN_RE.match(next_line) else ""
            idx += 2 if current_chapter_en else 1
            continue

        if APPENDIX_ZH_RE.match(line):
            current_appendix_zh = line
            current_appendix_en = next_line if APPENDIX_EN_RE.match(next_line) else ""
            idx += 2 if current_appendix_en else 1
            continue

        if _is_regulation_heading(line):
            idx += 1
            continue

        lang = classify_text(line)
        if lang not in {"zh", "mixed_zh"}:
            idx += 1
            continue

        zh_parts = [line]
        idx += 1
        while idx < len(content_lines):
            candidate = content_lines[idx]
            if _is_regulation_heading(candidate):
                break
            if classify_text(candidate) in {"zh", "mixed_zh"}:
                zh_parts.append(candidate)
                idx += 1
                continue
            break

        en_parts: List[str] = []
        while idx < len(content_lines):
            candidate = content_lines[idx]
            if _is_regulation_heading(candidate):
                break
            if classify_text(candidate) in {"en", "mixed_en"}:
                en_parts.append(candidate)
                idx += 1
                continue
            break

        zh_text = normalize_text(" ".join(zh_parts))
        en_text = normalize_text(" ".join(en_parts))
        if not zh_text:
            continue
        zh_compact = re.sub(r"\s+", "", zh_text)
        en_compact = re.sub(r"\s+", "", en_text)
        if len(zh_compact) < 6 and len(en_compact) < 20:
            continue

        clause_counter += 1
        section_path = " / ".join(
            part
            for part in [current_part_zh, current_chapter_zh or current_appendix_zh]
            if part
        )
        source_label = " > ".join(
            item
            for item in [source_title, current_part_zh, current_chapter_zh or current_appendix_zh, f"条款 {clause_counter}"]
            if item
        )
        records.append(
            {
                "record_id": f"{source_id}_clause_{clause_counter:04d}",
                "record_type": "regulation_clause",
                "source_id": source_id,
                "source_title": source_title,
                "source_file": document_config["file"],
                "section_part_zh": current_part_zh,
                "section_part_en": current_part_en,
                "section_chapter_zh": current_chapter_zh,
                "section_chapter_en": current_chapter_en,
                "appendix_zh": current_appendix_zh,
                "appendix_en": current_appendix_en,
                "section_path": section_path,
                "clause_index": clause_counter,
                "zh_text": zh_text,
                "en_text": en_text,
                "term_zh": "",
                "term_en": "",
                "category_zh": "",
                "category_en": "",
                "source_label": source_label,
                "retrieval_text": normalize_text(" ".join([source_title, section_path, zh_text, en_text])),
            }
        )

    if not records:
        raise ValueError("No regulation clauses parsed from source document.")
    return records


def _normalize_glossary_anchor_key(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text)).lower()


def _match_glossary_category(line: str) -> Tuple[str, str] | None:
    normalized_line = _normalize_glossary_anchor_key(line)
    for zh_name, en_name in GLOSSARY_CATEGORY_ANCHORS:
        zh_key = _normalize_glossary_anchor_key(zh_name)
        en_key = _normalize_glossary_anchor_key(en_name)
        anchor_key = zh_key + en_key
        repeated_anchor_key = anchor_key * 2
        if normalized_line in {anchor_key, repeated_anchor_key}:
            return normalize_text(zh_name), normalize_text(en_name)
    return None


def _prepare_glossary_lines(file_path: str) -> List[Tuple[int, str]]:
    blocks = read_docx_blocks(file_path)
    paragraphs = [normalize_text(block.text) for block in blocks if block.block_type == "paragraph" and block.text]

    start_index = 0
    for idx, line in enumerate(paragraphs):
        if "通用词汇" in line and "General" in line:
            start_index = idx
            break

    merged: List[Tuple[int, str]] = []
    for idx, line in enumerate(paragraphs[start_index:], start=start_index):
        if not line or _is_likely_noise(line):
            continue
        if merged and not has_cjk(line):
            line_no, previous = merged[-1]
            merged[-1] = (line_no, normalize_text(f"{previous} {line}"))
            continue
        merged.append((idx + 1, line))
    return merged


def _split_glossary_pairs(line: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for zh_term, en_term in GLOSSARY_PAIR_RE.findall(line):
        zh_clean = normalize_text(zh_term)
        en_clean = normalize_text(en_term)
        zh_compact = re.sub(r"\s+", "", zh_clean)
        if (
            zh_clean
            and en_clean
            and len(zh_compact) >= 2
            and has_english(en_clean)
            and en_clean not in {"(", ")", "/", ";", ":"}
        ):
            pairs.append((zh_clean, en_clean))
    return pairs


def _is_low_quality_glossary_pair(term_zh: str, term_en: str) -> bool:
    zh_compact = re.sub(r"\s+", "", term_zh)
    en_compact = re.sub(r"\s+", "", term_en)
    if len(zh_compact) < 2 or len(en_compact) < 2:
        return True
    if INDEX_MARKER_RE.match(zh_compact):
        return True
    if "索引" in zh_compact:
        return True
    if "续上表" in zh_compact:
        return True
    if "常用缩写" in zh_compact:
        return True
    if "相关国际组织名称" in zh_compact:
        return True
    if "常用缩写" in zh_compact and "abbreviation" in en_compact.lower():
        return True
    if en_compact in {"a", "an", "the"}:
        return True
    if re.fullmatch(r"[A-Za-z]", en_compact):
        return True
    if re.search(r"[^\w\s/;,\-().:+]", term_en):
        return True
    if en_compact.count("(") != en_compact.count(")"):
        return True
    return False


def build_glossary_records(document_config: Dict[str, str]) -> List[Dict[str, str]]:
    lines = _prepare_glossary_lines(document_config["file"])
    source_id = document_config["source_id"]
    source_title = document_config["title"]

    current_category_zh = "通用词汇"
    current_category_en = "General vocabulary"
    term_counter = 0
    records: List[Dict[str, str]] = []

    for _line_no, line in lines:
        if INDEX_MARKER_RE.match(re.sub(r"\s+", "", line)):
            break
        if "附件1 常用缩写" in line or "附件2相关国际组织名称" in line or "相关国际组织名称" in line:
            break
        category = _match_glossary_category(line)
        if category is not None:
            current_category_zh, current_category_en = category
            continue

        for term_zh, term_en in _split_glossary_pairs(line):
            if _is_low_quality_glossary_pair(term_zh, term_en):
                continue
            term_counter += 1
            source_label = " > ".join(
                item for item in [source_title, current_category_zh or "未分类术语", f"术语 {term_counter}"] if item
            )
            records.append(
                {
                    "record_id": f"{source_id}_term_{term_counter:05d}",
                    "record_type": "glossary_term",
                    "source_id": source_id,
                    "source_title": source_title,
                    "source_file": document_config["file"],
                    "section_part_zh": "",
                    "section_part_en": "",
                    "section_chapter_zh": "",
                    "section_chapter_en": "",
                    "appendix_zh": "",
                    "appendix_en": "",
                    "section_path": current_category_zh,
                    "clause_index": "",
                    "zh_text": term_zh,
                    "en_text": term_en,
                    "term_zh": term_zh,
                    "term_en": term_en,
                    "category_zh": current_category_zh,
                    "category_en": current_category_en,
                    "source_label": source_label,
                    "retrieval_text": normalize_text(
                        " ".join([source_title, current_category_zh, current_category_en, term_zh, term_en])
                    ),
                }
            )

    if not records:
        raise ValueError("No glossary terms parsed from source document.")
    return records


def build_all_records(config: Dict[str, Dict[str, str]]) -> List[Dict[str, str]]:
    records = []
    records.extend(build_regulation_records(config["documents"]["regulation"]))
    records.extend(build_glossary_records(config["documents"]["glossary"]))
    return records
