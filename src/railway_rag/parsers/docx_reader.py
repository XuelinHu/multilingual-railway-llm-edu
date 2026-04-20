from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from railway_rag.utils import normalize_text


WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


@dataclass
class DocxBlock:
    block_type: str
    text: str
    rows: List[List[str]]


def _paragraph_text(node: ET.Element) -> str:
    parts: List[str] = []
    for item in node.iter():
        tag = item.tag.rsplit("}", 1)[-1]
        if tag == "t" and item.text:
            parts.append(item.text)
        elif tag == "tab":
            parts.append("\t")
        elif tag in {"br", "cr"}:
            parts.append("\n")
    return normalize_text("".join(parts))


def _table_rows(node: ET.Element) -> List[List[str]]:
    rows: List[List[str]] = []
    for table_row in node.findall("w:tr", WORD_NS):
        row: List[str] = []
        for cell in table_row.findall("w:tc", WORD_NS):
            cell_texts: List[str] = []
            for para in cell.findall("w:p", WORD_NS):
                text = _paragraph_text(para)
                if text:
                    cell_texts.append(text)
            if cell_texts:
                row.append(" | ".join(cell_texts))
        if row:
            rows.append(row)
    return rows


def read_docx_blocks(path: str | Path) -> List[DocxBlock]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    try:
        with ZipFile(file_path) as archive:
            xml_bytes = archive.read("word/document.xml")
    except KeyError as exc:
        raise ValueError(f"Invalid DOCX structure, missing word/document.xml: {file_path}") from exc

    root = ET.fromstring(xml_bytes)
    body = root.find("w:body", WORD_NS)
    if body is None:
        raise ValueError(f"Invalid DOCX structure, missing body element: {file_path}")

    blocks: List[DocxBlock] = []
    for child in body:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = _paragraph_text(child)
            if text:
                blocks.append(DocxBlock(block_type="paragraph", text=text, rows=[]))
        elif tag == "tbl":
            rows = _table_rows(child)
            if rows:
                flat_rows = [" | ".join(row) for row in rows if row]
                blocks.append(DocxBlock(block_type="table", text="\n".join(flat_rows), rows=rows))
    return blocks


def flatten_table_rows(blocks: Iterable[DocxBlock]) -> List[str]:
    lines: List[str] = []
    for block in blocks:
        if block.block_type == "table":
            for row in block.rows:
                lines.append(" | ".join(row))
    return lines
