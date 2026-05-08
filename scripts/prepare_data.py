from __future__ import annotations

import argparse
from pathlib import Path

from railway_llm_edu.data.aligner import align_adjacent_paragraphs
from railway_llm_edu.data.chunker import build_chunks
from railway_llm_edu.data.cleaner import clean_paragraphs
from railway_llm_edu.data.docx_parser import parse_docx
from railway_llm_edu.data.instruction_builder import (
    build_alignment_instructions,
    build_regulation_instructions,
    build_term_instructions,
)
from railway_llm_edu.data.terminology import extract_terms
from railway_llm_edu.utils import read_yaml, stable_split, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = read_yaml(args.config)

    processed_dir = Path(cfg["processed_dir"])
    instruction_dir = Path(cfg["instruction_dir"])
    all_paragraphs = []

    for doc_cfg in cfg["documents"]:
        paragraphs = parse_docx(
            doc_cfg["path"],
            doc_cfg["id"],
            min_chars=cfg["cleaning"].get("min_paragraph_chars", 2),
        )
        all_paragraphs.extend(paragraphs)

    cleaned = clean_paragraphs(
        all_paragraphs,
        min_chars=cfg["cleaning"].get("min_paragraph_chars", 2),
    )
    chunks = build_chunks(
        cleaned,
        max_chars=cfg["chunking"].get("max_chars", 900),
        overlap_chars=cfg["chunking"].get("overlap_chars", 120),
    )
    glossary_doc_ids = {doc["id"] for doc in cfg["documents"] if doc.get("type") == "glossary"}
    glossary_paragraphs = [para for para in cleaned if para.doc_id in glossary_doc_ids]
    terms = extract_terms(glossary_paragraphs)
    aligned = align_adjacent_paragraphs(cleaned)

    write_jsonl(processed_dir / "paragraphs.jsonl", cleaned)
    write_jsonl(processed_dir / "chunks.jsonl", chunks)
    write_jsonl(processed_dir / "terms.jsonl", terms)
    write_jsonl(processed_dir / "aligned.jsonl", aligned)

    samples = []
    samples.extend(build_term_instructions(terms))
    samples.extend(build_regulation_instructions(chunks))
    samples.extend(build_alignment_instructions(aligned))
    train, valid = stable_split(
        samples,
        train_ratio=cfg["instructions"].get("train_ratio", 0.9),
        seed=cfg["instructions"].get("seed", 42),
    )
    write_jsonl(instruction_dir / "train.jsonl", train)
    write_jsonl(instruction_dir / "valid.jsonl", valid)
    write_jsonl(instruction_dir / "all.jsonl", samples)

    print(
        f"paragraphs={len(cleaned)} chunks={len(chunks)} terms={len(terms)} "
        f"aligned={len(aligned)} instructions={len(samples)}"
    )


if __name__ == "__main__":
    main()
