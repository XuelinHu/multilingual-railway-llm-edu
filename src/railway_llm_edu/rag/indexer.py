from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from railway_llm_edu.utils import ensure_dir, read_jsonl, read_yaml


def build_faiss_index(config_path: str | Path) -> None:
    cfg = read_yaml(config_path)
    rows = read_jsonl(cfg["processed_chunks"])
    index_dir = ensure_dir(cfg["index_dir"])

    model = SentenceTransformer(cfg["embedding_model"], device=cfg.get("device", "cuda"))
    texts = [row["text"] for row in rows]
    embeddings = model.encode(
        texts,
        batch_size=cfg.get("batch_size", 16),
        normalize_embeddings=cfg.get("normalize_embeddings", True),
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="writing metadata"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
