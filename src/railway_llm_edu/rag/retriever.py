from __future__ import annotations

from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from railway_llm_edu.schemas import RetrievalHit
from railway_llm_edu.utils import read_jsonl, read_yaml


class FaissRetriever:
    def __init__(self, config_path: str | Path):
        self.cfg = read_yaml(config_path)
        self.index_dir = Path(self.cfg["index_dir"])
        self.rows = read_jsonl(self.index_dir / "metadata.jsonl")
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.model = SentenceTransformer(
            self.cfg["embedding_model"],
            device=self.cfg.get("device", "cuda"),
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        top_k = top_k or self.cfg["retrieval"]["top_k"]
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            row = self.rows[int(idx)]
            if float(score) < self.cfg["retrieval"].get("min_score", 0.0):
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    score=float(score),
                    metadata={
                        "doc_id": row["doc_id"],
                        "source_path": row["source_path"],
                        "article_no": row.get("article_no"),
                        **row.get("metadata", {}),
                    },
                )
            )
        return hits


def format_context(hits: list[RetrievalHit], max_chars: int = 3500) -> str:
    blocks: list[str] = []
    total = 0
    for i, hit in enumerate(hits, start=1):
        citation = f"[{i}] {hit.metadata.get('doc_id')}/{hit.chunk_id}"
        block = f"{citation}\n{hit.text}"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n".join(blocks)
