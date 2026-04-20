from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class VectorStore:
    """Lightweight sparse vector store based on TF-IDF."""

    def __init__(self, vectorizer: TfidfVectorizer, matrix, records: List[Dict[str, str]]):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.records = records

    @classmethod
    def build(cls, records: List[Dict[str, str]], vectorizer_config: Dict[str, object]) -> "VectorStore":
        texts = [record["retrieval_text"] for record in records]
        vectorizer = TfidfVectorizer(
            analyzer=vectorizer_config.get("analyzer", "char"),
            ngram_range=tuple(vectorizer_config.get("ngram_range", [2, 4])),
            min_df=vectorizer_config.get("min_df", 1),
            lowercase=vectorizer_config.get("lowercase", True),
            sublinear_tf=vectorizer_config.get("sublinear_tf", True),
        )
        matrix = vectorizer.fit_transform(texts)
        return cls(vectorizer=vectorizer, matrix=matrix, records=records)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"vectorizer": self.vectorizer, "matrix": self.matrix, "records": self.records},
            target,
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        payload = joblib.load(Path(path))
        return cls(payload["vectorizer"], payload["matrix"], payload["records"])

    def search(self, query: str, top_k: int = 5, record_types: List[str] | None = None) -> List[Dict[str, object]]:
        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.matrix).ravel()
        ranked_indices = scores.argsort()[::-1]

        hits: List[Dict[str, object]] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            record = dict(self.records[index])
            if record_types and record.get("record_type") not in record_types:
                continue
            record["score"] = round(score, 6)
            hits.append(record)
            if len(hits) >= top_k:
                break
        return hits
