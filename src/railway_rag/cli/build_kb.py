from __future__ import annotations

import argparse

from railway_rag.config import load_config
from railway_rag.pipeline.builders import build_all_records
from railway_rag.retrieval.vector_store import VectorStore
from railway_rag.utils import write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multilingual railway JSON corpus and vector index.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    records = build_all_records(config)

    write_jsonl(records, config["paths"]["unified_jsonl"])
    write_json(records, config["paths"]["unified_pretty_json"])

    vector_store = VectorStore.build(records, config["retrieval"]["vectorizer"])
    vector_store.save(config["paths"]["vector_index"])

    regulation_count = sum(1 for record in records if record["record_type"] == "regulation_clause")
    glossary_count = sum(1 for record in records if record["record_type"] == "glossary_term")
    print(f"Built {len(records)} records.")
    print(f"  regulation clauses: {regulation_count}")
    print(f"  glossary terms: {glossary_count}")
    print(f"  unified jsonl: {config['paths']['unified_jsonl']}")
    print(f"  vector index: {config['paths']['vector_index']}")


if __name__ == "__main__":
    main()
