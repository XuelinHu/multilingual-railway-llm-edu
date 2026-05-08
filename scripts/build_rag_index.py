from __future__ import annotations

import argparse

from railway_llm_edu.rag.indexer import build_faiss_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()
    build_faiss_index(args.config)


if __name__ == "__main__":
    main()
