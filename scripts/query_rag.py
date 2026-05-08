from __future__ import annotations

import argparse
import json

from railway_llm_edu.rag.generator import RagGenerator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--config", default="configs/rag.yaml")
    args = parser.parse_args()
    result = RagGenerator(args.config).answer(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
