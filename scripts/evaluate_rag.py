from __future__ import annotations

import argparse
import json

from railway_llm_edu.eval.evaluator import run_rag_eval


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()
    report = run_rag_eval(args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
