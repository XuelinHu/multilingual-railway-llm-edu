from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_rag.agent.baselines import available_baselines
from railway_rag.cli.ask import answer_query
from railway_rag.config import load_config
from railway_rag.retrieval.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one QA baseline over the evaluation set.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--eval-file", default="data/eval/eval_samples.jsonl", help="Evaluation JSONL file.")
    parser.add_argument("--baseline", required=True, choices=available_baselines(), help="Baseline preset.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k.")
    args = parser.parse_args()

    config = load_config(args.config)
    vector_store = VectorStore.load(config["paths"]["vector_index"])
    top_k = args.top_k or int(config["retrieval"].get("top_k", 5))
    eval_path = Path(args.eval_file)
    output_dir = Path("experiments") / args.baseline
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"

    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            answer = answer_query(config, vector_store, row["question"], top_k=top_k, baseline_name=args.baseline)
            payload = {
                "sample_id": row["sample_id"],
                "baseline": args.baseline,
                "question": row["question"],
                "task_type": row["task_type"],
                "risk_level": row["risk_level"],
                "answerability": row["answerability"],
                "answer_gold": row["answer_gold"],
                "prediction": answer["answer"],
                "citations": answer["citations"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"baseline={args.baseline}")
    print(f"samples={len(rows)}")
    print(f"predictions={output_path}")


if __name__ == "__main__":
    main()
