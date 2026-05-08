from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from railway_llm_edu.eval.metrics import citation_coverage, has_fabricated_regulation_number
from railway_llm_edu.rag.generator import RagGenerator
from railway_llm_edu.utils import ensure_dir, read_jsonl, read_yaml


def run_rag_eval(config_path: str | Path) -> dict:
    cfg = read_yaml(config_path)
    questions = read_jsonl(cfg["question_file"])
    generator = RagGenerator(cfg["rag_config"])
    report_dir = ensure_dir(cfg["report_dir"])

    rows = []
    for q in tqdm(questions, desc="evaluating"):
        result = generator.answer(q["question"])
        retrieved_text = "\n".join(hit["text"] for hit in result["citations"])
        rows.append(
            {
                **q,
                "prediction": result["answer"],
                "citations": result["citations"],
                "fabricated_regulation_number": has_fabricated_regulation_number(
                    result["answer"], retrieved_text
                ),
            }
        )

    report = {
        "count": len(rows),
        "citation_coverage": citation_coverage([r["prediction"] for r in rows]),
        "fabricated_regulation_number_rate": (
            sum(r["fabricated_regulation_number"] for r in rows) / len(rows) if rows else 0
        ),
    }
    with (report_dir / "rag_eval_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (report_dir / "rag_eval_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
