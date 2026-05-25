from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_ORDER = [
    "Recall@5",
    "MRR",
    "nDCG@5",
    "EM",
    "F1",
    "Rouge-L",
    "terminology_consistency",
    "citation_correctness",
    "abstention_correctness",
]


def load_metrics(path: str | Path) -> dict:
    file_path = Path(path)
    return json.loads(file_path.read_text(encoding="utf-8"))


def format_delta(base_value: float | None, target_value: float | None) -> str:
    if base_value is None or target_value is None:
        return "-"
    delta = target_value - base_value
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline and LoRA experiment metrics.")
    parser.add_argument("--baseline-dir", required=True, help="Directory with baseline metric JSON files.")
    parser.add_argument("--target-dir", required=True, help="Directory with target metric JSON files.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    target_dir = Path(args.target_dir)
    output_path = Path(args.output)

    baseline_retrieval = load_metrics(baseline_dir / "retrieval_metrics.json") if (baseline_dir / "retrieval_metrics.json").exists() else {}
    baseline_answer = load_metrics(baseline_dir / "answer_metrics.json") if (baseline_dir / "answer_metrics.json").exists() else {}
    target_retrieval = load_metrics(target_dir / "retrieval_metrics.json") if (target_dir / "retrieval_metrics.json").exists() else {}
    target_answer = load_metrics(target_dir / "answer_metrics.json") if (target_dir / "answer_metrics.json").exists() else {}

    baseline_metrics = dict(baseline_retrieval)
    baseline_metrics.update(baseline_answer)
    target_metrics = dict(target_retrieval)
    target_metrics.update(target_answer)

    lines = [
        "# Experiment Comparison",
        "",
        f"- baseline_dir: `{baseline_dir}`",
        f"- target_dir: `{target_dir}`",
        "",
        "| Metric | Baseline | Target | Delta |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRIC_ORDER:
        base_value = baseline_metrics.get(metric)
        target_value = target_metrics.get(metric)
        base_text = f"{base_value:.6f}" if isinstance(base_value, (int, float)) else "-"
        target_text = f"{target_value:.6f}" if isinstance(target_value, (int, float)) else "-"
        delta_text = format_delta(base_value if isinstance(base_value, (int, float)) else None, target_value if isinstance(target_value, (int, float)) else None)
        lines.append(f"| {metric} | {base_text} | {target_text} | {delta_text} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report={output_path}")


if __name__ == "__main__":
    main()
