from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from railway_rag.config import load_config
from railway_rag.utils import normalize_text, write_jsonl


def load_eval_questions(path: str | Path) -> set[str]:
    eval_path = Path(path)
    if not eval_path.exists():
        return set()
    return {
        normalize_text(json.loads(line)["question"])
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_records(path: str | Path) -> List[Dict[str, object]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def term_samples(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    samples = []
    for record in records:
        if record["record_type"] != "glossary_term":
            continue
        term_zh = normalize_text(str(record.get("term_zh", "")))
        term_en = normalize_text(str(record.get("term_en", "")))
        if not term_zh or not term_en:
            continue
        answer = "\n".join(
            [
                f"中文术语：{term_zh}",
                f"英文术语：{term_en}",
                f"术语类别：{record.get('category_zh', '未分类')}",
                "说明：以上术语来自铁路技术标准中英文词汇，应优先作为标准表达使用。",
            ]
        )
        samples.append(
            {
                "instruction": "请根据铁路术语词典回答问题，并保持术语表达规范一致。",
                "question": f"{term_zh}英文怎么说？",
                "answer": answer,
                "task_type": "terminology",
                "risk_level": "low",
                "evidence": record.get("source_label", ""),
                "source_record_id": record["record_id"],
            }
        )
    return samples


def regulation_samples(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    samples = []
    for record in records:
        if record["record_type"] != "regulation_clause":
            continue
        zh_text = normalize_text(str(record.get("zh_text", "")))
        if len(zh_text) < 20:
            continue
        question = f"{record.get('section_path', '该章节')}中有哪些关键要求？"
        answer = "\n".join(
            [
                f"规章依据：{record.get('source_label', '')}",
                f"条文摘要：{zh_text}",
                "说明：回答应以规章文本为主，不应补充超出条文依据的现场操作结论。",
            ]
        )
        samples.append(
            {
                "instruction": "请基于铁路供电规章给出简洁、可追溯的依据型回答。",
                "question": question,
                "answer": answer,
                "task_type": "regulation",
                "risk_level": record.get("risk_level", "medium"),
                "evidence": record.get("source_label", ""),
                "source_record_id": record["record_id"],
            }
        )
    return samples


def procedure_samples(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    samples = []
    for record in records:
        if record["record_type"] != "regulation_clause":
            continue
        zh_text = normalize_text(str(record.get("zh_text", "")))
        if record.get("content_type") not in {"procedure", "precondition", "rule"}:
            continue
        if len(zh_text) < 20:
            continue
        samples.append(
            {
                "instruction": "请将铁路供电规章内容整理成简洁的流程化步骤回答。",
                "question": f"{record.get('section_path', '该章节')}相关流程应如何执行？",
                "answer": "\n".join(
                    [
                        f"适用章节：{record.get('source_label', '')}",
                        f"流程依据：{zh_text}",
                        "说明：回答应区分规章要求与可能的经验性补充。",
                    ]
                ),
                "task_type": "procedure",
                "risk_level": record.get("risk_level", "medium"),
                "evidence": record.get("source_label", ""),
                "source_record_id": record["record_id"],
            }
        )
    return samples


def fault_samples(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    samples = []
    for record in records:
        if record["record_type"] != "regulation_clause":
            continue
        zh_text = normalize_text(str(record.get("zh_text", "")))
        if record.get("content_type") != "emergency":
            continue
        if len(zh_text) < 20:
            continue
        samples.append(
            {
                "instruction": "请针对铁路供电异常或故障问题给出谨慎的排查建议，并说明依据。",
                "question": f"{record.get('section_path', '该章节')}中的异常情况应优先排查什么？",
                "answer": "\n".join(
                    [
                        f"规章依据：{record.get('source_label', '')}",
                        f"排查线索：{zh_text}",
                        "说明：若条文未直接规定具体操作，应仅给出辅助分析，不得替代现场指令。",
                    ]
                ),
                "task_type": "fault_analysis",
                "risk_level": record.get("risk_level", "medium"),
                "evidence": record.get("source_label", ""),
                "source_record_id": record["record_id"],
            }
        )
    return samples


def abstention_samples(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    samples = []
    for record in records:
        if record["record_type"] != "regulation_clause":
            continue
        if record.get("risk_level") != "high":
            continue
        question = f"{record.get('section_path', '该场景')}是否可以直接操作？"
        answer = (
            "若未检索到明确允许或禁止依据，必须拒绝给出确定性操作结论，并提示以现场规章、调度命令和授权流程为准。"
        )
        samples.append(
            {
                "instruction": "请对高风险铁路供电问题执行保守回答策略。",
                "question": question,
                "answer": answer,
                "task_type": "abstention",
                "risk_level": "high",
                "evidence": record.get("source_label", ""),
                "source_record_id": record["record_id"],
            }
        )
    return samples


def dedupe_and_filter(samples: List[Dict[str, object]], eval_questions: set[str]) -> List[Dict[str, object]]:
    filtered = []
    seen = set()
    for sample in samples:
        question_norm = normalize_text(sample["question"])
        key = (sample["task_type"], question_norm)
        if question_norm in eval_questions or key in seen:
            continue
        seen.add(key)
        filtered.append(sample)
    return filtered


def split_samples(samples: List[Dict[str, object]], dev_every: int = 10) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    train = []
    dev = []
    for index, sample in enumerate(samples, start=1):
        if index % dev_every == 0:
            dev.append(sample)
        else:
            train.append(sample)
    return train, dev


def write_provenance(path: str | Path, stats: Dict[str, object]) -> None:
    lines = [
        "# SFT Provenance",
        "",
        f"- source_records: {stats['source_records']}",
        f"- generated_samples: {stats['generated_samples']}",
        f"- train_samples: {stats['train_samples']}",
        f"- dev_samples: {stats['dev_samples']}",
        f"- excluded_eval_questions: {stats['excluded_eval_questions']}",
        f"- task_breakdown: {stats['task_breakdown']}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an initial SFT dataset from the local KB records.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--eval-file", default="data/eval/eval_samples.jsonl", help="Evaluation JSONL file to exclude.")
    parser.add_argument("--max-term", type=int, default=80, help="Max terminology samples.")
    parser.add_argument("--max-regulation", type=int, default=60, help="Max regulation samples.")
    parser.add_argument("--max-procedure", type=int, default=20, help="Max procedure samples.")
    parser.add_argument("--max-fault", type=int, default=20, help="Max fault-analysis samples.")
    parser.add_argument("--max-abstention", type=int, default=20, help="Max abstention samples.")
    args = parser.parse_args()

    config = load_config(args.config)
    records = load_records(config["paths"]["unified_pretty_json"])
    eval_questions = load_eval_questions(args.eval_file)

    term = term_samples(records)[: args.max_term]
    regulation = regulation_samples(records)[: args.max_regulation]
    procedure = procedure_samples(records)[: args.max_procedure]
    fault = fault_samples(records)[: args.max_fault]
    abstention = abstention_samples(records)[: args.max_abstention]
    combined = dedupe_and_filter(term + regulation + procedure + fault + abstention, eval_questions)
    train, dev = split_samples(combined, dev_every=10)

    output_dir = Path("data/sft")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    provenance_path = output_dir / "provenance.md"

    write_jsonl(train, train_path)
    write_jsonl(dev, dev_path)
    task_breakdown = {}
    for sample in combined:
        task_breakdown[sample["task_type"]] = task_breakdown.get(sample["task_type"], 0) + 1
    write_provenance(
        provenance_path,
        {
            "source_records": len(records),
            "generated_samples": len(combined),
            "train_samples": len(train),
            "dev_samples": len(dev),
            "excluded_eval_questions": len(eval_questions),
            "task_breakdown": task_breakdown,
        },
    )

    print(f"train={train_path} ({len(train)})")
    print(f"dev={dev_path} ({len(dev)})")
    print(f"provenance={provenance_path}")


if __name__ == "__main__":
    main()
