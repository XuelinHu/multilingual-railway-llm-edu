from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from railway_rag.utils import normalize_text


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text).lower()
    tokens: List[str] = []
    buffer = []
    for char in normalized:
        if "\u3400" <= char <= "\u9fff":
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
            tokens.append(char)
        elif char.isalnum():
            buffer.append(char)
        else:
            if buffer:
                tokens.append("".join(buffer))
                buffer = []
    if buffer:
        tokens.append("".join(buffer))
    return tokens


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(a: List[str], b: List[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(prediction: str, gold: str) -> float:
    pred_tokens = tokenize(prediction)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def terminology_consistency(sample: Dict[str, object], prediction: str) -> float:
    keywords = [normalize_text(str(item)).lower() for item in sample.get("keywords", [])]
    if not keywords:
        return 0.0
    normalized_prediction = normalize_text(prediction).lower()
    matched = sum(1 for keyword in keywords if keyword and keyword in normalized_prediction)
    return matched / len(keywords)


def citation_correctness(sample: Dict[str, object], citations: List[Dict[str, object]]) -> float:
    if sample["answerability"] == "abstain":
        return 1.0
    if not citations:
        return 0.0
    evidence = normalize_text(str(sample.get("evidence", ""))).lower()
    evidence_tokens = [token for token in evidence.replace("：", " ").replace("/", " ").replace("、", " ").split() if token]
    source_text = " ".join(normalize_text(str(citation.get("source_label", ""))).lower() for citation in citations)
    matched = sum(1 for token in evidence_tokens if token in source_text)
    return 1.0 if matched > 0 else 0.0


def abstention_correctness(sample: Dict[str, object], prediction: str) -> float:
    abstain_markers = ["未检索到明确", "不能据此判断", "以现场规章", "调度命令", "拒绝", "不得将该回答视为现场操作授权"]
    normalized_prediction = normalize_text(prediction)
    predicted_abstain = any(marker in normalized_prediction for marker in abstain_markers)
    expected_abstain = sample["answerability"] == "abstain"
    return 1.0 if predicted_abstain == expected_abstain else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated answers against the eval set.")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL file.")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No predictions found.")

    details_path = predictions_path.with_name("answer_eval_details.jsonl")
    metrics_path = predictions_path.with_name("answer_metrics.json")
    error_cases_path = predictions_path.with_name("error_cases.jsonl")

    em_total = 0.0
    f1_total = 0.0
    rouge_total = 0.0
    term_total = 0.0
    citation_total = 0.0
    abstention_total = 0.0

    with details_path.open("w", encoding="utf-8") as handle, error_cases_path.open("w", encoding="utf-8") as error_handle:
        for row in rows:
            em = exact_match(row["prediction"], row["answer_gold"])
            f1 = token_f1(row["prediction"], row["answer_gold"])
            rouge = rouge_l(row["prediction"], row["answer_gold"])
            term_score = terminology_consistency(row, row["prediction"])
            citation_score = citation_correctness(row, row["citations"])
            abstention_score = abstention_correctness(row, row["prediction"])

            em_total += em
            f1_total += f1
            rouge_total += rouge
            term_total += term_score
            citation_total += citation_score
            abstention_total += abstention_score

            payload = {
                "sample_id": row["sample_id"],
                "baseline": row["baseline"],
                "task_type": row["task_type"],
                "EM": round(em, 6),
                "F1": round(f1, 6),
                "Rouge-L": round(rouge, 6),
                "terminology_consistency": round(term_score, 6),
                "citation_correctness": round(citation_score, 6),
                "abstention_correctness": round(abstention_score, 6),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if f1 < 0.3 or abstention_score < 1.0 or citation_score < 1.0:
                error_payload = {
                    "sample_id": row["sample_id"],
                    "baseline": row["baseline"],
                    "question": row["question"],
                    "task_type": row["task_type"],
                    "risk_level": row["risk_level"],
                    "answerability": row["answerability"],
                    "answer_gold": row["answer_gold"],
                    "prediction": row["prediction"],
                    "citations": row["citations"],
                    "EM": round(em, 6),
                    "F1": round(f1, 6),
                    "Rouge-L": round(rouge, 6),
                    "citation_correctness": round(citation_score, 6),
                    "abstention_correctness": round(abstention_score, 6),
                }
                error_handle.write(json.dumps(error_payload, ensure_ascii=False) + "\n")

    count = len(rows)
    metrics = {
        "baseline": rows[0]["baseline"],
        "samples": count,
        "EM": round(em_total / count, 6),
        "F1": round(f1_total / count, 6),
        "Rouge-L": round(rouge_total / count, 6),
        "terminology_consistency": round(term_total / count, 6),
        "citation_correctness": round(citation_total / count, 6),
        "abstention_correctness": round(abstention_total / count, 6),
        "details_path": str(details_path),
        "error_cases_path": str(error_cases_path),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
