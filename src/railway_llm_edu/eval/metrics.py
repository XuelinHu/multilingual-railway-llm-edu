from __future__ import annotations

import re
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU


def accuracy(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    correct = sum(p.strip().upper() == r.strip().upper() for p, r in zip(predictions, references, strict=False))
    return correct / len(predictions)


def rouge_l(predictions: list[str], references: list[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = [
        scorer.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(predictions, references, strict=False)
    ]
    return sum(scores) / len(scores) if scores else 0.0


def bleu(predictions: list[str], references: list[str]) -> float:
    if not predictions:
        return 0.0
    return BLEU().corpus_score(predictions, [references]).score


def citation_coverage(answers: list[str]) -> float:
    if not answers:
        return 0.0
    cited = sum(bool(re.search(r"\[\d+\]|依据|引用|citation", answer, re.I)) for answer in answers)
    return cited / len(answers)


def has_fabricated_regulation_number(answer: str, retrieved_text: str) -> bool:
    nums = set(re.findall(r"第[一二三四五六七八九十百零〇\d]+[章节条款]", answer))
    if not nums:
        return False
    return any(num not in retrieved_text for num in nums)
