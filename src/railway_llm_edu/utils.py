from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, TypeVar

import yaml

T = TypeVar("T")


def read_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[dict | object]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_split(items: list[T], train_ratio: float, seed: int) -> tuple[list[T], list[T]]:
    copied = list(items)
    random.Random(seed).shuffle(copied)
    cut = int(len(copied) * train_ratio)
    return copied[:cut], copied[cut:]


def normalize_space(text: str) -> str:
    return " ".join(text.replace("\u3000", " ").split())
