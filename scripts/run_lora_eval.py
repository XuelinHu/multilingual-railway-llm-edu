from __future__ import annotations

import argparse
import json
from pathlib import Path

from railway_rag.config import load_config


def _resolve_optional_imports() -> dict:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise SystemExit(
            "Missing inference dependencies. Install them with `pip install -r requirements-train.txt` before running LoRA evaluation."
        ) from exc

    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
    }


def _torch_dtype(torch_module, name: str):
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def _load_training_snapshot(adapter_dir: Path) -> dict:
    snapshot_path = adapter_dir / "run_config_snapshot.json"
    if not snapshot_path.exists():
        raise SystemExit(f"Missing training snapshot: {snapshot_path}")
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def _build_quantization_config(imports: dict, config: dict):
    quant = config.get("quantization", {})
    if not quant.get("enabled", False):
        return None
    torch = imports["torch"]
    return imports["BitsAndBytesConfig"](
        load_in_4bit=bool(quant.get("load_in_4bit", True)),
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=_torch_dtype(torch, quant.get("bnb_4bit_compute_dtype", "bfloat16")),
    )


def _build_prompt(question: str) -> str:
    return (
        "<|system|>\n"
        "你是一个面向铁路牵引供电问答的高可信助手，需要优先保持术语规范、规章依据和风险提示。\n"
        "<|user|>\n"
        f"请回答下面的问题：\n{question}\n"
        "<|assistant|>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LoRA adapter inference on the held-out eval set.")
    parser.add_argument("--config", required=True, help="Path to base YAML config.")
    parser.add_argument("--adapter-dir", required=True, help="Path to LoRA adapter output directory.")
    parser.add_argument("--eval-file", default="data/eval/eval_samples.jsonl", help="Evaluation JSONL file.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length.")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and configuration without loading the model.")
    args = parser.parse_args()

    base_config = load_config(args.config)
    adapter_dir = Path(args.adapter_dir).resolve()
    if not adapter_dir.exists():
        raise SystemExit(f"Adapter directory does not exist: {adapter_dir}")
    train_config = _load_training_snapshot(adapter_dir)
    eval_path = Path(args.eval_file)
    rows = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    output_dir = Path("experiments") / "lora_eval" / adapter_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "predictions.jsonl"

    summary = {
        "adapter_dir": str(adapter_dir),
        "base_model": train_config["model"]["name_or_path"],
        "eval_samples": len(rows),
        "output_path": str(output_path),
        "max_new_tokens": args.max_new_tokens,
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    imports = _resolve_optional_imports()
    torch = imports["torch"]
    tokenizer = imports["AutoTokenizer"].from_pretrained(train_config["model"]["name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = _build_quantization_config(imports, train_config)
    base_model = imports["AutoModelForCausalLM"].from_pretrained(
        train_config["model"]["name_or_path"],
        torch_dtype=_torch_dtype(torch, train_config["model"].get("torch_dtype", "bfloat16")),
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=bool(train_config["model"].get("trust_remote_code", False)),
    )
    model = imports["PeftModel"].from_pretrained(base_model, adapter_dir)
    model.eval()

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            prompt = _build_prompt(row["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[0][inputs["input_ids"].shape[1] :]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            payload = {
                "sample_id": row["sample_id"],
                "baseline": f"lora::{adapter_dir.name}",
                "question": row["question"],
                "task_type": row["task_type"],
                "risk_level": row["risk_level"],
                "answerability": row["answerability"],
                "answer_gold": row["answer_gold"],
                "prediction": prediction,
                "citations": [],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"adapter={adapter_dir}")
    print(f"samples={len(rows)}")
    print(f"predictions={output_path}")


if __name__ == "__main__":
    main()
