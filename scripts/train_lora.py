from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from railway_rag.config import load_config
from railway_rag.training.datasets import build_text_dataset


def _resolve_optional_imports() -> dict:
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install them with `pip install -r requirements-train.txt` before running LoRA/QLoRA training."
        ) from exc

    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


class TokenizedDataset:
    def __init__(self, rows: list[dict[str, str]], tokenizer, max_seq_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        text = self.rows[index]["text"]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
        )
        encoded["labels"] = list(encoded["input_ids"])
        return encoded


def _torch_dtype(torch_module, name: str):
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def _build_quantization_config(imports: dict, config: dict):
    quant = config.get("quantization", {})
    if not quant.get("enabled", False):
        return None
    torch = imports["torch"]
    compute_dtype = _torch_dtype(torch, quant.get("bnb_4bit_compute_dtype", "bfloat16"))
    return imports["BitsAndBytesConfig"](
        load_in_4bit=bool(quant.get("load_in_4bit", True)),
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _load_training_config(path: str | Path) -> dict:
    config = load_config(path)
    base_dir = Path(config["base_dir"])
    paths = config.setdefault("paths", {})
    for key in ("train_sft", "dev_sft", "training_output_dir"):
        value = paths.get(key)
        if value:
            resolved = Path(value)
            paths[key] = str((base_dir / resolved).resolve() if not resolved.is_absolute() else resolved)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LoRA/QLoRA railway QA adapter.")
    parser.add_argument("--config", required=True, help="Path to training YAML config.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and dataset without starting training.")
    args = parser.parse_args()

    config = _load_training_config(args.config)
    imports = _resolve_optional_imports()
    torch = imports["torch"]

    train_rows = build_text_dataset(config["paths"]["train_sft"])
    dev_rows = build_text_dataset(config["paths"]["dev_sft"])
    if not train_rows or not dev_rows:
        raise SystemExit("SFT train/dev data is empty. Run `scripts/build_sft_data.py` first.")

    random.seed(int(config["training"].get("seed", 42)))

    tokenizer = imports["AutoTokenizer"].from_pretrained(
        config["model"]["name_or_path"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = int(config["training"].get("max_seq_length", 1024))
    train_dataset = TokenizedDataset(train_rows, tokenizer, max_seq_length=max_seq_length)
    dev_dataset = TokenizedDataset(dev_rows, tokenizer, max_seq_length=max_seq_length)

    summary = {
        "model": config["model"]["name_or_path"],
        "train_samples": len(train_dataset),
        "dev_samples": len(dev_dataset),
        "output_dir": config["paths"]["training_output_dir"],
        "quantization_enabled": bool(config["quantization"].get("enabled", False)),
        "lora_enabled": bool(config["lora"].get("enabled", True)),
        "max_seq_length": max_seq_length,
    }

    output_dir = Path(config["paths"]["training_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config_snapshot.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dry_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    quantization_config = _build_quantization_config(imports, config)
    model = imports["AutoModelForCausalLM"].from_pretrained(
        config["model"]["name_or_path"],
        trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
        torch_dtype=_torch_dtype(torch, config["model"].get("torch_dtype", "bfloat16")),
        quantization_config=quantization_config,
        device_map="auto",
    )

    if quantization_config is not None:
        model = imports["prepare_model_for_kbit_training"](model)

    if config["lora"].get("enabled", True):
        lora_config = imports["LoraConfig"](
            r=int(config["lora"].get("r", 16)),
            lora_alpha=int(config["lora"].get("alpha", 32)),
            lora_dropout=float(config["lora"].get("dropout", 0.05)),
            bias=str(config["lora"].get("bias", "none")),
            task_type="CAUSAL_LM",
            target_modules=list(config["lora"].get("target_modules", [])),
        )
        model = imports["get_peft_model"](model, lora_config)

    training_args = imports["TrainingArguments"](
        output_dir=str(output_dir),
        num_train_epochs=float(config["training"].get("num_train_epochs", 2)),
        per_device_train_batch_size=int(config["training"].get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(config["training"].get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(config["training"].get("gradient_accumulation_steps", 8)),
        learning_rate=float(config["training"].get("learning_rate", 2e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
        warmup_ratio=float(config["training"].get("warmup_ratio", 0.03)),
        logging_steps=int(config["training"].get("logging_steps", 10)),
        eval_steps=int(config["training"].get("eval_steps", 50)),
        save_steps=int(config["training"].get("save_steps", 50)),
        save_total_limit=int(config["training"].get("save_total_limit", 2)),
        lr_scheduler_type=str(config["training"].get("lr_scheduler_type", "cosine")),
        gradient_checkpointing=bool(config["training"].get("gradient_checkpointing", True)),
        bf16=config["model"].get("torch_dtype", "bfloat16") == "bfloat16",
        fp16=config["model"].get("torch_dtype") == "float16",
        eval_strategy="steps",
        save_strategy="steps",
        report_to=[],
        seed=int(config["training"].get("seed", 42)),
    )

    trainer = imports["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=imports["DataCollatorForLanguageModeling"](tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
