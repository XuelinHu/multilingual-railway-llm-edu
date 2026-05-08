from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from railway_llm_edu.rag.retriever import FaissRetriever, format_context
from railway_llm_edu.utils import read_yaml


class RagGenerator:
    def __init__(self, config_path: str | Path):
        self.cfg = read_yaml(config_path)
        gen_cfg = self.cfg["generation"]
        model_path = Path(gen_cfg["model_name_or_path"])
        model_name = str(model_path) if model_path.exists() else gen_cfg["fallback_model_name_or_path"]
        self.retriever = FaissRetriever(config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    def answer(self, question: str) -> dict:
        hits = self.retriever.search(question)
        context = format_context(hits, self.cfg["retrieval"]["max_context_chars"])
        messages = [
            {
                "role": "system",
                "content": (
                    "你是多语种铁道职业教育问答助手。只能根据给定资料回答；"
                    "如资料不足，说明无法从当前知识库确认。回答末尾列出引用编号。"
                ),
            },
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg["generation"].get("max_new_tokens", 768),
            temperature=self.cfg["generation"].get("temperature", 0.2),
            top_p=self.cfg["generation"].get("top_p", 0.9),
            do_sample=self.cfg["generation"].get("temperature", 0.2) > 0,
        )
        text = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return {
            "answer": text.strip(),
            "citations": [hit.model_dump(mode="json") for hit in hits],
        }
