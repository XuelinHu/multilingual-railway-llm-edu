from __future__ import annotations

import hashlib

from railway_llm_edu.schemas import InstructionSample, TermEntry, TextChunk

SYSTEM_PROMPT = (
    "你是面向留学生职业教育的多语种铁道知识教学助手。"
    "回答必须准确、可教学、必要时中英双语解释；涉及规章依据时必须引用给定资料。"
)


def _sid(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.md5(text.encode('utf-8')).hexdigest()[:12]}"


def build_term_instructions(terms: list[TermEntry]) -> list[InstructionSample]:
    samples: list[InstructionSample] = []
    for term in terms:
        samples.append(
            InstructionSample(
                id=_sid("term-zh2en", term.term_id),
                task_type="terminology_translation",
                source_ids=[term.term_id],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请给出铁道术语“{term.zh}”的英文表达，并做简要教学解释。"},
                    {
                        "role": "assistant",
                        "content": f"{term.zh} 的英文表达是 {term.en}。\n教学提示：该术语用于铁道专业场景，学习时应同时记忆中文概念和英文专业表达。",
                    },
                ],
            )
        )
        samples.append(
            InstructionSample(
                id=_sid("term-en2zh", term.term_id),
                task_type="terminology_translation",
                source_ids=[term.term_id],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f'What is the Chinese railway term for "{term.en}"? Explain briefly.'},
                    {
                        "role": "assistant",
                        "content": f'"{term.en}" 对应的中文铁道术语是“{term.zh}”。It should be learned as a domain-specific railway expression.',
                    },
                ],
            )
        )
    return samples


def build_regulation_instructions(chunks: list[TextChunk]) -> list[InstructionSample]:
    samples: list[InstructionSample] = []
    for chunk in chunks:
        if chunk.doc_id == "railway_glossary":
            continue
        source = f"[依据: {chunk.doc_id}/{chunk.chunk_id}]"
        samples.extend(
            [
                InstructionSample(
                    id=_sid("explain", chunk.chunk_id),
                    task_type="regulation_explanation",
                    source_ids=[chunk.chunk_id],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"请用适合留学生课堂教学的方式解释以下规章内容：\n{chunk.text}"},
                        {
                            "role": "assistant",
                            "content": f"这段规章的核心要求可以概括为：\n1. 明确管理或作业对象。\n2. 按规定执行运行维护、检查或记录要求。\n3. 对关键安全环节保持可追溯。\n{source}",
                        },
                    ],
                ),
                InstructionSample(
                    id=_sid("quiz", chunk.chunk_id),
                    task_type="quiz_generation",
                    source_ids=[chunk.chunk_id],
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"基于以下资料生成1道单选题，包含答案和解析：\n{chunk.text}"},
                        {
                            "role": "assistant",
                            "content": "题目：根据资料，相关人员在学习或执行该规章时最应关注什么？\nA. 只记忆英文名称\nB. 忽略现场记录\nC. 理解职责、流程和安全要求\nD. 仅关注设备外观\n答案：C\n解析：规章类知识的教学重点是职责边界、执行流程、安全要求和依据追溯。\n"
                            + source,
                        },
                    ],
                ),
            ]
        )
    return samples


def build_alignment_instructions(aligned: list[dict]) -> list[InstructionSample]:
    samples: list[InstructionSample] = []
    for item in aligned:
        samples.append(
            InstructionSample(
                id=_sid("align", item["id"]),
                task_type="bilingual_alignment",
                source_ids=item["source_ids"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请将以下中文铁道资料翻译为英文：\n{item['zh']}"},
                    {"role": "assistant", "content": item["en"]},
                ],
            )
        )
    return samples
