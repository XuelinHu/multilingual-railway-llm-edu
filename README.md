# multilingual-railway-llm-edu

第一阶段可运行版本：多语种铁道知识 RAG 原型系统。

当前版本完成：

- 解析 `docx` 中的中英双语规章与术语内容
- 将规章按章节和段落条款切分
- 将术语表按中英术语对齐
- 生成统一 `JSON/JSONL`
- 构建轻量本地向量检索知识库
- 提供命令行问答程序，并输出引用来源

## 目录结构

```text
multilingual-railway-llm-edu/
├── configs/
│   └── default.yaml
├── corpus/
│   ├── 规章43：ECRL牵引供电设备运行维护管理办法（修订）_zh2en_transResult.docx
│   └── 铁路中英文词汇（全）.docx
├── output/
├── requirements.txt
├── README.md
└── src/
    └── railway_rag/
        ├── __init__.py
        ├── config.py
        ├── utils.py
        ├── cli/
        │   ├── __init__.py
        │   ├── ask.py
        │   └── build_kb.py
        ├── parsers/
        │   ├── __init__.py
        │   └── docx_reader.py
        ├── pipeline/
        │   ├── __init__.py
        │   └── builders.py
        ├── qa/
        │   ├── __init__.py
        │   └── answering.py
        └── retrieval/
            ├── __init__.py
            └── vector_store.py
```

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 构建知识库

```bash
PYTHONPATH=src python -m railway_rag.cli.build_kb --config configs/default.yaml
```

## 命令行问答

```bash
PYTHONPATH=src python -m railway_rag.cli.ask --config configs/default.yaml --query "接触网运行维修应坚持什么方针？"
PYTHONPATH=src python -m railway_rag.cli.ask --config configs/default.yaml --query "“牵引供电”英文怎么说？"
```

## 说明

- `docx` 解析基于 Python 标准库 `zipfile + xml.etree.ElementTree`，不依赖 `python-docx`
- 检索采用本地 `TF-IDF` 稀疏向量，轻量、稳定、可离线运行
- 统一 JSON 保留中英文文本、章节路径、术语类别、来源文件等字段，便于后续无缝接入指令微调或 QLoRA 数据整理
