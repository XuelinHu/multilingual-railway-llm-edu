# multilingual-railway-llm-edu

<p align="center">
  <img height="20" src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&amp;logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/transformers-configured-FFD21E?logo=huggingface&amp;logoColor=black" />
  <img height="20" src="https://img.shields.io/badge/QLoRA-configured-A020F0" />
  <img height="20" src="https://img.shields.io/badge/FAISS-configured-0467DF" />
  <img height="20" src="https://img.shields.io/badge/fastapi-0.111.0-009688?logo=fastapi&amp;logoColor=white" />
  <img height="20" src="https://img.shields.io/badge/ruff-configured-D7FF64?logo=ruff&amp;logoColor=black" />
</p>

<<<<<<< HEAD
面向留学生职业教育场景的多语种铁道知识教学大模型系统。系统围绕“中英双语术语学习、规章解读、教学问答、课堂出题、基于知识依据的回答”构建，默认优先适配单卡 RTX 3090 24GB。

## 1. 项目总体架构

```text
multilingual-railway-llm-edu/
├── corpus/                         # 原始 docx 语料
├── configs/                        # 数据、训练、RAG、评测配置
├── data/
│   ├── raw/                        # 可选：原始文件镜像
│   ├── processed/                  # paragraphs/chunks/terms/aligned
│   ├── instructions/               # QLoRA 指令微调数据
│   ├── rag/                        # FAISS 索引与元数据
│   └── eval/                       # 评测题集
├── outputs/
│   ├── checkpoints/                # LoRA checkpoint
│   └── reports/                    # 评测报告
├── scripts/                        # 可执行命令入口
├── src/railway_llm_edu/
│   ├── data/                       # docx 解析、清洗、切分、术语、对齐、样本构建
│   ├── training/                   # QLoRA/SFT 训练
│   ├── rag/                        # embedding、FAISS、检索、带引用生成
│   └── eval/                       # 客观、主观、可信性、安全性评测
└── tests/                          # 后续单元测试
```

核心链路：

```text
DOCX 语料
  -> 解析段落和表格
  -> 文本清洗
  -> 条款/语义 chunk
  -> 术语抽取与中英对齐
  -> 指令样本构建
  -> QLoRA 微调

DOCX 语料
  -> chunk
  -> embedding
  -> FAISS vector store
  -> top-k retrieval
  -> RAG prompt with citation
  -> 教学问答/规章解释
```

## 2. 数据处理流程设计

已知语料：

- `ECRL牵引供电设备运行维护管理办法（修订）_zh2en_transResult.docx`
- `铁路中英文词汇（全）.docx`

处理步骤：

1. `docx` 解析：读取普通段落和表格行，保留 `doc_id/source_path/paragraph_id/style/table_id`。
2. 文本清洗：去控制字符、页码、重复段落，统一空白和中英文标点前空格。
3. 条款切分：优先识别 `第X章/节/条/款`、`Article N`、编号列表；chunk 默认 `900` 字，`120` 字重叠。
4. 术语抽取：对表格、tab 分隔、行内中英对照进行规则抽取，输出 `zh/en/source_path/term_id`。
5. 双语对齐：先处理同一行的中英分隔，再处理相邻中文段落和英文段落。
6. 指令样本构建：
   - 术语中译英、英译中
   - 规章课堂解释
   - 基于资料出单选题
   - 中英翻译/对齐样本
   - 带依据回答样本

输出文件：

- `data/processed/paragraphs.jsonl`
- `data/processed/chunks.jsonl`
- `data/processed/terms.jsonl`
- `data/processed/aligned.jsonl`
- `data/instructions/train.jsonl`
- `data/instructions/valid.jsonl`

## 3. 训练流程设计

基础模型建议：

- 显存更稳：`Qwen/Qwen2.5-3B-Instruct`
- 效果更强：`Qwen/Qwen2.5-7B-Instruct`

单卡 RTX 3090 24GB 推荐参数：

- 4bit NF4 量化加载
- LoRA rank `16`，alpha `32`，dropout `0.05`
- batch size `1`
- gradient accumulation `16`
- max sequence length `2048`
- gradient checkpointing 开启
- optimizer `paged_adamw_8bit`
- bf16 优先；如环境不支持可改 fp16

训练命令：

```bash
pip install -r requirements.txt
$env:PYTHONPATH="src"
python scripts/prepare_data.py --config configs/data.yaml
python scripts/train_qlora.py --config configs/train_qlora.yaml
```

如改用 7B，在 `configs/train_qlora.yaml` 中设置：

```yaml
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
gradient_accumulation_steps: 32
max_seq_length: 1536
```

## 4. RAG 知识库流程设计

RAG 使用 `data/processed/chunks.jsonl` 作为知识单元。

1. chunk：复用数据处理阶段 chunk，保留 `chunk_id/doc_id/article_no/source_path`。
2. embedding：默认 `BAAI/bge-m3`，适合中英混合检索。
3. vector store：FAISS `IndexFlatIP`，embedding 归一化后用内积近似 cosine。
4. retrieval：默认 top-k `5`，按分数过滤。
5. citation：生成上下文时使用 `[1] doc_id/chunk_id` 形式，要求回答末尾列出引用编号。

命令：

```bash
$env:PYTHONPATH="src"
python scripts/build_rag_index.py --config configs/rag.yaml
python scripts/query_rag.py "请解释牵引供电设备运行维护管理的教学重点" --config configs/rag.yaml
```

## 5. 评测体系设计

客观题：

- 术语选择题准确率
- 规章事实判断题准确率
- 中英术语匹配准确率

主观题：

- 参考答案 ROUGE-L/BLEU
- 教师人工评分：准确性、完整性、双语表达、课堂可讲性
- 学生视角评分：易懂性、术语解释清晰度、示例有效性

教学可用性：

- 是否能按“概念解释 -> 规章依据 -> 课堂示例 -> 小测题”组织答案
- 是否适合留学生语言水平
- 是否能输出中英双语关键术语

可信性：

- 引用覆盖率：回答是否包含引用编号或依据说明
- groundedness：关键结论是否可在检索资料中找到
- 规章编号一致性：是否编造不存在的章节条款

安全性：

- 资料不足时是否拒绝编造
- 是否避免给出危险作业指令
- 是否提示以正式规章和教师要求为准

命令：

```bash
$env:PYTHONPATH="src"
python scripts/evaluate_rag.py --config configs/eval.yaml
```

## 6. 代码文件说明

- `src/railway_llm_edu/data/docx_parser.py`：解析 docx 段落和表格。
- `src/railway_llm_edu/data/cleaner.py`：清洗噪声、页码和重复段落。
- `src/railway_llm_edu/data/chunker.py`：按条款和长度构建 chunk。
- `src/railway_llm_edu/data/terminology.py`：从词汇表抽取中英术语对。
- `src/railway_llm_edu/data/aligner.py`：构建轻量中英段落对齐数据。
- `src/railway_llm_edu/data/instruction_builder.py`：生成 SFT 指令样本。
- `src/railway_llm_edu/training/qlora.py`：QLoRA 微调入口。
- `src/railway_llm_edu/rag/indexer.py`：embedding 与 FAISS 建库。
- `src/railway_llm_edu/rag/retriever.py`：top-k 检索与上下文格式化。
- `src/railway_llm_edu/rag/generator.py`：带引用的 RAG 回答。
- `src/railway_llm_edu/eval/`：评测指标与评测运行器。
- `scripts/serve_rag.py`：FastAPI 形式的 RAG 问答服务。

## 7. 开发顺序

1. 数据处理最小闭环：跑通 `prepare_data.py`，人工抽查 `terms.jsonl/chunks.jsonl/aligned.jsonl`。
2. 术语规则增强：根据 `铁路中英文词汇（全）.docx` 的真实排版补充抽取规则。
3. 指令样本质检：抽样检查训练集，删除模板化过强、依据不足的样本。
4. RAG 建库：使用 bge-m3 建 FAISS，验证中英查询召回。
5. RAG 提示词优化：固定“资料不足不编造”和引用格式。
6. 小规模 QLoRA：先用 3B 跑 100-300 steps 验证 loss、显存和输出风格。
7. 完整 QLoRA：扩展到全量数据，再根据显存决定 3B 或 7B。
8. 评测集建设：由教师补充客观题、主观题、规章依据题和安全拒答题。
9. 端到端评测：比较 base model、RAG、QLoRA、QLoRA+RAG。
10. 服务化：增加 FastAPI、前端课堂工具、教师题库导出。

## 8. 当前可运行命令

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="src"
python scripts/prepare_data.py --config configs/data.yaml
python scripts/build_rag_index.py --config configs/rag.yaml
python scripts/query_rag.py "What is the teaching focus of railway traction power maintenance?"
```

启动 API：

```powershell
$env:PYTHONPATH="src"
python scripts/serve_rag.py --config configs/rag.yaml --port 8000
```

Linux/macOS:
=======
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
>>>>>>> 6afbc0f723dd75dc2e2a721739bd855fadbe6ebd

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
<<<<<<< HEAD
export PYTHONPATH=src
python scripts/prepare_data.py --config configs/data.yaml
python scripts/build_rag_index.py --config configs/rag.yaml
python scripts/query_rag.py "请解释牵引供电设备运行维护的课堂教学重点"
```
=======
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
>>>>>>> 6afbc0f723dd75dc2e2a721739bd855fadbe6ebd
