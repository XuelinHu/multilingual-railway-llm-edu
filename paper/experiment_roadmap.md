# Experiment Roadmap

## Goal

Build a reproducible experiment pipeline for the railway power QA project, from evaluation set construction to baseline comparison and later LoRA-based domain adaptation.

## Step 1. Build the Evaluation Set

Create a dedicated evaluation dataset under `data/eval/` that does not overlap with any future SFT set.

Each sample should include:

- `sample_id`
- `question`
- `answer_gold`
- `evidence`
- `task_type`
- `risk_level`
- `answerability`
- `keywords`
- `notes`

Task buckets:

- terminology
- regulation
- procedure
- fault_analysis
- bilingual
- abstention

Deliverables:

- `data/eval/README.md`
- `data/eval/schema.md`
- `data/eval/eval_samples.jsonl`

Double-check before moving on:

1. Validate every sample against the schema fields.
2. Ensure task coverage and risk labels are balanced enough for baseline comparison.

## Step 2. Make Baselines Switchable

Turn the current QA pipeline into configurable baselines instead of one fixed path.

Target baselines:

1. Base retrieval-only answer formatting
2. RAG baseline
3. RAG + rerank
4. RAG + terminology enhancement
5. RAG + terminology enhancement + risk-aware calibration

Deliverables:

- config flags for baseline switches
- runner script for selecting a baseline

Double-check before moving on:

1. Confirm each baseline can run on the same eval input.
2. Confirm the output format is stable across baselines.

## Step 3. Implement Retrieval Evaluation

Turn `scripts/eval_rag.py` into a real retrieval evaluator.

Metrics:

- Recall@k
- MRR
- nDCG

Deliverables:

- retrieval evaluator
- machine-readable results under `experiments/`

Double-check before moving on:

1. Re-run on the same input and confirm deterministic outputs.
2. Spot-check individual cases against retrieved citations.

## Step 4. Implement Answer Evaluation

Turn `scripts/eval_answer.py` into a real answer evaluator.

Metrics:

- EM
- token-level F1
- Rouge-L
- terminology consistency
- citation correctness
- abstention correctness

Deliverables:

- answer evaluator
- saved reports per baseline

Double-check before moving on:

1. Verify score computation on handcrafted examples.
2. Verify the script distinguishes answerable vs. abstention cases.

## Step 5. Add Experiment Logging

Standardize result storage for later error analysis and paper writing.

Deliverables:

- per-baseline run logs
- summary metrics
- error case dump

Double-check before moving on:

1. Confirm every run records question, answer, citations, and risk level.
2. Confirm output paths do not leak ignored local artifacts into Git.

## Step 6. Build SFT Data

Turn `scripts/build_sft_data.py` into a real data construction pipeline after the no-training baselines are stable.

Deliverables:

- `data/sft/` outputs
- split rules
- provenance notes

Double-check before moving on:

1. Ensure SFT and eval sets do not overlap.
2. Ensure evidence fields remain traceable to source documents.

## Step 7. Run Training Experiments

After baseline and evaluation are stable, add LoRA/QLoRA experiments on a 7B-class model suitable for RTX 3090 24GB.

Double-check before moving on:

1. Confirm training configuration fits hardware limits.
2. Compare trained results against non-training baselines using the same eval set.

## Current Execution Rule

Work sequentially:

1. Finish one step.
2. Check it twice.
3. Only then start the next step.

