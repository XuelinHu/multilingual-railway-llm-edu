# Experiment Comparison

- baseline_dir: `experiments/rag_term_risk`
- target_dir: `experiments/lora_eval/qwen2_5_7b_lora`

| Metric | Baseline | Target | Delta |
|---|---:|---:|---:|
| Recall@5 | 0.800000 | - | - |
| MRR | 0.700000 | - | - |
| nDCG@5 | 0.692812 | - | - |
| EM | 0.000000 | 0.000000 | +0.000000 |
| F1 | 0.164491 | 0.075746 | -0.088745 |
| Rouge-L | 0.111569 | 0.064357 | -0.047212 |
| terminology_consistency | 0.000000 | 0.000000 | +0.000000 |
| citation_correctness | 0.166667 | 0.166667 | +0.000000 |
| abstention_correctness | 1.000000 | 0.833333 | -0.166667 |
