# Evaluation Set

This directory stores the held-out evaluation set for railway power QA experiments.

Rules:

- Keep this set separate from any future SFT data.
- Use one JSON object per line in `eval_samples.jsonl`.
- Keep evidence short, explicit, and traceable.
- Cover all supported task types.
- Include both answerable and abstention-oriented cases.

Files:

- `schema.md`: field definitions
- `eval_samples.jsonl`: initial evaluation samples

