# Evaluation Sample Schema

Each line in `eval_samples.jsonl` must be a JSON object with the following fields:

- `sample_id`: stable sample identifier
- `question`: user question
- `answer_gold`: reference answer or abstention target
- `evidence`: reference evidence string or expected citation clue
- `task_type`: one of `terminology`, `regulation`, `procedure`, `fault_analysis`, `bilingual`, `abstention`
- `risk_level`: one of `low`, `medium`, `high`
- `answerability`: one of `answerable`, `abstain`
- `keywords`: list of key terms used for retrieval and analysis
- `notes`: short annotation for experimenters

Guidelines:

- `answer_gold` should be concise and evaluation-friendly.
- `evidence` should point to a regulation clause, term pair, or an explicit abstention reason.
- `risk_level` should reflect deployment risk, not question difficulty.
- `notes` can explain ambiguity or expected system behavior.

