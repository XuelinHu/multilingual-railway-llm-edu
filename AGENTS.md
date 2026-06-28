# Agent Instructions

## Default Conda Environment
- Environment name: `rc-llm-eval`
- Environment path: `/home/xuelin/miniconda3/envs/rc-llm-eval`
- Prefer running Python commands with `conda run -n rc-llm-eval ...` or `/home/xuelin/miniconda3/envs/rc-llm-eval/bin/python`.

<!-- codex-agent-runtime:start -->

## Runtime Ports And Database Configuration

- Keep this section aligned with the root README when database names, ports, or service defaults change.
- Do not copy secrets from local `.env` files into commits; document only placeholders or compose defaults.

### Database
- No external SQL database is configured by default.
- RAG retrieval uses local processed data and FAISS/vector-index artifacts under `data/` and configured paths.

### Default Ports
- Optional FastAPI RAG service uses `--port 8000` in the documented command.

### Notes For Codex Agents
- The root README currently contains pre-existing merge-conflict markers; this update only appends runtime/database notes.
- Before committing, check `git status --short --branch` and avoid staging unrelated runtime artifacts.

### Source Files Checked
- `README.md`
- `scripts/serve_rag.py`
- `configs/rag.yaml`

<!-- codex-agent-runtime:end -->

## GitHub Commit Language

- Use English for all GitHub commit messages and pull/push related commit notes.
