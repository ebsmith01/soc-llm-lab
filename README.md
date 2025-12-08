# SOC LLM Lab

## Overview
This repository hosts experiments, infra modules, and reference agents focused on bringing large language models into the Security Operations Center (SOC). Each project directory is self-contained but shares common tooling (Python 3.11, Ruff, pytest) so ideas can move from notebooks to deployable services quickly.

## Repo Layout
- `projects/secure-rag-soc` — Retrieval-augmented FastAPI service geared toward secure incident response workflows.
- `projects/agent-lite` — Minimal HTTPX/Pydantic agent skeleton for rapid prototyping.
- `libs`, `infra`, `evals`, `notes`, `docs` — Shared components, deployment artifacts, evaluation harnesses, and design notes.

## Getting Started
1. **Install Python 3.11** (Homebrew example): `brew install python@3.11`
2. **Create a virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   ```
3. **Install project dependencies**
   ```bash
   pip install -e projects/secure-rag-soc
   pip install -e projects/agent-lite
   pip install pre-commit
   ```
4. **Environment variables** — copy `.env.example` to `.env` and fill in required secrets (e.g., `OPENAI_API_KEY`, `CHUNKS_PATH`). Optional: set `ENABLE_DENSE_RETRIEVAL=1` if you want SentenceTransformer embeddings downloaded; leave unset to stick with offline BM25 retrieval.

## Formatting & Hooks
Once a `.pre-commit-config.yaml` is added to the repo root, run:
```bash
pre-commit install
pre-commit run --all-files
```
This will keep formatting consistent before every commit.

## Running Things
- `make run` / `make ingest` / `make test` delegate into `projects/secure-rag-soc`.
- Individual projects expose their own entry points—consult their `README.md` files for details.

## Contributing
Open PRs per project, keep changes scoped, and document novel evaluations in `notes/` so other analysts can learn from prior runs.
