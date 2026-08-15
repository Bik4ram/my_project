# Claude Code Instructions

## Project

Production-grade Semiconductor RAG Platform.

Purpose:
Index RTL/UVM/SystemVerilog repositories and answer engineering questions.

---

## Environment

OS:
Ubuntu 24.04 (WSL)

Python:
3.12

Database:
PostgreSQL 16
pgvector

LLM:
Ollama

Embedding Model:
nomic-embed-text

---

## Common Commands

Activate environment

```bash
source .venv/bin/activate
```

Index repository

```bash
python run_local_qa.py index \
    --project-id rtl_uvm \
    --root ~/projects/RTL_UVM
```

Query

```bash
python run_local_qa.py ask \
    --project-id rtl_uvm \
    --question "<question>"
```

---

## Development Rules

Always identify the root cause.

Never remove functionality just to make errors disappear.

Keep fixes production quality.

After every fix:

1. Run formatting if applicable.
2. Run the failing command again.
3. Verify the issue is resolved.

Do not stop after fixing the first error.

Continue until the pipeline works end-to-end.

Explain every change before finishing.

---

## Coding Style

- Small commits
- Clear function names
- Type hints where appropriate
- Keep existing architecture
- Avoid unnecessary refactoring
