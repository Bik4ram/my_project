# Offline Migration Guide

SemRAG is a fully self-contained, reproducible, offline-capable
development and coding-agent platform:

- **LLM** — exactly one model: `qwen3-coder:30b`, served by **local Ollama**
  at `http://localhost:11434`. No cloud LLM anywhere in the stack.
- **Embeddings** — exactly one model: `nomic-embed-text`, also local Ollama.
- **Runtime** — zero external network calls. All services talk over
  `127.0.0.1`/`localhost`; nothing dials out (verified — see Network audit).
- **Entry point** — OpenCode configured with the local Ollama provider and a
  portable SemRAG MCP server.

> **Why `qwen3-coder:30b` specifically.** The coding-agent workflow in
> SemRAG (retrieve → Plan-1 → read-before-write → Plan-2 strict-JSON → apply
> → validate → repair) needs a model with strong instruction following, JSON
> reliability, and enough reasoning budget. Smaller local models were
> tested end-to-end: `qwen3:4b` could not produce a valid Plan-2 change-set
> across multiple budget/configurations (truncation, empty content, wrong
> schema). The system is therefore tuned for and only officially supports
> `qwen3-coder:30b`.

---

## 1. Model contract

| Role          | Model              | Why                                            |
|---------------|--------------------|------------------------------------------------|
| LLM           | `qwen3-coder:30b`  | Only supported model; coding-agent contract    |
| Embedding     | `nomic-embed-text` | Only supported embedding model (768-d)         |

Both must be resolvable from local Ollama. Smaller models (e.g. `qwen3:4b`)
are **development-only** fallbacks; a deployment is not `OFFLINE_READY`
until `qwen3-coder:30b` is present.

### Hardware requirements for qwen3-coder:30b

The 30B model is ~18 GB in Q4 quant. You need one of:

- a GPU with **≥ 16 GB VRAM** (fast inference), or
- **≥ 24 GB free system RAM** for a CPU-only run (slow but workable; a single
  generation can take many minutes — this is why the timeouts below are
  large).

The reference machine this work was validated on (`NVIDIA GeForce RTX 3050
4GB Laptop`, 11 GiB RAM) **cannot** run `qwen3-coder:30b`. That is an honest
hardware blocker, not a configuration gap; the acceptance test reports it as
`BLOCKED` rather than pretending it passes.

---

## 2. Two-machine provisioning model

Internet is only needed on **Machine A** (build). **Machine B** (target)
must run everything offline.

```
Machine A (Internet)                      Machine B (offline)
────────────────────                      ──────────────────
scripts/build_offline_bundle.sh ────────▶ copy repo ──────▶ scripts/install_offline.sh
  • regenerates requirements-offline.txt                       • venv from local wheels only
  • downloads every pinned wheel                               • .env from .env.offline.example
  • builds the semrag wheel                                    • opencode.json paths rewritten
  • (--with-models) exports Ollama blobs       ─────────────▶ scripts/import_models_offline.sh
                                                               scripts/setup_postgres.sh
                                                               scripts/start_offline.sh
                                                               scripts/offline_acceptance_test.sh
```

If Machine B can reach a private artifact mirror, you can skip the wheel
bundle; the bundle keeps it fully air-gapped.

---

## 3. Step-by-step

### 3.1 Build the bundle (Machine A)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .
./scripts/build_offline_bundle.sh --with-models
```

Produces:

- `offline_wheels/` — every pinned wheel (`requirements-offline.txt`, which
  is regenerated from the venv as the exact snapshot) plus `semrag`'s wheel.
- `offline_models/` — portable Ollama blobs for `qwen3-coder:30b` and
  `nomic-embed-text` (~18 GB with `--with-models`).

Copy the whole repository directory to Machine B.

### 3.2 Install (Machine B, no Internet)

```bash
./scripts/install_offline.sh
```

- Creates `.venv` and installs **only** from `offline_wheels/`
  (`--no-index`, so pip can never reach the network).
- Installs the `semrag` wheel, writes `.env` from
  `.env.offline.example`, and rewrites `opencode.json` so the MCP launcher
  points at this machine's repo.

### 3.3 PostgreSQL + pgvector (Machine B)

```bash
./scripts/setup_postgres.sh
```

Idempotent: creates role `semrag`, database `semrag`, and the `vector`
extension. It tries `sudo -u postgres`, an explicit postgres password
(`PGPASSWORD=...`), root, then peer auth, and prints instructions if it
cannot reach a superuser.

### 3.4 Models (Machine B)

```bash
./scripts/import_models_offline.sh     # copies offline_models into local Ollama store
ollama list                            # verify both models appear
```

Restart Ollama if it was running during the import.

### 3.5 Start the stack (Machine B)

```bash
./scripts/start_offline.sh
```

Loads `.env` into the process environment (critical: the validation runner
reads `SEMRAG_UVM_ROOT` via `os.getenv`, and pydantic-settings does **not**
re-export `.env` variables), runs `alembic upgrade head`, verifies Ollama and
the models, and starts the API (uvicorn, `127.0.0.1:8000`) plus the dedicated
indexing worker (`python -m semrag.worker`). `--stop` stops both.

### 3.6 Acceptance test (Machine B)

```bash
./scripts/offline_acceptance_test.sh
```

Simulates an air gap **without root**: every Python process it starts runs
with a `sitecustomize` that refuses any non-loopback connect. It then
exercises the API, worker, validation, indexing, and the MCP server under
that block. Exit codes:

- `0` → `OFFLINE_READY: YES`
- `1` → `OFFLINE_READY: NO` (a check failed)
- `3` → `OFFLINE_READY: NO` (blocked: plumbing verified, but a required
  model/check is missing — e.g. `qwen3-coder:30b` not imported)

---

## 4. What was changed for offline (this repo)

### Configuration

- `opencode.json` — default model `ollama/qwen3-coder:30b`; cloud providers
  disabled (`disabled_providers`); MCP server invoked through the portable
  launcher `scripts/opencode_mcp_semrag.sh` with generous timeouts
  (`SEMRAG_MCP_TIMEOUT_SECONDS=7200`, `SEMRAG_MCP_INDEX_MAX_WAIT_SECONDS=1800`).
- `.env.example` — `SEMRAG_LLM_MODEL=qwen3-coder:30b`,
  `SEMRAG_LLM_NUM_PREDICT=8192`, `SEMRAG_LLM_TIMEOUT_SECONDS=1800`, plus
  `SEMRAG_UVM_ROOT` for portable validation.
- `.env.offline.example` — complete offline template (same LLM/embedding
  contract, dedicated-worker pattern `SEMRAG_API_RUNS_WORKER=false`, local
  trust/cors, and the MCP timeout variables).
- `src/semrag/core/settings.py` — default `llm_model` is now
  `qwen3-coder:30b`.

### Code

- `semrag_mcp_server.py` — replaced the hard-coded `600.0` timeouts with
  env-configurable `SEMRAG_MCP_TIMEOUT_SECONDS` (per request) and
  `SEMRAG_MCP_INDEX_MAX_WAIT_SECONDS` (job poll cap). Long coding-agent runs
  (10–30+ min on a 30B model) no longer abort.
- `src/semrag/validation/runner.py` + `config.py` — external dependency
  paths are portable: absolute, `$VAR`-expanded (e.g.
  `${SEMRAG_UVM_ROOT}`), or relative to the validation config directory.
  `configs/validation/rtl_uvm.json` now uses `"${SEMRAG_UVM_ROOT}"` for the
  UVM library and `"stubs/rtl_uvm"` (config-relative) for the lint stubs, so
  no machine-specific absolute paths are baked in.

### Tooling (new, in `scripts/`)

| Script | Purpose |
|---|---|
| `build_offline_bundle.sh` | regenerate the lock, download all wheels, build the semrag wheel, optionally export models |
| `install_offline.sh` | offline venv + wheel install + `.env` + `opencode.json` rewrite |
| `prepare_models.sh` / `import_models_offline.sh` | export / import the exact model set |
| `setup_postgres.sh` | idempotent role/db/pgvector creation |
| `start_offline.sh` | env load + migrations + API + worker |
| `offline_acceptance_test.sh` | air-gapped end-to-end verification |
| `opencode_mcp_semrag.sh` | portable MCP launcher used by opencode |

- `requirements-offline.txt` — regenerated from the validated venv
  (107 packages), dropping stale/unused entries (`networkx`, `graphifyy`,
  `tree-sitter-*`, etc.) and the non-portable editable `-e` line.
- `.gitignore` — `offline_wheels/` and `offline_bundle/` are build artifacts;
  `.env.offline.example` is committable.

---

## 5. Network audit

Scanned the runtime surface (`src/`, `semrag_mcp_server.py`, `scripts/`) for
URLs:

- Only `http://127.0.0.1` / `http://localhost` are used at runtime.
- The single external URL in the tree is `https://api.openai.com`, and it is
  a **default constant** in the optional OpenAI-compatible provider
  (`src/semrag/reasoning/llm/openai_llm.py`) that is **not on the ollama
  runtime path** and never dialed by the default configuration.
- `anthropic` is a declared-but-unimplemented provider that raises at
  construction (covered by a unit test) — no Anthropic SDK is installed.
- No `openai`/`anthropic`/`boto3`/`groq` Python packages are installed or
  imported anywhere.
- Tracing is disabled by default (`SEMRAG_TRACING_ENABLED=false`); the OTLP
  exporter is a no-op then.

The acceptance test re-checks all of this automatically under the socket
block.

---

## 6. Current status / honest reporting

`scripts/offline_acceptance_test.sh` on the reference machine reports:

```
summary: PASS=15 FAIL=0 BLOCKED=2
OFFLINE_READY: NO  (qwen3-coder:30b is not provisioned on this hardware)
```

Everything except the 30B model itself is verified offline (network
isolation, retrieval, validation, indexing, MCP, and full RAG with a
development fallback model). `OFFLINE_READY: YES` is achieved on a machine
that can run `qwen3-coder:30b` (see §1) after `import_models_offline.sh`
and a green acceptance run.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `required model missing: qwen3-coder:30b` | run `scripts/import_models_offline.sh` (bundle) or `ollama pull qwen3-coder:30b` (Internet) |
| `validation` fails with a file-not-found on `uvm_core` | `SEMRAG_UVM_ROOT` not in the API/worker environment — restart via `scripts/start_offline.sh` (it exports `.env`) |
| pip tries to reach the network during install | never use plain `pip install -r`; always `--no-index --find-links offline_wheels/` |
| long `semrag_code` run aborts | raise `SEMRAG_MCP_TIMEOUT_SECONDS` (default 7200 s) |
| index job never completes | ensure `scripts/start_offline.sh` started the worker (`SEMRAG_API_RUNS_WORKER=false`) |
| OpenCode doesn't connect to SemRAG MCP | confirm `opencode.json` → `mcp.semrag.command` points at `<repo>/scripts/opencode_mcp_semrag.sh` and the API is up on `127.0.0.1:8000`; restart opencode after editing config |
