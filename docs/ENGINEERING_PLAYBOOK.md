# SemRAG Engineering Playbook

> The day-to-day engineering manual for SemRAG. **Task-oriented.**
> Starts from "I need to do X" and ends with a verified, PR-ready change.

This is **not** architecture documentation, not an API reference,
not a package tour. It is the document you open when you have a
change to make and want to know:

- Where the change goes.
- Which Protocol / ABC to implement.
- Which factory entry / register call you must add so the new
  code is wired in.
- What tests verify the change.
- What you should **not** do — the mistakes already made and the
  architectural rules that exist for a reason.

If you want the deep WHY of an architectural decision, follow
the cross-references into the per-package READMEs. If you want the
file-level ownership of a subsystem, see
[`CODEBASE_GUIDE.md`](./CODEBASE_GUIDE.md). If you want end-to-end
lifecycle walkthroughs and the architectural rules that produce the
boundaries below, see
[`developer_guide.md`](./developer_guide.md).

**Audience:** software engineers with a fresh checkout who need to
ship a change without reading the whole codebase. **Not** for
operators (use [`DEPLOYMENT.md`](../DEPLOYMENT.md) and
[`docs/operations/`](./operations/)).

---

## Table of contents

1. Understanding the repository
2. Debugging workflow
3. Adding a new feature (general flow)
4. Adding a parser
5. Adding a chunker
6. Adding an embedding provider
7. Adding an embedding cache backend
8. Adding an LLM provider
9. Adding a prompt template
10. Adding a retrieval stage
11. Adding a reranker
12. Adding a new agent tool
13. Adding a REST endpoint
14. Adding middleware
15. Adding authentication / authorization
16. Adding Prometheus metrics
17. Adding tracing
18. Adding background jobs
19. Adding storage backends
20. Database migrations (Alembic)
21. Adding configuration
22. Adding evaluation metrics
23. Writing tests
24. Performance tuning
25. Memory optimization
26. Production debugging
27. Incident investigation
28. Reading logs
29. Understanding debug artifacts
30. Working with Docker
31. Working with Kubernetes
32. Working with Helm
33. Common engineering workflows
34. Common anti-patterns
35. Things you should NEVER do
36. Verification checklist before opening a PR
37. Code review checklist
38. Frequently modified files
39. Frequently misunderstood components
40. Glossary
41. Cross-reference index

---

## 1. Understanding the repository

**Files involved**

- [`../README.md`](../README.md) — quickstart, env vars, observability.
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — module layout and data-flow diagrams.
- [`./CODEBASE_GUIDE.md`](./CODEBASE_GUIDE.md) — file-by-file ownership.
- [`./developer_guide.md`](./developer_guide.md) — lifecycles, DI, security, observability.
- [`../src/semrag/README.md`](../src/semrag/README.md) — the package-level dependency graph (who imports whom).
- `../AGENTS.md` — contribution policy and the **priority order**: correctness > reliability > performance > readability > convenience.

**The mental model in five lines.** A request flows:
**HTTP → router → QueryService → HybridRetriever → PromptBuilder →
AgentLoop → LLM. Cite-from-context, validate, return.**

An indexing run flows:
**IngestionService → scanner → parsers → IR graph → chunkers →
embeddings → pgvector + IR rows in Postgres, one transaction.**

**Where the seams are.** Almost every "I want to swap X" question
maps to a Protocol or ABC:

| Concept                  | Protocol / ABC    | Lives in                                            |
| ------------------------ | ----------------- | --------------------------------------------------- |
| Parser                   | `Parser`          | `ingestion/parsers/interfaces.py`                   |
| Chunker                  | `Chunker`         | `ingestion/chunking/interfaces.py`                  |
| EmbeddingProvider        | `EmbeddingProvider` (`Protocol`) | `ingestion/embedding/interfaces.py` |
| EmbeddingCache           | `EmbeddingCache` (`Protocol`)    | `ingestion/embedding/interfaces.py` |
| EmbeddingCache (cache backends) | `InMemoryEmbeddingCache`, `JSONFileEmbeddingCache`, `RedisEmbeddingCache` factories | `ingestion/embedding/cache.py` |
| LLMProvider              | `LLMProvider` (`Protocol`) | `reasoning/llm/interfaces.py`                       |
| Reranker                 | `Reranker` (`Protocol`) | `retrieval/reranking.py`                            |
| VectorStore              | `VectorStore` (`Protocol`) | `storage/vector/interfaces.py`                      |
| GraphStore               | `GraphStore` (`Protocol`) | `storage/graph/interfaces.py`                       |
| JobStore                 | `JobStore` (`ABC`)| `jobs/store.py`                                     |
| QueryCache               | `QueryCache` (`ABC`) | `api/query_cache.py`                                |
| RateLimiter              | `RateLimiter` (`ABC`)| `api/rate_limit.py`                                 |
| APIKeyStore              | `APIKeyStore` (`ABC`)| `api/api_key.py`                                    |

Every "where do I plug in a new X?" question has the same shape:
implement the Protocol / ABC, register in the factory. For details
per type, jump to the relevant section in this playbook.

**Verification.** Before you write code, prove you understand the
seams:

```bash
# Every interface in one command
grep -rn "^class.*\(Protocol\|ABC\)" src/semrag/ | sort
# The dependency rule: a domain subpackage never imports from
# services/. Sanity-check it.
python -c "
from semrag.retrieval import pipeline
import inspect
src = inspect.getsourcefile(pipeline)
print(src)
forbidden = ['semrag.services', 'semrag.api', 'semrag.jobs']
for f in forbidden:
    assert f not in open(src).read(), f'{src} must not import {f}'
print('ok')
"
```

For implementation details, see [`./developer_guide.md`](./developer_guide.md).

---

## 2. Debugging workflow

Use this whenever a symptom lands in your lap. The whole chain is:
reproduce → localize → bisect → instrument → verify.

**Files involved**

- `src/semrag/core/observability.py` — structured logging.
- `src/semrag/core/metrics.py` — Prometheus registry.
- `src/semrag/core/tracing.py` — OpenTelemetry facade.
- `src/semrag/core/debug_artifacts.py` — per-run artifact writer.
- `src/semrag/core/query_trace.py` — per-query accumulated trace.

**Step 1 — Reproduce.** Always write the failing command first,
before touching code:

```bash
# Activate env
source .venv/bin/activate
# CLI path
python run_local_qa.py ask --project-id rtl_uvm --question "..."
# API path
curl -fsS -H "x-api-key: dev-local-key" \
  -H "content-type: application/json" \
  -d '{"question":"..."}' \
  http://localhost:8000/v1/projects/rtl_uvm/answer | jq
```

Pick the path the operator actually hits. CLI bugs and API bugs
can have completely different root causes (`cli.py` builds its own
service; routes use `get_query_service`).

**Step 2 — Localize.** Set the log level to TRACE and capture the
structured log:

```bash
SEMRAG_LOG_LEVEL=TRACE \
SEMRAG_LOG_FORMAT=json \
  python run_local_qa.py ask --project-id rtl_uvm --question "..." 2>&1 \
  | tee /tmp/semrag-run.json
```

Every stage emits a `stage=... event=start|finish elapsed_ms=...`
log line. The first stage that **stops** is where the bug lives. See
[`developer_guide.md` §14](./developer_guide.md#14-debugging-workflow)
for the full per-symptom triage table.

**Step 3 — Bisect.** For the failure mode, the article-shaped
rule of thumb is:

- Wrong/empty chunks → read `src/semrag/retrieval/pipeline.py`.
- Chunks look right, the prompt is wrong → read `src/semrag/reasoning/prompting/pipeline.py`.
- Prompt looks right, the LLM lies → read the citation validator
  + `src/semrag/evaluation/citation_validator.py`.
- Status code is 5xx → read `src/semrag/api/exception_handler.py`
  + the matching router.

**Step 4 — Instrument.** If the bug isn't on the obvious path, set
debug-artifacts and read the bundle:

```bash
SEMRAG_LOG_LEVEL=DEBUG python run_local_qa.py ask ...
# Bundle under .semrag/debug/<run_id>/
ls .semrag/debug/<run_id>/
```

See §29 below for what each file in that directory means.

**Step 5 — Verify.** After the fix:

```bash
# Lint
ruff check src/ tests/
# Type check
mypy src/semrag
# Unit tests
pytest tests/unit/<package> -x
# The integration suite for the touched layer
pytest tests/integration/ -x
```

**Common mistakes**

- Guessing without reproducing. The repo's
  [`../AGENTS.md`](../AGENTS.md) is explicit: "Never guess." Every
  diagnostic should cite a log line, an HTTP response, or a
  test failure.
- Reading `print()`-style debug output instead of the structured
  log. `core/observability.py` emits JSON in production; human-readable
  in development. Don't grep for `INFO:` — grep for `stage=`.
- Editing the symptom in the wrong place. A report names a symptom;
  the root cause is two packages upstream. See
  [`../src/semrag/README.md`](../src/semrag/README.md) for the
  dependency diagram.

For implementation details, see [`./developer_guide.md` §14](./developer_guide.md#14-debugging-workflow).

---

## 3. Adding a new feature (general flow)

The generic recipe applies to every "add X" in this playbook:

**1. Identify the Protocol.** Find the Protocol or ABC the rest
of the system already depends on. If you find yourself inventing a
new abstraction, **stop** — the existing one probably covers you
with one new method.

**2. Implement the concrete class.** One file. One class. Implement
the Protocol methods. If a constructor takes collaborators, take them
as named keyword arguments; do **not** reach for module-level
singletons.

**3. Register in the factory.** Every Protocol has exactly one
factory function. Add a case to that function, gated by a Settings
knob if the new backend needs runtime configuration. Add the Settings
knob (see §21).

**4. Add tests.** Mirror the existing `tests/unit/<package>/test_*.py`
shape: one happy path, one failure path, one boundary-case path.
Mock the dependencies; do not hit a real Postgres / LLM in unit
tests. See §23.

**5. Document.** Run `make format` then `make lint` then `make typecheck`.
The pre-commit hooks (or CI) enforce style. Then update the relevant
package README — keep the changes small, link to the dev guide where
appropriate. The playbook entry for your feature type (parser,
provider, etc.) tells you **which** files to touch.

**6. Verify end-to-end.** Use the same `python -m semrag …` /
`curl …` commands that exercise the feature in production. Add a
load-test scenario only if the change introduces a new performance
shape.

**Common mistakes**

- Skipping the factory registration. Adding the class without
  wiring it in means the new code never runs.
- Adding a Settings knob nobody asked for. Add only what the
  factory's new branch needs to discriminate.
- Forgetting to update tests that pin the previous factory output
  (e.g. tests that assert `build_embedding_provider` returns a
  specific concrete type).

For implementation details, see [`./developer_guide.md` §13](./developer_guide.md#13-extension-points).

---

## 4. Adding a parser

When you have a new file type (or a new file extension for an
existing language family) that the system should index.

**Files involved**

- `src/semrag/ingestion/parsers/interfaces.py` — the `Parser` Protocol
  and `ParserRegistry`.
- `src/semrag/ingestion/parsers/pipeline.py` — build_default_registry().
- `src/semrag/ingestion/parsers/<language>/parser.py` — your new
  module (one new file per language family).
- `src/semrag/core/settings.py` — settings knobs only if the
  parser needs a runtime flag.
- `tests/unit/parsers/<language>/` — mirror the package layout.

**Workflow**

1. Read the existing parser closest to yours in
   `src/semrag/ingestion/parsers/<existing language>/`. The shape is
   the same; copy a small one (e.g. `python_tcl`) to start.
2. Implement `Parser` (returns a list of `ParseResult` or your
   domain-specific shape the IR builder consumes).
3. Register in `build_default_registry()` (or in
   `ingestion/parsers/pipeline.py` per the existing convention).
4. Add a unit test under `tests/unit/parsers/<language>/` that
   parses a 50-line fixture and asserts the structural counts.

**Verification**

```bash
# Python path
python -c "
from semrag.ingestion.parsers.pipeline import build_default_registry
reg = build_default_registry()
# Assert your parser is registered for the right extension(s).
"
# End-to-end
python run_local_qa.py index --project-id <id> --root <path-to-fixture>
```

**Common mistakes**

- Putting the parser outside the language-specific subpackage.
  Adding `parsers/foo.py` directly is allowed for one-off cases,
  but a language family should get its own subdirectory so tests
  and fixtures are colocated.
- Returning rich Python objects the IR builder doesn't know. The
  IR builder accepts a fixed `ParseResult` shape — extend that
  shape and the IR consumer in lockstep.

For implementation details, see
[`../src/semrag/ingestion/README.md`](../src/semrag/ingestion/README.md#ingestionparsers).

---

## 5. Adding a chunker

When the default window chunker isn't right for a new content type
(e.g. a new language family whose structure you can exploit).

**Files involved**

- `src/semrag/ingestion/chunking/interfaces.py` — the `Chunker`
  Protocol and `ChunkerRegistry`.
- `src/semrag/ingestion/chunking/pipeline.py` — the chunker pipeline.
- `src/semrag/ingestion/chunking/<chunker>.py` — your new chunker (one
  file per chunker; co-locate language-specific chunks inside
  their language subpackage).
- `tests/unit/chunking/` — one test file per chunker.

**Workflow**

1. Read an existing language-aware chunker (`python_chunker.py` or
   `systemverilog_chunker.py`) to see how the Protocol maps to a
   node-driven walk.
2. Implement `chunk(node, project_id, …) -> Iterable[Chunk]`. Use
   `id_scheme.chunk_id(project_id, relative_path, chunk_kind,
   disambiguator)` to get a deterministic ID — **never** roll your
   own hash.
3. Register in `ChunkerRegistry` (the registry is keyed by file
   extension / language and selected by `parsers/pipeline.py`).
4. Verify chunk IDs match across re-index runs. The chunk ID is
   the upsert key in `pgvector_store.upsert_chunks`, so a
   non-deterministic ID causes duplicate embeddings.

**Verification**

```bash
python -c "
from semrag.ingestion.chunking.pipeline import chunk_project
chunks = list(chunk_project('path/to/fixture.py', ...))
ids = [c.chunk_id for c in chunks]
# Run twice; assert ids are identical.
"
```

**Common mistakes**

- Inventing a new chunk-ID scheme. Use
  `chunk_id(...)` in `ingestion/chunking/id_scheme.py`. Same content
  → same ID is the contract that makes incremental indexing work.
- Returning chunks without `line_start` and `line_end`. The
  citation validator and the prompt builder both depend on those
  fields to produce `[Source: <path>, lines X-Y]` references.

For implementation details, see
[`../src/semrag/ingestion/README.md`](../src/semrag/ingestion/README.md#ingestionchunking).

---

## 6. Adding an embedding provider

When you want to point the system at a vector model the codebase
doesn't yet support.

**Files involved**

- `src/semrag/ingestion/embedding/interfaces.py` — `EmbeddingProvider`
  Protocol.
- `src/semrag/ingestion/embedding/factory.py` — `build_embedding_provider(settings)`.
- `src/semrag/ingestion/embedding/<provider>.py` — your new
  provider (e.g. `cohere_provider.py`, `openai_provider.py`).
- `src/semrag/core/settings.py` — add `SEMRAG_EMBEDDING_PROVIDER`
  cases and any provider-specific settings (model name, base URL,
  API key, dimension).
- `tests/unit/embedding/test_<provider>.py`.

**Workflow**

1. Implement `EmbeddingProvider.embed_texts(texts) -> list[list[float]]`.
   The batched shape is important: the embedding pipeline calls
   `embed_texts` with a `list[str]` and expects one `list[float]`
   per input. If your underlying API is per-text, batch internally.
2. Implement the optional health probe (`probe()` if your provider
   has an obvious health endpoint).
3. Add a case in `build_embedding_provider` keyed on the
   `EmbeddingProviderType` enum (or whatever factory the codebase
   uses today; see §21 for the Settings side).
4. Add the provider-specific settings in `core/settings.py` — base
   URL, model name, dimension, API key. **All** of them; the API
   key is read from the secret manager at runtime, not from a file.
5. If the new provider changes embedding `dimension`, ensure you
   set `SEMRAG_EMBEDDING_DIMENSION` and that existing indexes are
   re-created (a different dimension is a breaking change for
   the pgvector column; see §20).

**Verification**

```bash
# Unit test
pytest tests/unit/embedding/test_<provider> -x
# CLI path (real provider)
SEMRAG_EMBEDDING_PROVIDER=<your-provider> \
SEMRAG_EMBEDDING_MODEL=<model-name> \
SEMRAG_EMBEDDING_DIMENSION=<dim> \
  python run_local_qa.py ask --project-id <id> --question "..."
```

**Common mistakes**

- Implementing per-text calls in a way that hits the rate limit.
  Use the batched shape; the codebase already calls
  `embed_texts(texts)` with thousands of chunks in a single
  ingestion run.
- Returning numpy arrays instead of plain `list[float]`. The
  Protocol uses Python lists so the payload is serializable.
- Forgetting the cache. Read `EmbeddingCache` Protocol; the
  pipeline always wraps the provider with a cache. See §7.

For implementation details, see
[`../src/semrag/ingestion/README.md`](../src/semrag/ingestion/README.md#ingestionembedding).

---

## 7. Adding an embedding cache backend

When the existing `InMemoryEmbeddingCache`, `JSONFileEmbeddingCache`,
or `RedisEmbeddingCache` doesn't fit your deployment.

**Files involved**

- `src/semrag/ingestion/embedding/interfaces.py` — `EmbeddingCache` Protocol.
- `src/semrag/ingestion/embedding/cache.py` — the three existing backends and
  the factory.
- `src/semrag/core/settings.py` — `SEMRAG_EMBEDDING_CACHE_BACKEND` knob.

**Workflow**

1. Implement `EmbeddingCache.get(key) -> list[float] | None`,
   `put(key, embedding, ttl=...)`, and `close()` (a no-op is fine
   in-memory; mandatory for file- and network-backed stores).
2. Add a case in the cache factory (`EmbeddingCacheFactory`
   in `cache.py`).
3. Wire the new backend in the embedding pipeline constructor
   (`ingestion/embedding/pipeline.py`).
4. The cache key is whatever the pipeline passes — usually
   `(provider, model, text_hash)`. Do not invent your own key
   format that diverges from `hashlib` of `text.encode("utf-8")`.

**Verification**

```bash
pytest tests/unit/embedding/test_cache.py -x
# End-to-end with a small corpus
python run_local_qa.py index --project-id <id> --root <small fixture>
# Re-run; the second run should hit the cache for already-embedded chunks.
```

**Common mistakes**

- Implementing `close()` that doesn't release file handles / Redis
  connections. Tests that exercise the same cache instance across
  multiple ingestion runs will leak.
- Caching without a TTL or size cap. Without one, the cache
  consumes unbounded memory; in production this is a memory-
  exhaustion incident waiting to happen.

For implementation details, see
[`../src/semrag/ingestion/README.md`](../src/semrag/ingestion/README.md#ingestionembedding).

---

## 8. Adding an LLM provider

When you want to plug in a model API the codebase doesn't yet
support (Anthropic, Cohere, a custom HTTP endpoint, etc.).

**Files involved**

- `src/semrag/reasoning/llm/interfaces.py` — `LLMProvider` Protocol and
  `CompletionResult` / `StreamChunk` / `ToolCall` shapes.
- `src/semrag/reasoning/llm/factory.py` — `build_llm(settings)`.
- `src/semrag/reasoning/llm/<provider>.py` — your new provider.
- `src/semrag/core/settings.py` — provider type, model, base URL,
  API key, and any provider-specific knobs.

**Workflow**

1. Implement `complete(messages, tools=None) -> CompletionResult`
   and `stream(messages, tools=None) -> Iterator[StreamChunk]`.
   Both must produce the **same** first token at the same logical
   point in the response — agents calling `stream` and the prompt
   path calling `complete` should be interchangeable.
2. Map provider-side tool calls to/from the codebase's `ToolCall`
   shape. The codebase's `ToolCall` is a pydantic model with
   `name`, `args`, `id` — the wire format varies by provider.
3. Add a case in `build_llm` keyed on the `LLMProviderType` enum.
4. Add provider settings in `core/settings.py`.

**Verification**

```bash
# Unit: mock the HTTP layer, exercise the message mapping
pytest tests/unit/reasoning/llm/test_<provider>.py -x
# Integration: real model
SEMRAG_LLM_PROVIDER=<your-provider> \
SEMRAG_LLM_MODEL=<model> \
SEMRAG_LLM_BASE_URL=<url> \
  python run_local_qa.py ask --project-id <id> --question "..."
```

**Common mistakes**

- Not implementing `stream` or implementing it differently from
  `complete`. The agent loop always calls `stream`; a `stream`
  that returns the full text in one chunk defeats the streaming
  UX and inflates time-to-first-token.
- Not handling the missing-API-key path. The factory should raise a
  `RuntimeConfigurationError` (defined in `api/startup_validation.py`)
  at construction time with an actionable message, not at first call.
- Mapping tool-call args as `dict[str, Any]` instead of through the
  `ToolCall.args` field. The agent loop expects `args` to be a
  JSON-serializable dict; if your provider returns XML or a custom
  format, normalize in the mapper.

For implementation details, see
[`../src/semrag/reasoning/README.md`](../src/semrag/reasoning/README.md).

---

## 9. Adding a prompt template

When you want to change the system prompt, add a per-feature
template (e.g. "summary-style" vs "explanation-style"), or wire a
new template-selection rule into `PromptBuilder`.

**Files involved**

- `src/semrag/reasoning/prompting/templates.py` — the canonical
  `SYSTEM_PROMPT` (and any future role-specific templates).
- `src/semrag/reasoning/prompting/pipeline.py` — `PromptBuilder`,
  token budgeting order, citation envelope.
- `src/semrag/reasoning/prompting/context_assembler.py` — the order
  in which system / history / retrieved context are packed.
- `src/semrag/core/settings.py` — any new template-selection knob.

**Workflow**

1. Add the new template as a module-level constant in `templates.py`.
   Don't construct templates at runtime; they are baked at startup.
2. If the template selection is **configurable**, expose a
   `PromptTemplateType` enum and a Settings field that drives the
   branch in `PromptBuilder.build(...)`.
3. Re-verify the token-budget order. `context_assembler.py` allocates
   tokens in a deliberate order: **system + history first, retrieved
   context last.** Changing the order breaks the design; the LLM
   sees the budget shrink from the chunks, not from the system prompt.

**Verification**

```bash
# Snapshot the rendered prompt for a known query
SEMRAG_LOG_LEVEL=DEBUG python -c "
from semrag.api.dependencies import get_query_service
svc = next(get_query_service(None))
import asyncio
ans = asyncio.run(svc.ask('project-id', 'test'))
"
cat .semrag/debug/<run_id>/prompt/final_prompt.txt | head -50
# Token count
SEMRAG_LOG_LEVEL=DEBUG ... | grep token_count=
```

**Common mistakes**

- Adding HTML / Markdown that the model's tokenizer treats
  inconsistently. Stick to plain text and let the formatter add
  the `[Source: …]` envelope.
- Treating the system prompt as user-editable per request. It
  isn't; the system prompt is part of the security model. Any
  per-request customization belongs in the user message.
- Forgetting that the prompt builder estimates tokens with
  `token_estimator.py`, which is a heuristic. Run with the actual
  model's tokenizer in `tests/integration/` to spot drift.

For implementation details, see
[`../src/semrag/reasoning/README.md`](../src/semrag/reasoning/README.md#reasoningprompting).

---

## 10. Adding a retrieval stage

When you want to add a new ranking signal to
`HybridRetriever.retrieve()` (e.g. a domain-specific reranker, a
query expansion, a metadata-based boost).

**Files involved**

- `src/semrag/retrieval/pipeline.py` — the orchestrator.
- `src/semrag/retrieval/fusion.py` — if the new signal participates
  in fusion.
- `src/semrag/retrieval/intent.py` + `retrieval/ranking/intent.py` —
  if the new signal is intent-gated.
- `src/semrag/retrieval/source_authority.py` +
  `retrieval/ranking/source.py` — if the new signal is
  document/source-based.
- `src/semrag/retrieval/ranking/diversity.py` — if the new signal is
  a selection-time diversity rule.
- `src/semrag/retrieval/reranking.py` — if the new signal is a
  reranker (see §11).
- `src/semrag/core/settings.py` — feature flags.

**Workflow**

1. Decide where the stage sits in the pipeline. Existing order:
   vector + lexical → query expansion → RRF → noisy-source penalty →
   graph expansion → intent boost → source authority → optional
   rerank → diversity-aware selection → stable sort + top-k slice →
   hybrid reservation. See
   [`../src/semrag/retrieval/README.md`](../src/semrag/retrieval/README.md#4-end-to-end-pipeline)
   for the rationale on order.
2. Implement the stage as a pure function `(results, ctx) -> results`
   where `results: list[RetrievalResult]`. **Never** mutate a
   `RetrievalResult` in place — use `model_copy(update={...})` so
   the per-stage trace can record the score at the moment the
   stage ran.
3. Wire the stage into `HybridRetriever.retrieve()` between the
   two adjacent stages; record its cost in the
   `RankingTrace.ChunkTrace` (a new field).
4. Add a feature flag in `core/settings.py` (default off) and
   thread it through the `HybridRetriever` constructor. New
   ranking stages default to off until the eval harness proves
   they help.

**Verification**

```bash
# Unit: stage alone, on a fixture of 10 results
pytest tests/unit/retrieval/test_pipeline.py -x
# Integration: against the live retriever with feature flag on
SEMRAG_LOG_LEVEL=DEBUG \
SEMRAG_ENABLE_<NEW_STAGE>=true \
  python run_benchmarks.py
# Eval: did precision/recall go up vs. the baseline?
cat benchmark_results/<run>/summary.md
```

**Common mistakes**

- Re-running RRF after the new stage. RRF is run **once**, after
  the two first-party signals (vector + lexical) — adding a
  third RRF on the augmented list compresses all scores into a
  narrow band and breaks the intent boost / reranker.
- Mutating `fused_score` directly. The contract is
  `fused_score * (1 + weight * stage_score)` — replace, never
  overwrite.
- Ignoring the existing intent boost / reranker ordering. The
  pipeline stacks signals in a specific order because each one
  acts on the previous one's output.

For implementation details, see
[`../src/semrag/retrieval/README.md`](../src/semrag/retrieval/README.md).

---

## 11. Adding a reranker

When you want to swap or extend `LexicalOverlapReranker` /
`HTTPRerankerClient`.

**Files involved**

- `src/semrag/retrieval/reranking.py` — `Reranker` Protocol and the
  existing implementations.
- `src/semrag/core/settings.py` — `RERANKER` enum.

**Workflow**

1. Implement `Reranker.rerank(query, results, top_k) -> list[RetrievalResult]`.
   **The contract is multiplicative, not replacement.** The
   module docstring spells out why: a strong-RRF chunk that the
   reranker scores low must still rank ahead of a weak-RRF chunk
   the reranker scores high. Use the same
   `fused_score * (1 + weight * reranker_score)` formula the
   existing rerankers use.
2. Wire your reranker into the `Reranker` enum in `core/settings.py`
   and the factory branch in `retrieval/reranking.py`.
3. The `HTTPRerankerClient` shows the retry / timeout pattern for
   network-backed rerankers. Copy it before adding your own HTTP
   logic.

**Verification**

```bash
pytest tests/unit/retrieval/test_reranking.py -x
# Sanity: reranker must not reduce NDCG@10 on the golden benchmark
SEMRAG_LOG_LEVEL=DEBUG python run_benchmarks.py | grep ndcg
```

**Common mistakes**

- Replacing `fused_score` instead of multiplying it. This is the
  single most common bug in custom rerankers and is the example
  called out in the existing `LexicalOverlapReranker` docstring.
- Returning more than `top_k` results. The pipeline slices
  afterwards, but your reranker should respect `top_k` so the
  cost is bounded.

For implementation details, see
[`../src/semrag/retrieval/README.md`](../src/semrag/retrieval/README.md#3-module-responsibilities).

---

## 12. Adding a new agent tool

When you want the agent loop to be able to call something it can't
today (e.g. `list_directory`, `get_symbol_definition`,
`run_simulation`).

**Files involved**

- `src/semrag/reasoning/agents/tools.py` — `Tool` Protocol,
  `ToolRegistry`, and the three existing tools.
- `src/semrag/services/services.py` — `_build_tools(project_id)`
  constructs the registry; add your tool there.
- `src/semrag/reasoning/llm/interfaces.py` — the `ToolCall` /
  tool-message contract the LLM uses; your tool's schema must be
  expressible in the LLM's native tool-call shape.
- `tests/unit/agents/test_tools.py`.

**Workflow**

1. Implement `Tool.name`, `Tool.description`, `Tool.args_schema`
   (pydantic), and `Tool.run(args, ctx) -> str`.
2. The three existing tools show the closure pattern — your tool's
   `run` typically closes over collaborators (the retriever, the
   session, the project root). Pass them via closure, not via
   constructor-imported singletons.
3. Register in `_build_tools` (services layer), gating by Settings
   if the tool needs a feature flag (e.g. `ReadFileTool` is gated
   by `project_root`).
4. Update the agent loop's system-prompt addition if your tool
   should be advertised differently. Default: list every tool in
   the registry with a short name + description.

**Verification**

```bash
# Unit: tool.run with a fixture of args
pytest tests/unit/agents/test_tools.py -x
# Integration: question that triggers your tool
python run_local_qa.py ask --project-id <id> --question "<question>"
# Confirm the tool actually ran
cat .semrag/debug/<run_id>/llm/response.json | grep <your_tool_name>
```

**Common mistakes**

- Calling the LLM from inside a tool. The agent loop owns the LLM.
  A tool that needs an LLM should request it via the tool's
  collaborators, not import the LLM provider directly.
- Returning non-serializable objects. The tool's `run` returns
  JSON-serializable data (string, dict, list). The agent loop
  re-feeds the result into the next LLM prompt.

For implementation details, see
[`../src/semrag/reasoning/README.md`](../src/semrag/reasoning/README.md#reasoningagents).

---

## 13. Adding a REST endpoint

When you want a new operation exposed over HTTP (a new resource, a
new verb, a new shape).

**Files involved**

- `src/semrag/api/routers/<resource>.py` — your new router.
- `src/semrag/api/app.py` — `app.include_router(...)` to register.
- `src/semrag/api/schemas.py` — request/response pydantic models.
- `src/semrag/api/dependencies.py` — DI provider functions; add
  yours here.
- `src/semrag/api/auth.py` + `api/api_key.py` + `api/project_auth.py` —
  authentication / authorization wiring.

**Workflow**

1. Define the request/response models in `schemas.py`. Don't
   import raw ORM models into the router — Pydantic models are the
   wire format.
2. Add a dependency provider in `dependencies.py` for whatever the
   route needs (DB session, query service, ingestion service, etc.).
3. Write the router. **Always:**
   - Apply `Depends(require_api_key)` for auth.
   - Apply `Depends(get_project_authorizer())` for project-scoped
     routes (per-project API keys).
   - Apply `Depends(enforce_rate_limit)` if the endpoint costs
     work.
   - Map service exceptions to HTTP via the global exception
     handler; don't catch them in the handler.
4. Register the router in `api/app.py` under `app.include_router(...)`.
5. Add an entry to `OpenAPI` tags.

**Verification**

```bash
# OpenAPI freshness
python -c "
import json, urllib.request
print(json.dumps(json.loads(urllib.request.urlopen('http://localhost:8000/openapi.json').read()), indent=2)[:500])
" | grep "<your-route>"
# Curl
curl -sS -H "x-api-key: dev-local-key" \
  -H "content-type: application/json" \
  -d '<your request>' \
  http://localhost:8000/<your-route> | jq
# Auto-generated client (the API ships an OpenAPI client)
make generate-client
```

**Common mistakes**

- Returning a raw ORM object from a handler. Pydantic models only.
- Adding a custom exception class. The codebase has a standard
  error envelope; raise an existing exception and let the
  global handler format it.
- Forgetting to add a rate-limit dependency on expensive endpoints.
  Per-key 60/min default; tighten per endpoint if needed.
- Putting business logic in the router. The router's job is HTTP:
  parse, validate, dispatch, format. Logic belongs in
  `services/` or the appropriate domain subpackage.

For implementation details, see
[`../src/semrag/api/README.md`](../src/semrag/api/README.md).

---

## 14. Adding middleware

When you want to add a cross-cutting concern over the request
pipeline (e.g. an audit logger, a custom header, a feature-flag
gate).

**Files involved**

- `src/semrag/api/middleware.py` — the six existing middlewares and
  the registration order.
- `src/semrag/api/app.py` — middlewares are added via `app.add_middleware(...)`.

**Workflow**

1. Subclass `BaseHTTPMiddleware` from Starlette. The existing
   `MetricsMiddleware`, `LoggingMiddleware`, etc. show the pattern.
2. Place the middleware in `middleware.py` and register in
   `api/app.py` in the correct **ordering**. Ordering matters:
   `RequestID` must run before `Logging` (so logs carry the id);
   `Metrics` must run before `Logging` (so 4xx/5xx don't pollute
   the success metrics). Verify by reading the existing
   `add_middleware(...)` calls in order.
3. Avoid blocking I/O. Middlewares run per-request; long operations
   tax the event loop.

**Verification**

```bash
pytest tests/unit/api/test_middleware.py -x
# Inspect headers in the response
curl -i -H "x-api-key: dev-local-key" \
  http://localhost:8000/livez | grep -i "<your-header>"
```

**Common mistakes**

- Forgetting to thread `request.state` values into downstream
  middlewares (e.g. the request ID set by `RequestIDMiddleware`
  must be readable by `LoggingMiddleware`).
- Adding an exception handler inside the middleware. Use the
  global `app.exception_handler(...)` registration in `app.py`.

For implementation details, see
[`../src/semrag/api/README.md`](../src/semrag/api/README.md).

---

## 15. Adding authentication / authorization

The codebase ships API keys (header or bearer) and per-project
authorization. New auth flows integrate at the dependency layer.

**Files involved**

- `src/semrag/api/auth.py` — `require_api_key` dependency.
- `src/semrag/api/api_key.py` — `APIKey` model, `APIKeyStore` ABC,
  `hash_api_key`.
- `src/semrag/api/api_key_store.py` — `InMemoryAPIKeyStore`,
  `PostgresAPIKeyStore`, `seed_initial_keys`.
- `src/semrag/api/project_auth.py` — `get_project_authorizer()`.
- `src/semrag/core/settings.py` — `SEMRAG_API_KEYS`,
  `SEMRAG_API_KEY_STORE`, `SEMRAG_ENVIRONMENT`,
  `SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION`.

**Workflow — adding a new auth flow**

1. Identify the **boundary** you want to enforce: per-route, per-
   project, or per-method. The codebase already supports all three;
  add yours at the same granularity.
2. If your flow needs an external store, implement `APIKeyStore`
   (or a sibling ABC) and register in the same factory slot.
3. Add a Settings knob if the flow is feature-gated.
4. Write the dependency (`require_*`) and apply it to routes via
   `Depends(...)`.

**Verification**

```bash
pytest tests/unit/api/test_auth.py -x
# Forbidden path
curl -i http://localhost:8000/v1/projects/x/index
# Authorized path
curl -i -H "x-api-key: <key>" \
  http://localhost:8000/v1/projects/x/index
# Per-project auth
curl -i -H "x-api-key: <key>" -H "x-project-key: <proj-key>" \
  http://localhost:8000/v1/projects/<other-project>/index
```

**Common mistakes**

- Storing raw API keys. Use `hash_api_key` from
  `src/semrag/api/api_key.py`. Plaintext-on-disk is a security
  incident.
- Skipping `SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION`. The startup
  validation rejects the literal `dev-local-key` default in
  production; don't disable this.
- Adding auth that runs *after* the rate limiter. The order is
  auth → rate-limit → handler; rate-limiting unauthenticated
  traffic lets attackers burn the budget.

For implementation details, see
[`../src/semrag/api/README.md`](../src/semrag/api/README.md) and
[`developer_guide.md` §8](./developer_guide.md#8-security-model).

---

## 16. Adding Prometheus metrics

When you want a new counter, gauge, histogram, or summary
exposed at `/metrics`.

**Files involved**

- `src/semrag/core/metrics.py` — the central registry and the
  existing ~25 metrics.
- `src/semrag/core/observability.py` — labels and conventions.
- `deploy/monitoring/prometheus/rules/semrag.yml` — alert rules
  (add a new rule if the metric deserves an SLO).
- `deploy/monitoring/grafana/dashboards/` — JSON dashboards (add a
  panel if the metric deserves one).

**Workflow**

1. In `core/metrics.py`, add the new metric. Use the matching
   type:
   - **Counter** — monotonically increasing (requests, errors).
   - **Gauge** — can go up or down (queue depth, pool size).
   - **Histogram** — duration / size distributions.
   - **Summary** — pre-computed quantiles (rarely used here).
2. Apply consistent **label cardinality**. Labels like `path`,
   `status`, `provider`, `model`, `chunk_type` are fine. **Do
   not** use labels with unbounded cardinality (request IDs,
   user IDs, paths with query strings).
3. Instrument the call site with `with metric.time():` (or the
   appropriate context manager) — do not write
   `metric.observe(time.time() - t)` ad-hoc; the context manager
   handles exceptions.
4. Add a corresponding `deploy/monitoring/prometheus/rules/semrag.yml`
   alert if the metric has an SLO.

**Verification**

```bash
# Unit: metric is registered and updates on call
pytest tests/unit/core/test_metrics.py -x
# Live: hit the endpoint, scrape /metrics
curl -fsS http://localhost:8000/metrics | grep "<your_metric>"
```

**Common mistakes**

- High-cardinality labels. **Every unique label value creates a
  new time series.** A label like `request_id` would create one
  series per request and OOM the scraper.
- Adding the metric to a test fixture that isn't torn down. Tests
  re-register metrics; check the existing patterns for using
  `REGISTRY.unregister(...)` on setup.
- Putting metrics in places that already have one. Don't add a
  second duration metric for the same operation.

For implementation details, see
[`developer_guide.md` §9.2](./developer_guide.md#92-metrics-coremetricspy).

---

## 17. Adding tracing

When you want a new span in the OpenTelemetry trace, or want to
trace a code path that isn't yet traced.

**Files involved**

- `src/semrag/core/tracing.py` — the `tracer(name)` helper and
  the no-op fallback.
- `src/semrag/core/observability.py` — trace-context propagation
  into structured logs.
- `deploy/monitoring/` — the trace sink (Tempo / Jaeger / OTLP
  collector) is configured out-of-tree.

**Workflow**

1. Acquire a tracer at module scope:
   ```python
   from semrag.core.tracing import tracer
   _tracer = tracer("semrag.<package>.<module>")
   ```
2. Wrap the operation:
   ```python
   with _tracer.start_as_current_span("operation_name"):
       ...
   ```
3. Add attribute annotations only for **bounded-cardinality**
   values: `db.system=postgres`, `rag.intent=architecture`, etc.
   Avoid user input.
4. Propagate context across boundaries. The codebase has helper
   utilities in `core/tracing.py` (`current_trace_id_hex`,
   `current_span_id_hex`) — use these rather than rolling your
   own context extraction.

**Verification**

```bash
SEMRAG_TRACING_ENABLED=true \
SEMRAG_TRACING_EXPORTER=otlp \
SEMRAG_TRACING_ENDPOINT=http://localhost:4317 \
  python run_local_qa.py ask --project-id <id> --question "..."
# Confirm in Tempo/Jaeger UI
```

**Common mistakes**

- Adding unbounded attributes (`user_id`, `chunk_id`). Bounded
  attributes only.
- Forgetting to enable the tracer (`Settings.tracing_enabled`).
  The facade is a no-op when the SDK is unconfigured — silent
  failures look like "the trace disappeared".
- Tracing inside a tight loop. Spans are not free; wrap the
  outer operation only.

For implementation details, see
[`developer_guide.md` §9.3](./developer_guide.md#93-tracing-coretracingpy).

---

## 18. Adding background jobs

When you want a long-running operation that shouldn't block the
HTTP request (indexing, bulk re-embed, recompute).

**Files involved**

- `src/semrag/jobs/store.py` — `JobStore` ABC, `InMemoryJobStore`,
  `PostgresJobStore`.
- `src/semrag/jobs/executor.py` — `JobExecutor` ABC,
  `ThreadPoolJobExecutor`, `ExecutorFactory`.
- `src/semrag/jobs/worker.py` — the worker entry point
  (`python -m semrag.worker`), `indexing_worker`.
- `src/semrag/jobs/manager.py` — the `JobManager` and per-stage
  progress calculator.
- `src/semrag/jobs/models.py` — job state machine
  (QUEUED → RUNNING → DONE / FAILED / CANCELLED).
- `src/semrag/core/settings.py` — `SEMRAG_JOB_STORE_BACKEND`,
  `SEMRAG_API_RUNS_WORKER`.

**Workflow — adding a new job type**

1. Reuse the existing job state machine; new job types don't
   need a new state, just a new `kind` discriminator.
2. Add a `run_*` function in `jobs/worker.py` (or a new module
   under `jobs/`) that does the work; call it from `indexing_worker`'s
   dispatch on the `kind`.
3. If the job needs new progress fields (per-stage weights),
   update `PipelineStage` in `jobs/pipeline.py` accordingly.
4. Update `JobManager.create_job(...)` to accept the new job
   kind.
5. By default, the API does NOT run jobs in-process
   (`SEMRAG_API_RUNS_WORKER=false`). Keep this default; new
   jobs run in the worker process.

**Verification**

```bash
pytest tests/unit/jobs/test_worker.py -x
# Submit a real job via the API
curl -sS -H "x-api-key: dev-local-key" \
  -H "content-type: application/json" \
  -d '<create-job request>' \
  http://localhost:8000/v1/projects/<id>/index
# Tail the worker logs
docker compose logs -f worker
# Job state
curl -sS -H "x-api-key: dev-local-key" \
  http://localhost:8000/v1/jobs/<job_id> | jq
```

**Common mistakes**

- Adding a job kind with a new state. Don't. Reuse
  QUEUED / RUNNING / DONE / FAILED / CANCELLED.
- Running the job synchronously in the HTTP handler. The whole
  reason the job system exists is to keep handlers cheap.
- Skipping the progress calculator. The worker calls
  `_update_job_progress` per stage; without it the API can't
  report a meaningful percentage.

For implementation details, see
[`../src/semrag/jobs/README.md`](../src/semrag/jobs/README.md).

---

## 19. Adding storage backends

The codebase has two storage seams: vector stores and graph
stores. Both are Protocols.

**Files involved — vector**

- `src/semrag/storage/vector/interfaces.py` — `VectorStore` Protocol
  and `VectorSearchResult`, `VectorSearchFilter`.
- `src/semrag/storage/vector/factory.py` — selection by Settings.
- `src/semrag/storage/vector/<backend>.py` — your new backend.
- `src/semrag/core/settings.py` — `VECTOR_STORE_BACKEND`.

**Files involved — graph**

- `src/semrag/storage/graph/interfaces.py` — `GraphStore` Protocol.
- `src/semrag/storage/vector/` (yes, vector path) — the
  Postgres-backed retrieval path uses `PostgresGraphQueries`
  directly (`storage/postgres/graph_queries.py`); the
  `GraphStore` Protocol is the optional backend that retrieval
  can fall back to when configured.

**Workflow**

1. Implement the Protocol's full surface. `VectorStore` has
   `upsert_chunks`, `search`, `delete_by_*`. Read the docstrings
   on each method; the contracts are precise (e.g. `upsert_chunks`
   must use the deterministic `chunk_id` to upsert not append).
2. Add a case in the factory function.
3. The codebase ships pgvector today. Adding a different backend
   usually means: a new dependency in `requirements.txt`, an
   Alembic migration if the schema differs, and an integration
   test that runs against the real backend in CI.

**Verification**

```bash
# Unit test against your backend (or a fake)
pytest tests/unit/storage/vector/test_<backend>.py -x
# Integration (CI on a labeled job)
pytest tests/integration/storage/test_<backend>_integration.py -x
```

**Common mistakes**

- Returning scores on a different scale than pgvector. The
  Protocol assumes higher-is-better uniformly. If your backend's
  natural scale is distance (lower-is-better), invert in the
  mapping layer (`score = 1 - distance`).
- Using session/stateful connections without lifecycle management.
  The factory receives a `Session` or connection at construction
  time and **must** release it on `close()`. Tests that exercise
  the same store across multiple ingestion runs will leak.

For implementation details, see
[`../src/semrag/storage/README.md`](../src/semrag/storage/README.md).

---

## 20. Database migrations (Alembic)

**Files involved**

- `alembic.ini` — Alembic config.
- `alembic/env.py` — env glue.
- `alembic/versions/<rev>_<description>.py` — your migration.
- `src/semrag/storage/postgres/models.py` — the Base/ORM models.

**Workflow**

1. Update the SQLAlchemy ORM model in `storage/postgres/models.py`
   (or whichever model file) **first**. This is what Alembic's
   `--autogenerate` compares against.
2. Generate the migration:
   ```bash
   alembic revision --autogenerate -m "<short description>"
   ```
3. Inspect the generated file. Autogenerate doesn't catch:
   - Enum value changes (it skips the values, only sees the type).
   - Renames (it sees a drop + add).
   - CHECK constraints.
   - Postgres-specific features (pgvector index parameters, JSONB
     operators).
   Edit the file to fix the gaps.
4. Test both upgrade and downgrade:
   ```bash
   alembic upgrade head
   alembic downgrade -1
   alembic upgrade head
   ```
5. Run the migration against a non-empty database to confirm it
   handles existing data; the existing
   `tests/integration/migrations/` pattern covers this.

**Verification**

```bash
# Migration is reversible
alembic upgrade head && alembic downgrade base && alembic upgrade head
# pgvector change specifically (it requires the extension)
psql -d <db> -c '\dx'  # vector must be listed
```

**Common mistakes**

- Skipping the downgrade. Always include `downgrade()`. The
  rollback runbook depends on it.
- Forgetting that pgvector is an **extension**, not a type. New
  columns that use `Vector(N)` need `CREATE EXTENSION IF NOT
  EXISTS vector;` first; the database container's health check
  verifies this — if the extension is missing, the API fails
  /readyz.
- Editing the model without regenerating the migration. The two
  must land together.
- Adding a non-trivial enum change without a `using=` clause on
  PostgreSQL — Postgres requires explicit cast logic.

For implementation details, see
[`../src/semrag/storage/README.md`](../src/semrag/storage/README.md).

---

## 21. Adding configuration

Every `SEMRAG_*` env var is declared in `src/semrag/core/settings.py`.
**No deploy shape adds a new env var without first adding it to
`Settings`** — the runtime validator will silently ignore it otherwise.

**Files involved**

- `src/semrag/core/settings.py` — `Settings` (pydantic-settings model).
- The `startup_validation` module (under `api/`) — runtime
  validation that fails fast at startup.

**Workflow**

1. Identify the **category** of the setting (LLM, embedding, storage,
   ingestion, etc.). Place it under the matching section in
   `Settings`.
2. Pick the right pydantic type: `str` with `Literal[...]` for
   enums; `int` with a sensible default; `bool` for feature flags.
   Add `description=` so the OpenAPI / `/version` rendering works.
3. Validation: add a `@field_validator` (or `@model_validator`)
   for cross-field constraints (e.g.
   `SEMRAG_REJECT_DEV_KEYS_IN_PRODUCTION=true` requires
   `SEMRAG_ENVIRONMENT != development`).
4. Surface in the right places:
   - `DEPLOYMENT.md` §"Environment variables" — the operator-facing
     table.
   - The Helm `values.yaml` comment block — the values that map.
   - `.env.example` (or equivalent) — dev defaults.

**Verification**

```bash
# Pydantic-settings auto-generates this; verify it still validates
python -c "from semrag.core.settings import Settings; Settings()"
# Validator paths
SEMRAG_ENVIRONMENT=production SEMRAG_API_KEYS=dev-local-key python -c "..."
# The startup validation flow
python run_local_qa.py ask --project-id <id> --question "..."
```

**Common mistakes**

- Using an env var that doesn't exist in `Settings` as the source
  of truth. Always edit `Settings` first; the deploy shape just
  exposes it.
- Confusing **server-side** config (`Settings.read_at_startup()`)
  with **runtime mutation**. Settings may change at runtime for
  some knobs (`Settings.llm_provider`, etc.); most are read once.
  Read the existing `@property` patterns before adding a mutator.
- Forgetting to update `.env.example` and `DEPLOYMENT.md`. New
  knob without docs = landmine for the next operator.

For implementation details, see
[`developer_guide.md` §10](./developer_guide.md#10-configuration).

---

## 22. Adding evaluation metrics

**Files involved**

- `src/semrag/evaluation/metrics.py` — the eight retrieval-metric
  primitives.
- `src/semrag/evaluation/models.py` — `RetrievalMetrics` and the
  `AggregateReport` shape.
- `src/semrag/evaluation/harness.py` — `EvaluationHarness.evaluate_query`.
- `src/semrag/evaluation/benchmarks.py` — the canned `GoldenQuery` set.
- `src/semrag/evaluation/evaluator.py` — the end-to-end evaluator
  + LLM judge.

**Workflow**

1. Add the metric primitive in `metrics.py` as a **pure function**.
   No I/O, no settings, no logging. Take Python collections in,
   return `float`. Match the conventions:
   - Empty relevant / empty expected → defined-but-default result
     (`recall_at_k` returns 0.0 on empty relevant;
     `file_recall` returns 1.0 on empty expected).
   - "No hit" sentinel → use `float("inf")` to match `average_rank`;
     the aggregate filters inf out.
2. Add a field to `RetrievalMetrics`.
3. Populate in `EvaluationHarness.evaluate_query(...)`.
4. Add a `mean_*` field to `AggregateReport` and the aggregation
   in `evaluate_all`.
5. If the metric also applies to the end-to-end evaluator, add
   a new LLM-judge prompt dimension (see §22b below).

**22b. Adding an LLM-judge dimension**

1. Add the field to `LLMJudgeScores` in `evaluator.py`.
2. Add a line to `_JUDGE_PROMPT`.
3. Add a `mean_*` field to `EvalAnswerReport`.
4. Aggregate over `llm_scores.valid == True` only — the existing
   pattern filters parse failures.

**Verification**

```bash
pytest tests/unit/evaluation/test_metrics.py -x
# End-to-end: a real benchmark run, compare to baseline
python run_benchmarks.py
cat benchmark_results/<run>/summary.md | grep <new metric>
```

**Common mistakes**

- Adding a metric that requires LLM calls. The harness is
  LLM-free by design; LLM-judged metrics go in the
  `EndToEndEvaluator`.
- Treating the LLM judge as ground truth. The judge is best-effort;
  use it as one signal among several.
- Adding a metric to `RetrievalMetrics` without updating
  `EvaluationHarness` — the field stays `0.0` forever.

For implementation details, see
[`../src/semrag/evaluation/README.md`](../src/semrag/evaluation/README.md).

---

## 23. Writing tests

**Files involved**

- `tests/unit/<package>/` — unit tests, mirroring the source layout.
- `tests/integration/` — cross-package tests, gated behind env vars.
- `tests/load/`, `tests/perf/` — load + performance tests.
- `tests/conftest.py` — shared fixtures (in-memory settings, mock
  LLM, Postgres test database).

**Workflow**

1. **Mirror the source layout.** A new chunker module gets a new
   test file under `tests/unit/chunking/`. The convention makes
   missing tests visible.
2. **One file per source module.** A test file with 30 functions
   covering 8 source modules is a smell; split.
3. **Use `MockLLM` for LLM tests.** Don't hit a real Ollama in
   unit tests. `MockLLM` is in `reasoning/llm/mock_llm.py`.
4. **Use in-memory / JSON-file backends for storage tests.** Don't
   talk to real Postgres unless the test is in `tests/integration/`.
5. **Test the boundary, not the internals.** A route handler test
   asserts the response status + payload shape; an LLM provider
   test asserts the message mapping; a metric test asserts the
   metric was registered and increments on call.

**Verification**

```bash
# Quick
pytest tests/unit/<package> -x
# With coverage for what changed
pytest tests/unit/<package> -x --cov=src/semrag/<package> --cov-report=term-missing
# Lint / format
make format lint typecheck
```

**Common mistakes**

- Hitting the real LLM in unit tests. Tests should be
  deterministic; scripted responses in `MockLLM`.
- Calling `time.sleep(...)` to test async behavior. Use
  `asyncio.sleep(0)` for cooperative yields.
- Asserting on error messages. Assert on error types / status
  codes; messages change.
- Skipping the rollout of test parametrization. The codebase has
  ample use of `pytest.mark.parametrize`; copy the pattern.

For implementation details, see
[`developer_guide.md` §12](./developer_guide.md#12-testing-strategy).

---

## 24. Performance tuning

**Files involved**

- `src/semrag/retrieval/pipeline.py` — retrieval latency
  (`retrieval.vector.duration_ms` per stage).
- `src/semrag/services/services.py` — end-to-end query latency.
- `src/semrag/core/metrics.py` — histograms you can graph.
- `deploy/monitoring/grafana/dashboards/semrag-retrieval.json` —
  the latency dashboard.

**Workflow — when the SLO is breached**

1. Identify the stage from the histogram. The retrieval
   dashboard splits per stage (`retrieval.vector`,
   `retrieval.lexical`, `retrieval.query_expansion.lexicalN`,
   `retrieval.rrf`, `retrieval.noisy_source_penalty`,
   `retrieval.graph_expansion`, `retrieval.intent_boost`,
   `retrieval.doc_type_prior`, `retrieval.rerank`,
   `retrieval.diversity`).
2. The likely culprits in order:
   - Vector search: HNSW query latency. Check
     `retrieval/vector_search_adapter.py`.
   - Lexical search: Postgres FTS. Check
     `retrieval/lexical_search.py`.
   - LLM call: dominant for complex queries. Switch providers
     or trim the prompt.
   - Reranker: the `lexical_overlap` Jaccard reranker is the default
     and is cheap; the HTTP cross-encoder client is heavier — measure
     it separately if enabled.
3. Fix one stage; re-measure. Don't stack changes.

**Tuning knobs (Settings)**

| Knob                                | Effect                                                      |
| ------------------------------------ | ----------------------------------------------------------- |
| `RERANKER`                           | `lexical_overlap` (default), `http`, or `none` (skip)     |
| `NOISY_SOURCE_PENALTY`               | Penalty applied to log/coverage chunks                       |
| `ENABLE_GRAPH_EXPANSION`             | Skip graph expansion when knowledge is shallow               |
| `ENABLE_INTENT_BOOST`                | Skip the per-chunk-type bias                                |
| `HYBRID_RESERVATION_SLOTS`           | Slots reserved for top-1 vector/lexical                     |
| `LLM_NUM_PREDICT`                    | Hard cap on response tokens                                |
| `LLM_CONTEXT_WINDOW`                  | Total prompt+response budget                                |
| `LLM_RESERVED_FOR_RESPONSE`          | Reservation inside the budget for the response             |

**Verification**

```bash
# Before / after on the same benchmark
python run_benchmarks.py --tag baseline
# Apply the tuning
python run_benchmarks.py --tag tuned
# Compare
python -m semrag.benchmark.diff baseline tuned
```

**Common mistakes**

- "Optimization" that breaks the eval harness. Re-run the
  benchmark; if NDCG@10 drops, revert.
- Adding caches without TTLs or hit-rate telemetry. A cache that
  doesn't move hit-rate up is a memory leak.

For implementation details, see
[`../src/semrag/retrieval/README.md`](../src/semrag/retrieval/README.md).

---

## 25. Memory optimization

**Files involved**

- `src/semrag/ingestion/embedding/pipeline.py` — batched embedding.
- `src/semrag/ingestion/chunking/pipeline.py` — chunking yield size.
- `src/semrag/api/query_cache.py` — TTLQueryCache vs Redis backend.
- `src/semrag/api/rate_limit.py` — same.

**Workflow — under memory pressure**

1. Check `core/metrics.py` for `DB_POOL_CONNECTIONS{state=*}` — if
   `overflow` is high, the pool is exhausted; reduce
   `SEMRAG_DB_POOL_SIZE` is rarely the answer (lower concurrency is).
2. Check ingestion batch size. Large corpora embed in batches of
   `EMBEDDING_BATCH_SIZE` — reduce if OOM.
3. The `InMemoryQueryCache` and `InMemoryEmbeddingCache` grow
   unbounded without TTL. Switch to `RedisQueryCache` /
   `RedisEmbeddingCache` for shared state.
4. The `DebugArtifactWriter` writes per-run bundles. If the
   default retention is too long for the disk, lower
   `SEMRAG_DEBUG_ARTIFACT_RETENTION_DAYS`.

**Verification**

```bash
# RSS by pod
kubectl top pods -n semrag -l component=api
# Heap by Python type (in DEBUG with tracemalloc)
SEMRAG_LOG_LEVEL=DEBUG python -c "
import tracemalloc; tracemalloc.start()
... do work ...
snap = tracemalloc.take_snapshot()
for stat in snap.statistics('lineno')[:20]: print(stat)
"
```

**Common mistakes**

- "I'll just add more memory." That hides the leak. Find what
  holds the references (the cache TTL is missing; the artifact
  retention is wrong; the request-scoped state is module-level).
- Disabling the embedding cache "to save memory". The opposite is
  true — without it, every re-index re-embeds the entire corpus.

For implementation details, see
[`docs/operations/memory-exhaustion.md`](./operations/memory-exhaustion.md).

---

## 26. Production debugging

Use this when a production alert fires. The on-call runbooks are
the operational manual; this is the engineering view.

**Files involved**

- `src/semrag/core/observability.py` — JSON logs in production.
- `src/semrag/core/metrics.py` — the `/metrics` endpoint.
- `src/semrag/core/tracing.py` — OpenTelemetry traces.
- `src/semrag/core/debug_artifacts.py` — per-run bundles.
- `src/semrag/api/health.py` — `/livez`, `/readyz`, `/version`.
- `docs/operations/` — the failure-mode runbooks.

**Workflow**

1. **Confirm the alert.** `kubectl -n semrag logs -l component=api
   --tail=200` for the structured logs.
2. **Check the version.** `curl /version` should match the
   expected commit. A drift here is a deployment artifact issue.
3. **Read `/readyz`.** One of the checks has likely flipped to
   "down" — the body shows which.
4. **Pivot on the runbook for the failure mode.** Each
   `docs/operations/<failure>.md` has Symptom / Triage / Mitigation
   / Verification. Don't invent a new diagnostic; the runbooks
   reflect past incidents.
5. **If it's a regression**, rollback first
   (`docs/operations/rollback.md`); debug second.

**Verification**

```bash
# Critical endpoints always
curl /livez /readyz /version
# Logs by stage
kubectl -n semrag logs -l component=api --tail=200 | jq 'select(.stage != null)'
```

**Common mistakes**

- Restarting the pod before reading `/readyz`. The output tells
  you which dependency is broken. Restarting a misconfigured pod
  just restarts the failure.
- Reading human-formatted logs in production. Production uses
  `SEMRAG_LOG_FORMAT=json`; the structured fields are what you
  filter on.

For implementation details, see
[`docs/operations/`](./operations/).

---

## 27. Incident investigation

**Files involved**

- `src/semrag/core/query_trace.py` — the per-query trace accumulator.
- `src/semrag/core/debug_artifacts.py` — the per-run writer.
- `src/semrag/retrieval/pipeline.py` — the per-stage ranking trace.
- `deploy/monitoring/grafana/dashboards/` — dashboards to correlate.
- `docs/operations/escalation.md` — when and how to escalate.

**Workflow — for a single bad query**

1. Reconstruct the query in the dev environment with the same
   `SEMRAG_*` settings as production.
2. Set `SEMRAG_LOG_LEVEL=DEBUG` and run the query. Open
   `.semrag/debug/<run_id>/` and read top to bottom:
   `query_trace.json` → `retrieval/ranking_trace.json` →
   `prompt/final_prompt.txt` → `llm/request.json` +
   `llm/response.json`.
3. Compare to a known-good query from the same project to spot
   structural differences.

**Workflow — for a sustained incident**

1. Pull the alert from Prometheus. The
   `deploy/monitoring/prometheus/rules/semrag.yml` shows what
   each alert actually fires on.
2. Pivot to the corresponding dashboard
   (`semrag-api.json`, `semrag-retrieval.json`, etc.).
3. Run a synthetic repro using the
   `tests/load/` or `tests/perf/` scripts if available.

**Common mistakes**

- Investigating an incident without reading the runbook first. The
  runbooks reflect prior post-mortems; the next person will add
  your findings to the runbook when you file post-mortem.
- Modifying production config without a corresponding change in
  `core/settings.py`. Production may have env overrides, but the
  validator and the doctype live in `Settings`.

For implementation details, see
[`developer_guide.md` §14](./developer_guide.md#14-debugging-workflow).

---

## 28. Reading logs

The codebase emits **structured** logs through
`core/observability.py`. Production uses `SEMRAG_LOG_FORMAT=json`
for ingestion; dev uses `human`.

**Files involved**

- `src/semrag/core/observability.py` — the JSON / human formatters
  and the `stage(...)` context manager.
- `src/semrag/api/middleware.py` — `LoggingMiddleware` adds
  request-context fields.
- `src/semrag/api/exception_handler.py` — exception envelopes.

**Workflow**

1. **Production:** every log line is a JSON object. Filter on
   fields, not substrings:
   ```bash
   kubectl -n semrag logs -l component=api --tail=200 \
     | jq 'select(.event=="error" and .stage=="query.agent")'
   ```
2. **Development:** `SEMRAG_LOG_FORMAT=human` is human-readable,
   but the same fields are present. Look for `stage=... event=...`
   prefixes.
3. **Stage markers.** The four events you'll see for every stage:
   `event=start`, `event=finish elapsed_ms=...`, `event=error`,
   and `event=...` with extra `stats` fields for accumulated
   counters.
4. **Request correlation.** `RequestIDMiddleware` sets a
   `request_id` field at the top of every line for that request;
   filter on it to get all logs for one call.

**Common mistakes**

- `grep -i "error"` on JSON logs. The field is `event=error`, not
  the literal word.
- Logging large response bodies. The middleware redacts auth
  headers but doesn't redact LLM responses; check the level
  (INFO+ should never include them).

For implementation details, see
[`developer_guide.md` §9.1](./developer_guide.md#91-logging-coreobservabilitypy).

---

## 29. Understanding debug artifacts

`DebugArtifactWriter` writes a per-run bundle to
`SEMRAG_DEBUG_ARTIFACTS_DIR` (default `.semrag/debug/<run_id>/`).
The bundle is the primary diagnostics surface.

**Files involved**

- `src/semrag/core/debug_artifacts.py` — the writer.
- `src/semrag/core/query_trace.py` — the per-query trace accumulator.
- `src/semrag/services/services.py` — per-stage `write_*` calls.

**Layout (verified against `services.py` and `core/debug_artifacts.py`)**

| File under `debug/<run_id>/`               | Written by  | What it contains                                              |
| ------------------------------------------ | ----------- | ------------------------------------------------------------- |
| `scan/manifest.json`                       | Ingestion   | The full scan manifest (file metadata + change-set).          |
| `parse/<rel_path>.json`                    | Ingestion   | One per parsed file, a coarse summary (`_summarize_parse`).   |
| `ir/graph.json`                            | Ingestion   | The full IR graph (`model_dump(mode="json")`).                |
| `knowledge_graph/graph_summary.json`       | Ingestion   | `{"node_count", "edge_count"}`.                               |
| `chunks/chunk_metadata.json`               | Ingestion   | One line per chunk: chunk_id, relative_path, chunk_type, line range. |
| `embeddings/embedding_stats.json`          | Ingestion   | Model name + embedded-chunk counts.                           |
| `timings/pipeline.json`                    | Both        | Marker; per-stage durations come from `event=finish` log lines. |
| `retrieval/ranking_trace.json`             | Query       | `RankingTrace` (per-chunk per-stage scores).                  |
| `retrieval/top_results.json`               | Query       | Top-K after dedup + structural filter.                        |
| `prompt/final_prompt.txt`                  | Query       | The full prompt string sent to the LLM.                       |
| `llm/request.json` + `llm/response.json`   | Query       | System + user + provider + chunk IDs; final answer + iteration count. |
| `query_trace.json`                         | Query       | `QueryTrace` (preprocessing, embedding, retrieval, context assembly, prompt, LLM, citations, final answer). |

**Workflow**

1. Set `SEMRAG_LOG_LEVEL=DEBUG` to enable the writer.
2. Run the request.
3. `cat .semrag/debug/<run_id>/query_trace.json | jq`
4. If retrieval is wrong: `cat .../retrieval/ranking_trace.json | jq`
5. If prompt is wrong: `cat .../prompt/final_prompt.txt | head -50`
6. If LLM is wrong: `cat .../llm/request.json | jq` then
   `cat .../llm/response.json | jq`

**Common mistakes**

- Skipping the writer. It's a no-op at INFO+; if you don't set
  DEBUG, the bundle is empty.
- Not cleaning up. `SEMRAG_DEBUG_ARTIFACT_RETENTION_DAYS` defaults
  to a small number; check the setting before leaving artifacts.

For implementation details, see
[`developer_guide.md` §9.4](./developer_guide.md#94-debug-artifacts-coredebugartifactspy).

---

## 30. Working with Docker

**Files involved**

- `Dockerfile` — multi-stage production image, `tini` PID 1, non-root.
- `docker-compose.yml` — single-host stack with profile-gated
  Redis / Ollama.
- `docker-compose.override.yml` — local-dev overlay (auto-merged).
- `Makefile` — `make docker-build`, `make docker-run`.

**Workflow — full local stack**

```bash
cp .env.example .env
docker compose up -d                   # api + postgres
docker compose --profile ollama up -d  # local LLM
docker compose logs -f api
curl http://localhost:8000/livez
```

**Workflow — single container**

```bash
docker build -t semrag:dev .
docker run --rm -p 8000:8000 --env-file .env semrag:dev
```

**Production deploys**

```bash
# Production compose = docker-compose.yml ONLY (no override)
docker compose -f docker-compose.yml up -d
```

**Common mistakes**

- Using `docker compose up` (which merges the override) in
  production. Production deploys use
  `docker compose -f docker-compose.yml`.
- Editing the running container. The image is the artifact;
  changes belong in `Dockerfile`.
- Forgetting `depends_on: postgres: { condition: service_healthy }`
  on a new service. Without it, the API can race Postgres startup.

For implementation details, see
[`../DEPLOYMENT.md`](../DEPLOYMENT.md) and
[`../deploy/README.md`](../deploy/README.md).

---

## 31. Working with Kubernetes

**Files involved**

- `deploy/k8s/` — eight numbered raw manifests.
- `src/semrag/core/health.py` and `src/semrag/api/health.py` —
  the `/livez` / `/readyz` contracts.

**Workflow — apply**

```bash
kubectl apply -f deploy/k8s/
# Order: namespace → configmap → secret → pvc → deployment →
# service → ingress → worker-deployment.
```

**Workflow — debug a pod**

```bash
kubectl -n semrag get pods -l component=api
kubectl -n semrag logs -l component=api --tail=200 --previous
kubectl -n semrag describe pod <pod-name>
kubectl -n semrag exec -it <pod-name> -- python -c "import semrag; print(semrag.__file__)"
kubectl -n semrag port-forward <pod-name> 8000:80
curl localhost:8000/livez /readyz /version
```

**Common mistakes**

- Mounting the source tree as a bind mount in production.
  Bind mounts are developer-only.
- Skipping `kubectl rollout restart` after a ConfigMap-only change
  on the raw manifests. The Helm chart's `checksum/config`
  annotation does this automatically; the raw manifests' annotation
  is empty.

For implementation details, see
[`../deploy/README.md`](../deploy/README.md).

---

## 32. Working with Helm

**Files involved**

- `deploy/helm/semrag/Chart.yaml`, `values.yaml`, `templates/`.
- `src/semrag/core/settings.py` — all `SEMRAG_*` values land here.

**Workflow — install**

```bash
kubectl create secret generic semrag-secrets \
  --from-literal=POSTGRES_PASSWORD=... \
  --from-literal=SEMRAG_API_KEYS=key1,key2

helm install semrag deploy/helm/semrag/ \
  --namespace semrag --create-namespace \
  --set existingSecret=semrag-secrets \
  --set image.tag=v0.1.0
```

**Workflow — upgrade**

```bash
helm upgrade semrag deploy/helm/semrag/ --reuse-values \
  --set image.tag=v0.1.1
```

**Workflow — lint**

```bash
helm lint deploy/helm/semrag/
helm template semrag deploy/helm/semrag/ | kubeconform -strict -summary -
```

**Common mistakes**

- Setting `POSTGRES_PASSWORD` in `values.yaml` for production. Use
  `--set existingSecret=…` and create the Secret out-of-band.
- Modifying `templates/` instead of `values.yaml`. Templates are
  the rendered shape; values are the knobs.

For implementation details, see
[`../deploy/README.md`](../deploy/README.md).

---

## 33. Common engineering workflows

The repo's `AGENTS.md` spells out the editorial workflow for fixing
bugs:

1. Reproduce.
2. Find root cause.
3. Implement fix.
4. Re-run the failing test.
5. Verify no regressions.

The complementary workflows every engineer eventually needs:

**Adding a Settings knob (cross-cutting).** Settings change →
document in `developer_guide.md` §10 → update `DEPLOYMENT.md`
§"Environment variables" → update `.env.example` → update
`deploy/README.md` if it lists the knob → update the chart
`values.yaml` comment.

**Shipping a new model backend (LLM / embedding).** Implement the
Protocol → register in factory → add Settings keys → add unit tests
with a mock HTTP layer → add integration test under
`tests/integration/` (CI on a labeled job) → update the relevant
package README → add the knob to `DEPLOYMENT.md`.

**Adding a new query path that uses the LLM.** Use
`QueryService.ask(...)` end-to-end. **Don't** call the LLM provider
directly from a router — the prompt builder, the agent loop, and the
citation validator are upstream of your path; bypassing them skips
the groundedness guarantees.

**Diagnosing a flaky test.** Confirm it's truly flaky (`pytest -x
--count=N` or just rerun). Open the test in isolation; mock every
external dependency; check the timestamps and time-dependent logic.

**Migrating a model.** When `nomic-embed-text` changes dimension
or the LLM provider upgrades the model, the change is **two
places**: the model name in `values.yaml` / `.env`, and the
embeddings column dimension in Postgres (`pgvector` columns have
fixed dimensions — re-create the index).

For implementation details, see
[`../AGENTS.md`](../AGENTS.md) and
[`./developer_guide.md`](./developer_guide.md).

---

## 34. Common anti-patterns

These are mistakes the codebase has already made once or is set up
to make; each section explains WHY and what to do instead.

1. **Importing a concrete backend into business code.**
   - **Why it's wrong:** couples the business logic to the
     backend choice; defeats the Protocol pattern.
   - **Do instead:** depend on the Protocol (`EmbeddingProvider`,
     `VectorStore`, etc.); wire the concrete at the edge.

2. **Catching a broad exception to "log and continue".**
   - **Why it's wrong:** hides failures from `RuntimeConfigurationError`,
     `LLMProviderError`, etc. Operators lose the actionable message.
   - **Do instead:** let the exception propagate; the global
     exception handler formats it.

3. **Reading settings at import time.**
   - **Why it's wrong:** makes the value a module-level singleton
     — tests can't change `Settings` between cases.
   - **Do instead:** read inside the function (or constructor);
     `get_settings()` per request.

4. **Adding `time.sleep(...)` to coordinate async work.**
   - **Why it's wrong:** blocks the event loop; tests flake.
   - **Do instead:** use `await asyncio.sleep(0)` to yield; use
     barriers if you genuinely need synchronization.

5. **Mutating a `RetrievalResult` in place.**
   - **Why it's wrong:** corrupts the per-stage `RankingTrace` and
     surprises downstream stages that hold references.
   - **Do instead:** `result.model_copy(update={...})` for every
     score change.

6. **Adding a `Settings` field to a service constructor and reading
   it at import time.** Same anti-pattern as #3, scoped.

7. **Returning raw ORM objects from a route.**
   - **Why it's wrong:** the wire format must be a Pydantic model;
     raw ORM objects leak DB columns.
   - **Do instead:** Pydantic `model_validate(...)` at the
     boundary.

8. **Inlining a custom RAG loop because "the agent loop is too
   slow".**
   - **Why it's wrong:** you lose the citation validator, the
     prompt budgeter, and the tool registry. The agent loop is
     bounded by `MAX_AGENT_ITERATIONS`; tune there first.
   - **Do instead:** ship the question to `QueryService.ask(...)`.

9. **Adding a new "fast path" that bypasses the retriever.**
   - **Why it's wrong:** the eval harness's `primary_file_recall`
     drops to zero. The retriever's job is grounded retrieval;
     bypassing it returns hallucinated answers.

10. **Storing secrets in `values.yaml`.** Use `existingSecret`.

11. **Disabling the embedding cache in production "to save
    memory".** The opposite: the cache prevents re-embedding the
    entire corpus on every re-index.

12. **Logging the LLM response at INFO.** Belongs at DEBUG, with
    redaction of auth headers / API keys. Production logs must
    not leak prompts in plaintext.

13. **Adding a Settings knob without a corresponding
    `values.yaml` / `DEPLOYMENT.md` / `.env.example` entry.** The
    knob becomes undocumented, and operators discover it via
    `cat logs`.

For the full list, see
[`./developer_guide.md` §15](./developer_guide.md#15-common-architectural-rules).

---

## 35. Things you should NEVER do

The repo's [`../AGENTS.md`](../AGENTS.md) is explicit. These are
the **bright-line** rules — no exceptions.

- **Never delete failing code to make tests pass.** The test is
  the spec; fix the code, not the test.
- **Never guess.** Reproduce the bug, find the root cause, then
  edit.
- **Never swallow exceptions silently.** If you must catch,
  re-raise with context (`raise RuntimeError("...") from exc`).
- **Never import a concrete backend into business code.** Use
  the Protocol.
- **Never read settings at import time.** Use `get_settings()`
  per request.
- **Never mutate a `RetrievalResult` in place.** Use
  `model_copy(update={...})`.
- **Never add a new `SEMRAG_*` env var without adding it to
  `core/settings.py` first.** Deploy-time env vars that
  bypass the validator are an incident waiting to happen.
- **Never add a new transition to the job state machine.** Reuse
  QUEUED / RUNNING / DONE / FAILED / CANCELLED.
- **Never run indexing in-process.** The
  `SEMRAG_API_RUNS_WORKER=false` default exists for a reason;
  keep it.
- **Never commit real `POSTGRES_PASSWORD` / `SEMRAG_API_KEYS`.**
  Use placeholders + external Secrets.
- **Never ship a custom reranker that replaces `fused_score`
  instead of multiplying it.** This is the single most common
  custom-reranker bug; the existing docstring calls it out.
- **Never ship a Prometheus metric with high-cardinality labels
  (request IDs, user IDs).**

---

## 36. Verification checklist before opening a PR

Run before you push. CI may catch some of this, but a personal
run is faster.

```bash
# 1. Format and lint
make format lint
# 2. Type check
make typecheck
# 3. Unit tests
pytest tests/unit -x
# 4. Tests for the package you touched
pytest tests/unit/<package> -x --cov=src/semrag/<package>
# 5. Integration tests if you touched a Protocol or factory
pytest tests/integration -x
# 6. The end-to-end smoke (CLI)
SEMRAG_LOG_LEVEL=INFO \
  python run_local_qa.py ask --project-id <id> --question "..."
# 7. Verify health endpoints
curl -fsS http://localhost:8000/livez
curl -fsS http://localhost:8000/readyz
curl -fsS http://localhost:8000/version
# 8. Verify metrics endpoint
curl -fsS http://localhost:8000/metrics | grep -c "^semrag_"
# 9. If you touched a deployment shape, lint it
helm lint deploy/helm/semrag/
kubeconform -strict -summary deploy/k8s/*.yaml
# 10. If you added a Settings knob, verify the validator
python -c "from semrag.core.settings import Settings; Settings()"
# 11. If you added a migration, verify upgrade + downgrade
alembic upgrade head && alembic downgrade -1 && alembic upgrade head
```

If any step fails, **fix it before opening the PR**. "It's a
small fix" is the most common path to a 3am page.

---

## 37. Code review checklist

Use this list when reviewing a PR:

**Architecture**

- Does the change depend on a Protocol, not a concrete backend?
- Is the new code added at the right layer (router / service /
  package)?
- Are new settings added to `core/settings.py`, not to a deploy
  shape?
- Does the migration (if any) include a working `downgrade()`?

**Correctness**

- Is there a unit test that asserts the right behavior?
- Is there a regression test (the bug a real user reported)?
- Does the change handle the empty-input case (no results, empty
  query, no LLM)?

**Reliability**

- Did the author add a debug-artifacts entry for the new stage?
- Did the author add a Prometheus metric / log line for the new
  failure mode?
- If the change is async-safe, are there race condition tests?

**Readability**

- Are the new public functions named for the operation they perform
  (`embed_texts`, `upsert_chunks`, `validate_citations`), not for
  the implementation?
- Is the docstring accurate (no aspirational docs)?
- Are type hints correct (no `Any` where a concrete type fits)?

**Operational**

- Is there a runbook entry if this is a new failure mode?
- Does the Helm `values.yaml` comment reflect any new knob?
- Does the OpenAPI schema match what the route actually returns?

**Hard rules — every PR**

- No raw ORM objects in route handlers.
- No module-level `Settings()` reads.
- No new concrete backend imported into business code.
- No deleting tests to make them pass.
- No new `SEMRAG_*` env vars outside `Settings`.

---

## 38. Frequently modified files

These files change most often; reading them end-to-end once pays
back the time tenfold:

| File                                                        | Why it changes often                                              |
| ----------------------------------------------------------- | ---------------------------------------------------------------- |
| `src/semrag/core/settings.py`                               | Every new env var.                                                |
| `src/semrag/core/metrics.py`                                | Every new observability surface.                                  |
| `src/semrag/services/services.py`                           | Composition changes; new pipeline stage; new DI wiring.          |
| `src/semrag/retrieval/pipeline.py`                          | Ranking changes; new stage; new flags.                           |
| `src/semrag/reasoning/prompting/templates.py`                | System prompt tuning.                                             |
| `src/semrag/reasoning/agents/agent_loop.py`                  | Tool-call iteration rules; budget caps.                           |
| `src/semrag/reasoning/agents/tools.py`                       | New tools.                                                        |
| `src/semrag/reasoning/llm/factory.py`                       | New LLM provider registration.                                    |
| `src/semrag/ingestion/embedding/factory.py`                 | New embedding provider registration.                              |
| `src/semrag/api/routers/*.py`                               | New endpoints; per-route authorization changes.                  |
| `src/semrag/api/middleware.py`                              | New middleware; ordering changes.                                |
| `src/semrag/api/dependencies.py`                            | New dependency providers; singleton lifecycle.                   |
| `src/semrag/jobs/worker.py`                                 | New job kinds; cancellation rules.                                |
| `alembic/versions/<next>.py`                                | Schema changes.                                                  |
| `deploy/helm/semrag/values.yaml`                            | Default values; new knobs; resource limits.                      |
| `deploy/monitoring/prometheus/rules/semrag.yml`             | New alert rules; threshold tuning.                               |
| `docs/operations/*.md`                                      | New runbooks; updated triage steps.                              |

For the full per-file ownership table, see
[`./CODEBASE_GUIDE.md`](./CODEBASE_GUIDE.md).

---

## 39. Frequently misunderstood components

These are the parts where new engineers (and sometimes old ones)
get the abstraction wrong.

**The agent loop is not a chatbot.** It's a **bounded loop** that
calls the LLM, dispatches tool calls, and ends when the LLM
returns no tool calls or hits `MAX_AGENT_ITERATIONS`. It does not
remember state between requests. Treating it like a stateful
chat agent is the most common architectural mistake.

**The retriever is LLM-free.** `HybridRetriever` doesn't call the
LLM. Period. The "LLM judge" in evaluation is a *separate* call
after retrieval. If your design calls for an LLM during retrieval,
that's a tool — route through `SearchCodebaseTool`.

**The settings object is mutable on purpose.** Some knobs are
read once at startup (LLM base URL); others are read per request
(cache backend). The mutation points are deliberate; the
constructor pattern of every service re-reads the relevant
fields per call.

**The job system runs separately from the API.** The API process
has `SEMRAG_API_RUNS_WORKER=false`. Indexing is the worker's
job. Skipping this split — running indexing in-process — is
what the `08-job-backlog.md` runbook exists to recover from.

**`/livez` and `/readyz` are different probes.** `/livez` is
"the Python process is alive" (cheap, always 200 unless the
process is dead). `/readyz` is "every dependency is reachable"
(expensive, used by the orchestrator's readiness probe, **not**
by Docker's `HEALTHCHECK`).

**The "Citation count" report metric is not the same as
"groundedness".** The citation validator checks that every
`[Source: …]` reference corresponds to a path that was in the
context — that catches fabricated paths. It does NOT catch
hallucinated *prose around* a real citation; that's the LLM
judge's job.

**`MockLLM` is for tests, never for production.** Its scripted
responses don't parse as the judge's JSON.

For implementation details, see
[`developer_guide.md` §15](./developer_guide.md#15-common-architectural-rules).

---

## 40. Glossary

The codebase uses specific terms in precise ways. The alphabetized
short definitions; cross-references go deeper.

- **Agent loop** — the bounded iterative loop in
  `reasoning/agents/agent_loop.py` that orchestrates LLM ↔ tool
  calls. Not a chatbot.
- **Authorship settings / model author settings** — within the
  embedding provider, the namespace of per-model configuration
  (model name, dimension, base URL). See `core/settings.py`
  §Embedding.
- **Cancelable** — a job can be moved from `RUNNING` to
  `CANCELLED` by a separate process (the API). The worker honors
  this by checking state between stages.
- **Citation envelope** — the `[Source: <path>]` /
  `[Source: <path>, lines X-Y]` syntax. The prompt builder
  produces it; the citation validator consumes it.
- **Compliance / redaction** — the per-log-field redaction done
  by `core/observability.py`. Auth headers / API keys are
  redacted; user content is not (it's optional at DEBUG).
- **Coroutine-safe session** — the SQLAlchemy session is not
  thread-safe or async-safe across event loops; one session per
  request.
- **Dedup** — `_deduplicate_chunks(...)` in `services.py` strips
  pure-log noise and collapses near-identical chunks within a
  file. Applied after retrieval, before the prompt.
- **Debug artifacts** — the per-run bundle under
  `.semrag/debug/<run_id>/`. See §29.
- **Embedding cache** — `EmbeddingCache` Protocol; backed by
  memory / JSON-file / Redis.
- **Eval** — `src/semrag/evaluation/`. The harness (retrieval
  quality) and the evaluator (full-pipeline quality + LLM judge).
- **Existing Secret** — the Helm chart's `existingSecret:` value;
  tells the chart to skip rendering `templates/secret.yaml` and
  reference a Secret managed elsewhere.
- **Fusion** — `reciprocal_rank_fusion(...)` in
  `retrieval/fusion.py`; combines vector + lexical by rank
  position, not raw score.
- **GoldenQuery** — the eval package's data model; one benchmark
  question with chunk-ID + primary-files + secondary-files
  ground truth.
- **Graph expansion** — `expand_with_graph_context(...)` in
  `retrieval/graph_expansion.py`; pulls in structurally related
  chunks via IR traversal.
- **Health endpoints** — `/livez`, `/readyz`, `/version`. See
  §39.
- **Hybrid reservation** — the rule guaranteeing the top-1 by
  vector and the top-1 by lexical survive into the returned list.
- **Intent** — `QueryIntent` enum; coarse classification that
  drives the per-chunk-type boost.
- **JSONL benchmark loader** — not implemented today; benchmarks
  are Python literals in `benchmarks.py`.
- **LLM judge** — the second-pass LLM call in
  `EvaluationHarness.evaluate_all(..., run_llm_judge=True)` that
  scores 1-10 on five dimensions.
- **Main / worker split** — the API process and the worker
  process; connected only via the Postgres job store.
- **OpenTelemetry** — the tracing SDK wrapped by `core/tracing.py`.
  No-op when `Settings.tracing_enabled=false`.
- **Pipeline stage** — a labeled `with stage(...)` block in
  `core/observability.py`. The unit of timing + logging.
- **Pipeline stage weights** — `PipelineStage` in `jobs/pipeline.py`
  drives the per-stage progress percentage in the job system.
- **Plugin** — colloquial for "anything implementing a Protocol or
  ABC". The codebase has no dynamic plugin loader; new backends
  are added at the factory.
- **Prompt budget** — the `LLM_CONTEXT_WINDOW` minus the
  reservation, allocated in a deliberate order (system +
  history first, retrieved context last).
- **Query cache** — `QueryCache` ABC in `api/query_cache.py`;
  backed by TTL memory or Redis.
- **Reranker** — `Reranker` Protocol; `LexicalOverlapReranker`
  (default) or `HTTPRerankerClient`. Multiplies, never replaces
  the upstream score.
- **RRF** — Reciprocal Rank Fusion; rank-based combiner.
- **Service** — the composition root (`services/services.py`).
  Routes call services, never primitives.
- **Settings** — `Settings` pydantic-settings model in
  `core/settings.py`. The validator lives here.
- **Stage / event** — `stage=... event=start|finish|error` log
  fields; per-stage telemetry.
- **Streaming** — `LLMProvider.stream(...)` plus the agent loop's
  per-iteration yielding. The same loop produces the streamed
  response; no separate agent.
- **Tool calls** — `ToolCall` pydantic model in
  `reasoning/llm/interfaces.py`; the wire shape between the LLM
  and the agent loop.
- **Top-k reservation** — see **Hybrid reservation**.
- **Trusted Hosts** — `SEMRAG_TRUSTED_HOSTS` middleware config;
  required when behind a reverse proxy that rewrites `Host`.
- **Worker** — `python -m semrag.worker`; runs the indexing
  pipeline in `src/semrag/jobs/worker.py`.
- **Working directory** — `WORKDIR /app` in the Dockerfile; the
  only writable directory under the read-only root FS.

---

## 41. Cross-reference index

**Source package READMEs (per-package mental models)**

- [`../src/semrag/README.md`](../src/semrag/README.md) — the dependency graph at a glance.
- [`../src/semrag/core/README.md`](../src/semrag/core/README.md) — settings, observability, debug artifacts.
- [`../src/semrag/services/README.md`](../src/semrag/services/README.md) — `IngestionService`, `QueryService`.
- [`../src/semrag/ingestion/README.md`](../src/semrag/ingestion/README.md) — scanner, parsers, IR, chunking, embedding.
- [`../src/semrag/retrieval/README.md`](../src/semrag/retrieval/README.md) — hybrid retrieval, fusion, rerank.
- [`../src/semrag/reasoning/README.md`](../src/semrag/reasoning/README.md) — LLMs, agents, prompting.
- [`../src/semrag/storage/README.md`](../src/semrag/storage/README.md) — Postgres, pgvector, graph.
- [`../src/semrag/api/README.md`](../src/semrag/api/README.md) — FastAPI routers, middleware, auth, DI.
- [`../src/semrag/jobs/README.md`](../src/semrag/jobs/README.md) — async indexing-job lifecycle.
- [`../src/semrag/evaluation/README.md`](../src/semrag/evaluation/README.md) — harness + evaluator + judge.

**Repository-root docs**

- [`../README.md`](../README.md) — quickstart.
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — module layout + data-flow diagrams.
- [`../DEPLOYMENT.md`](../DEPLOYMENT.md) — every deployment environment, env-var reference.
- [`../RUN_API.md`](../RUN_API.md) — running the API in dev.
- [`../AGENTS.md`](../AGENTS.md) — contribution policy + priority order.
- [`../SETUP_POSTGRES.md`](../SETUP_POSTGRES.md) — local Postgres + pgvector setup.

**`docs/` docs**

- [`./README.md`](./README.md) — documentation index.
- [`./CODEBASE_GUIDE.md`](./CODEBASE_GUIDE.md) — file-by-file ownership.
- [`./developer_guide.md`](./developer_guide.md) — lifecycles, DI, security, observability, debugging.
- [`./operations/`](./operations/) — on-call runbooks (database-outage, llm-outage, embedding-failure, indexing-failure, job-backlog, high-latency, memory-exhaustion, rollback, escalation).

**Deployment assets**

- [`../deploy/README.md`](../deploy/README.md) — Docker, Compose, K8s, Helm, monitoring assets.

**Tests**

- [`../tests/README.md`](../tests/README.md) — test layout, fixtures, integration gating.

---

## Skipped (and why)

These are engineering topics intentionally **not** included because
the codebase does not implement them. Each is a "do not invent" -
if a future task seems to require one of these, the right response
is to escalate, not to invent.

- **Service mesh integration (Istio / Linkerd).** No mesh is
  installed in the deployed shapes; documenting mesh-aware
  middleware rules would invent topology that doesn't exist.
- **Plugin discovery.** The codebase uses compile-time factory
  registration, not runtime plugin discovery. Documenting a
  dynamic plugin loader would describe code that doesn't exist.
- **Multi-tenant isolation policies.** Every project is
  authenticated, but there is no tenant-level quota or
  isolation-policy framework today.
- **Hot-reload for settings.** Settings are read on each service
  construction; there is no file-watch or signal-driven reload.
- **Blue-green / canary release.** The Helm chart and raw K8s
  manifests ship `RollingUpdate` only.
- **A schema-evolution migration generation tool.** Alembic
  `--autogenerate` is the existing story; we don't have a custom
  evolution framework.
- **Native multi-region / multi-master database support.** Postgres
  HA, read replicas, cross-region failover — out of scope today.
- **An in-tree rate-limit "tenant" abstraction.** The current
  rate limit is per-API-key; tenant-aware limits would require a
  new abstraction.
- **A CI pipeline definition.** The repo carries the artifacts a
  pipeline applies (`deploy/`), not the pipeline itself.
- **A custom tracing exporter for OpenTelemetry.** The facade is
  configured per environment via `SEMRAG_TRACING_*`; the
  exporter is OTLP / Jaeger / Tempo configured out-of-tree.
- **A "knowledge graph" UI.** The IR graph is a server-side
  construct; no in-tree UI exists.

---

## Add when

These are engineering topics that should be added to this playbook
**only when the corresponding feature actually lands** in the
codebase. None of the below are present today; the absence from
the playbook is intentional.

- **CI workflow guidance.** Add a "Run the CI pipeline locally"
  section once a CI configuration ships in
  `.github/workflows/` or equivalent.
- **Schema-evolution framework beyond Alembic.** Add when a custom
  migration generator or rollback strategy is introduced.
- **Custom tracing / metrics exporters.** Add a section per
  exporter when one is added.
- **Multi-tenant operations.** Add when per-tenant quotas /
  isolation policies ship.
- **Blue-green / canary deployment.** Add when `Rollout` resources
  (Argo Rollouts or similar) replace the chart's
  `RollingUpdate`.
- **In-tree plugin loader.** Add a full extension-point workflow
  when runtime plugin discovery is introduced; until then, the
  factory pattern in §3 is the source of truth.
- **Service-mesh integration.** Add a "Working with Istio" /
  "Working with Linkerd" section when a mesh is installed in the
  chart.
- **Native multi-region support.** Add when cross-region failover
  / read replicas ship in `deploy/`.
- **A custom reranker harness.** The harness exists; if a
  per-reranker evaluation pass is added, document the workflow
  alongside §11.
- **A doc generation step.** Add a "Generate API reference" /
  "Generate READMEs from source" workflow once the tooling
  ships.
