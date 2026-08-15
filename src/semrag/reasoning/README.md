# `src/semrag/reasoning` — LLM-facing layers

The LLM half of the platform. Owns everything from "we have a list of
chunks and a question" to "we have a final answer plus the tool calls
that produced it." This is where the system crosses the LLM-free
boundary: nothing in `storage/`, `retrieval/`, or `ingestion/` calls
an LLM; everything that does lives here.

Three subpackages with distinct responsibilities:

- `reasoning/llm/` — `LLMProvider` Protocol, concrete providers
  (Ollama, OpenAI, OpenAI-compatible, mock), and the factory that
  picks one based on `Settings`.
- `reasoning/agents/` — the ReAct-style agent loop that drives
  tool-calling conversations, the `Tool` Protocol, the three real
  tools, and the `ToolRegistry` that dispatches them.
- `reasoning/prompting/` — token budgeting, context assembly,
  prompt templates, citation formatting, and the data models that
  describe a built prompt.

This is **not** an API reference, **not** a prompt catalog, and
**not** the place to learn how `Settings` is wired. For those, see
[`docs/developer_guide.md`](../../../docs/developer_guide.md) and
[`src/semrag/services/README.md`](../services/README.md).

---

## 1. Purpose of the package

The reasoning package exists to answer one question:

> Given a user query, a list of retrieved chunks, and a configured
> LLM, produce a final answer — with citations — that the user can
> trust.

Three things make that harder than a single LLM call:

1. **Token budgets are real.** A 7B local model on CPU spends 80+
   seconds processing 6K input tokens and blows the 90-second
   timeout. Reasoning owns the budget allocation that decides what
   makes it into the prompt and what gets dropped.
2. **The first retrieval often isn't enough.** Repository-architecture
   questions need a file index in the prompt; questions whose
   retrieval returned nothing need a tool call to enumerate the
   project. Reasoning owns the agent loop that drives those
   follow-ups.
3. **Citations must round-trip.** The LLM must output `[Source: …]`
   references that the citation validator in
   `src/semrag/evaluation/citation_validator.py` can check against
   the actual retrieved context. Reasoning owns both halves: the
   formatter that puts the citations in the prompt and the prompt
   instructions that tell the model to emit them.

Everything else in the package — the heuristic token estimator, the
multi-provider retry loop, the tool execution dispatcher, the
conversation-history trimmer — exists to make the first three points
land well in practice.

---

## 2. Package structure

```
reasoning/
├── __init__.py               (empty)
│
├── llm/
│   ├── __init__.py           (empty)
│   ├── interfaces.py         LLMProvider Protocol + CompletionResult / StreamChunk / ToolCall
│   ├── _openai_compatible_impl.py  OpenAICompatibleLLM + _post_with_retries_shared + LLMProviderError
│   ├── openai_compatible_llm.py    re-export shim (historical)
│   ├── openai_llm.py         OpenAILLM  (hosted OpenAI; subclass of OpenAICompatibleLLM)
│   ├── ollama_llm.py         OllamaLLM  (Ollama /api/chat; subclass of OpenAICompatibleLLM)
│   ├── mock_llm.py           MockLLM    (deterministic in-process; tests + CI)
│   └── factory.py            build_llm(settings) + LLMConfigurationError + startup probe
│
├── agents/
│   ├── __init__.py           (empty)
│   ├── tools.py              Tool Protocol + SearchCodebaseTool + ReadFileTool + ListFilesTool + ToolRegistry
│   └── agent_loop.py         AgentLoop + AgentResult + DEFAULT_MAX_ITERATIONS = 8
│
└── prompting/
    ├── __init__.py           (empty)
    ├── models.py             ConversationTurn + ChatMessage + AssembledPrompt
    ├── token_estimator.py    TokenEstimator Protocol + HeuristicTokenEstimator + CHARS_PER_TOKEN_ESTIMATE = 4.0
    ├── context_assembler.py  assemble_context (greedy by fused-score order)
    ├── citation.py           format_citation_header + format_chunk_with_citation
    ├── templates.py          SYSTEM_PROMPT + build_messages
    └── pipeline.py           PromptBuilder (orchestrator)
```

Every `__init__.py` is empty. Callers always import from the
submodule path (`semrag.reasoning.llm.factory.build_llm`,
`semrag.reasoning.agents.agent_loop.AgentLoop`). Same convention used
by every other package — see
[`src/semrag/storage/README.md`](../storage/README.md) for the
rationale.

---

## 3. Module responsibilities

### `llm/interfaces.py`

The `LLMProvider` Protocol — the only thing the rest of the codebase
binds to. Four members:

- `provider_name` — short string used in metrics tags.
- `model_name` — the configured model identifier.
- `complete(messages, tools=None) -> CompletionResult` — one
  non-streaming round-trip.
- `stream(messages, tools=None) -> Iterator[StreamChunk]` — streaming
  generator.

Plus three Pydantic models: `ToolCall` (id, name, arguments),
`CompletionResult` (content, tool_calls, finish_reason, optional
prompt_tokens / completion_tokens / total_tokens), and
`StreamChunk` (delta_content, delta_tool_calls, finish_reason).

Token-usage fields are optional because some providers (older Ollama
builds, `MockLLM`) do not report them. The reasoning layer surfaces
them as metrics when present and ignores them when absent — no
separate API.

### `llm/_openai_compatible_impl.py`

`OpenAICompatibleLLM` + `LLMProviderError` +
`_post_with_retries_shared`. The base class for every OpenAI-shaped
provider (vLLM, llama.cpp HTTP server, OpenAI-compatible cloud
shims, hosted OpenAI, and — for Ollama ≥ 0.5.0 — the same
`/v1/chat/completions` path).

Owns:

- **Retry loop.** `_post_with_retries_shared` does up to
  `max_retries` attempts with exponential backoff (`2 ** attempt`
  seconds), emits `llm.request.duration_ms` per attempt tagged with
  `op=provider_name` and `stream=...`, and raises `LLMProviderError`
  on the last failure.
- **Non-streaming `complete()`.** POSTs to
  `/v1/chat/completions`, parses `choices[0].message` and the
  `usage` block, surfaces token counts as metrics when present.
- **Streaming `stream()`.** POSTs with `stream=True`, parses
  SSE `data: ...` lines, accumulates tool-call fragments by index
  (arguments arrive as JSON fragments that only parse once
  concatenated), and yields one `StreamChunk` per line.
- **Default endpoint.** `_endpoint(stream)` returns
  `/v1/chat/completions`; subclasses override this.

`OpenAILLM` (`openai_llm.py`) is a one-line subclass that only
overrides `provider_name` to return `"openai"`. The hosted-OpenAI
shape is identical to the OpenAI-compatible base; splitting them
keeps each class small and clarifies that OpenAI is *not* a
self-hosted server (no `/v1` shim concerns, no Ollama-flavoured
streaming).

### `llm/ollama_llm.py`

`OllamaLLM` — Ollama's native `/api/chat` client. The reason this
exists separately from the OpenAI-compatible base is concrete:
**Ollama 0.3x builds (verified against 0.32.5) return 404 on
`/v1/chat/completions` even though `/v1/models` returns 200.** The
Ollama client uses `/api/chat` instead, flattens Ollama's
`{message: {...}, done: true}` response into the same shape the
shared parser expects, and uses NDJSON for streaming (one JSON
object per line) instead of SSE.

Owns:

- **Endpoint and response shape.** `_endpoint()` returns
  `/api/chat`; `_build_payload()` produces the Ollama-native
  request body, including `options.num_predict` when configured.
- **Token-usage mapping.** Ollama reports `prompt_eval_count` /
  `eval_count`; the client maps these to the same `usage` shape and
  emits the same `llm.tokens.*` metrics so dashboards don't need
  provider-specific tag values.
- **`num_predict` cap.** Capping the output length is the single
  biggest latency lever against `qwen2.5:7b` on CPU — an uncapped
  response can run 60+ seconds generating bullet points the user
  never asked for. The cap is operator-configurable via
  `Settings.llm_num_predict`.
- **NDJSON streaming parser.** Accumulates tool calls by index,
  yields `StreamChunk` per line, terminates when `event["done"]` is
  true.

Inherits retry/timeout/header logic from the OpenAI-compatible base
via `_post_with_retries_shared`.

### `llm/mock_llm.py`

`MockLLM` — deterministic, in-process LLM for tests and CI. Returns
scripted `CompletionResult`s in order regardless of messages.
Streaming emits a single chunk containing the whole response. No
network, no real model.

Why a real (network-attached) provider isn't the test default:
every CI run that talks to a real LLM is slow, flaky, costs money,
and leaks prompt text. Provider behavior (request shaping, retry
policy, response parsing) is tested in isolation against
`responses`-faked HTTP. This is the same boundary `MockLLM` has had
since the embedding provider tests in Part 9.

### `llm/factory.py`

`build_llm(settings) -> LLMProvider` + `LLMConfigurationError` +
`_probe(client)`. The single factory entry point; every other call
site goes through it.

`build_llm` reads `settings.llm_provider` and dispatches:

- `"ollama"` → `OllamaLLM` (with `num_predict=settings.llm_num_predict`)
- `"openai"` → `OpenAILLM`
- `"openai_compatible"` or `"vllm"` → `OpenAICompatibleLLM` (the
  two are aliased because vLLM exposes the OpenAI-compatible API)
- `"anthropic"` → raises `LLMConfigurationError` (intentional —
  Anthropic's Messages API has a different shape; the concrete
  class will be added when the deployment target requires it)
- `"mock"` → `MockLLM`
- anything else → raises `LLMConfigurationError`

The factory reads from `Settings` rather than accepting loose
kwargs on purpose: one source of truth means the CLI and the API
can't drift, and a misconfiguration (typo in `llm_provider`, model
not set) fails at startup, not in the middle of a request.

`_probe(client)` runs one minimal `complete([{"role": "user",
"content": "ping"}])` call to confirm the endpoint actually
responds. Catches URL typos, missing API keys, and Ollama-version
404s before the first user waits for one. Controlled by
`settings.llm_probe_at_startup`.

### `agents/tools.py`

`Tool` Protocol + `SearchCodebaseTool` + `ReadFileTool` +
`ListFilesTool` + `ToolRegistry`. Three real tools, each wrapping
something already built rather than reimplementing it. Every tool
exposes an OpenAI-compatible function-calling schema
(`to_openai_schema()`) so the same definition feeds both
`LLMProvider.complete(tools=...)` and the agent loop's own
dispatch — one definition, not two that can drift apart.

- **`SearchCodebaseTool`** — `name = "search_codebase"`. Wraps
  `HybridRetriever.retrieve`; the `query` argument is embedded
  with the configured `embed_query_fn`, retrieval runs, and the
  results are formatted with `format_chunk_with_citation` before
  being returned to the agent. This is the only tool that can
  produce a citation-eligible result.
- **`ReadFileTool`** — `name = "read_file"`. Reads real file
  content from the project root, optionally bounded by
  `line_start` / `line_end`. Has a `MAX_CHARS = 20_000` cap (an
  agent handing back an entire large file defeats the point of
  retrieval). Refuses to read paths that resolve outside the
  project root.
- **`ListFilesTool`** — `name = "list_files"`. Enumerates files
  via `FileRow` (the storage layer's current-state file table),
  optionally filtered by `category`. This is what the agent
  reaches for when answering "overview" or "architecture"
  questions with empty initial retrieval.

`ToolRegistry` is the dispatcher: it builds the list of OpenAI
schemas for the LLM call, and on the response side it looks up
the tool by name, executes it, and returns a string result.
Tool exceptions are caught and returned as error strings
(`"Error: invalid arguments for tool 'X': ..."`,
`"Error executing tool 'X': ..."`) so a tool failure does not
crash the agent loop. Per-tool metrics
(`agent.tool.duration_ms`, `agent.tool.ok`,
`agent.tool.bad_args`, `agent.tool.error`,
`agent.tool.unknown`) are emitted on every call.

### `agents/agent_loop.py`

`AgentLoop` + `AgentResult` + `DEFAULT_MAX_ITERATIONS = 8`. The
ReAct-style loop the LLM-facing layer is built around. One method:
`run(messages) -> AgentResult`.

```text
for iteration in range(1, max_iterations + 1):
    result = llm.complete(messages, tools=tool_schemas)
    if not result.tool_calls:
        return AgentResult(final_answer=result.content, ...)
    for tool_call in result.tool_calls:
        tool_result = tools.execute(tool_call.name, tool_call.arguments)
        append tool result to messages
return AgentResult(final_answer=None, hit_iteration_limit=True)
```

Three design points that matter:

- **`max_iterations` is a hard cap.** A model can get stuck calling
  tools indefinitely (e.g. repeatedly searching without ever
  synthesizing an answer); bounding it means a misbehaving model
  degrades to a clear "ran out of turns" result with
  `final_answer=None` and `hit_iteration_limit=True`, not an
  unbounded loop.
- **Tool calls' prose echo is dropped.** When the LLM returns both
  `content` (a prose prefix) and `tool_calls`, the prose is
  preserved only if non-empty. A small local model (`qwen2.5:7b`)
  often fills `content` with garbage ("Ronaldo", random tokens)
  alongside a valid tool call; replaying that prose back into the
  next turn's context derails the next completion. Tool calls
  carry the intent; the prose before them is noise in that case.
- **Tool result is `role: "tool"` with `tool_call_id`.** The
  OpenAI-compatible schema for tool results requires the
  `tool_call_id` from the originating call; the loop preserves it
  on every result so the LLM can correlate results to calls.

Per-iteration metrics (`agent.run.duration_ms` with `tools=`
count, `agent.run.iterations`, `agent.run.final_answer`,
`agent.run.iteration_limit_hit`) and per-event traces
(`agent iteration final answer`, `agent iteration tool calls`,
`agent tool result`) are emitted at every step.

### `prompting/models.py`

Three Pydantic models:

- `ConversationTurn` — `role` (`"user"` or `"assistant"`,
  matches OpenAI-compatible chat roles) + `content`.
- `ChatMessage` — `role` + `content`. The shape of one message in
  the assembled prompt.
- `AssembledPrompt` — the result of a `PromptBuilder.build()` call.
  Carries `messages`, `included_chunks` (the citation/provenance
  record for this prompt — chunks that actually made it into the
  context), `excluded_chunk_count` (a truncation signal — how many
  retrieved chunks didn't fit the budget), and
  `estimated_total_tokens`.

### `prompting/token_estimator.py`

`TokenEstimator` Protocol + `HeuristicTokenEstimator` +
`CHARS_PER_TOKEN_ESTIMATE = 4.0`.

`TokenEstimator.estimate(text) -> int` is the single method.

The heuristic is documented in the module docstring: real BPE
tokenizers need downloadable vocabulary files (`tiktoken` fetches
from `openaipublic.blob.core.windows.net`, `tokenizers` downloads
the vocab itself — both blocked in this codebase's deployment
environment). What's built instead is a documented,
testable, conservative ratio of ~4 characters per token for
English-language text. The estimator rounds up
(`max(1, -(-len(text) // int(chars_per_token))` — the `-(-x // y)`
trick is a ceiling division in Python) so it never under-budgets;
under-budgeting is a much worse failure mode than over-budgeting
(it produces a prompt that's actually over the model's real token
limit).

The `TokenEstimator` Protocol is the seam: a real tokenizer
becomes a one-class change once one's model-specific vocabulary
is reachable (or by vendoring a vocab file directly rather than
fetching it at runtime).

### `prompting/context_assembler.py`

`assemble_context(results, budget_tokens, estimator, max_chunks=None)`
→ `(included_results, formatted_context_text, used_tokens)`. Greedy
packing by fused-score order: results are already ranked by
`HybridRetriever`, this takes them in that order, includes each
one whole (never truncating a chunk mid-text — a half-cut
assertion or function is worse than not including it at all) until
the next one wouldn't fit the remaining budget, then stops.

Two knobs:

- **`budget_tokens`** — the hard cap, computed by `PromptBuilder`
  as `model_context_window - reserved_for_response - system -
  query - history - file_index`. A chunk that would push over
  the budget is skipped entirely.
- **`max_chunks`** — an optional additional cap on the *count* of
  included chunks, regardless of budget. Greedy packing can admit
  a dozen chunks of real code if each is small — fine for a
  frontier model, but a 7B local LLM on CPU spends 80+ seconds
  processing 6K input tokens. Capping the count (in addition to
  the token budget) keeps prompts small enough for the slow path.

A chunk that itself exceeds the budget is dropped
(`if used_tokens + additional_cost > budget_tokens: continue`).
The budget gate runs before the `max_chunks` cap so a single
giant chunk doesn't push everything else out.

The separator `\n\n---\n\n` (`DEFAULT_CHUNK_SEPARATOR`) is what
joins included chunks in the formatted text. Its own token cost
is counted against the budget on every join.

### `prompting/citation.py`

Two functions: `format_citation_header(result)` and
`format_chunk_with_citation(result)`. The first produces the
`[Source: <path>, lines X-Y]` header (or just `[Source: <path>]`
when line range is unknown); the second prepends that header to
the chunk text. This is the format the prompt asks the LLM to
echo back, and the format the citation validator
(`src/semrag/evaluation/citation_validator.py`) checks against
the included chunks.

The format is deliberately plain and consistent — not the
`chunk_id` (an internal identifier meaningless to an engineer
reading the response), but a file path and line range, which is
what someone would actually open their editor to.

### `prompting/templates.py`

`SYSTEM_PROMPT` (a string constant) and `build_messages(...)`
that turns `(context_text, history, query, file_index?)` into a
list of OpenAI-shape messages.

The system prompt is where "hallucination prevention" and
"grounding" get enforced at the prompt level: answer only from
the provided context, cite every claim, say so explicitly when
the context doesn't contain an answer, place citations inline
after the sentence they support, and treat the FILE INDEX as
authoritative when one is provided. This is a prompt-level
control, not a guarantee; real hallucination measurement is the
eval harness's job.

The tool list is named inline in the system prompt so the LLM
knows what to reach for when the initial retrieval is empty
(e.g. "give me an overview of this repo"). Without that list,
the model often responds with a generic "no context was found"
rather than calling `list_files` to enumerate the project and
then `search_codebase` for representative chunks.

`build_messages` produces one of two shapes for the final user
message:

- **With context (or file index).** `FILE INDEX:\n<index>` and/or
  `CONTEXT:\n<context>` are joined above `QUESTION:\n<query>`.
- **Empty context.** The user message is replaced with an
  explicit instruction: "The initial retrieved context was empty
  for this question. Use the available tools … to gather context
  before answering." This is what makes the empty-retrieval
  case recover instead of degrading into an ungrounded
  response.

### `prompting/pipeline.py`

`PromptBuilder` + three defaults: `DEFAULT_MODEL_CONTEXT_WINDOW =
8192`, `DEFAULT_RESERVED_FOR_RESPONSE = 1024`,
`DEFAULT_MAX_HISTORY_TOKENS = 1024`. The orchestrator that turns
`(query, retrieval_results, history, file_index?)` into an
`AssembledPrompt` ready to send to the LLM.

`build()`:

1. Estimates the system prompt, query, file index, and trimmed
   history tokens.
2. Trims conversation history with `_trim_history` — keeps the
   most recent turns (sliding window), drops the oldest first.
3. Computes the context budget as
   `model_context_window - reserved_for_response - system - query -
   history - file_index`. Floored at 0.
4. Calls `assemble_context` to produce the formatted context
   text and the list of included chunks.
5. Calls `build_messages` to produce the OpenAI-shape message
   list.
6. Returns `AssembledPrompt` with the messages, included
   chunks, excluded count, and the total estimated tokens.

The budget allocation order is deliberate and matters: the
system prompt and conversation history are non-negotiable (an
LLM without its grounding instructions, or a conversation
missing its own prior turns, produces worse answers than one
with less retrieved context), so they're subtracted from the
total budget **first**, and whatever remains goes to retrieval
context, which degrades gracefully by including fewer chunks
rather than failing outright when the budget is tight.

`_trim_history` walks the history in reverse, keeps adding
turns until adding the next one would exceed
`max_history_tokens`, then returns the kept turns in their
original order. The most recent turns are almost always the
most relevant to interpreting the current question, so the
sliding window drops the oldest first.

---

## 4. End-to-end reasoning pipeline

How a query flows from `QueryService.ask()` to a final answer:

```text
query + project_id
       │
       ▼
   ┌──────────────────────────────────────────────┐
   │ retrieval.HybridRetriever.retrieve(...)      │  in src/semrag/retrieval/
   │ → list[RetrievalResult]                      │
   └────────────┬─────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────┐
   │ reasoning.prompting.PromptBuilder.build(     │  builds the prompt
   │     query, results, history, file_index?     │
   │ )                                            │
   │ → AssembledPrompt                            │
   └────────────┬─────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────┐
   │ reasoning.agents.AgentLoop.run(messages)     │  the LLM/tool loop
   │   for iter in 1..max_iterations:             │
   │     result = llm.complete(messages, tools)   │
   │     if not result.tool_calls:                │
   │       return AgentResult(                    │
   │         final_answer=result.content, ...     │
   │       )                                      │
   │     for tc in result.tool_calls:            │
   │       tools.execute(tc.name, tc.arguments)   │
   │       append role:tool message               │
   │   return AgentResult(hit_iteration_limit)    │
   └────────────┬─────────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────────────────────┐
   │ reasoning.llm.LLMProvider                    │  the configured provider
   │   ollama / openai / openai_compatible / mock │
   └──────────────────────────────────────────────┘
```

The retrieval → prompt → agent → LLM chain is the entire
reasoning path. The service layer composes them; this package
owns everything inside the box.

### 4.1 Tool execution model

When the LLM returns one or more `tool_calls`, the loop appends
the assistant message *with* the tool calls preserved (including
`id`, `name`, `arguments`) to the conversation, then for each
tool call:

1. `ToolRegistry.execute(name, arguments)` looks up the tool by
   name and calls its `execute(**arguments)`.
2. The string result is appended as
   `{"role": "tool", "tool_call_id": <id>, "content": <result>}`.
3. The next iteration's `llm.complete` sees the tool result and
   decides whether to call more tools or synthesize the answer.

A tool exception is caught by `ToolRegistry` and returned as an
error string, so a single tool failure does not crash the agent
loop — the LLM sees the error as a `role: "tool"` message and can
recover (e.g. by reformulating the search query or falling back
to a different tool).

### 4.2 Why the budget order is fixed

The system prompt and conversation history are subtracted from
the model context window **first**; only the remainder goes to
retrieval context. The reason is asymmetric degradation:

- **No system prompt** → the LLM stops citing, starts
  hallucinating, and the citation validator in evaluation starts
  failing.
- **No history** → multi-turn conversations break, the model
  loses context the user already paid for.
- **Less retrieved context** → the model just sees fewer chunks
  and produces a shorter answer. Still grounded, still
  citable, just less complete.

The same principle applies to `reserved_for_response`: every
prompt reserves 1024 tokens for the model's reply so the loop
doesn't get cut off mid-sentence by hitting the model's output
limit. Frontier models with larger context windows should bump
both numbers; the defaults target a 7B local model on CPU.

---

## 5. Prompt construction

`PromptBuilder.build(query, retrieval_results, conversation_history?,
file_index?, max_chunks?) -> AssembledPrompt`.

### 5.1 Budget allocation

```text
model_context_window   = 8192  (configurable; default targets 7B CPU)
reserved_for_response  = 1024  (configurable)
max_history_tokens     = 1024  (configurable)

system_tokens          = estimator.estimate(SYSTEM_PROMPT)
query_tokens           = estimator.estimate(query)
file_index_tokens      = estimator.estimate(file_index) if file_index else 0
history_tokens         = sum(estimator.estimate(t.content) for t in trimmed_history)

context_budget = max(0,
    model_context_window
    - reserved_for_response
    - system_tokens
    - query_tokens
    - history_tokens
    - file_index_tokens
)
```

Anything not allocated here is left on the floor — the model
context window is a hard cap and the prompt builder never
exceeds it. If the inputs already exceed the window (a giant
file index, a long history) `context_budget` floors at 0 and
the assembled context is empty.

### 5.2 History trimming

`_trim_history` walks the conversation in reverse, accumulating
turns into `kept` until the next turn would push
`used + cost > max_history_tokens`, then returns the kept turns
in their original order. The oldest turns are dropped first;
the most recent turns are kept. The trim respects the
`max_history_tokens` cap; the system prompt and the current
query are not counted against it.

### 5.3 Context assembly

`assemble_context` is called with the budget and the configured
`estimator`. See [§3 prompting/context_assembler.py](#3-module-responsibilities)
for the greedy-by-fused-score-order algorithm and the
`max_chunks` cap. The output is a list of `RetrievalResult`
objects that fit the budget, joined with
`DEFAULT_CHUNK_SEPARATOR = "\n\n---\n\n"`, and the
`included_chunks` field on the returned `AssembledPrompt` is
this list — the citation/provenance record for what actually
made it into the prompt.

### 5.4 Message construction

`build_messages(context_text, trimmed_history, query, file_index?)`
produces a list of OpenAI-shape messages:

```python
[
    {"role": "system", "content": SYSTEM_PROMPT},
    *history_turns,                                    # role: "user" / "assistant"
    {"role": "user", "content": <FILE INDEX + CONTEXT + QUESTION, or the empty-context prompt>},
]
```

Two shapes for the final user message — see
[`prompting/templates.py`](#3-module-responsibilities) for the
exact text. Both shapes end with the user's question as
`QUESTION: <query>` so the LLM always has the actual question
on the same line as the context that grounds it.

### 5.5 What the prompt builder does **not** do

- **Decide which LLM to call.** `LLMProvider` is a constructor
  argument to `AgentLoop`, not to `PromptBuilder`.
- **Run the agent loop.** `AgentLoop` consumes the
  `AssembledPrompt` (or just the message list); the builder
  does not orchestrate iterations.
- **Cache prompts.** The API-layer query cache wraps the
  reasoning path; the prompt builder is stateless.
- **Re-rank the retrieved chunks.** Retrieval already produced
  the order; the prompt builder trusts it.

---

## 6. Agent loop

`AgentLoop.run(messages: list[dict]) -> AgentResult`. The
`messages` argument is the list produced by `PromptBuilder` —
typically `[{"role": "system", ...}, ..., {"role": "user",
...}]`. The loop does not inspect message roles; it just
appends its own messages and asks the LLM to continue.

```text
AgentResult(
    final_answer: str | None,
    messages: list[dict],       # the full conversation including tool calls and results
    iterations_used: int,
    hit_iteration_limit: bool,
)
```

The loop's three termination conditions:

| Condition                              | `final_answer`     | `hit_iteration_limit` |
| -------------------------------------- | ------------------ | --------------------- |
| LLM returns no tool calls              | `result.content`   | `False`               |
| LLM ran out of iterations (default 8)  | `None`             | `True`                |
| LLM raised mid-loop                    | exception propagates | n/a                |

The first condition is the happy path: the LLM synthesizes an
answer from the context and any tool results, the loop appends
the assistant message, and returns.

The second is the safety net: a model that keeps calling tools
without ever answering is bounded. `hit_iteration_limit=True` is
a signal to the caller (and to operators) that the agent did
not converge. The LLM-facing service layer should surface this
clearly in the response; the agent loop does not invent a
fallback answer.

The third is genuinely an error: the agent loop does not catch
LLM provider exceptions. They propagate to the caller, which
translates them into a standard error envelope (see
[`src/semrag/api/README.md`](../api/README.md)).

### 6.1 Conversation state

`AgentResult.messages` is the full conversation including:

- The original system message and history (whatever the caller
  passed in).
- One assistant message per iteration that emitted tool calls
  (with `tool_calls` populated) or a final answer (with
  `content` populated).
- One `role: "tool"` message per tool result, with
  `tool_call_id` preserved.

The conversation is the agent's state. The agent loop does
**not** keep a separate `state` object; the messages list is
the source of truth, and the caller is responsible for
persisting it (across turns, across requests) if the
conversation is meant to be multi-turn.

For multi-turn conversations: the service layer keeps the
caller's `ConversationTurn` history, builds the new
`AssembledPrompt` with the same history, and calls
`AgentLoop.run` fresh each time. The agent loop itself is
stateless across calls; state lives in the messages list.

### 6.2 Why a hard `max_iterations`

The docstring is explicit: "a model can get stuck calling tools
indefinitely (e.g. repeatedly searching without ever
synthesizing an answer) - bounding it means a misbehaving model
degrades to a clear 'ran out of turns' result instead of an
unbounded loop." This is the same reason every agent framework
caps iterations; the value is the same eight. Tune via the
constructor argument `max_iterations`.

### 6.3 What the agent loop does **not** do

- **Persist conversations across calls.** Stateless across
  calls; the caller owns persistence.
- **Catch LLM provider errors.** They propagate. The reasoning
  package's job is to surface them, not to swallow them.
- **Inspect message content.** The loop only cares about
  `result.tool_calls`; everything else is passed through.
- **Run tools in parallel.** Tool calls in a single iteration
  run sequentially in the order the LLM returned them. The
  OpenAI schema allows parallel tool calls, but `ToolRegistry`
  executes them serially — see [§13 Common mistakes](#13-common-mistakes)
  for why.

---

## 7. Tool execution

`ToolRegistry.execute(name, arguments) -> str`. The
single-call interface the agent loop uses. Three real tools
ship today; the Protocol is the seam for adding more.

### 7.1 Tool Protocol

```python
class Tool(Protocol):
    name: str
    description: str
    def parameters_schema(self) -> dict: ...
    def execute(self, **kwargs) -> str: ...
    def to_openai_schema(self) -> dict: ...
```

`to_openai_schema()` is provided by the Protocol because every
existing tool has the same shape (`{"type": "function",
"function": {"name": ..., "description": ...,
"parameters": ...}}`). Tools inherit the Protocol's
implementation by calling `Tool.to_openai_schema(self)` —
explicit, not magic.

### 7.2 `SearchCodebaseTool`

The only tool that can produce a citation-eligible result.
Wraps `HybridRetriever.retrieve`; the `query` argument is
embedded with the configured `embed_query_fn` (provided at
construction), retrieval runs with
`VectorSearchFilter(project_id=...)`, and each result is
formatted with `format_chunk_with_citation` before being
joined with `DEFAULT_CHUNK_SEPARATOR`. The string returned to
the agent is therefore the same `[Source: …]\n<text>`
format the prompt itself uses — the LLM doesn't have to
re-parse anything.

### 7.3 `ReadFileTool`

Reads real file content from the configured project root.
Optional `line_start` / `line_end` slice the file to a range
(1-indexed, inclusive on both ends). `MAX_CHARS = 20_000`
caps the returned text — an agent handing back an entire
large file defeats the point of retrieval.

Two safety checks:

- **Path traversal.** `target.resolve()` must still start with
  the project root; if it doesn't, the tool returns an error
  string (`"Error: <path> resolves outside the project root
  - refusing to read it."`) rather than reading.
- **Missing files.** A non-existent path returns an error
  string; the LLM can recover by calling `list_files` to
  confirm what the repository contains.

### 7.4 `ListFilesTool`

Enumerates files via the `FileRow` table (the storage layer's
current-state file table, populated by the scanner at index
time). Optional `category` filter. Results are joined with
`\n`. The LLM uses this to ground "overview" and "architecture"
questions when the initial retrieval was empty.

### 7.5 Error handling

`ToolRegistry.execute` catches every exception from
`tool.execute(**arguments)`:

- `TypeError` (bad arguments) → returns
  `"Error: invalid arguments for tool '<name>': <exc>"`,
  emits `agent.tool.bad_args`.
- Any other exception → returns
  `"Error executing tool '<name>': <exc>"`, emits
  `agent.tool.error`.
- Unknown tool name → returns
  `"Error: unknown tool '<name>'."`, emits
  `agent.tool.unknown`.

The error string is appended as a normal `role: "tool"`
message; the LLM sees the failure and can decide whether to
retry, reformulate, or fall back to a different tool. A tool
crash never crashes the agent loop.

Successful calls emit `agent.tool.duration_ms` (with `tool=`
tag) and `agent.tool.ok`.

---

## 8. LLM providers and interfaces

The Protocol-based model-agnostic pattern. Nothing outside
`reasoning/llm/interfaces.py` imports a concrete provider class
directly; every other module binds to the Protocol.

### 8.1 The four providers

| Provider                | Class                | Endpoint                              | Streaming       | Token usage                |
| ----------------------- | -------------------- | ------------------------------------- | --------------- | -------------------------- |
| Ollama (≥ 0.5.0)        | `OpenAICompatibleLLM` | `POST /v1/chat/completions`           | SSE             | `usage.*` block            |
| Ollama (0.3x)           | `OllamaLLM`          | `POST /api/chat`                      | NDJSON          | `prompt_eval_count` / `eval_count` (mapped) |
| OpenAI (hosted)         | `OpenAILLM`          | `POST /v1/chat/completions`           | SSE             | `usage.*` block            |
| vLLM / llama.cpp / OAI-compat cloud | `OpenAICompatibleLLM` | `POST /v1/chat/completions` | SSE       | `usage.*` block            |
| Tests / CI              | `MockLLM`            | in-process                            | one chunk       | always None                |

Ollama 0.3x is the reason `OllamaLLM` exists as a separate
class. The factory selects it via `llm_provider == "ollama"`
regardless of version; the class itself targets `/api/chat`
which works on every Ollama build (verified against 0.32.5).

### 8.2 Token-usage handling

Every real provider exposes token usage in some form; the
completion path normalizes them into `CompletionResult` and
emits `llm.tokens.total` / `llm.tokens.prompt` /
`llm.tokens.completion` as `count` metrics tagged with
`provider=...` and `model=...`. Dashboards don't need to
branch on provider to chart token throughput.

`MockLLM` does not report token usage; the reasoning layer
silently skips the metrics when usage fields are absent (no
exception, no special-cased call site). The same code path
handles "real provider returned no usage" the same way.

### 8.3 Retry and timeout

Every real provider uses `_post_with_retries_shared` for
network calls. Defaults: `timeout_seconds=120.0` for the
OpenAI-compatible base, `300.0` for Ollama (longer for local
model cold-start); `max_retries=3` with exponential backoff
(`2 ** attempt` seconds). Failures emit
`llm.request.duration_ms` per attempt with `op=provider_name`
and `attempt=...` tags, and `LLMProviderError` is raised on
the last attempt.

The factory's `_probe()` does one minimal `complete` at
startup to catch URL typos, missing auth, and the Ollama-0.3x
`/v1/chat/completions` 404 before the first user request. Set
`SEMRAG_LLM_PROBE_AT_STARTUP=false` to disable.

### 8.4 Adding a new provider

Three steps:

1. **Implement `LLMProvider`** in
   `semrag/reasoning/llm/<provider>_llm.py`. For an
   OpenAI-shape provider, subclass `OpenAICompatibleLLM` and
   override `_endpoint()` and (if needed) `provider_name`.
2. **Add a case in `build_llm`** in
   `semrag/reasoning/llm/factory.py`. The settings key goes
   in `core/settings.py` (see [`docs/developer_guide.md` §5](../../../docs/developer_guide.md)).
3. **Add a test** in `tests/unit/reasoning/llm/` using
   `MockLLM` for control flow and an `httpx`-faked HTTP
   response for request-shape validation. (The same pattern
   the embedding provider tests in Part 9 use.)

That's it. `QueryService.ask()` calls
`LLMProvider.complete(...)` and `LLMProvider.stream(...)`;
it doesn't know or care which class is on the other side.

---

## 9. Context assembly

The prompting layer's job: turn `list[RetrievalResult]` into
the string that goes into the `CONTEXT:` section of the user
message, and produce the list of chunks that *actually*
ended up in the prompt (the citation record).

### 9.1 Greedy by fused-score order

`assemble_context` iterates `results` in their input order
(which is the `HybridRetriever`'s ranking) and admits each
chunk that fits the budget. A chunk that would push the
running total over `budget_tokens` is skipped (not
truncated). The first chunk to exceed the budget also stops
iteration. The output is `included_results` (in ranking
order), `formatted_context_text` (joined with the separator),
and `used_tokens`.

This is "context prioritization" and "token budgeting" from
the architecture doc, implemented plainly: prioritization is
"trust the ranking you were already given," and budgeting is
"stop before you go over." A smarter packing (e.g. skipping
one big chunk to fit two smaller lower-ranked ones) would
second-guess the retrieval ranking in a way that's hard to
justify without evidence it actually improves answer quality
— the eval harness's job to measure, not this module's to
assume.

### 9.2 Citation in context

Every included chunk is formatted by
`format_chunk_with_citation(result)`, which prepends
`format_citation_header(result)` to the chunk text. The
header is `[Source: <relative_path>, lines <start>-<end>]`
when both line numbers are known, or `[Source: <relative_path>]`
when either is `None`.

This is the format the prompt asks the LLM to echo back, and
the format the citation validator checks against. The LLM
should not invent a path that wasn't in the prompt; the
system prompt's "FILE INDEX" section makes this explicit.

### 9.3 Empty context

When `assemble_context` admits zero chunks (the budget was
zero, or every chunk was too large), `build_messages` takes
the empty-context branch and produces a user message that
explicitly tells the LLM to use tools:

> "The initial retrieved context was empty for this question.
> Use the available tools (search_codebase, list_files,
> read_file) to gather context before answering."

This is the recovery path for the empty-retrieval case —
without it, the LLM would either answer from general
knowledge (hallucination) or refuse to answer. With it, the
LLM reaches for `list_files` and `search_codebase` to
gather context and the conversation proceeds normally.

### 9.4 What context assembly does **not** do

- **Re-rank.** Retrieval's order is sacred.
- **Summarize.** Chunks are included whole. Summarization
  belongs in a future `Summarizer` Protocol, not here.
- **Deduplicate.** Two chunks with overlapping text both go
  in. Deduplication is the agent's job (the LLM can
  synthesize across redundant context), not the prompt
  builder's.
- **Embed.** `SearchCodebaseTool` embeds the query before
  retrieval; the prompt builder is downstream of that.

---

## 10. Citation handling

The citation contract is the single most important contract
in the reasoning layer. The prompt produces citations; the
LLM is supposed to echo them; the citation validator checks
that every `[Source: …]` in the LLM's response corresponds
to a real `RetrievalResult` in the prompt.

### 10.1 The format

```
[Source: <relative_path>, lines <start>-<end>]
<chunk text>
```

Two cases the formatter handles:

- Both `line_start` and `line_end` are non-`None` → header
  includes the line range.
- Either is `None` → header is just `[Source: <path>]`.

The format is plain ASCII. No markdown, no extra brackets,
no trailing punctuation. The system prompt asks for the
exact same format, and the citation validator
(`src/semrag/evaluation/citation_validator.py`) parses it
verbatim.

### 10.2 Why the citation contract is the validator's contract

The eval harness's citation validator runs after every
evaluation question and flags any `[Source: <path>]` that
doesn't appear in the prompt's `included_chunks` paths.
That's how "groundedness" is measured — not by the LLM
saying it's grounded, but by checking that the citations it
emitted correspond to real retrieved paths. A bad citation
is a failed eval; a missing citation is a failed eval; a
hallucinated citation is a failed eval.

This is the loop the prompt's "every factual sentence must
carry its own inline citation" instruction actually closes:
the LLM produces the citation, the validator checks the
citation, the eval reports groundedness, the metric is
real. The prompt-level control is necessary, not
sufficient.

### 10.3 What the reasoning layer does **not** do

- **Validate citations.** That's the eval layer's job.
- **Format citations for display in the response.** The
  reasoning layer produces the LLM's raw response; the API
  layer wraps it in the response envelope.
- **Inject citations the LLM didn't write.** The LLM is
  expected to cite; if it doesn't, the validator flags the
  answer. There is no "post-process to add citations"
  fallback.

---

## 11. Extension points

| Need                                          | Touch                                  | What to add                                   |
| --------------------------------------------- | -------------------------------------- | --------------------------------------------- |
| New LLM provider                              | `llm/<provider>_llm.py` + `llm/factory.py` | Implement `LLMProvider`; add case in `build_llm` |
| New tool                                      | `agents/tools.py`                      | Implement `Tool` Protocol; register in the service layer's `_build_tools()` |
| New prompt template                           | `prompting/templates.py`               | Modify `SYSTEM_PROMPT` constant or add a new `build_messages` variant |
| New token estimator                           | `prompting/token_estimator.py`         | Implement `TokenEstimator` Protocol; inject at `PromptBuilder` construction |
| Real tokenizer (tiktoken / tokenizers)        | `prompting/token_estimator.py`         | New class implementing `TokenEstimator`; same Protocol; no other change |
| New context-assembly strategy                 | `prompting/context_assembler.py`       | Replace `assemble_context`; `PromptBuilder` takes the function via DI if needed |
| New conversation-trimming strategy            | `prompting/pipeline.py`                | Replace `_trim_history` on the `PromptBuilder` (currently a method) |
| New token-budget defaults                     | `prompting/pipeline.py`                | `DEFAULT_*` constants; pass overrides at construction |
| New LLM-facing metric                         | `core/metrics.py`                      | Add a helper, never import a Prometheus collector from outside this file |

Two extension points are deliberately **not** available:

- **Replacing `LLMProvider` mid-conversation.** A `AgentLoop`
  instance is bound to one provider. Use a new `AgentLoop`
  per call (the service layer already does this).
- **Streaming `AgentLoop.run`.** The current loop calls
  `llm.complete()` (non-streaming). Streaming is exposed on
  `LLMProvider.stream` for direct callers; the agent loop
  does not yet drive a streaming iteration. Add when a
  caller (e.g. a chat UI) needs token-by-token output
  mid-iteration.

---

## 12. Common debugging paths

| Symptom                                                | Look first                                                                                  |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `final_answer=None`, `hit_iteration_limit=True`        | The agent ran out of iterations. Check `agent.run.iteration_limit_hit`; lower the prompt's `max_chunks` or tighten the tool list |
| Tool returns "Error: invalid arguments"               | The LLM produced a tool call with the wrong arg shape; check `agent.tool.bad_args` and the tool's `parameters_schema()` |
| LLM echoes garbage text alongside a tool call           | The loop drops the prose if it's empty; if the garbage is non-empty, the next iteration may derail. Tighten the system prompt or lower the model's temperature |
| Citations in the response don't correspond to prompt  | Citation validator (`src/semrag/evaluation/citation_validator.py`); the LLM is hallucinating paths |
| Prompt is way over the model's real token limit        | `HeuristicTokenEstimator` rounds up; for a 7B local model the 4-chars-per-token ratio overestimates English by ~25% — bump `model_context_window` down or trim the file index |
| Empty retrieval → LLM hallucinates                      | The empty-context branch in `templates.py` should kick in; if the LLM ignores it, the system prompt needs tightening |
| LLM call times out                                      | `llm.request.duration_ms` with the `op=` and `attempt=` tags; bump `Settings.llm_timeout_seconds` |
| LLM endpoint not reachable at startup                 | `_probe()` failed; check `llm probe failed` log line and the `provider=` / `model=` tags |
| Streaming chunks never finish                          | Provider's `event["done"]` is missing; check the parser in `ollama_llm.py` or `_openai_compatible_impl.py` |
| `LLMProviderError` mid-agent-run                       | Network or provider failure after retries; the agent loop does **not** catch this — it propagates to the API error envelope |
| Tool reads the wrong file                              | `ReadFileTool` resolved outside the project root; check the LLM's path argument |
| Conversation history is dropping important turns       | `max_history_tokens` is too low; raise the constructor argument on `PromptBuilder` |
| `Anthropic provider not yet implemented`              | Intentional; use `openai_compatible` with an OpenAI-compat shim in front of Claude for now |
| `MockLLM exhausted` warning                           | Test ran out of scripted responses; queue more with `mock.queue(...)` |

The agent loop emits per-iteration traces at DEBUG
(`agent iteration start`, `agent iteration final answer`,
`agent iteration tool calls`, `agent tool result`). Combined
with `agent.run.iterations` and the per-tool metrics, this
is enough to reconstruct any conversation post-hoc.

---

## 13. Common mistakes

1. **Forgetting to embed the query in `SearchCodebaseTool`.** The
   tool does the embedding internally; callers pass the raw
   query string. Calling `HybridRetriever.retrieve` directly
   from outside the agent loop means the caller has to embed
   first.
2. **Catching `LLMProviderError` inside a custom tool.** Tool
   errors are caught by `ToolRegistry` and returned as error
   strings; the agent loop sees them as `role: "tool"`
   messages. Letting an LLM provider exception escape a tool
   skips that recovery path.
3. **Returning non-strings from a tool's `execute()`.** The
   Protocol signature is `-> str`; the LLM will receive
   whatever you return as `content`. Non-string returns get
   stringified by the JSON encoder, but a structured return
   (dict) may be more useful as a future change — today the
   string is what the LLM sees.
4. **Running tools in parallel inside the agent loop.** The
   loop iterates `for tool_call in result.tool_calls:`
   sequentially. Even if the LLM emits parallel tool calls
   in one iteration, the loop runs them serially. Add a
   `concurrent.futures` pool here only when a real tool is
   the latency bottleneck — none are today.
5. **Putting chat history in the system message.** The
   system message is one-shot at the start of the prompt;
   `conversation_history` is the right place for prior
   turns. Putting history in the system message confuses
   the model and breaks the message-rotation logic
   (history is trimmed; system is not).
6. **Importing a concrete provider class outside
   `reasoning/llm/`.** Nothing outside the `llm/` subpackage
   imports `OllamaLLM`, `OpenAILLM`, `OpenAICompatibleLLM`,
   or `MockLLM` directly. Use the Protocol; let
   `build_llm(settings)` choose the concrete class.
7. **Setting `max_iterations` higher to "fix" a stuck agent.**
   The cap is a safety net, not a knob for fixing prompt
   quality. If the agent is stuck, the prompt or the tool
   list is the problem — raise `max_iterations` to defer
   the failure, not to fix it.
8. **Constructing a `PromptBuilder` per call inside a hot
   loop.** `PromptBuilder` is cheap (one `estimator`
   instance, four ints) but construction is not free; reuse
   across calls in the same request.
9. **Trusting `estimated_total_tokens` as a hard count.** It's
   a heuristic. The 4-chars-per-token ratio overestimates
   English by ~25% and code by more. The number is for
   budgeting, not for showing the user.
10. **Adding a new `QueryIntent` value to the boost table
    without updating the keyword vocabulary.** Intent
    classification and the boost table are coupled; an
    intent with a boost table but no keywords is a silent
    no-op, and an intent with keywords but no boost table
    can never be elevated. Both must be updated.
    (Cross-reference: [`src/semrag/retrieval/README.md`](../retrieval/README.md#6-query-classification).)
11. **Modifying `SYSTEM_PROMPT` to "fix" grounding.** The
    prompt is one control; real grounding is the eval
    harness's job. If a query needs more grounding, the
    right fix is in retrieval (more context) or in the
    citation validator (stricter check), not the prompt.
12. **Streaming from a non-streaming provider.** `MockLLM`
    does not truly stream; it emits one chunk per call. A
    caller that loops on `StreamChunk` to build a
    typewriter UI will see the full response appear in one
    tick. Use a real provider for streaming UX.

---

## 14. Architectural boundaries

What the reasoning package **does not** do:

- **Own HTTP concerns.** No router, no request/response
  models, no auth. API code lives in
  [`src/semrag/api/`](../api/README.md).
- **Own retrieval.** The `HybridRetriever` lives in
  [`src/semrag/retrieval/`](../retrieval/README.md). The
  reasoning layer consumes its output; it does not own
  the ranking, the fusion, or the graph expansion.
- **Own embedding generation.** The
  `SearchCodebaseTool` calls a caller-provided
  `embed_query_fn`; the embedding provider lives in
  [`src/semrag/ingestion/embedding/`](../ingestion/README.md).
- **Own database access directly.** SQL goes through
  `storage/`. `ListFilesTool` uses the storage layer's
  `FileRow` via the SQLAlchemy ORM; it does not construct
  engines or sessions.
- **Own the citation validator.** The validator lives in
  [`src/semrag/evaluation/`](../evaluation/README.md). The
  reasoning layer produces the citations; the eval layer
  checks them.
- **Own the prompt budget as a configuration surface.** The
  defaults are constants in
  `prompting/pipeline.py`; overrides are constructor
  arguments to `PromptBuilder`. `Settings` does not expose
  prompt-budget knobs (yet).
- **Own the conversation across calls.** The agent loop is
  stateless; the service layer persists the
  `ConversationTurn` history between calls.
- **Own token-usage-based rate limiting.** The API layer
  rate-limits per API key; the reasoning layer just
  reports `llm.tokens.*` metrics.
- **Cache LLM responses.** The API layer's query cache
  wraps the entire `QueryService.ask()` call (not just the
  LLM call). The reasoning layer is uncached.

What it **does** own:

- The agent loop and its iteration cap.
- The three real tools (`search_codebase`, `read_file`,
  `list_files`) and the `Tool` / `ToolRegistry` Protocols.
- The `LLMProvider` Protocol and every concrete
  implementation.
- The factory that picks the provider from `Settings`.
- The startup probe that confirms the provider is
  reachable.
- The retry / timeout / streaming logic for every real
  provider.
- The `PromptBuilder` orchestrator and the budget
  allocation order.
- The token estimator and its `TokenEstimator` Protocol.
- The context assembler, citation formatter, and prompt
  templates.
- The conversation-history trimmer.

---

## 15. Related documentation

- [`docs/developer_guide.md`](../../../docs/developer_guide.md) —
  the end-to-end reasoning lifecycle, observability surface
  (per-iteration agent traces, `agent.*` and `llm.*` metrics),
  the debug artifact directory, the prompt-template
  customization cookbook, and the architectural rules that
  produce the boundaries in [§14](#14-architectural-boundaries).
- [`docs/CODEBASE_GUIDE.md`](../../../docs/CODEBASE_GUIDE.md) —
  file-by-file catalog of every directory, including the
  reasoning package's per-file scope.
- [`src/semrag/README.md`](../README.md) — the source-package
  map; the reasoning package's role in the indexing →
  retrieval → reasoning pipeline.
- [`src/semrag/services/README.md`](../services/README.md) —
  how `QueryService.ask()` composes the prompt builder and
  the agent loop; how the API layer wires the
  `LLMProvider` and `ToolRegistry` per request.
- [`src/semrag/retrieval/README.md`](../retrieval/README.md) —
  the upstream consumer of this package's input; the
  `RetrievalResult` shape the prompt builder consumes.
- [`src/semrag/ingestion/README.md`](../ingestion/README.md) —
  the embedding provider that the
  `SearchCodebaseTool`'s `embed_query_fn` ultimately calls.
- [`src/semrag/evaluation/README.md`](../evaluation/README.md) —
  the citation validator that closes the citation
  contract loop.
- [`src/semrag/core/README.md`](../core/README.md) —
  `Settings`, metrics, logging, redaction; the env-var
  schema that drives the factory and the budget defaults.
- [`src/semrag/api/README.md`](../api/README.md) — DI wiring
  of `build_llm` and the `ToolRegistry` into FastAPI
  routes; the query cache that wraps the entire reasoning
  path.
- [`src/semrag/reasoning/agents/agent_loop.py`](./agents/agent_loop.py) —
  the agent loop; the only class services construct for
  the tool-calling flow.
- [`src/semrag/reasoning/prompting/pipeline.py`](./prompting/pipeline.py) —
  the prompt builder; the only class services construct
  for the prompt-assembly flow.
- [`src/semrag/reasoning/llm/factory.py`](./llm/factory.py) —
  the `build_llm` factory and the startup probe.

---

## Skipped (and why)

- **Streaming `AgentLoop.run`.** The current loop calls
  `llm.complete()` (non-streaming). The `LLMProvider.stream`
  method is exposed for direct callers (e.g. a chat UI) but
  the agent loop does not drive a streaming iteration. Add
  when a caller actually needs token-by-token output
  mid-iteration; until then, a streaming agent loop is
  speculative and adds a state-machine for no caller.
- **Real BPE tokenizers (tiktoken, HuggingFace
  tokenizers).** Both fetch vocabulary files from
  out-of-the-sandbox hosts (`openaipublic.blob.core.windows.net`
  for `tiktoken`, HuggingFace for `tokenizers`). The
  `TokenEstimator` Protocol is the seam; a real tokenizer
  is a one-class change once a vocabulary is reachable (or
  vendored).
- **Anthropic provider.** Anthropic's Messages API has a
  different shape than OpenAI's. The factory raises
  `LLMConfigurationError` rather than silently fall back to
  an OpenAI-shaped client (which would produce malformed
  requests). Add when the deployment target requires it; for
  now, an OpenAI-compat shim in front of Claude is the
  supported path.
- **Tool-call parallelism.** The agent loop runs tools in
  the order the LLM returned them, serially. Adding a
  `concurrent.futures` pool here is a real change with a
  real cost (error handling, exception types) and none of
  the three real tools are the latency bottleneck today.
  Add when a real tool is.
- **Conversational persistence inside the agent loop.**
  `AgentLoop` is stateless across calls; the messages list
  is the source of truth. The service layer persists
  `ConversationTurn` history between calls. Persistence
  belongs in the API/DB layer, not the agent loop.
- **Catching `LLMProviderError` in the agent loop.** The
  loop deliberately propagates provider errors; catching
  them would hide them from the API error envelope and
  from the eval harness's correctness checks. The reasoning
  layer's job is to surface failures, not swallow them.
- **A retry strategy for the agent loop itself.** The
  `AgentLoop` does not retry. The LLM provider has its own
  retry loop for network failures; the agent loop
  iterates, but does not retry the same iteration on
  failure. A user-facing retry belongs in the API layer.
- **Per-tool permissions.** Every tool is bound to the
  project_id of the request and the `ReadFileTool`
  enforces path traversal safety. Per-tool permission
  grants (e.g. "this key can call `search_codebase` but
  not `read_file`") are not modeled; the API layer
  enforces project-level authorization, not tool-level.
- **A second context-assembly strategy.** `assemble_context`
  is greedy-by-fused-score. A more elaborate packing
  (knapsack, summarization-before-include) would
  second-guess the retrieval ranking without evidence it
  improves answer quality. Add when the eval harness
  shows greedy is the bottleneck.
- **Deduplication of overlapping retrieved chunks.** Two
  chunks with overlapping text both go into the prompt.
  Deduplication is the LLM's job (it can synthesize
  across redundant context), not the prompt builder's.
- **Prompt-template per-query-type routing.** The current
  `SYSTEM_PROMPT` is one constant; `build_messages`
  branches on whether the context is empty. Per-query-type
  routing (architecture / debug / verification) is the
  retrieval layer's `QueryIntent` job, not the prompt
  layer's; the prompt is a single shared grounding
  contract.

---

## Add when

- **A new LLM provider.** Implement `LLMProvider` in
  `semrag/reasoning/llm/<provider>_llm.py`; add a case in
  `build_llm`; add a test using mocked HTTP. For
  OpenAI-shape providers, subclass `OpenAICompatibleLLM`
  and override `_endpoint()` and `provider_name`.
- **A new tool.** Implement the `Tool` Protocol in
  `semrag/reasoning/agents/tools.py`; register it in
  `QueryService._build_tools()` in
  `src/semrag/services/services.py`. The tool is a new
  constructor argument to `ToolRegistry`.
- **A real tokenizer.** Add a class implementing
  `TokenEstimator` in
  `semrag/reasoning/prompting/token_estimator.py` (the
  heuristic is the default; the real tokenizer is the
  production path once its vocabulary is reachable).
  Inject at `PromptBuilder` construction.
- **Streaming `AgentLoop.run`.** Add when a caller
  (chat UI, SSE endpoint) needs token-by-token output
  mid-iteration. The streaming accumulation logic is
  already on the provider; the agent loop needs a
  streaming iteration that yields chunks to the caller
  between tool calls.
- **A second context-assembly strategy.** Add when the
  eval harness shows greedy-by-fused-score is the
  bottleneck. Replace `assemble_context` at the
  `PromptBuilder` call site; the function is small and
  side-effect-free.
- **A new prompt template variant.** Add to
  `semrag/reasoning/prompting/templates.py` — either a
  new `build_messages` shape or a new `SYSTEM_PROMPT`
  constant. Add an eval that compares answer quality
  on a fixed question set before/after.
- **Per-tool permissions.** Add a `permissions` field to
  the `Tool` Protocol; enforce in `ToolRegistry.execute`.
  Pair with the API layer's per-key authorization.
- **The Anthropic Messages API.** Add a new concrete
  class in `semrag/reasoning/llm/anthropic_llm.py`
  implementing the Messages contract; add a case in
  `build_llm`. The factory's `"anthropic"` branch
  already raises `LLMConfigurationError` with the
  supported path documented.
- **A retry layer for `AgentLoop.run`.** Add when the
  API layer needs a user-facing retry. The retry must
  preserve the conversation state (re-run from
  `messages`, not from the first iteration) and
  surface a clear error to the user on the final
  failure.
