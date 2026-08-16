# `src/semrag/retrieval` — Query-time retrieval

The retrieval plane. Owns every step between a user's question and a
ranked list of chunks ready to hand to the prompting layer. Retrieval
is LLM-free by design: nothing in this package calls an LLM, and the
package's only consumers (`services.QueryService` and
`reasoning.agents.tools.search_codebase`) own what happens to the
returned list. The boundary is intentional — retrieval's job is
*finding*; the LLM's job is *answering*.

This document describes the **hybrid retrieval pipeline** that runs at
query time: vector + lexical + graph-expanded + intent-boosted +
optionally reranked, fused and trimmed down to a top-k that survives
the prompt budget.

This is **not** an API reference, **not** an embedding-guide, and **not**
the place to learn how `Settings` is wired. For those, see
[`docs/developer_guide.md`](../../../docs/developer_guide.md).

---

## 1. Purpose of the package

The retrieval package exists to answer one question:

> Given a user query, a project, and a previously-built index, return
> the *top-k* chunks most likely to help the LLM produce a correct,
> well-cited answer.

Three things make that harder than a single SQL query:

1. **Two different ranking scales.** Vector search returns cosine
   similarity; lexical search returns `ts_rank_cd`. They are not on
   comparable scales, and averaging them lets whichever signal happens
   to produce larger numbers dominate. Retrieval owns the fusion rule
   that sidesteps that.
2. **Source categories matter.** An answer to "what module drives the
   SDA line?" should be ranked from RTL, not from a sim log that
   happens to mention SDA. Retrieval owns the intent-aware boost that
   does this.
3. **Structural context is real evidence.** A module's
   `transitive_dependents` — who instantiates it — is a relevance
   signal the lexical/vector signals cannot recover from text alone.
   Retrieval owns the graph expansion that pulls those chunks in.

Everything else in the package — rerankers, hybrid reservation, noisy
source penalties, ranking traces — exists to make the first three
points land well in practice.

---

## 2. Package structure

```
retrieval/
├── __init__.py               (empty)
├── models.py                 RetrievalResult (the per-chunk result shape)
├── fusion.py                 reciprocal_rank_fusion (RRF combiner)
├── lexical_search.py         PostgresLexicalSearch (Postgres FTS leg)
├── vector_search_adapter.py  to_retrieval_results (VectorSearchResult -> RetrievalResult)
├── graph_expansion.py        expand_with_graph_context (IR-traversal leg)
├── intent.py                 classify_intent, is_structural_query (query-intent detection)
├── query_expansion.py        QueryExpansionConfig, expand_query (RI-05 deterministic expansion)
├── source_authority.py       SourceAuthorityConfig, classify_document (RI-02 source authority)
├── doc_type_prior.py         RI-01 compatibility shim over source_authority
├── reranking.py              Reranker Protocol + LexicalOverlapReranker + HTTPRerankerClient (RI-07)
├── ranking/                  post-fusion ranking stages
│   ├── intent.py             apply_intent_ranking (intent boost table)
│   ├── source.py             apply_source_ranking (noisy-source penalty + source authority)
│   ├── graph.py              apply_graph_expansion_ranking (penalized graph scoring)
│   └── diversity.py          apply_diversity (RI-03 diversity-aware selection)
└── pipeline.py               HybridRetriever, RankingTrace (the orchestrator)
```

`__init__.py` is empty. Callers always import from the submodule path
(`semrag.retrieval.pipeline.HybridRetriever`,
`semrag.retrieval.intent.classify_intent`). Same convention used by
[`src/semrag/storage/README.md`](../storage/README.md).

---

## 3. Module responsibilities

### `models.py`

`RetrievalResult` — the per-chunk result the pipeline produces and the
prompting layer consumes. Fields:

- `chunk_id`, `project_id`, `relative_path`, `chunk_type` — the
  identity tuple (matches `ChunkRow` in `storage/vector/pgvector_models.py`).
- `text` — the chunk body, ready for the prompt.
- `source_node_id` — the Part 4 `IRNode.id` this chunk derives from
  (`None` for chunk categories without an IR node). Used by
  `graph_expansion.py` to pull in structurally related chunks.
- `line_start`, `line_end` — for citation formatting.
- `fused_score` — the current score at the moment the result was
  produced. Mutates as the result passes through pipeline stages.
- `signal_scores` — a per-signal breakdown (`{"vector": 0.82,
  "lexical": 0.41, "graph_expansion": 1.0, "intent_boost": 0.30,
  "rerank_lexical_overlap": 0.27, ...}`). Kept for debugging
  regressions and for the prompting layer to surface provenance.

Deliberately similar to Part 6's `VectorSearchResult` but distinct —
it carries `source_node_id` and `signal_scores`, both of which the
vector result didn't need and the retrieval-stage result does.

### `fusion.py`

`reciprocal_rank_fusion(result_lists, k=60)` — RRF combiner.

Why rank-based and not score-based: vector cosine and `ts_rank_cd` are
on different scales (one bounded in [-1, 1], the other unbounded).
Summing or averaging raw scores lets whichever signal happens to
produce larger numbers dominate regardless of actual relevance. RRF
sidesteps that by using each list's *rank position*:

```
contribution = 1 / (k + rank)    # 1-indexed rank
```

A chunk ranked #1 by vector and #1 by lexical scores highest; a chunk
appearing in only one list still contributes, just less. `k=60` is the
default from the original Cormack et al. (2009) paper. Returns a
`RetrievalResult` list sorted by `fused_score` descending, with each
result's `signal_scores` updated to include every signal that
contributed.

`reciprocal_rank_fusion` is *not* called a second time on graph
expansions — see [§5 End-to-end pipeline](#5-end-to-end-pipeline) for
why.

### `lexical_search.py`

`PostgresLexicalSearch` — Postgres full-text search leg.

Implements lexeme extraction via `plainto_tsquery('english', ...)` and
a `ts_rank_cd`-backed ranker. Two design notes from the module
docstring that matter for understanding the rest of the pipeline:

- **This is Postgres FTS, not literal BM25.** Postgres's `ts_rank` /
  `ts_rank_cd` use a different (older, simpler) term-weighting scheme
  than BM25's saturating term-frequency formula. Same family
  ("lexical/keyword relevance ranking"), different math. The module is
  honest about this; the project name used "BM25" in earlier docs as
  shorthand for "the lexical half of hybrid retrieval" — that's the
  role this module fills.
- **OR semantics, not AND.** Earlier AND-style lexical matching
  produced no hits for 101 of 104 benchmark questions because real
  questions contain enough distinct terms that no single chunk holds
  them all. The current implementation extracts query lexemes, then
  ranks chunks by *how many* distinct query lexemes they contain
  (`sub.matched DESC`) before falling back to `ts_rank_cd` as a
  tie-breaker. This is a cheap, IDF-free stand-in for "covers more of
  the query."

The query returns `RetrievalResult` objects with `fused_score` set to
the ranker output and `signal_scores = {"lexical": <score>}`.

### `vector_search_adapter.py`

`to_retrieval_results(session, vector_results, signal_name="vector")` —
adapter from Part 6's `VectorSearchResult` to this part's
`RetrievalResult`.

Part 6's result doesn't carry `source_node_id` — it wasn't needed for
anything Part 6 itself does. Rather than modify Part 6's
already-shipped model, this adapter does one batch lookup of
`source_node_id` for the returned chunk_ids directly against the
`chunks` table. Same "small adapter at the boundary" pattern used
throughout the codebase for cross-part integration.

The signal name in `signal_scores` is configurable (`"vector"` by
default) so future fan-in from a different vector store can label its
own contributions distinctly.

### `graph_expansion.py`

`expand_with_graph_context(session, graph_queries, base_results,
project_id, max_depth=1, max_expansions_per_result=3)` — graph leg.

For each base result with a `source_node_id`, calls
`PostgresGraphQueries.transitive_dependents` (the same recursive-CTE
traversal used elsewhere; see [`src/semrag/storage/README.md` §3](../storage/README.md#3-responsibilities-of-every-module))
to find chunks structurally related to the top hits. Returns a list
of new `RetrievalResult` objects, each with a depth-decaying score
(`fused_score = 1.0 / (depth + 1)`) and `signal_scores =
{"graph_expansion": 1.0 / (depth + 1)}`.

Two design constraints that matter:

- **`max_depth=1` (default).** Direct relationships only, not deep
  transitive closure. Part 7's benchmark showed shallow queries are
  the fast, common case; deep traversal is reserved for explicit
  graph-traversal tools, not the retrieval path.
- **`max_expansions_per_result=3`.** Caps how many extra chunks one
  base hit can pull in, so graph expansion augments the top results
  rather than flooding them with a high-fan-out node's entire
  dependent set.

Chunks already in `base_results` are skipped — no duplicates.

### `intent.py`

`QueryIntent` enum + `classify_intent(query)` + `is_structural_query(query)`.
The ranking side (the boost table + `apply_intent_ranking`) lives in
`ranking/intent.py`.

Three things, all in one file because they are tightly coupled:

1. **`QueryIntent` classification.** Coarse intent classes
   (`ARCHITECTURE`, `VERIFICATION`, `DEBUG`, `GENERAL`) used to bias
   retrieval toward a category of source material. The classification
   is a keyword-counting heuristic over small fixed vocabularies —
   deliberately *not* an LLM call (each LLM classification doubles
   latency and the LLM is the same one we're trying to make produce
   better answers; a small keyword classifier has the same accuracy
   on this question shape and runs in microseconds, per the module
   docstring).
2. **`is_structural_query(query)`.** True when the user is asking
   about the *shape* of the project ("summarize this repository",
   "describe the architecture", "module hierarchy"). Detected by a
   small list of phrase hints + a few regexes for "summarize",
   "describe", "review", "explain" patterns. These questions need a
   file index in the prompt or the LLM will hallucinate structure
   from log noise — the prompting layer uses this signal to switch
   prompts, not the retrieval layer.
3. **`intent_keywords(intent)`.** Public accessor for the intent's
   domain terminology, reused by `query_expansion.py` (RI-05) so a
   query phrased differently from the repository's wording still
   retrieves relevant context on the lexical leg. GENERAL intents
   have no vocabulary.

The boost table that consumes a classified intent during ranking
lives in `ranking/intent.py` (`apply_intent_ranking`): a small
multiplicative nudge keyed on `(intent, chunk_type)`, capped at
`max_boost=0.5` so a strong genuine match in an unboosted category
still wins; the intent layer only matters near the boundary of
relevance.

Not a hard filter. Every chunk still gets through if it's the best
match. The boost is the smallest possible knob that does the right
thing; tuning the magnitudes is the evaluation harness's job (see
`docs/developer_guide.md`).

### `query_expansion.py`

`QueryExpansionConfig` + `expand_query(query_text, *, intent, config)` +
`build_query_expansion_config(settings)` — deterministic query expansion
(RI-05).

Purpose: a question phrased differently from the repository's wording
(e.g. "why did the test fail?" vs. log text "UVM_ERROR ... mismatch")
retrieves poorly on the **lexical** leg, which matches literal lexemes.
The vector leg already tolerates paraphrase via the embedding; the
expansion fixes the lexical-leg gap by generating up to
`max_expansions` additional search formulations, each = the original
query + a bounded group of the intent's repository terminology (reusing
the `_INTENT_KEYWORDS` vocabulary from `intent.py`). The formulations
are searched by the existing `PostgresLexicalSearch` and fused into the
same RRF as the original vector + lexical lists (new signal names
`lexical_exp1`, `lexical_exp2`, ...), so no fusion or ranking stage
changes.

Properties (all enforced by tests):

- **Original query always searched.** `expand_query` never replaces the
  query; the pipeline searches the original text first and expansions
  are additive signals.
- **Deterministic + zero-network.** A pure function of
  `(query_text, intent, config)`; no LLM, no embedding calls, no clock,
  so no added latency and no new failure modes — the extra cost is a
  bounded set of FTS queries per query.
- **Bounded.** Output ≤ `max_expansions` formulations, each ≤
  `max_expansion_length` characters. The retrieval pool is bounded at
  `(1 + max_expansions) * lexical_top_k + vector_top_k` candidates
  before RRF dedups by `chunk_id`.
- **No trivial expansions.** Terms already present in the query are not
  re-appended; GENERAL intents (no vocabulary), structural queries,
  blank queries, and queries that already cover every domain term all
  return `[]`.
- **Fail-open.** The pipeline wraps expansion in try/except: an
  exception logs, sets `RankingTrace.expansion_failed`, and retrieval
  continues with the original query only.

Trace: `RankingTrace` gains `expansions`, `expansion_latency_ms`,
`expansion_failed`, `expansion_enabled`; each `ChunkTrace` gains
`retrieved_by_original`, `retrieved_by_expansion`, `expansion_queries`.
`QueryService` copies the expansion metadata onto the `QueryTrace`
stage 4 (`query_expansion`) slot.

Config (Settings, all `SEMRAG_`-prefixed): `query_expansion_enabled`
(default `True`), `query_expansion_max_expansions` (3),
`query_expansion_max_expansion_length` (80),
`query_expansion_timeout_ms` (`None`; reserved for a future
external/LLM mechanism, unused by the deterministic implementation).

### `source_authority.py`

`SourceAuthorityConfig` + `classify_document(...)` + `apply_source_authority(...)`
— the document-aware ranking prior (RI-02).

`classify_document` maps a chunk's `(chunk_type, relative_path)` to a
semantic document category (RTL, testbench, assertion, log, docs, …);
`apply_source_authority` nudges each result's `fused_score` by a
per-category authority weight. The rule tables are data, not code:
`SourceAuthorityConfig` (Settings var `SEMRAG_SOURCE_AUTHORITY_OVERRIDES`)
is a serializable dict of category→weight so operators can re-tune
authority without touching the pipeline.

Recorded under **two** `signal_scores` keys — the canonical
`source_authority` and the legacy `doc_type_prior` — so the pipeline
trace and the RB-02 / RB-03 analytics keep working unchanged. Results
are re-sorted by `fused_score` desc after application. Weights are
small multiplicative nudges bounded by `max_boost`
(`SEMRAG_SOURCE_AUTHORITY_MAX_BOOST`, default 0.5); a zero-weight
category leaves a chunk untouched (no signal key).

### `doc_type_prior.py`

RI-01 compatibility shim. `apply_doc_type_prior` and `DEFAULT_MAX_BOOST`
now delegate to `source_authority.apply_source_authority` with the
default authority config (whose weights are the verbatim translation of
the original RI-01 tables). Behavior is unchanged; the module exists so
existing imports keep working.

### `ranking/`

Post-fusion ranking adjustments, orchestrated by `pipeline.py`. The
package split exists because these stages were added as their own
deliverables (RI-02/03) and are individually testable:

- **`ranking/source.py`** — `apply_source_ranking(...)`. Orchestrates
  the two source-based adjustments: the **noisy-source penalty**
  (downweights files matching `noisy_source_path_terms`, default
  `/log/`) and **source authority** (RI-02). The pipeline calls it
  twice: once early with the penalty only, once after intent boost
  with the full authority config.
- **`ranking/graph.py`** — `apply_graph_expansion_ranking(...)`.
  Merges graph-expanded chunks into the fused pool with a penalized,
  depth-decayed score (`max_base_score * GRAPH_EXPANSION_PENALTY *
  depth_weight`) — see [§5](#5-why-graph-expansion-is-not-a-second-rrf).
- **`ranking/intent.py`** — `apply_intent_ranking(results, intent)`.
  The intent-boost table and its application; consumes the intent from
  `intent.py:classify_intent`.
- **`ranking/diversity.py`** — `apply_diversity(results, top_k,
  config)` (RI-03). Compositionally diverse top-k **selection**, not
  filtering: the highest-scored chunk is always kept, and later slots
  prefer chunks from files/categories not yet represented (bounded by
  `max_per_file`, `max_per_category`, `neighbour_window`,
  `pool_multiplier`). Runs after every other ranking signal is final
  and immediately before the stable top-k slice. See
  [§11](#11-ranking-and-reranking).

### `reranking.py`

`Reranker` Protocol + two real implementations + `RerankerError`.

```python
class Reranker(Protocol):
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]: ...
```

Both implementations follow the same rule: **the reranker nudges
within the upstream signal; it does not veto.** The score combination
is a multiplicative boost on `fused_score`, not a replacement. The
upstream RRF/intent score is preserved as the primary signal; the
reranker contributes `fused_score * (1 + weight * reranker_score)`.
A chunk with strong RRF + low reranker score still ranks ahead of a
chunk with weak RRF + high reranker score (e.g. a noisy log file
that mentions the query terms in passing). The full rationale is in
the module docstring.

Implementations:

- **`LexicalOverlapReranker`** — Jaccard similarity between query
  tokens and each candidate's tokens. Real, local, dependency-free
  reranker. Default weight `0.3` (caps the reranker's swing at ~30%
  on a perfect Jaccard hit). Falls back to raw Jaccard if the
  upstream `fused_score` is uniformly zero (test fixtures /
  degenerate fusion). Fully tested.
- **`HTTPRerankerClient`** — real HTTP client against the
  Cohere/TEI-style `POST /rerank` contract
  (`{"query": ..., "documents": [...]}` → `{"results": [...]}`) that
  vLLM and Text-Embeddings-Inference both implement. Real-client
  code with retries; tested against mocked HTTP. The default weight
  is `1.0` (higher than Jaccard's because cross-encoders are more
  accurate, so their contribution can be larger without hijacking the
  ranking).

A real cross-encoder model requires downloadable weights, which is
blocked in this codebase's deployment environment (same restriction as
Part 9's embedding provider). The HTTP client is the production
substitute; `LexicalOverlapReranker` is the local fallback for
sandboxed / offline runs. The Protocol is the seam; both
implementations are real, not placeholders.

### `pipeline.py`

`HybridRetriever` + `RankingTrace` + `ChunkTrace` — the orchestrator.
This is the only class a service-layer caller constructs.

Owns the multi-stage flow (vector + lexical → query expansion →
RRF → noisy-source penalty → graph expansion → intent boost →
source authority → optional rerank → diversity-aware selection →
stable top-k → hybrid reservation), the per-stage `RankingTrace`
collector, the metrics emissions (`retrieval.vector.duration_ms`,
`retrieval.lexical.duration_ms`,
`retrieval.query_expansion.lexicalN.duration_ms`,
`retrieval.rrf.duration_ms`, `retrieval.noisy_source_penalty.duration_ms`,
`retrieval.graph_expansion.duration_ms`,
`retrieval.intent_boost.duration_ms`,
`retrieval.doc_type_prior.duration_ms`, `retrieval.rerank.duration_ms`,
`retrieval.diversity.duration_ms`, plus count metrics),
and the hybrid top-k reservation that protects strong signal-1
chunks from being demoted by downstream tie-breaks.

Construction takes a `Session` (shared with all repositories via DI,
see [`src/semrag/api/README.md`](../api/README.md)), an optional
`Reranker`, and the feature flags / configs for each stage
(`enable_graph_expansion`, `enable_intent_boost`,
`enable_doc_type_prior`, `noisy_source_penalty` +
`noisy_source_path_terms`, `SourceAuthorityConfig`,
`DiversityConfig`, `QueryExpansionConfig`). The default values match
the configuration used in production.

The `RankingTrace` collector is exposed in two ways: returned via the
optional `ranking_trace` parameter on `retrieve`, and emitted
automatically as `<debug_artifacts_dir>/retrieval_trace.json` when
`Settings.log_level` is DEBUG or TRACE (the writer is a no-op at
INFO+; production has no overhead). The trace is the primary
diagnostic surface for ranking regressions — see
[§10 Common debugging paths](#10-common-debugging-paths).

---

## 4. End-to-end pipeline

The full `HybridRetriever.retrieve()` flow, in order. Every stage is
individually swappable; the orchestrator depends on Protocols
(`VectorStore`, `Reranker`) and concrete adapters (`PostgresLexicalSearch`,
`PostgresGraphQueries`), not on engine internals.

```text
query_text + query_embedding + project_id
            │
            ▼
    ┌────────────────────────────────────────┐
    │ 1. Vector search (top_k=20)            │  PgvectorStore.search
    │    signal: "vector"                    │  adapter: to_retrieval_results
    └────────────┬───────────────────────────┘
                 │
    ┌────────────▼───────────────────────────┐
    │ 2. Lexical search (top_k=20)           │  PostgresLexicalSearch.search
    │    signal: "lexical"                   │  (Postgres FTS, OR-semantics)
    └────────────┬───────────────────────────┘
                 │
    ┌────────────▼───────────────────────────┐
    │ 2.5. Query expansion (RI-05)           │  expand_query + extra lexical
    │    additional lexical formulations,    │  searches (signals lexical_expN)
    │    folded into the same RRF            │
    └────────────┬───────────────────────────┘
                 │
    ┌────────────▼───────────────────────────┐
    │ 3. RRF fusion (k=60)                   │  reciprocal_rank_fusion
    │    score = sum 1/(k+rank) per signal   │  (vector + lexical + expansions)
    └────────────┬───────────────────────────┘
                 │
    ┌────────────▼───────────────────────────┐
    │ 3.5. Noisy-source penalty              │  ranking/source.py
    │    downweight files matching           │  default term: "/log/"
    │    noisy_source_path_terms             │
    └────────────┬───────────────────────────┘
                 │
    ┌────────────▼───────────────────────────┐
    │ 4. Graph expansion (max_depth=1,       │  expand_with_graph_context
    │    max_expansions=3)                   │  on top 5 fused results
    │    depth-decayed score, NOT re-fused   │  (penalized against top base
    └────────────┬───────────────────────────┘  score, see §5)
                 │
    ┌────────────▼───────────────────────────┐
    │ 5. Intent boost                        │  classify_intent +
    │    multiplicative, capped at +50%      │  ranking/intent.py
    └────────────┬───────────────────────────┘
                 │
    ┌────────────▼───────────────────────────┐
    │ 5.5. Source authority (RI-02)          │  ranking/source.py +
    │    document-aware nudge by             │  source_authority.py
    │    (chunk_type, relative_path)         │
    └────────────┬───────────────────────────┘
                 │
    ┌────────────▼───────────────────────────┐
    │ 6. Rerank (optional)                   │  LexicalOverlapReranker or
    │    multiplicative, weight-bounded      │  HTTPRerankerClient
    │    NEVER a fused_score replacement     │  (fail-open, RI-07)
    └────────────┬───────────────────────────┘
                 │
    ┌────────────▼───────────────────────────┐
    │ 6.5. Diversity-aware selection (RI-03) │  ranking/diversity.py
    │    keep highest-scored chunk; bound    │  (selection, not filtering)
    │    per-file / per-category repetition  │
    └────────────┬───────────────────────────┘
                 │
    ┌────────────▼───────────────────────────┐
    │ 7. Stable sort (chunk_id tiebreak)     │  _stable_top_k
    │    + top_k slice                       │
    └────────────┬───────────────────────────┘
                 │
    ┌────────────▼───────────────────────────┐
    │ 8. Hybrid reservation                  │  _apply_hybrid_reservation
    │    reserve 2 slots for vector-top-1    │  (top_k=2, not duplicating)
    │    and lexical-top-1                   │
    └────────────┬───────────────────────────┘
                 │
                 ▼
        list[RetrievalResult]   (the top-k)
```

### 4.1 Why these stages, in this order

Each stage has a single, narrow reason to exist:

- **Vector + lexical first** because they are the two independent
  relevance signals the index has. Running them in parallel is
  conceptually what happens; in code they run sequentially against
  the same `Session` (genuine parallelism is a future performance
  concern, not a correctness one — the `Session` is not
  thread-safe).
- **Query expansion between lexical and RRF** because the expansion
  is lexical-only and additive — it produces additional lists that
  participate in the same RRF, no second fusion or ranking stage
  needed. The original query is always searched; expansions never
  replace it.
- **RRF after both signals (and their expansions) are in hand**
  because fusion needs all the lists to score against.
- **Noisy-source penalty after RRF, before graph expansion**, so the
  penalty downweights verbose transcripts that over-match on both
  signals and lets the structural-context pass elevate real source
  files.
- **Graph expansion after RRF and before intent boost / rerank** so
  the boost and reranker can act on the augmented candidate set, not
  on a smaller pool that ignored structural context.
- **Intent boost before source authority** so the two compose
  multiplicatively (both are bounded nudges; neither can dominate
  the RRF signal on its own).
- **Source authority after intent boost and before rerank** so the
  reranker's high-resolution signal still gets a clean look at the
  top candidates.
- **Rerank after the coarse query-side signals** because the boost is
  a coarse, cheap, query-side signal; the reranker is a
  fine-grained, candidate-side signal. The two compose naturally in
  that order.
- **Diversity-aware selection after every ranking signal is final**
  so the final context is compositionally diverse without touching
  any upstream relevance signal.
- **Hybrid reservation last** so a chunk that was #1 by either
  signal alone can't be demoted by reranker / RRF tie-breaks and
  drop out of the LLM context.
- **Stable sort (chunk_id tiebreak) before top-k slice** so re-running
  the same query produces the same order even when many chunks tie
  at the bottom (common with RRF on long ranked lists).

### 4.2 Default sizes

| Constant                    | Value | Location                              | Meaning                                    |
| --------------------------- | ----- | ------------------------------------- | ------------------------------------------ |
| `DEFAULT_VECTOR_TOP_K`      | 20    | `pipeline.py`                         | Candidates pulled from vector search       |
| `DEFAULT_LEXICAL_TOP_K`     | 20    | `pipeline.py`                         | Candidates pulled from lexical search      |
| `DEFAULT_FINAL_TOP_K`       | 10    | `pipeline.py` (passed in by caller)   | Chunks returned to the caller              |
| Graph expansion fan-in      | top 5 | `pipeline.py:320` (`fused[:5]`)       | Base hits considered for graph expansion   |
| `DEFAULT_MAX_DEPTH`         | 1     | `graph_expansion.py`                  | Direct relationships only                  |
| `DEFAULT_MAX_EXPANSIONS_PER_RESULT` | 3 | `graph_expansion.py`            | Cap on extras per base hit                 |
| `HYBRID_RESERVATION_SLOTS`  | 2     | `pipeline.py`                         | Slots reserved for signal-1 chunks         |
| `GRAPH_EXPANSION_PENALTY`   | 0.5   | `pipeline.py`                         | Penalty against top base chunk's RRF score |
| `DEFAULT_LEXICAL_RERANK_WEIGHT` | 0.3 | `reranking.py`                      | Jaccard reranker multiplier                |
| `HTTPRerankerClient.DEFAULT_WEIGHT` | 1.0 | `reranking.py`                  | Cross-encoder reranker multiplier          |
| `DEFAULT_RRF_K`             | 60    | `fusion.py`                           | RRF constant                               |
| `max_boost` (intent)        | 0.5   | `ranking/intent.py`                   | Cap on intent multiplier                   |
| `max_boost` (source authority) | 0.5 | `source_authority.py`              | Cap on authority multiplier                |
| `DEFAULT_SOURCE_AUTHORITY`  | —     | `source_authority.py`                 | Default category→weight table (RI-02)      |
| Diversity limits            | 3 / 6 | `ranking/diversity.py` / Settings   | `max_per_file` / `max_per_category`        |

These constants are not configuration knobs exposed via `Settings`;
they are deliberately named, located, and documented in-place.
Changing them changes the ranking shape — read the comment above each
constant before touching it.

---

## 5. Why graph expansion is not a second RRF

The pipeline runs RRF once, between vector + lexical only. Graph
expansions are added afterwards with a hand-set score
`max_base_score * GRAPH_EXPANSION_PENALTY * depth_weight`. The
comment in `pipeline.py:325-340` is explicit and worth quoting in
full:

> A second RRF would recalculate ALL scores from scratch using only
> rank position, which:
>   (a) loses the carefully-combined vector+lexical signal,
>   (b) compresses the score range (~0.006-0.033 → ~0.010-0.016),
>   (c) lets graph expansion chunks (typically 5-15) compete at the
>       same level as top base chunks,
>   (d) makes intent boost and reranker ineffective on the
>       compressed range.

A graph expansion chunk then has a *penalized* score relative to the
strongest base chunk, decayed by depth. This keeps expansions below
the strongest base chunks while still allowing
structurally-relevant context (e.g. a module that instantiates a
queried component) to appear when intent boost or reranker elevates
them.

---

## 6. Query classification

Two distinct classifications run over the query before retrieval
proceeds. They are independent and answer different questions:

| Function                | Output                                       | Used by                                                |
| ----------------------- | -------------------------------------------- | ------------------------------------------------------ |
| `classify_intent`       | `QueryIntent` ∈ {ARCHITECTURE, VERIFICATION, DEBUG, GENERAL} | `apply_intent_ranking` (`ranking/intent.py`)       |
| `is_structural_query`   | `bool`                                       | Prompting layer (`reasoning/prompting/`) to switch to a file-index prompt |

`classify_intent` is a keyword-counting heuristic over fixed
vocabularies in `intent.py:_INTENT_KEYWORDS`. Ties broken by intent
priority (ARCHITECTURE > VERIFICATION > DEBUG) so an RTL/DV engineer
asking about an RTL-level signal lands on the design-heavy side —
asking about a module is more common than asking about a generic
debug log entry. Zero hits → `GENERAL`, which has an empty boost
table (no effect).

`is_structural_query` matches against a small list of phrase hints
plus regexes for "summarize", "describe", "review", "explain",
"complete/whole/overall/high-level architecture". Word-boundary
matching ensures a question about the "architecture of i2c_master"
still asks about a specific module and is *not* classified as
structural.

`is_structural_query` is computed in this package but consumed by the
prompting layer; this is the only place retrieval produces a value
the prompt uses to make a decision. The boundary is documented so
the next person to add a new structural signal knows where to put
it.

---

## 7. Hybrid search

"Hybrid search" in this codebase means: vector + lexical +
intent + (optional) query expansion + graph + source authority +
(optional) rerank + diversity, combined by the pipeline in
[`pipeline.py`](#3-module-responsibilities). Not two signals with a
configurable third — every signal in the diagram above is on by
default except reranking (constructor parameter) and the noisy-source
penalty (constructor parameter, default off).

The specific fusion rule for the first two signals is RRF
(`fusion.py:reciprocal_rank_fusion`). The fusion rule for the
remaining signals is **hand-set, hand-scored, hand-capped** — see
[§5 Why graph expansion is not a second RRF](#5-why-graph-expansion-is-not-a-second-rrf)
for graph, and the `apply_intent_ranking` / `apply_source_authority` /
`LexicalOverlapReranker` docstrings for the rest. The "hybrid" word
covers the composition *policy*, not a single shared algorithm.

---

## 8. Lexical search

Implemented by `PostgresLexicalSearch` (`lexical_search.py`).

Two design choices that shape its output:

- **Lexeme extraction via `plainto_tsquery('english', ...)`.** Keeps
  Postgres's stemming (consistent with how chunks are indexed in
  `chunks.text_search`). Multi-term queries produce a list of
  stemmed lexemes, then `array_to_string(..., ' | ')` builds an
  OR-tsquery that matches any chunk containing *any* of the
  lexemes.
- **OR semantics, ranked by coverage.** `sub.matched DESC` (count of
  distinct query lexemes a chunk contains), then `ts_rank_cd DESC`
  as a tie-breaker. This is what the rtl_uvm benchmark showed was
  missing from the original AND semantics — real questions have
  enough distinct terms that no single chunk contains all of them.

Score returned: `sub.matched` (an integer cast to float in the
`RetrievalResult` constructor). `signal_scores` is set to
`{"lexical": sub.matched}`.

The filter columns are limited to what `VectorSearchFilter` exposes
(`project_id`, `chunk_type`, `relative_path`) — those are the only
indexed filter columns on `chunks`. Richer filters belong in SQL
directly, not in the lexical search interface.

---

## 9. Vector retrieval

`PgvectorStore.search(query_embedding, top_k, filters)` returns
`VectorSearchResult` ordered by `cosine_distance` ascending, with
`score = 1 - cosine_distance` so callers work in higher-is-better
uniformly. See
[`src/semrag/storage/README.md` §6.3](../storage/README.md#63-search-semantics)
for the storage-side details.

The retrieval package doesn't touch the vector store directly; it
goes through `vector_search_adapter.to_retrieval_results(...)` to
convert the storage shape to `RetrievalResult` and attach
`source_node_id` via one batch lookup against `chunks.source_node_id`.

Top-k default is 20 (`DEFAULT_VECTOR_TOP_K`). The actual final
top-k is the constructor argument to `HybridRetriever.retrieve()`
(default 10, `DEFAULT_FINAL_TOP_K`).

---

## 10. Graph expansion

`expand_with_graph_context(session, graph_queries, base_results,
project_id, max_depth=1, max_expansions_per_result=3)`. Called from
the pipeline on the top 5 fused results.

For each base result with a `source_node_id`, fetches
`transitive_dependents(project_id, node_id, max_depth=1)` — the
backward traversal (who instantiates this module?). The first
`max_expansions_per_result` related node IDs are looked up in
`chunks` for actual chunk rows (the IR has many chunks per node);
results whose `chunk_id` is already in `base_results` are skipped
(no duplicates).

Each new `RetrievalResult` is created with
`fused_score = 1.0 / (depth + 1)` and
`signal_scores = {"graph_expansion": 1.0 / (depth + 1)}`. After the
pipeline combines these with the base fused pool, the final score
is replaced with the penalized top-base score as described in
[§5](#5-why-graph-expansion-is-not-a-second-rrf).

Two important non-features:

- **No expansion over chunks without a `source_node_id`.** The check
  is `if base_result.source_node_id is None: continue` at the top
  of the loop. Chunks in categories without an IR node (e.g. some
  document chunks) silently get no graph boost.
- **No bidirectional traversal.** The pipeline uses
  `transitive_dependents` (backward). `transitive_dependencies`
  exists in `PostgresGraphQueries` but is not wired into the
  retrieval path; it's available to other graph-traversal tools
  (CLI, agent tools) that want to ask "what does this depend on."

---

## 11. Ranking and reranking

The pipeline runs several distinct ranking layers after RRF fusion,
each a bounded multiplicative nudge on `fused_score`:

1. **RRF fusion** (`fusion.py`) — combines vector + lexical (and any
   query-expansion lists) by rank position. The base fused score.
2. **Intent boost** (`ranking/intent.py:apply_intent_ranking`) —
   multiplies `fused_score` by `1 + min(boost, max_boost)` (default
   `max_boost=0.5`) for chunks whose `chunk_type` prefix matches
   the boost table for the classified intent. Applied to the
   RRF-fused pool.
3. **Source authority** (`source_authority.py`, applied via
   `ranking/source.py:apply_source_ranking`) — multiplies
   `fused_score` by `1 + weight * authority` for the chunk's
   classified document category. Recorded under both
   `source_authority` and the legacy `doc_type_prior` keys. Applied
   after intent boost, before the reranker.
4. **Reranker** (optional, `reranking.py`) — the constructor-injected
   `Reranker` instance. Multiplies `fused_score` by
   `1 + weight * reranker_score`. The default `LexicalOverlapReranker`
   weight is 0.3; `HTTPRerankerClient`'s default is 1.0. The reranker's
   score is stored in `signal_scores` under a `rerank_*` key
   (lexical overlap → `rerank_lexical_overlap`, HTTP cross-encoder →
   `rerank_cross_encoder`). Fail-open (RI-07): a reranker that raises
   falls back to the pre-reranker ranking and sets
   `RankingTrace.rerank_failed`.

The pipeline sorts after every stage that can change order, so each
stage's effect is independently observable in the trace.

### Diversity-aware selection (RI-03)

After reranking, the pipeline runs `apply_diversity` (`ranking/diversity.py`)
to pick a compositionally diverse top-k **by selection, not
filtering**. The highest-scored chunk is always selected; each later
slot prefers a chunk from a file / category that is not yet
represented, bounded by `max_per_file` (default 3), `max_per_category`
(6), and a `neighbour_window` (25) over the rerank pool sized at
`pool_multiplier` (2) × final `top_k`. The selection decision is
recorded per candidate (`signal_scores["diversity"]` = 1.0 selected /
0.0 rejected) plus a `diversity_reason`, so the trace explains why a
chunk that was in the rerank pool did or didn't reach the context.
Diversity never alters the relevance scores; it only decides *which*
of the top candidates appear, so it cannot be "gamed" by re-scoring.

### Hybrid top-k reservation

After the stable top-k slice, the pipeline applies
`_apply_hybrid_reservation` to guarantee the top-1 from vector search
and the top-1 from lexical search survive into the returned list.
Without this, a chunk that was #1 by either signal alone can be
demoted by reranker / RRF tie-breaks and drop out of the LLM context.
Reservation only kicks in if the candidate isn't already in the top-k,
and never duplicates. The reserved chunk is moved to the front (in
reservation order, not original-rank order), the rest of the list
keeps its order minus the inserted ones, and the result is trimmed
to `top_k`.

The reservation is bounded at 2 slots (`HYBRID_RESERVATION_SLOTS`)
so it doesn't cannibalize the reranker's top-k decisions. With a
final top-k of 10, the maximum reservation impact is 20% of the
returned list.

### Stable sort

`_stable_top_k` sorts by `fused_score` desc with a deterministic
tie-break on `chunk_id` so re-running the same query produces the
same order even when many chunks tie at the bottom (common with RRF
on long ranked lists). Applied after diversity selection, before
reservation.

---

## 12. Context selection and handoff

The pipeline returns `list[RetrievalResult]` (or `(list, RankingTrace)`
when `return_trace=True`). It does **not** build the prompt, format
citations, or estimate tokens — those concerns are the prompting
layer's, in `src/semrag/reasoning/prompting/`.

The handoff surface is just the list of `RetrievalResult`. Three
fields the prompting layer uses:

- `text` — the chunk body, dropped into the prompt.
- `relative_path`, `line_start`, `line_end` — for citation
  formatting. The LLM is expected to output `[Source: <path>:<line>]`
  references that the citation validator in
  `src/semrag/evaluation/citation_validator.py` checks against this
  set.
- `signal_scores` — the prompting layer may include a "why this
  chunk" footer for debugging; not currently used in production
  prompts.

What the pipeline does **not** do:

- **Token budgeting.** The prompting layer allocates tokens in a
  deliberate order (system + history first, retrieved context last)
  and trims. Retrieval returns the top-k it was asked for, period.
- **Citation formatting.** The prompt builder formats the
  `[Source: …]` envelope. The retrieval result has the raw data; the
  format is the prompt builder's.
- **Deduplication across `search_codebase` calls.** The agent loop
  may call `HybridRetriever` multiple times; deduplication across
  those calls is the agent's responsibility, not retrieval's.

---

## 13. Extension points

The retrieval package is pluggable in five places. Every place is
either a Protocol, a constructor parameter, or a single-function call
site that the new code can replace.

| Need                                          | Touch                                  | What to add                                   |
| --------------------------------------------- | -------------------------------------- | --------------------------------------------- |
| New reranker                                  | `reranking.py`                         | Implement `Reranker` Protocol, inject at construction |
| New vector backend                            | `storage/vector/` (not here)           | Implement `VectorStore` Protocol; pipeline already depends on Protocol |
| New lexical backend                           | `lexical_search.py`                    | Implement equivalent of `PostgresLexicalSearch.search`; swap at the pipeline construction site |
| New graph backend                             | `storage/graph/` (not here)            | Implement `GraphStore` Protocol; wire in `pipeline.py` if `PostgresGraphQueries` is no longer the default |
| New intent or boost table                     | `intent.py` + `ranking/intent.py`      | Add a `QueryIntent` value, a keywords tuple, a boost table entry |
| New fusion strategy                           | `fusion.py` + `pipeline.py`            | Replace `reciprocal_rank_fusion` at the call site; the trace still records per-signal scores |
| New ranking-stage diagnostic                  | `pipeline.py`                          | Add a field to `ChunkTrace`, populate it in the relevant stage |
| New context-handoff format                    | `reasoning/prompting/` (not here)      | Out of scope — retrieval's handoff surface is the `RetrievalResult` list |

Two extension points are deliberately *not* available:

- **Replacing `PgvectorStore` mid-pipeline.** A pipeline instance is
  bound to one `Session` and one `PgvectorStore`. Switching mid-run
  would require a new pipeline instance.
- **Swapping `Reranker` mid-call.** A pipeline instance is bound to
  one reranker. Use `HybridRetriever(reranker=...)` per call.

---

## 14. Common debugging paths

The primary diagnostic surface is the per-chunk `RankingTrace`,
emitted to `<debug_artifacts_dir>/retrieval_trace.json` when
`SEMRAG_LOG_LEVEL=DEBUG` (see `docs/developer_guide.md` for the
`debug_artifacts_dir` location). The trace fields are described in
the `pipeline.py` module docstring; the short version:

- `vector_score`, `lexical_score` (raw signals, may be None)
- `vector_rank`, `lexical_rank` (1-indexed position in their lists)
- `rrf_score` (the fused RRF output)
- `intent_boost` (multiplier applied by intent layer)
- `post_intent_score` (`rrf_score * (1 + intent_boost)`)
- `graph_expansion_score` (separate signal contributed by graph expansion)
- `reranker_score` (rescored by the reranker, if enabled)
- `doc_type_prior` / `source_authority` (multipliers applied by the
  source-authority stage; the same nudge under both keys)
- `diversity` (1.0 selected / 0.0 rejected by diversity-aware
  selection) + `diversity_reason` (which limit rejected it)
- `retrieved_by_original` / `retrieved_by_expansion` +
  `expansion_queries` (which query the chunk came from; RI-05)
- `final_score`, `final_rank`, `reached_llm`

| Symptom                                                    | Look first                                                                                  |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Wrong chunks returned for a known query                    | `retrieval_trace.json`: check `vector_score` and `lexical_score` per chunk — which signal led? |
| Correct signals, wrong final ranking                       | `retrieval_trace.json`: check `intent_boost` and `reranker_score` — did they over-amplify?  |
| Chunk that was #1 by vector OR lexical dropped out         | `_apply_hybrid_reservation` ran; check `reached_llm` — was it reserved?                     |
| Sim log chunks dominating real source                      | `noisy_source_penalty` is off by default; enable it on the pipeline constructor with `noisy_source_penalty=0.5` and `noisy_source_path_terms=("/log/",)` |
| Structural context missing (e.g. "who uses i2c_master?")   | `enable_graph_expansion=True`; check `graph_expansion_score` per chunk                       |
| Query classified as the wrong intent                       | Extend `_INTENT_KEYWORDS` in `intent.py`; the table is the single source of truth for intent vocabulary |
| Reranker is vetoing instead of nudging                     | The Protocol contract is multiplicative, not replacement; check the new reranker's score-combination code |
| HNSW index slow / missing                                  | Storage side — see [`src/semrag/storage/README.md` §12](../storage/README.md#12-common-debugging-paths) |
| `connection refused` on first query                        | `SEMRAG_DATABASE_URL`; `pool_pre_ping` doesn't connect until the first checkout             |
| Two chunks tied at the boundary, ordering flips between runs | `_stable_top_k` is working correctly; check the trace's `final_rank` and confirm the `chunk_id` tiebreak is deterministic |

The trace is also logged at DEBUG via `_log_ranking_summary` — a
short one-line-per-chunk summary in the standard log stream when
`Settings.log_level=DEBUG`. Combine with the JSON artifact for a
regression-friendly diagnostic.

---

## 15. Common mistakes

1. **Mutating a `RetrievalResult` in place.** Every stage that
   changes a result's score uses `result.model_copy(deep=True)` and
   assigns a new `fused_score`. In-place mutation would corrupt the
   trace (the trace records the score at the moment the stage ran,
   not the post-stage score) and would surprise the agent loop if
   it kept a reference to the result across `search_codebase` calls.
2. **Calling `Reranker.rerank` and treating its output as
   independent of the input.** The reranker's output is a
   re-scored, re-ordered version of its input — not a new ranking
   from scratch. Discard the input after rerank; keep the output.
3. **Adding a new `QueryIntent` value without adding keywords AND a
   boost table entry.** Both are coupled; an intent with keywords
   but no boost table is a silent no-op, and an intent with a boost
   table but no keywords can never be classified into.
4. **Replacing `fused_score` in a custom reranker instead of
   multiplying it.** This is what the original `LexicalOverlapReranker`
   used to do, and the module docstring documents why it was wrong
   (repetitive log chunks won over real RTL chunks). Use
   `fused_score * (1 + weight * reranker_score)`.
5. **Importing from `semrag.retrieval` instead of submodule paths.**
   `__init__.py` is empty by design. Import
   `semrag.retrieval.pipeline.HybridRetriever`, not
   `semrag.retrieval.HybridRetriever`.
6. **Constructing a `HybridRetriever` per call inside a hot loop.**
   The retriever holds a `Session`, a `PgvectorStore`, a
   `PostgresLexicalSearch`, a `PostgresGraphQueries`, and an optional
   `Reranker`. Construct once per request lifecycle; reuse across
   calls in the same request. The same `Session` is shared with the
   other repositories in the request, so the connection-pool cost is
   per-session, not per-retriever.
7. **Setting `top_k` larger than the caller's prompt budget.** The
   retriever will return what you ask for; the prompting layer will
   trim what it can't fit. The two layers do not negotiate.
8. **Filtering on JSONB columns in `lexical_search`.** The filter
   interface is `VectorSearchFilter` (project_id, chunk_type,
   relative_path); richer filters belong in raw SQL, not here.
9. **Treating `signal_scores` as a sorted list.** It's a dict;
   insertion order is preserved but it's a key→score mapping, not
   a ranking. Use `fused_score` for ordering; use `signal_scores`
   for provenance.

---

## 16. Architectural boundaries

What the retrieval package **does not** do:

- **Own HTTP concerns.** No router, no request/response models, no
  auth. API code lives in [`src/semrag/api/`](../api/README.md).
- **Own LLM calls.** Retrieval is LLM-free by design. The agent
  loop in `src/semrag/reasoning/agents/` is the only path that
  calls an LLM.
- **Own prompt assembly.** Token budgeting, citation formatting,
  prompt templates — all in
  [`src/semrag/reasoning/prompting/`](../reasoning/README.md).
- **Own database access directly.** SQL goes through
  `storage/postgres/` (`PostgresGraphQueries`) and
  `storage/vector/` (`PgvectorStore`). The retrieval package
  imports from these; it doesn't construct engines or sessions.
- **Own the embedding model.** Embeddings are computed by
  `ingestion/embedding/` providers; the caller passes the
  `query_embedding` into `HybridRetriever.retrieve()`.
- **Cache or rate-limit.** The query cache, rate limiter, and Redis
  client live in `src/semrag/api/`. The retrieval layer is not a
  cache and is not rate-limited.
- **Decide *what* to do with the returned chunks.** That's the
  service layer (`src/semrag/services/services.py:QueryService.ask()`)
  or the agent loop (`src/semrag/reasoning/agents/`).
- **Own the IR builder, parsers, or chunkers.** The IR graph is
  built in `src/semrag/ingestion/ir/`; retrieval reads from it via
  `PostgresGraphQueries` and is otherwise blind to how it was built.

What it **does** own:

- The hybrid retrieval pipeline (`pipeline.py:HybridRetriever`).
- The RRF combiner and its `k` constant (`fusion.py`).
- The lexical-search leg over Postgres FTS (`lexical_search.py`).
- The graph-expansion leg over `PostgresGraphQueries`
  (`graph_expansion.py`).
- The query-intent detection (`intent.py`) and the intent-boost
  application (`ranking/intent.py`).
- The source-authority model and weights (`source_authority.py`) and
  its orchestration with the noisy-source penalty (`ranking/source.py`).
- The diversity-aware selection stage (`ranking/diversity.py`).
- The reranker Protocol and the two real implementations
  (`reranking.py`).
- The `RetrievalResult` model (`models.py`) and the
  `VectorSearchResult → RetrievalResult` adapter
  (`vector_search_adapter.py`).
- The per-stage ranking trace (`pipeline.py:RankingTrace`,
  `ChunkTrace`).
- The hybrid top-k reservation rule
  (`pipeline.py:_apply_hybrid_reservation`) and the noisy-source
  penalty (constructor parameter on `HybridRetriever`).

---

## 17. Related documentation

- [`docs/developer_guide.md`](../../../docs/developer_guide.md) — the
  end-to-end retrieval lifecycle, observability surface, metrics
  (`retrieval.*`), the per-query trace accumulator, the debug
  artifact directory, and the architectural rules that produce the
  boundaries in [§16](#16-architectural-boundaries).
- [`docs/CODEBASE_GUIDE.md`](../../../docs/CODEBASE_GUIDE.md) —
  file-by-file catalog of every directory, including the retrieval
  package's per-file scope.
- [`src/semrag/storage/README.md`](../storage/README.md) — the
  persistence adapters retrieval reads from
  (`PgvectorStore`, `PostgresGraphQueries`, `chunks` table, HNSW
  index, GIN index on `text_search`).
- [`src/semrag/services/README.md`](../services/README.md) — how
  `QueryService.ask()` composes the retrieval pipeline; how the
  query path shares a single read session across all the
  repositories.
- [`src/semrag/reasoning/README.md`](../reasoning/README.md) — how
  the agent loop's `search_codebase` tool routes through
  `HybridRetriever`; how the prompting layer consumes the returned
  `RetrievalResult` list.
- [`src/semrag/evaluation/README.md`](../evaluation/README.md) —
  the eval harness and citation validator; the only consumer of
  retrieval that asserts *quality* rather than correctness.
- [`src/semrag/api/README.md`](../api/README.md) — DI wiring of
  `HybridRetriever` into FastAPI routes; the query cache and rate
  limiter that wrap retrieval in production.
- [`src/semrag/retrieval/pipeline.py`](./pipeline.py) — the
  orchestrator; the only class services construct.
- [`src/semrag/retrieval/reranking.py`](./reranking.py) — the
  `Reranker` Protocol and the two real implementations.
- [`src/semrag/retrieval/intent.py`](./intent.py) — the
  `QueryIntent` enum, the keywords table, the structural-query
  detector, and the intent-vocabulary accessor.
- [`src/semrag/retrieval/ranking/intent.py`](./ranking/intent.py) —
  the intent-boost table and `apply_intent_ranking`.

---

## Skipped (and why)

- **Real cross-encoder reranker weights.** A cross-encoder reranker
  model requires downloadable weights (HuggingFace or similar),
  which is blocked in this codebase's deployment environment (the
  same restriction as Part 9's embedding provider). The HTTP client
  is the production substitute; `LexicalOverlapReranker` is the
  local fallback. Add a real cross-encoder when the deployment
  environment allows weight downloads and the eval harness shows
  Jaccard rerank is measurably insufficient.
- **Literal BM25.** Postgres's `ts_rank_cd` is the lexical
  ranker, not BM25. A real BM25 implementation (or the
  `pg_search`/ParadeDB extension) is a legitimate future upgrade
  if eval shows Postgres's native ranking falling short — not
  assumed necessary without that evidence. The lexical module's
  docstring is explicit about this.
- **Parallel vector + lexical execution.** The pipeline runs both
  signals sequentially against the same `Session`. Genuine
  parallelism would require either a second `Session` (separate
  connection-pool cost) or async orchestration. Add when profiling
  shows the sequential pair is a latency bottleneck, not before.
- **`GraphStore` Protocol in `graph_expansion.py`.** The
  expansion module imports `PostgresGraphQueries` directly today.
  Wiring it through the `GraphStore` Protocol would add an
  indirection with no production caller on the other side; it
  would be a placeholder, not an extension point. When a
  deployment actually switches to Neo4j, switch the import to a
  Protocol-resolved injection.
- **Per-row similarity threshold at the vector store.** The
  `HybridRetriever` applies `top_k` and the reranker, not a
  per-chunk score threshold. The vector store returns the
  requested top-k; filtering by threshold is the caller's
  responsibility (and currently nobody needs it).
- **Sparse-vector search.** pgvector is dense-only. If sparse
  vectors (SPLADE, BM25 alone) become a first-class signal at
  scale, that's a new `VectorStore` implementation, not a
  pgvector feature.
- **In-process LLM-based query classification.** `classify_intent`
  is a keyword-counting heuristic by design — see
  [`intent.py` module docstring](./intent.py). Each LLM
  classification call doubles latency for a small keyword-classifier
  accuracy gain on this question shape.
- **Dedup across `search_codebase` calls in the agent loop.**
  Retrieval returns the top-k it was asked for; the agent loop
  owns cross-call deduplication. The boundary keeps retrieval
  LLM-free and stateless across calls.

---

## Add when

- **Real cross-encoder reranker.** Add when the deployment
  environment allows weight downloads and the eval harness shows
  Jaccard rerank measurably insufficient. Implement a third
  `Reranker` in `reranking.py`; no other change required.
- **A real BM25 leg.** Add when the eval harness shows
  `PostgresLexicalSearch`'s `ts_rank_cd` is the bottleneck. Either
  swap the implementation (keep the interface) or add a second
  signal; the pipeline's RRF is signal-agnostic.
- **`GraphStore` Protocol in `graph_expansion.py`.** Add when a
  production deployment runs Neo4j. Switch the import to a
  Protocol-resolved injection; the call site is a one-line change.
- **Parallel vector + lexical execution.** Add when profiling on
  realistic production query volume shows the sequential pair is
  the latency bottleneck. The RRF stage is independent of which
  signals are parallel; nothing downstream needs to change.
- **A new `QueryIntent`.** Add a value to the enum, a keywords
  tuple in `_INTENT_KEYWORDS`, and a boost table entry in
  `_DEFAULT_BOOST_TABLE`. The two tables are coupled; both must
  be updated.
- **A new ranking-stage diagnostic.** Add a field to `ChunkTrace`,
  populate it in the relevant stage, and update the `write_to`
  payload in `RankingTrace.write_to`. The trace format is the
  regression-detection surface for ranking quality; new stages
  must extend it.
- **A new reranker.** Implement the `Reranker` Protocol, add a
  factory function (probably a new module under
  `src/semrag/retrieval/`), and inject it at `HybridRetriever`
  construction.
