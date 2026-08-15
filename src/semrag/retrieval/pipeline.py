"""
Hybrid retrieval pipeline.

Assembles every piece this part built into the flow Part 1's architecture
document laid out in Section 3 (retrieval plane): vector search + lexical
search run in parallel (conceptually — sequentially here, since both hit
the same Postgres connection; genuine parallelism is a Part 15
performance concern, not a correctness one), fused with RRF, optionally
expanded with graph context, then reranked. Every stage is individually
swappable (`VectorStore`, lexical search, `Reranker` are all behind
interfaces this pipeline depends on, not concrete classes it owns) so a
future stage upgrade (a real cross-encoder once one's reachable, Qdrant
if Part 6's decision ever gets revisited) doesn't require touching this
orchestration.

Query expansion (RI-05) sits between lexical search and RRF (stage 2.5):
it adds a bounded number of additional lexical searches — the original
query plus intent-specific repository terminology — and folds their
results into the same RRF, so a query phrased differently from the
codebase's wording still matches the chunks a DV engineer would
recognize. Deterministic, zero-network, fail-open; see
``query_expansion.py`` for the strategy and the latency/back-compat
constraints it obeys.

Per-stage ranking trace
-----------------------

Every stage records per-chunk scores in a ``RankingTrace`` dict. The
trace is exposed in two ways:

- Returned to the caller via the optional ``ranking_trace`` parameter on
  ``retrieve``. Services / tests capture it for assertions or for
  inclusion in debug bundles.
- Emitted automatically when ``Settings.log_level`` is DEBUG or TRACE,
  as ``<debug_artifacts_dir>/retrieval_trace.json``. The writer is
  derived from the singleton ``Settings`` and is a no-op at INFO+, so
  production has no overhead.

The trace fields are:

- chunk_id, relative_path, chunk_type
- vector_score, lexical_score (raw signals, may be None)
- vector_rank, lexical_rank (1-indexed position in their lists, None if absent)
- rrf_score (Reciprocal Rank Fusion output)
- intent_boost (multiplier applied by intent layer)
- post_intent_score (rrf_score * (1 + intent_boost))
- graph_expansion_score (separate signal contributed by graph expansion)
- reranker_score (rescored by the reranker, if enabled)
- diversity (1.0 selected / 0.0 rejected by the diversity-aware
  selection stage) + diversity_reason (which limit rejected it)
- retrieved_by_original / retrieved_by_expansion (which query the chunk
  came from; RI-05 query expansion) + expansion_queries (the expansion
  formulation(s) that retrieved it)
- final_score (the score used to order the returned list)
- final_rank (1-indexed position in the returned list, None if dropped)
- reached_llm (True if this chunk is in the returned top-k)

The trace makes ranking regressions diagnosable: ``final_rank`` vs the
prior stage's rank pinpoints exactly which stage demoted a good chunk,
and the score columns explain why.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from semrag.retrieval.fusion import reciprocal_rank_fusion
from semrag.retrieval.graph_expansion import (
    DEFAULT_DEPENDENCIES_MAX_DEPTH,
    DEFAULT_DEPENDENCIES_MAX_EXPANSIONS_PER_RESULT,
    DEFAULT_DEPENDENCIES_PER_DEPTH_LIMIT,
    DEFAULT_MAX_DEPTH,
    DEFAULT_MAX_EXPANSIONS_PER_RESULT,
    expand_with_graph_context,
)
from semrag.retrieval.intent import classify_graph_direction, classify_intent
from semrag.retrieval.lexical_search import PostgresLexicalSearch
from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.query_expansion import QueryExpansionConfig, expand_query
from semrag.retrieval.ranking.diversity import (
    DEFAULT_DIVERSITY,
    DiversityConfig,
    apply_diversity,
)
from semrag.retrieval.ranking.graph import apply_graph_expansion_ranking
from semrag.retrieval.ranking.intent import apply_intent_ranking
from semrag.retrieval.ranking.source import apply_source_ranking
from semrag.retrieval.reranking import Reranker
from semrag.retrieval.source_authority import SourceAuthorityConfig
from semrag.retrieval.vector_search_adapter import to_retrieval_results
from semrag.storage.postgres.graph_queries import PostgresGraphQueries
from semrag.storage.vector.models import VectorSearchFilter
from semrag.storage.vector.pgvector_store import PgvectorStore

DEFAULT_VECTOR_TOP_K = 20
DEFAULT_LEXICAL_TOP_K = 20
DEFAULT_FINAL_TOP_K = 10

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ranking trace
# ---------------------------------------------------------------------------

@dataclass
class ChunkTrace:
    """Per-chunk trace record. Stages append their values as they run;
    the final trace is dumped via ``DebugArtifactWriter.write_json``."""
    chunk_id: str
    relative_path: str = ""
    chunk_type: str = ""
    # RI-06 metadata, additive and backward compatible: filled from the
    # retrieval result the first time a chunk is seen (vector or lexical
    # leg), so the trace exposes what kind of source produced each hit.
    language: str = ""
    category: str = ""
    parser_name: str | None = None
    chunker_name: str = ""
    symbol_name: str | None = None
    vector_score: float | None = None
    lexical_score: float | None = None
    vector_rank: int | None = None
    lexical_rank: int | None = None
    rrf_score: float | None = None
    intent_boost: float | None = None
    post_intent_score: float | None = None
    doc_type_prior: float | None = None
    source_authority: float | None = None
    graph_expansion_score: float | None = None
    graph_seed_score: float | None = None
    graph_direction: str | None = None
    reranker_score: float | None = None
    diversity: float | None = None
    diversity_reason: str | None = None
    retrieved_by_original: bool = False
    retrieved_by_expansion: bool = False
    expansion_queries: list[str] = field(default_factory=list)
    final_score: float | None = None
    final_rank: int | None = None
    reached_llm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RankingTrace:
    """Collector for per-chunk trace data across all retrieval stages.
    Lives only for the duration of a single ``retrieve`` call.

    In addition to the per-chunk records, the trace carries query-level
    expansion metadata (RI-05): the expansion formulations that were
    generated (``expansions``), how long generation took
    (``expansion_latency_ms``), whether generation failed open
    (``expansion_failed``), and whether expansion was configured on
    (``expansion_enabled``).
    """

    def __init__(self, query: str) -> None:
        self.query = query
        self.expansions: list[str] = []
        self.expansion_latency_ms: float | None = None
        self.expansion_failed: bool = False
        self.expansion_enabled: bool = False
        # RI-07 fail-open flag: set when the configured reranker raised
        # and the pre-reranker ranking was used instead. Additive.
        self.rerank_failed: bool = False
        self._chunks: dict[str, ChunkTrace] = {}

    def get_or_create(self, chunk_id: str) -> ChunkTrace:
        if chunk_id not in self._chunks:
            self._chunks[chunk_id] = ChunkTrace(chunk_id=chunk_id)
        return self._chunks[chunk_id]

    def chunks(self) -> list[ChunkTrace]:
        return list(self._chunks.values())

    def write_to(self, writer: Any, *, category: str = "retrieval", name: str = "ranking_trace") -> None:
        """Write the trace via the DebugArtifactWriter. No-op when the
        writer isn't active."""
        payload = {
            "query": self.query,
            "expansions": self.expansions,
            "expansion_latency_ms": self.expansion_latency_ms,
            "expansion_failed": self.expansion_failed,
            "expansion_enabled": self.expansion_enabled,
            "rerank_failed": self.rerank_failed,
            "chunks": [c.to_dict() for c in self._chunks.values()],
        }
        writer.write_json(category, name, payload)


TraceSink = Callable[[RankingTrace], None]


class _NullTraceSink:
    """Default trace sink used when the caller doesn't supply one."""
    def __call__(self, trace: RankingTrace) -> None:  # pragma: no cover
        return None


# ---------------------------------------------------------------------------
# Hybrid top-k reservation
# ---------------------------------------------------------------------------

# Number of slots to reserve in the final top-k so a chunk that wins by
# vector score OR lexical score alone still has a seat, even if RRF /
# reranker ranked it lower. Without this, a chunk that was #1 in vector
# search but didn't appear in lexical search (typical for semantically-
# similar but lexically-distant hits) can be pushed out by chunks that
# happen to appear in BOTH lists but at lower absolute rank. 2 slots
# keeps the reservation narrow — more would start cannibalizing the
# reranker's top-k decisions.
HYBRID_RESERVATION_SLOTS = 2

# The graph-expansion score assignment lives in ``ranking/graph.py``;
# graph expansions are now scored seed-relative (``graph_seed_score`` *
# depth weight) rather than with a fixed penalty relative to the top base
# chunk. There is no longer a ``GRAPH_EXPANSION_PENALTY`` constant.


def _stable_top_k(results: list[RetrievalResult], top_k: int | None = None) -> list[RetrievalResult]:
    """Stable sort by ``fused_score`` desc with a deterministic tie-break
    on ``chunk_id`` so re-running the same query produces the same order
    even when many chunks tie at the bottom (common with RRF on long
    ranked lists)."""
    sorted_results = sorted(
        results,
        key=lambda r: (r.fused_score, r.chunk_id),
        reverse=True,
    )
    if top_k is not None:
        return sorted_results[:top_k]
    return sorted_results


# ---------------------------------------------------------------------------
# Ranking diagnostics
# ---------------------------------------------------------------------------


def _log_ranking_summary(query: str, trace: RankingTrace, final: list[RetrievalResult]) -> None:
    """Log a concise ranking summary at DEBUG level so future ranking
    regressions are easy to spot in logs. No-op at INFO+."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    all_chunks = trace.chunks()
    reached_ids = {r.chunk_id for r in final}
    top5 = [c for c in sorted(all_chunks, key=lambda c: c.rrf_score or 0.0, reverse=True)[:5]]
    lines = [
        f"query={query!r}",
        f"total_candidates={len(all_chunks)}",
        f"final_top_k={len(final)}",
    ]

    seen_top5 = 0
    for c in top5:
        reached = c.chunk_id in reached_ids
        tag = "✓" if reached else "✗"
        lines.append(
            f"  top5_rrf={tag} id={c.chunk_id} src={c.relative_path} "
            f"rrf={c.rrf_score or 0:.4f} intent={c.intent_boost or 0:.2f} "
            f"rerank={c.reranker_score or 0:.2f} final={c.final_score or 0:.4f}"
        )
        if reached:
            seen_top5 += 1
    lines.append(f"top5_rrf_in_final={seen_top5}/{len(top5)}")

    # Chunks that reached final but had low RRF — these are the ones
    # that graph expansion, intent boost, reranker, or reservation
    # pulled up (potential regressions to watch).
    low_rrf = [c for c in all_chunks if c.chunk_id in reached_ids and (c.rrf_score or 0.0) > 0.0 and (c.rrf_score or 0.0) < 0.01 and c.final_rank is not None and c.final_rank <= len(final)]
    if low_rrf:
        for c in sorted(low_rrf, key=lambda x: x.final_rank or 999):
            lines.append(
                f"  low_rrf_in_final rank={c.final_rank} id={c.chunk_id} "
                f"rrf={c.rrf_score:.4f} final={c.final_score:.4f}"
            )

    logger.debug("Ranking trace summary\n\t" + "\n\t".join(lines))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class HybridRetriever:
    def __init__(
        self,
        session: Session,
        reranker: Reranker | None = None,
        enable_graph_expansion: bool = True,
        enable_intent_boost: bool = True,
        enable_doc_type_prior: bool = True,
        trace_sink: TraceSink | None = None,
        noisy_source_path_terms: tuple[str, ...] = ("/log/",),
        noisy_source_penalty: float = 0.0,
        source_authority_config: SourceAuthorityConfig | None = None,
        diversity_config: DiversityConfig | None = None,
        query_expansion_config: QueryExpansionConfig | None = None,
        graph_expansion_max_depth: int = DEFAULT_MAX_DEPTH,
        graph_expansion_dependencies_max_depth: int = DEFAULT_DEPENDENCIES_MAX_DEPTH,
        graph_expansion_max_per_result: int = DEFAULT_MAX_EXPANSIONS_PER_RESULT,
        graph_expansion_dependencies_max_per_result: int = DEFAULT_DEPENDENCIES_MAX_EXPANSIONS_PER_RESULT,
        graph_expansion_dependencies_per_depth_limit: int = DEFAULT_DEPENDENCIES_PER_DEPTH_LIMIT,
    ):
        self._session = session
        self._vector_store = PgvectorStore(session)
        self._lexical_search = PostgresLexicalSearch(session)
        self._graph_queries = PostgresGraphQueries(session)
        self._reranker = reranker
        self._enable_graph_expansion = enable_graph_expansion
        self._enable_intent_boost = enable_intent_boost
        self._enable_doc_type_prior = enable_doc_type_prior
        self._noisy_source_path_terms = noisy_source_path_terms
        self._noisy_source_penalty = noisy_source_penalty
        self._source_authority_config = source_authority_config
        self._diversity_config = diversity_config or DEFAULT_DIVERSITY
        self._query_expansion_config = query_expansion_config
        self._graph_expansion_max_depth = graph_expansion_max_depth
        self._graph_expansion_dependencies_max_depth = graph_expansion_dependencies_max_depth
        self._graph_expansion_max_per_result = graph_expansion_max_per_result
        self._graph_expansion_dependencies_max_per_result = graph_expansion_dependencies_max_per_result
        self._graph_expansion_dependencies_per_depth_limit = graph_expansion_dependencies_per_depth_limit
        self._trace_sink: TraceSink = trace_sink or _NullTraceSink()

    def set_trace_sink(self, sink: TraceSink | None) -> None:
        """Replace the trace sink. Useful for tests; production wiring
        passes it in via the constructor."""
        self._trace_sink = sink or _NullTraceSink()

    def retrieve(
        self,
        query_text: str,
        query_embedding: list[float],
        project_id: str,
        top_k: int = DEFAULT_FINAL_TOP_K,
        filters: VectorSearchFilter | None = None,
        return_trace: bool = False,
    ) -> list[RetrievalResult] | tuple[list[RetrievalResult], RankingTrace]:
        from semrag.core.observability import count, timer

        filters = filters or VectorSearchFilter(project_id=project_id)
        trace = RankingTrace(query=query_text)

        # Stage 1: vector search
        with timer("retrieval.vector.duration_ms", project_id=project_id):
            raw_vector = self._vector_store.search(query_embedding, top_k=DEFAULT_VECTOR_TOP_K, filters=filters)
            vector_results = to_retrieval_results(self._session, raw_vector, signal_name="vector")
        for rank, r in enumerate(vector_results, start=1):
            ct = trace.get_or_create(r.chunk_id)
            ct.relative_path = r.relative_path
            ct.chunk_type = r.chunk_type
            ct.language = r.language
            ct.category = r.category
            ct.parser_name = r.parser_name
            ct.chunker_name = r.chunker_name
            ct.symbol_name = r.symbol_name
            ct.vector_score = float(r.fused_score)
            ct.vector_rank = rank
            ct.retrieved_by_original = True

        # Stage 2: lexical search
        with timer("retrieval.lexical.duration_ms", project_id=project_id):
            lexical_results = self._lexical_search.search(query_text, top_k=DEFAULT_LEXICAL_TOP_K, filters=filters)
        for rank, r in enumerate(lexical_results, start=1):
            ct = trace.get_or_create(r.chunk_id)
            ct.relative_path = ct.relative_path or r.relative_path
            ct.chunk_type = ct.chunk_type or r.chunk_type
            ct.language = ct.language or r.language
            ct.category = ct.category or r.category
            ct.parser_name = ct.parser_name or r.parser_name
            ct.chunker_name = ct.chunker_name or r.chunker_name
            ct.symbol_name = ct.symbol_name or r.symbol_name
            ct.lexical_score = float(r.fused_score)
            ct.lexical_rank = rank
            ct.retrieved_by_original = True

        count("retrieval.vector.hits", len(vector_results), project_id=project_id)
        count("retrieval.lexical.hits", len(lexical_results), project_id=project_id)

        # Stage 2.5: query expansion (RI-05). Deterministic, intent-driven
        # lexical expansion: the original query text is ALWAYS searched
        # (above), and each additional formulation is an extra bounded
        # lexical search whose results are fused through the same RRF.
        # See ``query_expansion.py`` for the generation strategy; the
        # expansion never replaces the original query and never touches
        # the vector leg (no extra embeddings = no added network latency).
        expansion_queries: list[str] = []
        expansion_lists: dict[str, list[RetrievalResult]] = {}
        expansion_enabled = (
            self._query_expansion_config is not None
            and self._query_expansion_config.enabled
        )
        if expansion_enabled and self._query_expansion_config is not None:
            intent = classify_intent(query_text)
            _expand_start = time.perf_counter()
            try:
                expansion_queries = expand_query(
                    query_text, intent=intent, config=self._query_expansion_config
                )
            except Exception:  # noqa: BLE001 — fail-open, never break retrieval
                trace.expansion_failed = True
                expansion_queries = []
                logger.exception("query expansion failed; continuing with original query")
            trace.expansion_latency_ms = (time.perf_counter() - _expand_start) * 1000
            for idx, formulation in enumerate(expansion_queries, start=1):
                with timer(f"retrieval.query_expansion.lexical{idx}.duration_ms", project_id=project_id):
                    exp_results = self._lexical_search.search(
                        formulation, top_k=DEFAULT_LEXICAL_TOP_K, filters=filters
                    )
                signal_name = f"lexical_exp{idx}"
                for r in exp_results:
                    r.signal_scores = {signal_name: float(r.fused_score)}
                expansion_lists[signal_name] = exp_results
                for r in exp_results:
                    ct = trace.get_or_create(r.chunk_id)
                    ct.relative_path = ct.relative_path or r.relative_path
                    ct.chunk_type = ct.chunk_type or r.chunk_type
                    ct.retrieved_by_expansion = True
                    if formulation not in ct.expansion_queries:
                        ct.expansion_queries.append(formulation)
            count("retrieval.query_expansion.formulations", len(expansion_queries), project_id=project_id)
            count("retrieval.query_expansion.hits", sum(len(v) for v in expansion_lists.values()), project_id=project_id)
        trace.expansions = expansion_queries
        trace.expansion_enabled = expansion_enabled

        # Stage 3: RRF fusion
        with timer("retrieval.rrf.duration_ms", project_id=project_id):
            fused = reciprocal_rank_fusion(
                {"vector": vector_results, "lexical": lexical_results, **expansion_lists}
            )
        for r in fused:
            ct = trace.get_or_create(r.chunk_id)
            ct.rrf_score = float(r.fused_score)

        # Stage 3.5: noisy-source penalty. Verbose transcript files (sim
        # logs) over-match natural-language queries on both signals, so
        # downweight them before graph expansion / intent / rerank so they
        # can't crowd real source files out of the final top-k. See
        # ``ranking/source.apply_source_ranking``.
        if self._noisy_source_penalty > 0.0:
            with timer("retrieval.noisy_source_penalty.duration_ms", project_id=project_id):
                fused = apply_source_ranking(
                    fused,
                    noisy_source_path_terms=self._noisy_source_path_terms,
                    noisy_source_penalty=self._noisy_source_penalty,
                    enable_doc_type_prior=False,
                )

        # Stage 4: graph expansion
        if self._enable_graph_expansion:
            with timer("retrieval.graph_expansion.duration_ms", project_id=project_id):
                graph_direction = classify_graph_direction(query_text)
                expansions = expand_with_graph_context(
                    self._session, self._graph_queries, fused[:5], project_id,
                    direction=graph_direction,
                    max_depth=self._graph_expansion_max_depth,
                    dependencies_max_depth=self._graph_expansion_dependencies_max_depth,
                    max_expansions_per_result=self._graph_expansion_max_per_result,
                    dependencies_max_expansions_per_result=self._graph_expansion_dependencies_max_per_result,
                    dependencies_per_depth_limit=self._graph_expansion_dependencies_per_depth_limit,
                )
            count("retrieval.graph_expansion.hits", len(expansions), project_id=project_id)
            if expansions:
                # Ranking adjustments — seed-relative scoring, merge, and
                # re-sort — live in
                # ``ranking.graph.apply_graph_expansion_ranking`` (see its
                # module docstring for why no second RRF is run).
                fused = apply_graph_expansion_ranking(fused, expansions)
                for r in fused:
                    ct = trace.get_or_create(r.chunk_id)
                    gexp = r.signal_scores.get("graph_expansion")
                    if gexp is not None:
                        ct.graph_expansion_score = float(gexp)
                        ct.graph_seed_score = r.signal_scores.get("graph_seed_score")
                        ct.graph_direction = r.signal_scores.get("graph_direction")

        # Identity set for the diversity stage: every source_node_id the
        # graph expansion produced an expanded candidate for. Lets
        # ``apply_diversity`` recognize an ordinary retrieval copy of a
        # structurally-related node (same ``source_node_id``) as
        # graph-relevant, without any new traversal or name matching.
        # Empty when graph expansion is disabled or produced nothing.
        expansion_source_node_ids = frozenset(
            e.source_node_id for e in expansions if e.source_node_id
        ) if self._enable_graph_expansion else frozenset()

        # Stage 5: intent boost
        if self._enable_intent_boost:
            with timer("retrieval.intent_boost.duration_ms", project_id=project_id):
                intent = classify_intent(query_text)
                count("retrieval.intent_boost.intent", 1, intent=intent.value)
                fused = apply_intent_ranking(fused, intent)
            for r in fused:
                ct = trace.get_or_create(r.chunk_id)
                boost = r.signal_scores.get("intent_boost")
                if boost is not None:
                    ct.intent_boost = float(boost)
                    ct.post_intent_score = float(r.fused_score)

        # Stage 5.5: source authority. Document-aware nudge based on
        # ``chunk_type`` and ``relative_path`` (already on every result).
        # Runs after intent boost so the two compose multiplicatively, and
        # before the reranker so the reranker's high-resolution signal
        # still gets a clean look at the top candidates. Rule application
        # is orchestrated by ``ranking/source.apply_source_ranking``;
        # classification + weights live in ``source_authority.py``.
        if self._enable_doc_type_prior:
            with timer("retrieval.doc_type_prior.duration_ms", project_id=project_id):
                fused = apply_source_ranking(
                    fused, source_authority_config=self._source_authority_config
                )
            for r in fused:
                ct = trace.get_or_create(r.chunk_id)
                prior = r.signal_scores.get("doc_type_prior")
                if prior is not None:
                    ct.doc_type_prior = float(prior)
                authority = r.signal_scores.get("source_authority")
                if authority is not None:
                    ct.source_authority = float(authority)

        # Stage 6: reranker (or final cut). Fail-open contract (RI-07):
        # a configured reranker that raises must never abort retrieval —
        # fall back to the pre-reranker ranking and record the failure on
        # the trace so the outage is observable.
        ranked: list[RetrievalResult]
        rerank_failed = False
        diversity_enabled = self._diversity_config.enabled
        if self._reranker is not None:
            # When diversity selection is enabled, ask the reranker for a
            # larger pool (top_k * pool_multiplier) so the selection stage
            # has candidates outside the would-be top-k to diversify from.
            # The reranker itself is untouched — this only changes how many
            # candidates its existing ``rerank`` interface is asked for.
            rerank_k = (
                top_k * self._diversity_config.pool_multiplier
                if diversity_enabled
                else top_k
            )
            try:
                with timer("retrieval.rerank.duration_ms", project_id=project_id):
                    reranked = self._reranker.rerank(query_text, fused, top_k=rerank_k)
            except Exception:  # noqa: BLE001 — fail-open, never abort retrieval
                logger.exception("reranker failed; continuing with pre-reranker ranking")
                rerank_failed = True
                reranked = fused
            for r in reranked:
                ct = trace.get_or_create(r.chunk_id)
                # The reranker stashes its own score in a ``rerank_*``
                # signal key (cross-encoder HTTP uses
                # ``rerank_cross_encoder``; lexical overlap uses
                # ``rerank_lexical_overlap``). Pick whichever was set.
                for k, v in r.signal_scores.items():
                    if k.startswith("rerank_"):
                        ct.reranker_score = float(v)
                        break
                ct.final_score = float(r.fused_score)
            ranked = reranked
        else:
            for r in fused:
                ct = trace.get_or_create(r.chunk_id)
                ct.final_score = float(r.fused_score)
            ranked = fused
        trace.rerank_failed = rerank_failed

        # Stage 6.5: diversity-aware selection (RI-03). Runs after every
        # ranking signal is final (RRF, graph, intent, source authority,
        # reranker) and immediately before the stable top-k slice, so the
        # final context is compositionally diverse without touching any
        # upstream relevance signal. The highest-scored chunk is always
        # selected; diversity only influences later selections. See
        # ``ranking/diversity.py`` for the rule rationale.
        if diversity_enabled:
            with timer("retrieval.diversity.duration_ms", project_id=project_id):
                diversity_pool = ranked
                ranked, diversity_decisions = apply_diversity(
                    ranked, top_k=top_k, config=self._diversity_config,
                    graph_source_node_ids=expansion_source_node_ids,
                )
            # Record the selection decision for every evaluated candidate
            # (selected AND rejected), so the trace explains why a chunk
            # that was in the rerank pool did or didn't reach the context.
            for r in diversity_pool:
                ct = trace.get_or_create(r.chunk_id)
                dv = r.signal_scores.get("diversity")
                if dv is not None:
                    ct.diversity = float(dv)
                reason = diversity_decisions.get(r.chunk_id)
                if reason is not None:
                    ct.diversity_reason = reason

        # Stable sort + slice to top_k before hybrid reservation so the
        # ranking is reproducible independent of Postgres input order.
        ranked = _stable_top_k(ranked, top_k=top_k)

        # Hybrid top-k reservation: guarantee that the top-1 from vector
        # search and top-1 from lexical search survive into the returned
        # list. Without this, a chunk that was #1 by vector OR lexical
        # can be demoted by reranker / RRF tie-breaks and drop out of
        # the LLM context. Reservation only kicks in if the candidate
        # isn't already in the top-k, and never duplicates.
        ranked = self._apply_hybrid_reservation(
            ranked,
            vector_results=vector_results,
            lexical_results=lexical_results,
            fused_pool=fused,
        )

        # Mark final ranks + reached_llm, then emit trace.
        for rank, r in enumerate(ranked, start=1):
            ct = trace.get_or_create(r.chunk_id)
            ct.final_rank = rank
            ct.reached_llm = True

        try:
            self._trace_sink(trace)
        except Exception:  # noqa: BLE001 — observability best-effort
            logger.exception("trace sink raised; ignoring")

        _log_ranking_summary(query_text, trace, ranked)

        if return_trace:
            return ranked, trace
        return ranked

    # -----------------------------------------------------------------------

    def _apply_hybrid_reservation(
        self,
        ranked: list[RetrievalResult],
        *,
        vector_results: list[RetrievalResult],
        lexical_results: list[RetrievalResult],
        fused_pool: list[RetrievalResult] | None = None,
        slots: int = HYBRID_RESERVATION_SLOTS,
    ) -> list[RetrievalResult]:
        """Reserve up to ``slots`` slots for the chunks that topped the
        vector and lexical lists, so a chunk that was #1 by either signal
        alone (typical for semantically similar or exact-identifier
        matches that don't appear in both lists) can't be dropped by
        downstream tie-breaking. Insertion preserves the rest of the
        ranking — only the reservation slots get rearranged, and only
        when the reserved chunk isn't already in the top-k.
        """
        if not ranked or not slots:
            return ranked
        top_k = len(ranked)

        reserved_ids: list[str] = []
        for source in (vector_results, lexical_results):
            if not source:
                continue
            top = source[0].chunk_id
            if top not in reserved_ids:
                reserved_ids.append(top)
            if len(reserved_ids) >= slots:
                break

        existing = {r.chunk_id for r in ranked}
        # Pull the (already-ranked) reservation chunk out of the list
        # rather than re-fetching from the original signal — that way
        # it keeps whatever score the reranker/RRF gave it and the
        # final ordering is consistent with the rest of the top-k.
        reserver_lookup = {r.chunk_id: r for r in ranked}
        # Augment lookup with fused_pool and the original signals so a
        # chunk that was demoted below the post-rerank top-k can still
        # be re-inserted from the post-RRF/pre-rerank pool. ``setdefault``
        # keeps the most-recently-scored copy (ranked > fused_pool >
        # signals).
        fused_pool = fused_pool or []
        for r in fused_pool:
            reserver_lookup.setdefault(r.chunk_id, r)
        for src in (vector_results, lexical_results):
            for r in src:
                reserver_lookup.setdefault(r.chunk_id, r)

        to_insert: list[RetrievalResult] = []
        for cid in reserved_ids:
            if cid in existing:
                continue
            reserver = reserver_lookup.get(cid)
            if reserver is None:
                continue
            to_insert.append(reserver)
            if len(ranked) + len(to_insert) >= top_k:
                break

        if not to_insert:
            return ranked

        # Move reservation chunks to the front (in reservation order,
        # not original-rank order — so vector's #1 ranks ahead of
        # lexical's #1 if both were missing). The remainder keeps its
        # existing order minus the inserted ones.
        head = to_insert
        tail_ids = {r.chunk_id for r in to_insert}
        tail = [r for r in ranked if r.chunk_id not in tail_ids]
        merged = head + tail
        # Trim to top_k in case the reservation pushed us over (only
        # possible if the upstream ``ranked`` was already shorter than
        # top_k, which it normally is).
        return merged[:top_k]
