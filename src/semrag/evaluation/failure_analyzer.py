"""
Retrieval failure analyzer.

For each golden query with chunk-level ground truth
(``GoldenQuery.relevant_chunk_ids``), runs retrieval with
``return_trace=True`` and classifies, for every expected chunk that did
not reach the LLM context, WHICH ranking stage dropped it and WHY —
using only the signals the pipeline already records in
``RankingTrace`` / ``ChunkTrace``.

This module is purely analytical. It does not modify retrieval, ranking,
weighting, embedding, chunking, or reranking code, and it does not
duplicate ranking logic or trace data: the per-stage orderings are
reconstructed from the recorded trace columns, and the contribution
columns are re-exposed verbatim.

Attribution model
-----------------
The pipeline records a ``ChunkTrace`` per chunk with one score column per
stage (``rrf_score``, ``post_intent_score``, ``doc_type_prior``,
``graph_expansion_score``, ``reranker_score``, ``final_score``). The
analyzer rebuilds the ordering each stage would have produced from those
columns alone:

- RRF: ``rrf_score`` (base chunks only; expansions record ``None``).
- SOURCE (noisy-source penalty, only when configured): ``rrf_score *
  (1 - penalty)`` for chunks matching ``noisy_source_path_terms``. Not a
  trace column (the pipeline applies it in place before the graph
  merge), so it is reconstructed from the analyzer's config — the same
  values used to build the retriever.
- GRAPH: base chunks keep their source score; expansion chunks get
  ``graph_seed_score * graph_expansion_score`` when the trace recorded the
  producing seed's RRF score, otherwise ``max_base * graph_expansion_score``
  (the pipeline's ``apply_graph_expansion_ranking`` rule).
- INTENT: ``post_intent_score`` where recorded (the pipeline only
  records it for boosted chunks); otherwise the score is unchanged.
- DOC_PRIOR: ``intent_score * (1 + doc_type_prior)`` where a prior was
  applied.
- RERANK: ``final_score`` (the post-rerank fused score recorded at stage
  6 — the exact score that ordered the returned list; chunks outside the
  rerank top-k keep their pre-rerank score).
- DIVERSITY (only when the trace recorded diversity columns): the chunks
  that reached the LLM context (``reached_llm``), scored by ``final_score``.
  A chunk that was in the rerank pool but not in this set was dropped by
  the diversity-aware selection.

A chunk is attributed to the FIRST stage whose top-k no longer contains
it (change-detection attribution): the demotion stage is, by
construction, the stage that dropped the chunk out of the context
window. The reconstruction is deterministic — score desc, ``chunk_id``
as the tiebreak, mirroring ``_stable_top_k``.

Outputs
-------
``write_failure_reports`` persists ``retrieval_failure_analysis.json``
(machine-readable) plus ``retrieval_failure_analysis.md`` (human).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from semrag.evaluation.benchmark_runner import DEFAULT_BENCHMARK_TOP_K
from semrag.evaluation.models import (
    ExpectedChunkAnalysis,
    FailureAggregate,
    FailureCategory,
    GoldenQuery,
    QueryFailureAnalysis,
    RetrievalFailureReport,
    StageContributions,
)
from semrag.reasoning.prompting.pipeline import PromptBuilder
from semrag.retrieval.pipeline import ChunkTrace
from semrag.storage.vector.models import VectorSearchFilter

# Reconstructed stage order, RRF first. ``SOURCE`` exists only when a
# noisy-source penalty is configured; otherwise it is identical to RRF.
# ``DIVERSITY`` exists only when the query's trace recorded a diversity
# decision (any ``ChunkTrace.diversity`` non-None) — legacy traces without
# the diversity stage reconstruct the pre-RI-03 order exactly.
_STAGE_ORDER = ("RRF", "SOURCE", "GRAPH", "INTENT", "DOC_PRIOR", "RERANK", "DIVERSITY")

# The stage whose top-k demotion dropped a chunk -> the failure category
# it is attributed to. SOURCE (noisy penalty) and DOC_PRIOR are both
# "source" adjustments per ``ranking/source``.
_STAGE_TO_CATEGORY = {
    "SOURCE": FailureCategory.SOURCE_PRIOR,
    "GRAPH": FailureCategory.GRAPH_EXPANSION,
    "INTENT": FailureCategory.INTENT,
    "DOC_PRIOR": FailureCategory.SOURCE_PRIOR,
    "RERANK": FailureCategory.RERANKER,
    "DIVERSITY": FailureCategory.DIVERSITY,
}

# Diagnostic-only recommendations — "inspect X", never a code change.
_RECOMMENDATIONS = {
    FailureCategory.NOT_RETRIEVED: (
        "inspect base retrieval recall — the chunk never surfaced through "
        "vector, lexical, or graph expansion; confirm the chunk_id still "
        "exists in the current index"
    ),
    FailureCategory.RANKED_TOO_LOW: (
        "inspect base fusion (vector + lexical + RRF) — the chunk ranked "
        "below the context cut before any later stage intervened"
    ),
    FailureCategory.SOURCE_PRIOR: "inspect source ranking",
    FailureCategory.GRAPH_EXPANSION: "inspect graph expansion",
    FailureCategory.INTENT: "inspect intent ranking",
    FailureCategory.RERANKER: "inspect reranker",
    FailureCategory.DIVERSITY: (
        "inspect diversity-aware selection — check the per-file / "
        "per-category limits and neighbour window in DiversityConfig"
    ),
    FailureCategory.PROMPT_CUTOFF: "inspect prompt context budget",
    FailureCategory.UNKNOWN: "inspect the ranking trace for this query manually",
}


def build_stage_scores(
    trace_chunks: list[ChunkTrace],
    *,
    noisy_penalty: float = 0.0,
    noisy_terms: tuple[str, ...] = ("/log/",),
) -> dict[str, dict[str, float]]:
    """Reconstruct each ranking stage's ordering score from the recorded
    trace columns. Returns ``{stage: {chunk_id: score}}`` for every stage
    in ``_STAGE_ORDER``. See the module docstring for the per-stage
    reconstruction rules."""
    by_id = {c.chunk_id: c for c in trace_chunks}

    rrf: dict[str, float] = {}
    source: dict[str, float] = {}
    for chunk_id, chunk in by_id.items():
        if chunk.rrf_score is None:
            continue
        rrf[chunk_id] = chunk.rrf_score
        score = chunk.rrf_score
        if noisy_penalty > 0.0 and any(term in chunk.relative_path for term in noisy_terms):
            score = score * (1.0 - noisy_penalty)
        source[chunk_id] = score

    graph = dict(source)
    max_base = max(source.values()) if source else 0.0
    for chunk_id, chunk in by_id.items():
        if chunk.rrf_score is None and chunk.graph_expansion_score is not None:
            seed_score = chunk.graph_seed_score
            if seed_score is None:
                seed_score = max_base
            graph[chunk_id] = seed_score * chunk.graph_expansion_score

    intent: dict[str, float] = {}
    doc_prior: dict[str, float] = {}
    rerank: dict[str, float] = {}
    for chunk_id, chunk in by_id.items():
        intent_score = chunk.post_intent_score
        if intent_score is None:
            intent_score = graph.get(chunk_id)
        if intent_score is None:
            continue
        intent[chunk_id] = intent_score

        doc_score = (
            intent_score * (1.0 + chunk.doc_type_prior)
            if chunk.doc_type_prior is not None
            else intent_score
        )
        doc_prior[chunk_id] = doc_score

        rerank[chunk_id] = chunk.final_score if chunk.final_score is not None else doc_score

    stages: dict[str, dict[str, float]] = {
        "RRF": rrf,
        "SOURCE": source,
        "GRAPH": graph,
        "INTENT": intent,
        "DOC_PRIOR": doc_prior,
        "RERANK": rerank,
    }

    # DIVERSITY: the final selection. Reconstructed from the chunks that
    # actually reached the LLM context (the diversity stage's output).
    # A chunk that was in the rerank pool (final_score recorded) but is
    # not in this set was dropped by the diversity selection (or the final
    # selection generally) — change-detection attributes it to DIVERSITY.
    # The stage is only present when the trace recorded diversity columns,
    # so legacy traces reconstruct the exact pre-RI-03 orderings.
    if any(c.diversity is not None for c in trace_chunks):
        stages["DIVERSITY"] = {
            c.chunk_id: c.final_score
            for c in trace_chunks
            if c.reached_llm and c.final_score is not None
        }
    return stages


def order_stages(stage_scores: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """Deterministic descending stage orderings (score desc, ``chunk_id``
    tiebreak), mirroring the pipeline's ``_stable_top_k``."""
    return {stage: _order(scores) for stage, scores in stage_scores.items()}


def classify_failure(
    chunk_id: str,
    entry: ChunkTrace | None,
    stage_orders: dict[str, list[str]],
    top_k: int,
) -> FailureCategory | None:
    """Attribute a failure to the first stage whose top-k dropped the
    chunk. Returns ``None`` when the chunk reached the LLM context (not
    a failure) and ``NOT_RETRIEVED`` when it never entered the trace."""
    if entry is None:
        return FailureCategory.NOT_RETRIEVED
    if entry.reached_llm:
        return None

    # Walk only the stages that were actually present for this query (a
    # legacy trace has no DIVERSITY stage, preserving pre-RI-03 behavior).
    present = [s for s in _STAGE_ORDER if s in stage_orders]
    if chunk_id in stage_orders["RRF"][:top_k]:
        prev_ok = True
        stages = [s for s in present if s != "RRF"]
    elif chunk_id in stage_orders["GRAPH"][:top_k]:
        # Expansion-only chunk: it only exists from the graph merge on.
        prev_ok = True
        stages = [s for s in present if s in ("INTENT", "DOC_PRIOR", "RERANK", "DIVERSITY")]
    else:
        return FailureCategory.RANKED_TOO_LOW

    for stage in stages:
        in_top = chunk_id in stage_orders[stage][:top_k]
        if prev_ok and not in_top:
            return _STAGE_TO_CATEGORY[stage]
        prev_ok = in_top
    return FailureCategory.UNKNOWN


class RetrievalFailureAnalyzer:
    """Runs golden queries through a retriever with the ranking trace on
    and classifies every expected chunk that failed to reach the LLM
    context. Mirrors ``RetrievalBenchmark`` wiring (retriever +
    ``embed_query_fn`` callable) but reads the trace instead of
    computing retrieval-quality metrics."""

    def __init__(
        self,
        retriever,
        embed_query_fn,
        top_k: int = DEFAULT_BENCHMARK_TOP_K,
        noisy_source_path_terms: tuple[str, ...] = ("/log/",),
        noisy_source_penalty: float = 0.0,
        prompt_builder: PromptBuilder | None = None,
    ):
        self._retriever = retriever
        self._embed_query_fn = embed_query_fn
        self._top_k = top_k
        self._noisy_terms = noisy_source_path_terms
        self._noisy_penalty = noisy_source_penalty
        self._prompt_builder = prompt_builder

    def analyze(self, golden_queries: list[GoldenQuery]) -> RetrievalFailureReport:
        per_query = [self._analyze_query(q) for q in golden_queries]
        total_expected = sum(q.expected_chunk_count for q in per_query)
        total_failures = sum(q.failure_count for q in per_query)
        counts = {category.value: 0 for category in FailureCategory}
        for query in per_query:
            for failure in query.failures:
                if failure.failure_category is not None:
                    counts[failure.failure_category.value] += 1
        return RetrievalFailureReport(
            query_count=len(per_query),
            expected_chunk_count=total_expected,
            failure_count=total_failures,
            aggregate=FailureAggregate(total_failures=total_failures, counts=counts),
            per_query=per_query,
        )

    # ------------------------------------------------------------------

    def _analyze_query(self, golden_query: GoldenQuery) -> QueryFailureAnalysis:
        expected = sorted(golden_query.relevant_chunk_ids)
        query_embedding = self._embed_query_fn(golden_query.query_text)
        results, trace = self._retriever.retrieve(
            golden_query.query_text,
            query_embedding,
            project_id=golden_query.project_id,
            top_k=self._top_k,
            filters=VectorSearchFilter(project_id=golden_query.project_id),
            return_trace=True,
        )

        trace_chunks = trace.chunks()
        trace_map = {c.chunk_id: c for c in trace_chunks}
        stage_scores = build_stage_scores(
            trace_chunks,
            noisy_penalty=self._noisy_penalty,
            noisy_terms=self._noisy_terms,
        )
        stage_orders = order_stages(stage_scores)

        actual_top_k = [r.chunk_id for r in results]
        excluded_ids: set[str] = set()
        if self._prompt_builder is not None and results:
            prompt = self._prompt_builder.build(golden_query.query_text, results)
            excluded_ids = set(actual_top_k) - {c.chunk_id for c in prompt.included_chunks}

        failures = [
            self._analyze_expected(
                golden_query.query_id,
                chunk_id,
                trace_map.get(chunk_id),
                stage_scores,
                stage_orders,
                excluded_ids,
            )
            for chunk_id in expected
        ]

        return QueryFailureAnalysis(
            query_id=golden_query.query_id,
            query_text=golden_query.query_text,
            top_k=self._top_k,
            expected_chunk_count=len(expected),
            failure_count=sum(1 for f in failures if f.failure_category is not None),
            actual_top_k=actual_top_k,
            failures=failures,
        )

    def _analyze_expected(
        self,
        query_id: str,
        chunk_id: str,
        entry: ChunkTrace | None,
        stage_scores: dict[str, dict[str, float]],
        stage_orders: dict[str, list[str]],
        excluded_ids: set[str],
    ) -> ExpectedChunkAnalysis:
        if entry is None:
            return ExpectedChunkAnalysis(
                query_id=query_id,
                chunk_id=chunk_id,
                failure_category=FailureCategory.NOT_RETRIEVED,
                primary_cause=(
                    "The chunk never surfaced through vector, lexical, or "
                    "graph expansion during retrieval."
                ),
                recommendation=_RECOMMENDATIONS[FailureCategory.NOT_RETRIEVED],
            )

        category = classify_failure(chunk_id, entry, stage_orders, self._top_k)

        # Retrieved into context but dropped by the prompt budget is still
        # a failure — it was retrieved, then excluded from the prompt.
        if category is None and entry.reached_llm and chunk_id in excluded_ids:
            category = FailureCategory.PROMPT_CUTOFF

        stage_ranks: dict[str, int] = {}
        for stage, order in stage_orders.items():
            position = _position(order, chunk_id)
            if position is not None:
                stage_ranks[stage] = position

        return ExpectedChunkAnalysis(
            query_id=query_id,
            chunk_id=chunk_id,
            relative_path=entry.relative_path,
            chunk_type=entry.chunk_type,
            retrieved=True,
            retrieved_stage=_retrieved_stage(entry),
            final_rank=entry.final_rank,
            reached_llm=entry.reached_llm,
            failure_category=category,
            primary_cause=(
                _primary_cause(chunk_id, entry, category, stage_scores, self._top_k)
                if category is not None
                else ""
            ),
            recommendation=_RECOMMENDATIONS[category] if category is not None else "",
            stage_ranks=stage_ranks,
            contributions=StageContributions(
                vector_score=entry.vector_score,
                vector_rank=entry.vector_rank,
                lexical_score=entry.lexical_score,
                lexical_rank=entry.lexical_rank,
                rrf_score=entry.rrf_score,
                graph_expansion_score=entry.graph_expansion_score,
                intent_boost=entry.intent_boost,
                doc_type_prior=entry.doc_type_prior,
                reranker_score=entry.reranker_score,
                diversity=entry.diversity,
                diversity_reason=entry.diversity_reason,
            ),
        )


def to_json_dict(report: RetrievalFailureReport) -> dict:
    """JSON-safe dict of the report (enums -> values, None -> null)."""
    return _json_safe(report.model_dump(mode="json"))


def format_markdown(report: RetrievalFailureReport) -> str:
    """Human-readable markdown report: the failure summary block plus a
    per-query section for every query with at least one failure."""
    lines: list[str] = ["# Retrieval Failure Analysis", ""]
    lines.append("## Failure Summary")
    lines.append("")
    for category in FailureCategory:
        count = report.aggregate.counts.get(category.value, 0)
        label = category.value
        lines.append(f"{label} {' ' * max(1, 20 - len(label))} {count}")
    lines.append("")
    lines.append(
        f"Total failures: {report.aggregate.total_failures} across "
        f"{report.query_count} queries ({report.expected_chunk_count} expected chunks)."
    )

    for query in report.per_query:
        if query.failure_count == 0:
            continue
        lines.append("")
        lines.append(f"## {query.query_id} — {query.query_text!r} (top-{query.top_k})")
        lines.append("")
        if query.actual_top_k:
            lines.append("Actual top-k: " + ", ".join(query.actual_top_k))
            lines.append("")
        lines.append("| Expected chunk | Retrieved | Final rank | Failure | Primary cause |")
        lines.append("|---|---|---|---|---|")
        for failure in query.failures:
            if failure.failure_category is None:
                continue
            chunk = failure.chunk_id
            if failure.relative_path:
                chunk = f"{chunk} ({failure.relative_path})"
            rank = f"#{failure.final_rank}" if failure.final_rank is not None else "—"
            lines.append(
                f"| `{chunk}` | {'Yes' if failure.retrieved else 'No'} | {rank} | "
                f"{failure.failure_category.value} | {failure.primary_cause} |"
            )
        lines.append("")
        for failure in query.failures:
            if failure.failure_category is None:
                continue
            lines.append(f"### Ranking breakdown — `{failure.chunk_id}`")
            lines.append("")
            lines.append(f"- retrieved via: {failure.retrieved_stage}")
            stage_ranks = " ".join(
                f"{stage}={failure.stage_ranks[stage]}"
                for stage in _STAGE_ORDER
                if stage in failure.stage_ranks
            )
            lines.append(f"- stage ranks: {stage_ranks}")
            contrib = failure.contributions
            lines.append(
                "- vector: "
                f"{_fmt(contrib.vector_score)} (rank {_fmt(contrib.vector_rank, '{:d}')})"
                f" · lexical: {_fmt(contrib.lexical_score)} (rank {_fmt(contrib.lexical_rank, '{:d}')})"
            )
            lines.append(
                f"- RRF: {_fmt(contrib.rrf_score, '{:.4f}')}"
                f" · graph: {_fmt(contrib.graph_expansion_score, '{:.3f}')}"
                f" · intent: {_fmt(contrib.intent_boost, '{:+.2f}')}"
            )
            lines.append(
                f"- source: {_fmt(contrib.doc_type_prior, '{:+.2f}')}"
                f" · reranker: {_fmt(contrib.reranker_score, '{:.3f}')}"
            )
            if contrib.diversity is not None:
                lines.append(
                    f"- diversity: {_fmt(contrib.diversity, '{:.0f}')}"
                    f" ({contrib.diversity_reason or 'selected'})"
                )
            if failure.recommendation:
                lines.append(f"- recommended: {failure.recommendation}")
            lines.append("")

    return "\n".join(lines)


def write_failure_reports(
    report: RetrievalFailureReport,
    json_path: Path | str,
    md_path: Path | str | None = None,
) -> tuple[Path, Path | None]:
    """Persist the machine-readable JSON report and (optionally) the
    markdown report. Returns the paths written."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(to_json_dict(report), indent=2) + "\n")

    written_md: Path | None = None
    if md_path is not None:
        md_path = Path(md_path)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(format_markdown(report))
        written_md = md_path
    return json_path, written_md


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _order(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))


def _position(order: list[str], chunk_id: str) -> int | None:
    try:
        return order.index(chunk_id) + 1
    except ValueError:
        return None


def _retrieved_stage(entry: ChunkTrace) -> str:
    sources = []
    if entry.vector_rank is not None:
        sources.append("vector")
    if entry.lexical_rank is not None:
        sources.append("lexical")
    if entry.graph_expansion_score is not None:
        sources.append("graph_expansion")
    return "+".join(sources) if sources else "unknown"


def _boosted(chunk_id: str, stage_scores: dict[str, dict[str, float]]) -> bool:
    """True when the intent stage changed the chunk's score (a recorded
    ``post_intent_score`` that differs from its graph-stage score)."""
    graph_score = stage_scores["GRAPH"].get(chunk_id)
    intent_score = stage_scores["INTENT"].get(chunk_id)
    return graph_score is not None and intent_score is not None and intent_score != graph_score


def _primary_cause(
    chunk_id: str,
    entry: ChunkTrace,
    category: FailureCategory,
    stage_scores: dict[str, dict[str, float]],
    top_k: int,
) -> str:
    orders = order_stages(stage_scores)

    if category is FailureCategory.RANKED_TOO_LOW:
        rrf_position = _position(orders["RRF"], chunk_id)
        if rrf_position is not None:
            return (
                f"Ranked #{rrf_position} after RRF fusion (context cut at top-{top_k}); "
                "no later stage demoted it further."
            )
        return f"Never ranked inside the reconstructed top-{top_k} at any stage."

    if category is FailureCategory.GRAPH_EXPANSION:
        graph_order = orders["GRAPH"]
        position = _position(graph_order, chunk_id) or top_k
        expansion_above = sum(
            1
            for cid in graph_order[:position]
            if cid != chunk_id and cid not in stage_scores["RRF"]
        )
        return f"{expansion_above} graph-expansion chunk(s) ranked above it after the graph merge."

    if category is FailureCategory.INTENT:
        intent_order = orders["INTENT"]
        position = _position(intent_order, chunk_id) or top_k
        boosted_above = [
            cid
            for cid in intent_order[:position]
            if cid != chunk_id and _boosted(cid, stage_scores)
        ]
        if boosted_above:
            return f"Intent boost promoted {boosted_above[0]} above it."
        return "Ranked below the context cut after the intent stage re-sort."

    if category is FailureCategory.SOURCE_PRIOR:
        if chunk_id in orders["RRF"][:top_k] and chunk_id not in orders["SOURCE"][:top_k]:
            return "Noisy-source penalty reduced its RRF score below the context cut."
        prior = entry.doc_type_prior
        if prior is not None and chunk_id not in orders["DOC_PRIOR"][:top_k]:
            return f"Document-type prior of {prior:+.2f} demoted it below the context cut."
        return "Source adjustments demoted it below the context cut."

    if category is FailureCategory.RERANKER:
        score = entry.reranker_score
        if score is not None:
            return f"Reranker rescored competitors above it (own reranker score {score:.2f})."
        return "Competitors were rescored above it during the rerank stage."

    if category is FailureCategory.DIVERSITY:
        reason = entry.diversity_reason or "final selection"
        return f"Diversity-aware selection excluded it from the final context ({reason})."

    if category is FailureCategory.PROMPT_CUTOFF:
        return (
            f"Retrieved into context at rank #{entry.final_rank} but excluded by the "
            "prompt context budget."
        )

    return "No single stage demotion was identified from the reconstructed orderings; inspect the trace."


def _fmt(value, fmt: str = "{:.3f}") -> str:
    return fmt.format(value) if value is not None else "—"


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value
