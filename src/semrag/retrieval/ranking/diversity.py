"""
Diversity-aware selection (RI-03).

The last ranking adjustment before the final top-k cut: after every
relevance signal has been computed (RRF fusion, noisy-source penalty,
graph expansion, intent boost, source authority, reranker),
``apply_diversity`` selects the final top-k so the context handed to the
LLM is *compositionally diverse* — multiple complementary source files
and document categories rather than many near-identical chunks from a
single file.

Why this stage sits where it does
---------------------------------
The pipeline order is: vector + lexical -> RRF -> noisy-source penalty ->
graph expansion -> intent boost -> source authority -> reranker ->
DIVERSITY -> stable top-k slice. Diversity must run after every ranking
score is final and immediately before the top-k slice:

- Running it any earlier would let a later stage (intent boost, source
  authority, reranker) re-sort and re-populate the top-k with redundant
  chunks, undoing the selection.
- Running it after the top-k slice would leave no room to work — the
  slice has already fixed the context composition.

This module is intentionally independent of graph, intent, and source
authority: it does not read their outputs, only each chunk's own
``relative_path``, ``chunk_type``, line range, and final score. The one
shared helper it uses, ``classify_document``, is the pipeline's canonical
document-classification utility (source_authority.py); reusing it keeps
categories consistent across the two stages without coupling the
selection logic to the authority *weights*.

Selection rule
--------------
A deterministic greedy pass over candidates ordered by ``fused_score``
with the pipeline's ``chunk_id`` tiebreak. The highest-scored candidate
is always selected; each later candidate is selected unless it is
redundant given the chunks already chosen, per three configurable
signals:

- ``max_per_file`` — at most N chunks from one source file.
- ``max_per_category`` — at most N chunks from one document category
  (via ``classify_document``).
- ``graph_expansion_quota`` — graph-relevant candidates may exceed
  ``max_per_category`` by at most N in total per context. Graph-relevant
  means a candidate the pipeline explicitly marked as graph-expanded
  (``signal_scores["graph_expansion"]``, pulled in by ``graph_expansion.py``)
  *or* an ordinary retrieval copy of a node in the active graph-expansion
  source-node set (identity by ``source_node_id``). Such candidates are
  structurally related to the query and often land in an already-saturated
  category (e.g. UVM classes), so without an exemption they would be
  removed purely by the category cap even though they carry unique
  structural context. The exemption is narrow and bounded: only
  ``graph_expansion_quota`` of them may bypass the category cap, at most
  once per distinct ``source_node_id``. Ordinary candidates always obey
  the category cap, and every graph-relevant candidate still obeys
  ``max_per_file``, the neighbour window, and score order.
- ``neighbour_window`` — a chunk whose line range is within N lines of an
  already-selected chunk from the same file is suppressed (adjacent
  windows of the same document are near-duplicates).

Relevance is preserved: candidates are considered in score order, so a
strongly-scored non-redundant candidate is never displaced by a weaker
one, and the top-scored candidate can never be dropped. Diversity only
influences which *later* candidates fill the remaining slots.

Configurability
---------------
Every knob lives in ``DiversityConfig`` — future tuning (loosen a limit,
change the neighbour window, widen the reranker pool) is a config
change, never ranking code. ``pool_multiplier`` only matters when a
reranker is configured: the pipeline asks the reranker for
``top_k * pool_multiplier`` candidates so the selection has a larger
pool to diversify from. When ``enabled`` is false the pipeline skips the
stage entirely and the ranking is byte-for-byte the pre-diversity
ordering.

Explainability
--------------
Every evaluated candidate is annotated: selected candidates get
``signal_scores["diversity"] = 1.0`` with reason ``"selected"``; rejected
candidates get ``0.0`` plus the limit that rejected them (``file_limit``,
``category_limit``, ``neighbour``). The pipeline copies these into the
additive ``ChunkTrace.diversity`` / ``ChunkTrace.diversity_reason``
columns, so the RB-02 failure analyzer can attribute a chunk that was in
the rerank top-k but not in the final context to the DIVERSITY stage.
"""
from __future__ import annotations

from pydantic import BaseModel, field_validator

from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.source_authority import classify_document

# Selection reasons recorded for explainability.
_REASON_SELECTED = "selected"
_REASON_FILE_LIMIT = "file_limit"
_REASON_CATEGORY_LIMIT = "category_limit"
_REASON_NEIGHBOUR = "neighbour"


class DiversityConfig(BaseModel):
    """The single configurable place for diversity-aware selection.

    - ``enabled``: master switch — when false the pipeline skips the
      stage entirely (output identical to the pre-diversity ranking).
    - ``max_per_file``: max chunks from one ``relative_path``.
    - ``max_per_category``: max chunks from one document category
      (via ``classify_document``).
    - ``graph_expansion_quota``: max graph-expanded candidates (marked by
      ``signal_scores["graph_expansion"]``) that may exceed
      ``max_per_category`` per context. 0 disables the exemption.
    - ``neighbour_window``: line-range proximity within the same file
      that treats a chunk as a near-duplicate of an already-selected
      chunk (0 disables neighbour suppression).
    - ``pool_multiplier``: when a reranker is configured, request
      ``top_k * pool_multiplier`` candidates from it so the selection
      has a larger pool to diversify from.
    """

    enabled: bool = True
    max_per_file: int = 3
    max_per_category: int = 6
    graph_expansion_quota: int = 3
    neighbour_window: int = 25
    pool_multiplier: int = 3

    @field_validator("max_per_file", "max_per_category", "pool_multiplier")
    @classmethod
    def _at_least_one(cls, value: int) -> int:
        if value < 1:
            raise ValueError("diversity limits must be >= 1")
        return value

    @field_validator("graph_expansion_quota")
    @classmethod
    def _quota_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("graph_expansion_quota must be >= 0")
        return value

    @field_validator("neighbour_window")
    @classmethod
    def _window_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("neighbour_window must be >= 0")
        return value


DEFAULT_DIVERSITY = DiversityConfig()


def apply_diversity(
    results: list[RetrievalResult],
    *,
    top_k: int,
    config: DiversityConfig = DEFAULT_DIVERSITY,
    graph_source_node_ids: frozenset[str] | None = None,
) -> tuple[list[RetrievalResult], dict[str, str]]:
    """Deterministic diversity-aware selection over fully-ranked results.

    Returns ``(selected, decisions)`` where ``selected`` holds up to
    ``top_k`` candidates in score-desc order (chunk_id tiebreak, matching
    the pipeline's ordering invariant) and ``decisions`` maps every
    *evaluated* candidate's ``chunk_id`` to its reason (``"selected"`` or
    the rejecting limit). Candidates never reached because ``top_k`` was
    already filled have no entry — they were not rejected by diversity,
    just not needed. Each evaluated candidate's ``signal_scores["diversity"]``
    is set to 1.0 (selected) or 0.0 (rejected).

    ``graph_source_node_ids`` is the active graph-expansion source-node
    set: every ``source_node_id`` the pipeline's graph expansion produced
    an expanded candidate for (see ``graph_expansion.py``). It lets the
    selection recognize that an *ordinary* retrieval copy of a
    structurally-related node represents the same graph node as the
    graph-expanded copy — identity is ``source_node_id``, never file or
    symbol names.

    Category-cap exemption: a candidate that is *graph-relevant* — its
    ``signal_scores`` carries ``graph_expansion`` (explicitly marked by
    the pipeline) or its ``source_node_id`` is in
    ``graph_source_node_ids`` (an ordinary copy of a graph-relevant node)
    — may exceed ``max_per_category``, but at most
    ``config.graph_expansion_quota`` times per context and at most once
    per distinct ``source_node_id``. All other candidates always obey the
    category cap. Graph-relevant candidates still obey ``max_per_file``,
    the neighbour window, and score order, and the exemption never
    promotes a candidate that score order alone would not reach — it only
    lets an otherwise-competitive candidate past the category cap. When
    ``graph_source_node_ids`` is ``None`` or empty the behavior is
    byte-for-byte the graph-marked-only exemption.
    """
    if not config.enabled or not results or top_k < 1:
        return results[:top_k], {}

    pool = sorted(results, key=lambda r: (r.fused_score, r.chunk_id), reverse=True)

    selected: list[RetrievalResult] = []
    decisions: dict[str, str] = {}
    counts: dict[str, int] = {}
    graph_quota_used = 0
    quota_node_ids: set[str] = set()

    for result in pool:
        if len(selected) >= top_k:
            break

        file_key = f"file:{result.relative_path}"
        category_key = f"category:{classify_document(result.chunk_type, result.relative_path)}"
        file_count = counts.get(file_key, 0)
        category_count = counts.get(category_key, 0)
        is_graph_expanded = "graph_expansion" in result.signal_scores
        is_graph_relevant = is_graph_expanded or (
            graph_source_node_ids
            and result.source_node_id in graph_source_node_ids
        )

        if file_count >= config.max_per_file:
            result.signal_scores["diversity"] = 0.0
            decisions[result.chunk_id] = _REASON_FILE_LIMIT
            continue
        if category_count >= config.max_per_category:
            # Bounded graph-relevance exemption: candidates that are
            # graph-relevant (graph-marked, or an ordinary copy of a node
            # the pipeline's graph expansion reached) may exceed the
            # category cap, but only ``graph_expansion_quota`` of them in
            # total and at most once per distinct source_node_id.
            quota_eligible = (
                is_graph_relevant
                and graph_quota_used < config.graph_expansion_quota
                and (not result.source_node_id or result.source_node_id not in quota_node_ids)
            )
            if quota_eligible:
                graph_quota_used += 1
                if result.source_node_id:
                    quota_node_ids.add(result.source_node_id)
            else:
                result.signal_scores["diversity"] = 0.0
                decisions[result.chunk_id] = _REASON_CATEGORY_LIMIT
                continue
        if _is_neighbour(result, selected, config.neighbour_window):
            result.signal_scores["diversity"] = 0.0
            decisions[result.chunk_id] = _REASON_NEIGHBOUR
            continue

        selected.append(result)
        result.signal_scores["diversity"] = 1.0
        decisions[result.chunk_id] = _REASON_SELECTED
        counts[file_key] = file_count + 1
        counts[category_key] = category_count + 1

    selected.sort(key=lambda r: (r.fused_score, r.chunk_id), reverse=True)
    return selected, decisions


def _is_neighbour(
    result: RetrievalResult,
    selected: list[RetrievalResult],
    window: int,
) -> bool:
    """True when ``result`` comes from the same file as an already-selected
    chunk and its line range is within ``window`` lines of it. Chunks
    without line metadata are never treated as neighbours."""
    if window <= 0 or result.line_start is None or result.line_end is None:
        return False
    for other in selected:
        if other.relative_path != result.relative_path:
            continue
        if other.line_start is None or other.line_end is None:
            continue
        if not (
            result.line_start > other.line_end + window
            or result.line_end < other.line_start - window
        ):
            return True
    return False