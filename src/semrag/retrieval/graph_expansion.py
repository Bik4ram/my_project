"""
Graph-based context expansion.

Pulls in chunks structurally related to the top vector/lexical hits, via
Part 7's `PostgresGraphQueries`. Two traversal directions are supported:

- ``DEPENDENTS`` (reverse: ``transitive_dependents``) — "who uses X".
  This is the historical default and preserves existing behavior.
- ``DEPENDENCIES`` (forward: ``transitive_dependencies``) — "what does X
  use / depend on". Needed for transaction tracing and path questions
  ("trace the read from test through sequence, sequencer, driver and
  interface"), where the answer lives *downstream* of the seed, not
  upstream.
- ``BIDIRECTIONAL`` — both directions, deduplicated, for "connected to" /
  "relationship" phrasing.

Direction is chosen per-query by ``intent.classify_graph_direction`` and
recorded on each expansion as ``signal_scores["graph_direction"]`` —
pipeline metadata only; it never alters graph relationships or scoring
weights.

Bounded deliberately: ``max_depth=1`` for the reverse direction (direct
relationships only, the fast common case) and a separate, still-bounded
``dependencies_max_depth`` (default 4) for forward chains, so a
transaction trace can span the handful of hops a real RTL/UVM hierarchy
needs without enabling unbounded transitive closure. ``max_expansions_per_result``
caps how many extra chunks one base hit can pull in per direction, so
graph expansion augments the top results rather than flooding them with a
large fan-out node's entire neighbor set.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from semrag.retrieval.intent import GraphExpansionDirection
from semrag.retrieval.models import RetrievalResult
from semrag.storage.postgres.graph_queries import GraphPathResult, PostgresGraphQueries
from semrag.storage.vector.pgvector_models import ChunkRow

DEFAULT_MAX_DEPTH = 1
DEFAULT_DEPENDENCIES_MAX_DEPTH = 4
DEFAULT_MAX_EXPANSIONS_PER_RESULT = 3
# Default forward (dependencies) per-seed cap: equal to the maximum the
# other two forward bounds allow (depth 4 x per-depth 4 = 16), so the
# explicit cap never silently truncates a valid multi-hop trace chain
# (e.g. test -> seq -> sequencer -> driver -> iface, where the deep chain
# nodes sit behind fan-out-heavy shallower levels).
DEFAULT_DEPENDENCIES_MAX_EXPANSIONS_PER_RESULT = 16
# Per-depth cap for the forward (dependencies) direction. Prevents a
# shallow-but-fan-out-heavy level from crowding out the deeper chain nodes
# a transaction-trace query actually wants (e.g. sequencer/driver/iface at
# depth 2-4 under a test's dependency closure).
DEFAULT_DEPENDENCIES_PER_DEPTH_LIMIT = 4


def expand_with_graph_context(
    session: Session,
    graph_queries: PostgresGraphQueries,
    base_results: list[RetrievalResult],
    project_id: str,
    direction: GraphExpansionDirection = GraphExpansionDirection.DEPENDENTS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    dependencies_max_depth: int = DEFAULT_DEPENDENCIES_MAX_DEPTH,
    max_expansions_per_result: int = DEFAULT_MAX_EXPANSIONS_PER_RESULT,
    dependencies_max_expansions_per_result: int = DEFAULT_DEPENDENCIES_MAX_EXPANSIONS_PER_RESULT,
    dependencies_per_depth_limit: int = DEFAULT_DEPENDENCIES_PER_DEPTH_LIMIT,
) -> list[RetrievalResult]:
    """Expand ``base_results`` along ``direction``.

    Each seed is walked along the requested direction(s); expanded chunks
    carry ``graph_expansion`` (depth-decayed weight), ``graph_seed_score``
    (the producing seed's RRF score, per the R2D-1 seed-relative scoring
    rule) and ``graph_direction`` metadata.

    ``max_expansions_per_result`` bounds the reverse (dependents) walk per
    seed, preserving historical behavior. ``dependencies_max_expansions_per_result``
    bounds the forward (dependencies) walk separately — it needs room for
    the deeper chain nodes — and ``dependencies_per_depth_limit`` caps how
    many nodes per depth level the forward walk keeps, so shallow fan-out
    can't starve deeper trace nodes.
    """
    existing_chunk_ids = {r.chunk_id for r in base_results}
    expansions: list[RetrievalResult] = []

    for base_result in base_results:
        if base_result.source_node_id is None:
            continue  # nothing to traverse from — this chunk has no IR node

        related: dict[str, tuple[int, str]] = {}
        if direction in (GraphExpansionDirection.DEPENDENTS, GraphExpansionDirection.BIDIRECTIONAL):
            for r in sorted(
                graph_queries.transitive_dependents(
                    project_id, base_result.source_node_id, max_depth=max_depth
                ),
                key=lambda r: (r.depth, r.node_id),
            )[:max_expansions_per_result]:
                if r.node_id not in related or r.depth < related[r.node_id][0]:
                    related[r.node_id] = (r.depth, GraphExpansionDirection.DEPENDENTS.value)
        if direction in (GraphExpansionDirection.DEPENDENCIES, GraphExpansionDirection.BIDIRECTIONAL):
            per_depth_count: dict[int, int] = {}
            selected: list[GraphPathResult] = []
            for r in sorted(
                graph_queries.transitive_dependencies(
                    project_id, base_result.source_node_id, max_depth=dependencies_max_depth
                ),
                key=lambda r: (r.depth, r.node_id),
            ):
                if per_depth_count.get(r.depth, 0) >= dependencies_per_depth_limit:
                    continue
                per_depth_count[r.depth] = per_depth_count.get(r.depth, 0) + 1
                selected.append(r)
                if len(selected) >= dependencies_max_expansions_per_result:
                    break
            for r in selected:
                if r.node_id not in related or r.depth < related[r.node_id][0]:
                    related[r.node_id] = (r.depth, GraphExpansionDirection.DEPENDENCIES.value)

        if not related:
            continue

        ordered = sorted(related.items(), key=lambda kv: (kv[1][0], kv[0]))
        related_node_ids = [nid for nid, _ in ordered]
        rows = session.execute(
            select(
                ChunkRow.chunk_id, ChunkRow.project_id, ChunkRow.relative_path,
                ChunkRow.chunk_type, ChunkRow.text, ChunkRow.source_node_id,
                ChunkRow.line_start, ChunkRow.line_end,
                ChunkRow.language, ChunkRow.category, ChunkRow.parser_name,
                ChunkRow.chunker_name, ChunkRow.symbol_name,
            ).where(ChunkRow.source_node_id.in_(related_node_ids))
        ).all()

        for row in rows:
            if row.chunk_id in existing_chunk_ids:
                continue  # already surfaced by vector/lexical search directly
            existing_chunk_ids.add(row.chunk_id)
            depth, producing_direction = related[row.source_node_id]
            expansions.append(
                RetrievalResult(
                    chunk_id=row.chunk_id,
                    project_id=row.project_id,
                    relative_path=row.relative_path,
                    chunk_type=row.chunk_type,
                    text=row.text,
                    source_node_id=row.source_node_id,
                    line_start=row.line_start,
                    line_end=row.line_end,
                    language=row.language or "",
                    category=row.category or "",
                    parser_name=row.parser_name,
                    chunker_name=row.chunker_name or "",
                    symbol_name=row.symbol_name,
                    # Fixed, depth-decaying score — graph expansions are a
                    # separate signal in RRF fusion (see fusion.py), not
                    # competing on the same scale as vector/lexical scores.
                    # ``graph_seed_score`` records the RRF score of the base
                    # seed that produced this expansion, so the ranking
                    # stage (``ranking/graph.py``) can anchor the expansion
                    # to its own seed's relevance instead of the strongest
                    # base chunk.
                    fused_score=1.0 / (depth + 1),
                    signal_scores={
                        "graph_expansion": 1.0 / (depth + 1),
                        "graph_seed_score": float(base_result.fused_score),
                        "graph_direction": producing_direction,
                    },
                )
            )

    return expansions
