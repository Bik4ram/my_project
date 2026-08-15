"""
Graph traversal over Part 5's `ir_nodes`/`ir_edges` tables, using
PostgreSQL recursive CTEs — no new storage system, no new schema. This
exists specifically to answer the question this part's design doc opens
with: does a dedicated graph database add anything over what's already in
Postgres, for the traversal depths this system's actual data has (RTL
hierarchies are rarely more than 5-10 levels deep)? Measured, not assumed
— see `test_graph_queries.py`'s scale test and the design doc's benchmark
section for the numbers.

Every query here guards against cycles explicitly (`visited_ids` array,
`NOT node_id = ANY(visited_ids)` in the recursive term) because a
dependency graph is not guaranteed acyclic — a `HierarchyInstantiation`
loop is a real (if unusual) RTL bug pattern, and an unguarded recursive
CTE over cyclic data does not terminate.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.orm import Session


@dataclass
class GraphPathResult:
    node_id: str
    depth: int
    path: list[str]


class PostgresGraphQueries:
    def __init__(self, session: Session):
        self._session = session

    def transitive_dependencies(
        self, project_id: str, start_node_id: str, max_depth: int = 10
    ) -> list[GraphPathResult]:
        """Everything `start_node_id` (transitively) depends on, following
        edges in their natural direction (source -> target) — e.g. "every
        module this module instantiates, directly or indirectly."""
        return self._traverse(
            project_id, start_node_id, max_depth, source_col="source_id", target_col="target_id"
        )

    def transitive_dependents(
        self, project_id: str, start_node_id: str, max_depth: int = 10
    ) -> list[GraphPathResult]:
        """Everything that (transitively) depends on `start_node_id` —
        edges followed in reverse (target -> source) — e.g. "every module
        that instantiates this module, directly or indirectly." The two
        methods share one implementation with the edge direction swapped,
        rather than duplicated SQL, so a fix to the cycle-guard logic
        can't accidentally apply to only one direction."""
        return self._traverse(
            project_id, start_node_id, max_depth, source_col="target_id", target_col="source_id"
        )

    def _traverse(
        self,
        project_id: str,
        start_node_id: str,
        max_depth: int,
        source_col: str,
        target_col: str,
    ) -> list[GraphPathResult]:
        query = text(f"""
            WITH RECURSIVE traversal AS (
                SELECT
                    e.{target_col} AS node_id,
                    1 AS depth,
                    ARRAY[e.{source_col}, e.{target_col}]::varchar[] AS path
                FROM ir_edges e
                WHERE e.project_id = :project_id
                  AND e.{source_col} = :start_node_id
                  AND e.{target_col} IS NOT NULL

                UNION ALL

                SELECT
                    e.{target_col} AS node_id,
                    t.depth + 1,
                    t.path || e.{target_col}
                FROM ir_edges e
                JOIN traversal t ON e.{source_col} = t.node_id
                WHERE e.project_id = :project_id
                  AND e.{target_col} IS NOT NULL
                  AND NOT (e.{target_col} = ANY(t.path))  -- cycle guard
                  AND t.depth < :max_depth
            )
            SELECT DISTINCT ON (node_id) node_id, depth, path
            FROM traversal
            ORDER BY node_id, depth ASC
        """)
        rows = self._session.execute(
            query, {"project_id": project_id, "start_node_id": start_node_id, "max_depth": max_depth}
        ).all()
        return [GraphPathResult(node_id=row.node_id, depth=row.depth, path=list(row.path)) for row in rows]

    def shortest_path(
        self, project_id: str, source_id: str, target_id: str, max_depth: int = 15
    ) -> list[str] | None:
        """Shortest path by edge count (unweighted BFS via recursive CTE —
        the first time `target_id` appears in the frontier is guaranteed
        shortest, since depth increases monotonically one level per
        recursive step)."""
        query = text("""
            WITH RECURSIVE traversal AS (
                SELECT
                    e.target_id AS node_id,
                    1 AS depth,
                    ARRAY[e.source_id, e.target_id]::varchar[] AS path
                FROM ir_edges e
                WHERE e.project_id = :project_id
                  AND e.source_id = :source_id
                  AND e.target_id IS NOT NULL

                UNION ALL

                SELECT
                    e.target_id,
                    t.depth + 1,
                    t.path || e.target_id
                FROM ir_edges e
                JOIN traversal t ON e.source_id = t.node_id
                WHERE e.project_id = :project_id
                  AND e.target_id IS NOT NULL
                  AND NOT (e.target_id = ANY(t.path))
                  AND t.depth < :max_depth
                  -- Stop expanding a branch once it has reached the
                  -- target — cuts wasted work without affecting
                  -- correctness, since we only care about the first
                  -- (shortest) arrival.
                  AND NOT (:target_id = ANY(t.path))
            )
            SELECT path FROM traversal
            WHERE node_id = :target_id
            ORDER BY depth ASC
            LIMIT 1
        """)
        row = self._session.execute(
            query,
            {"project_id": project_id, "source_id": source_id, "target_id": target_id, "max_depth": max_depth},
        ).first()
        return list(row.path) if row else None
