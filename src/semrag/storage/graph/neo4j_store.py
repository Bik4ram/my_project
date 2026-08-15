"""
Neo4j-backed GraphStore.

**Status: written against the real `neo4j` Python driver (v5+ API,
installed and import-verified in this delivery) but NOT run against a
live Neo4j server** — this sandbox's network allowlist blocks
neo4j.com/dist.neo4j.org (verified directly: `curl` to neo4j.com returns
"Host not in allowlist"), and there is no apt package or GitHub-release
binary the way there was for Postgres/pgvector/Qdrant. This is stated
here as plainly as it was stated to the user when it was discovered — see
this part's design doc for the full explanation. Everything in this
module is real, standard Cypher and real driver calls; what's missing is
the live-execution verification every other storage module in this
series has had. Treat this as reviewed-but-unverified, and run
`tests/unit/graph/test_neo4j_graph_store.py` (currently skipped) against
a real instance before relying on it.

Design mirrors `PostgresGraphQueries` deliberately — same three methods,
same semantics — specifically so the two are a fair comparison (see
`interfaces.py`'s docstring) rather than differently-scoped
implementations that can't be measured against each other.

Cypher note: relationship variable-length hop counts (`*1..N`) cannot be
bound Cypher *parameters* — they must be literal integers in the query
text (a real, standard Cypher limitation, not an oversight here).
`max_depth` is validated as a positive int before being interpolated into
the query string, which is safe specifically because it's a
type-checked integer, never a raw string — this is not the same class of
risk as string-interpolating untrusted input.
"""
from __future__ import annotations

from neo4j import GraphDatabase

from semrag.ingestion.ir.models import IRGraph
from semrag.storage.graph.interfaces import GraphPathResult


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def ensure_constraints(self) -> None:
        """Idempotent — safe to call on every startup, matching Alembic's
        `upgrade head` being safe to run repeatedly. Neo4j has no
        migration-history table equivalent to Alembic's `alembic_version`
        for constraint/index management; `IF NOT EXISTS` is the
        idempotency mechanism here instead."""
        with self._driver.session() as session:
            session.run(
                "CREATE CONSTRAINT ir_node_unique IF NOT EXISTS "
                "FOR (n:IRNode) REQUIRE (n.project_id, n.id) IS UNIQUE"
            )

    def sync_graph(self, graph: IRGraph) -> None:
        """Upsert-and-prune, same philosophy as
        `IRGraphRepository.save()` (Part 5) and `PgvectorStore.upsert_chunks`
        (Part 6): after this call, Neo4j's state for this project matches
        `graph` exactly — nodes/relationships no longer present are
        deleted, not left stale. Uses `MERGE` (Neo4j's upsert primitive)
        for nodes and relationships, then a bounded `DETACH DELETE` pass
        for anything belonging to this project that wasn't touched by
        this sync.
        """
        with self._driver.session() as session:
            session.execute_write(self._sync_nodes_tx, graph)
            session.execute_write(self._sync_edges_tx, graph)
            session.execute_write(self._prune_stale_tx, graph)

    @staticmethod
    def _sync_nodes_tx(tx, graph: IRGraph) -> None:
        tx.run(
            """
            UNWIND $nodes AS node
            MERGE (n:IRNode {project_id: $project_id, id: node.id})
            SET n.kind = node.kind,
                n.relative_path = node.relative_path,
                n.name = node.name
            """,
            project_id=graph.project_id,
            nodes=[
                {"id": n.id, "kind": n.kind.value, "relative_path": n.relative_path, "name": n.name}
                for n in graph.nodes.values()
            ],
        )

    @staticmethod
    def _sync_edges_tx(tx, graph: IRGraph) -> None:
        # Every IREdgeKind maps to one generic :DEPENDS_ON relationship
        # type with a `kind` property, rather than 5 distinct relationship
        # types — matching PostgresGraphQueries treating all ir_edges
        # uniformly for traversal, so the two implementations answer the
        # same question the same way. Only resolved edges (real target)
        # become relationships; an unresolved edge has nothing to connect
        # to, same as Postgres's `target_id IS NOT NULL` filter.
        resolved = [e for e in graph.edges if e.resolved and e.target_id]
        tx.run(
            """
            UNWIND $edges AS edge
            MATCH (source:IRNode {project_id: $project_id, id: edge.source_id})
            MATCH (target:IRNode {project_id: $project_id, id: edge.target_id})
            MERGE (source)-[r:DEPENDS_ON {edge_id: edge.id}]->(target)
            SET r.kind = edge.kind
            """,
            project_id=graph.project_id,
            edges=[
                {"id": e.id, "source_id": e.source_id, "target_id": e.target_id, "kind": e.kind.value}
                for e in resolved
            ],
        )

    @staticmethod
    def _prune_stale_tx(tx, graph: IRGraph) -> None:
        current_node_ids = list(graph.nodes.keys())
        current_edge_ids = [e.id for e in graph.edges if e.resolved and e.target_id]
        tx.run(
            """
            MATCH ()-[r:DEPENDS_ON]->()
            WHERE NOT r.edge_id IN $current_edge_ids
              AND (startNode(r).project_id = $project_id OR endNode(r).project_id = $project_id)
            DELETE r
            """,
            project_id=graph.project_id,
            current_edge_ids=current_edge_ids,
        )
        tx.run(
            """
            MATCH (n:IRNode {project_id: $project_id})
            WHERE NOT n.id IN $current_node_ids
            DETACH DELETE n
            """,
            project_id=graph.project_id,
            current_node_ids=current_node_ids,
        )

    def transitive_dependencies(
        self, project_id: str, start_node_id: str, max_depth: int = 10
    ) -> list[GraphPathResult]:
        return self._traverse(project_id, start_node_id, max_depth, reverse=False)

    def transitive_dependents(
        self, project_id: str, start_node_id: str, max_depth: int = 10
    ) -> list[GraphPathResult]:
        return self._traverse(project_id, start_node_id, max_depth, reverse=True)

    def _traverse(
        self, project_id: str, start_node_id: str, max_depth: int, reverse: bool
    ) -> list[GraphPathResult]:
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")

        arrow = "<-[:DEPENDS_ON*1..{}]-" if reverse else "-[:DEPENDS_ON*1..{}]->"
        pattern = arrow.format(max_depth)
        query = f"""
            MATCH path = (start:IRNode {{project_id: $project_id, id: $start_node_id}}){pattern}(other:IRNode)
            WITH other, path, length(path) AS depth
            ORDER BY depth ASC
            WITH other, collect(path)[0] AS shortest, min(depth) AS depth
            RETURN other.id AS node_id, depth, [n IN nodes(shortest) | n.id] AS path
        """
        with self._driver.session() as session:
            result = session.run(query, project_id=project_id, start_node_id=start_node_id)
            return [
                GraphPathResult(node_id=record["node_id"], depth=record["depth"], path=record["path"])
                for record in result
            ]

    def shortest_path(
        self, project_id: str, source_id: str, target_id: str, max_depth: int = 15
    ) -> list[str] | None:
        if not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("max_depth must be a positive integer")

        query = f"""
            MATCH (a:IRNode {{project_id: $project_id, id: $source_id}}),
                  (b:IRNode {{project_id: $project_id, id: $target_id}})
            MATCH p = shortestPath((a)-[:DEPENDS_ON*1..{max_depth}]->(b))
            RETURN [n IN nodes(p) | n.id] AS path
        """
        with self._driver.session() as session:
            record = session.run(query, project_id=project_id, source_id=source_id, target_id=target_id).single()
            return record["path"] if record else None
