"""
IRGraph persistence.

Same upsert-and-prune pattern as `manifest_store.py`'s file sync, applied
to nodes and edges: Part 4's `IRBuilder` always computes a full graph from
complete inputs (see Part 4's design doc for why), so persisting it means
making the database match that full graph exactly — upsert everything
present, delete anything for this project that's no longer present. This
is what makes "a module was deleted" correctly propagate to the database
(its node row disappears) rather than accumulating stale nodes forever.

Ordering matters, in both directions, for the foreign key between edges
and nodes (`source_id`, `target_id`): stale edges must be deleted BEFORE
stale nodes are pruned (otherwise a lingering edge could reference a node
that's about to be deleted), and new edges must be inserted AFTER nodes
are synced (otherwise a new edge could reference a node that doesn't
exist yet). `save()` is sequenced accordingly: delete stale edges, sync
nodes, then upsert current edges.
"""
from __future__ import annotations

from sqlalchemy import delete
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from semrag.ingestion.ir.models import IRGraph
from semrag.storage.postgres.models import IREdgeRow, IRNodeRow, Project


class IRGraphRepository:
    def __init__(self, session: Session):
        self._session = session

    def save(self, graph: IRGraph) -> None:
        self._ensure_project(graph.project_id)

        # Ordering here is load-bearing, not stylistic — got this wrong on
        # the first pass and a live foreign-key violation caught it (see
        # this part's design doc, Section 5): edges reference nodes via FK,
        # so (1) stale edges must be deleted BEFORE stale nodes are pruned
        # (else a lingering edge could reference an about-to-be-deleted
        # node), and (2) new edges must be inserted AFTER nodes are synced
        # (else a new edge could reference a node that doesn't exist yet).
        # A single "sync edges then sync nodes" or "sync nodes then sync
        # edges" ordering satisfies one of these and violates the other —
        # deleting stale edges and inserting current edges have to happen
        # on opposite sides of the node sync.
        self._delete_stale_edges(graph)
        self._sync_nodes(graph)
        self._upsert_edges(graph)

    def _ensure_project(self, project_id: str) -> None:
        self._session.execute(
            pg_insert(Project)
            .values(project_id=project_id, root_path="")
            .on_conflict_do_nothing(index_elements=["project_id"])
        )

    def _sync_nodes(self, graph: IRGraph) -> None:
        """Insert or update nodes in batch."""
        if not graph.nodes:
            return
        nodes_to_upsert = []
        for node in graph.nodes.values():
            nodes_to_upsert.append(
                {
                    "id": node.id,
                    "project_id": node.project_id,
                    "kind": node.kind.value,
                    "relative_path": node.relative_path,
                    "name": node.name,
                    "parent_file_id": node.parent_file_id,
                    "attributes": node.attributes,
                }
            )
        # Use a single INSERT ... ON CONFLICT DO UPDATE for all nodes.
        stmt = pg_insert(IRNodeRow).values(nodes_to_upsert)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "kind": stmt.excluded.kind,
                "relative_path": stmt.excluded.relative_path,
                "name": stmt.excluded.name,
                "parent_file_id": stmt.excluded.parent_file_id,
                "attributes": stmt.excluded.attributes,
            },
        )
        self._session.execute(stmt)

        # Delete nodes that are no longer present.
        current_ids = set(graph.nodes.keys())
        delete_stmt = delete(IRNodeRow).where(IRNodeRow.project_id == graph.project_id)
        if current_ids:
            delete_stmt = delete_stmt.where(IRNodeRow.id.notin_(current_ids))
        self._session.execute(delete_stmt)

    def _delete_stale_edges(self, graph: IRGraph) -> None:
        """Delete edges that are no longer present."""
        if not graph.edges:
            return
        current_ids = {edge.id for edge in graph.edges}
        delete_stmt = delete(IREdgeRow).where(IREdgeRow.project_id == graph.project_id)
        if current_ids:
            delete_stmt = delete_stmt.where(IREdgeRow.id.notin_(current_ids))
        self._session.execute(delete_stmt)

    def _upsert_edges(self, graph: IRGraph) -> None:
        """Insert or update edges in batch."""
        if not graph.edges:
            return
        edges_to_upsert = []
        for edge in graph.edges:
            edges_to_upsert.append(
                {
                    "id": edge.id,
                    "project_id": graph.project_id,
                    "kind": edge.kind.value,
                    "source_id": edge.source_id,
                    "target_name": edge.target_name,
                    "target_id": edge.target_id,
                    "resolved": edge.resolved,
                    "ambiguous": edge.ambiguous,
                    "candidate_ids": edge.candidate_ids,
                }
            )
        stmt = pg_insert(IREdgeRow).values(edges_to_upsert)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "kind": stmt.excluded.kind,
                "source_id": stmt.excluded.source_id,
                "target_name": stmt.excluded.target_name,
                "target_id": stmt.excluded.target_id,
                "resolved": stmt.excluded.resolved,
                "ambiguous": stmt.excluded.ambiguous,
                "candidate_ids": stmt.excluded.candidate_ids,
            },
        )
        self._session.execute(stmt)

    def bulk_delete_stale_nodes(self, project_id: str, keep_ids: set[str]) -> int:
        """Bulk delete nodes for a project that aren't in ``keep_ids``.
        Returns the rowcount for observability. Stale edges are pruned
        first by ``save`` — callers using this in isolation are responsible
        for their own ordering."""
        from sqlalchemy import delete as _delete

        stmt = _delete(IRNodeRow).where(IRNodeRow.project_id == project_id)
        if keep_ids:
            stmt = stmt.where(IRNodeRow.id.notin_(keep_ids))
        result = self._session.execute(stmt)
        return result.rowcount or 0

    def load(self, project_id: str) -> IRGraph:
        """Reconstructs an IRGraph from stored rows — used by tests to
        verify round-trip fidelity, and available to any future caller
        (e.g. Part 10's knowledge graph sync) that needs the graph without
        re-running the builder."""
        from semrag.ingestion.ir.models import IREdge, IREdgeKind, IRNode, IRNodeKind
        from sqlalchemy import select

        node_rows = self._session.execute(
            select(IRNodeRow).where(IRNodeRow.project_id == project_id)
        ).scalars().all()
        edge_rows = self._session.execute(
            select(IREdgeRow).where(IREdgeRow.project_id == project_id)
        ).scalars().all()

        nodes = {
            row.id: IRNode(
                id=row.id,
                kind=IRNodeKind(row.kind),
                project_id=row.project_id,
                relative_path=row.relative_path,
                name=row.name,
                parent_file_id=row.parent_file_id,
                attributes=row.attributes or {},
            )
            for row in node_rows
        }
        edges = [
            IREdge(
                id=row.id,
                kind=IREdgeKind(row.kind),
                source_id=row.source_id,
                target_name=row.target_name,
                target_id=row.target_id,
                resolved=row.resolved,
                ambiguous=row.ambiguous,
                candidate_ids=row.candidate_ids or [],
            )
            for row in edge_rows
        ]
        return IRGraph(project_id=project_id, nodes=nodes, edges=edges)