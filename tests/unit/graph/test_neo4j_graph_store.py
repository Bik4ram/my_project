"""
Mirrors `tests/unit/storage/test_graph_queries.py` exactly (same graph
shapes, same assertions) so the two test suites double as the fair
comparison described in `interfaces.py`'s docstring — same questions,
asked of both engines. Skips at collection/fixture time (not a failure)
if no Neo4j instance is reachable — see `conftest.py`.
"""
from semrag.ingestion.ir.models import IREdge, IREdgeKind, IRGraph, IRNode, IRNodeKind


def _node(node_id: str, project_id: str = "proj1") -> IRNode:
    return IRNode(
        id=node_id, kind=IRNodeKind.SV_MODULE, project_id=project_id,
        relative_path=f"RTL/{node_id}.sv", name=node_id,
    )


def _edge(edge_id: str, source: str, target: str) -> IREdge:
    return IREdge(
        id=edge_id, kind=IREdgeKind.INSTANTIATES, source_id=source,
        target_name=target, target_id=target, resolved=True,
    )


def _seed(neo4j_store, project_id: str, node_ids: list[str], edges: list[tuple[str, str]]) -> None:
    nodes = {nid: _node(nid, project_id) for nid in node_ids}
    ir_edges = [_edge(f"e{i}", src, tgt) for i, (src, tgt) in enumerate(edges)]
    neo4j_store.sync_graph(IRGraph(project_id=project_id, nodes=nodes, edges=ir_edges))


def test_transitive_dependencies_linear_chain(neo4j_store):
    _seed(neo4j_store, "proj1", ["top", "mid", "leaf"], [("top", "mid"), ("mid", "leaf")])
    results = neo4j_store.transitive_dependencies("proj1", "top")
    by_id = {r.node_id: r for r in results}
    assert set(by_id.keys()) == {"mid", "leaf"}
    assert by_id["leaf"].depth == 2


def test_transitive_dependencies_diamond_shape(neo4j_store):
    _seed(
        neo4j_store, "proj1",
        ["top", "a", "b", "leaf"],
        [("top", "a"), ("top", "b"), ("a", "leaf"), ("b", "leaf")],
    )
    results = neo4j_store.transitive_dependencies("proj1", "top")
    by_id = {r.node_id: r for r in results}
    assert set(by_id.keys()) == {"a", "b", "leaf"}
    assert by_id["leaf"].depth == 2


def test_transitive_dependents_reverse_direction(neo4j_store):
    _seed(neo4j_store, "proj1", ["top", "mid", "leaf"], [("top", "mid"), ("mid", "leaf")])
    results = neo4j_store.transitive_dependents("proj1", "leaf")
    by_id = {r.node_id: r for r in results}
    assert set(by_id.keys()) == {"top", "mid"}


def test_shortest_path_found(neo4j_store):
    _seed(
        neo4j_store, "proj1",
        ["top", "a", "b", "leaf"],
        [("top", "a"), ("top", "b"), ("a", "leaf"), ("b", "leaf")],
    )
    path = neo4j_store.shortest_path("proj1", "top", "leaf")
    assert path is not None
    assert len(path) == 3


def test_shortest_path_returns_none_when_unreachable(neo4j_store):
    _seed(neo4j_store, "proj1", ["a", "b", "isolated"], [("a", "b")])
    assert neo4j_store.shortest_path("proj1", "a", "isolated") is None


def test_resync_prunes_removed_nodes(neo4j_store):
    _seed(neo4j_store, "proj1", ["a", "b"], [("a", "b")])
    _seed(neo4j_store, "proj1", ["a"], [])  # b removed
    results = neo4j_store.transitive_dependencies("proj1", "a")
    assert results == []
