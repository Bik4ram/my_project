import time

from semrag.ingestion.ir.models import IREdge, IREdgeKind, IRGraph, IRNode, IRNodeKind
from semrag.storage.postgres.graph_queries import PostgresGraphQueries
from semrag.storage.postgres.ir_repository import IRGraphRepository


def _node(node_id: str, project_id: str = "proj1") -> IRNode:
    return IRNode(
        id=node_id, kind=IRNodeKind.SV_MODULE, project_id=project_id,
        relative_path=f"RTL/{node_id}.sv", name=node_id,
    )


def _edge(edge_id: str, source: str, target: str | None, resolved: bool = True) -> IREdge:
    return IREdge(
        id=edge_id, kind=IREdgeKind.INSTANTIATES, source_id=source,
        target_name=target or "unresolved", target_id=target, resolved=resolved,
    )


def _seed_graph(db_session, project_id: str, node_ids: list[str], edges: list[tuple[str, str]]) -> None:
    nodes = {nid: _node(nid, project_id) for nid in node_ids}
    ir_edges = [_edge(f"e{i}", src, tgt) for i, (src, tgt) in enumerate(edges)]
    IRGraphRepository(db_session).save(IRGraph(project_id=project_id, nodes=nodes, edges=ir_edges))
    db_session.flush()


def test_transitive_dependencies_linear_chain(db_session):
    # top -> mid -> leaf
    _seed_graph(db_session, "proj1", ["top", "mid", "leaf"], [("top", "mid"), ("mid", "leaf")])
    queries = PostgresGraphQueries(db_session)

    results = queries.transitive_dependencies("proj1", "top")
    by_id = {r.node_id: r for r in results}

    assert set(by_id.keys()) == {"mid", "leaf"}
    assert by_id["mid"].depth == 1
    assert by_id["leaf"].depth == 2
    assert by_id["leaf"].path == ["top", "mid", "leaf"]


def test_transitive_dependencies_diamond_shape(db_session):
    # top instantiates both a and b; both a and b instantiate shared
    # "leaf" — leaf should appear exactly once, at its shortest depth.
    _seed_graph(
        db_session, "proj1",
        ["top", "a", "b", "leaf"],
        [("top", "a"), ("top", "b"), ("a", "leaf"), ("b", "leaf")],
    )
    queries = PostgresGraphQueries(db_session)
    results = queries.transitive_dependencies("proj1", "top")
    by_id = {r.node_id: r for r in results}

    assert set(by_id.keys()) == {"a", "b", "leaf"}
    assert by_id["leaf"].depth == 2  # not duplicated, not deeper than necessary


def test_transitive_dependents_reverse_direction(db_session):
    _seed_graph(db_session, "proj1", ["top", "mid", "leaf"], [("top", "mid"), ("mid", "leaf")])
    queries = PostgresGraphQueries(db_session)

    results = queries.transitive_dependents("proj1", "leaf")
    by_id = {r.node_id: r for r in results}

    assert set(by_id.keys()) == {"top", "mid"}
    assert by_id["mid"].depth == 1
    assert by_id["top"].depth == 2


def test_cyclic_graph_does_not_hang(db_session):
    # a -> b -> c -> a (a genuine cycle, e.g. a real RTL instantiation bug)
    _seed_graph(db_session, "proj1", ["a", "b", "c"], [("a", "b"), ("b", "c"), ("c", "a")])
    queries = PostgresGraphQueries(db_session)

    start = time.perf_counter()
    results = queries.transitive_dependencies("proj1", "a", max_depth=50)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0  # must terminate quickly, not hang or time out
    by_id = {r.node_id: r for r in results}
    assert set(by_id.keys()) == {"b", "c"}  # 'a' itself never re-appears via the cycle guard


def test_max_depth_is_respected(db_session):
    chain = [f"n{i}" for i in range(10)]
    edges = [(chain[i], chain[i + 1]) for i in range(9)]
    _seed_graph(db_session, "proj1", chain, edges)
    queries = PostgresGraphQueries(db_session)

    results = queries.transitive_dependencies("proj1", "n0", max_depth=3)
    assert set(r.node_id for r in results) == {"n1", "n2", "n3"}


def test_shortest_path_found(db_session):
    _seed_graph(
        db_session, "proj1",
        ["top", "a", "b", "leaf"],
        [("top", "a"), ("top", "b"), ("a", "leaf"), ("b", "leaf")],
    )
    queries = PostgresGraphQueries(db_session)
    path = queries.shortest_path("proj1", "top", "leaf")
    assert path is not None
    assert len(path) == 3  # top -> {a or b} -> leaf
    assert path[0] == "top"
    assert path[-1] == "leaf"


def test_shortest_path_returns_none_when_unreachable(db_session):
    _seed_graph(db_session, "proj1", ["a", "b", "isolated"], [("a", "b")])
    queries = PostgresGraphQueries(db_session)
    assert queries.shortest_path("proj1", "a", "isolated") is None


def test_unresolved_edges_are_not_traversed(db_session):
    # An INSTANTIATES edge to a module that doesn't exist anywhere in the
    # project (target_id is None) must not be followed — there's nothing
    # to follow it to.
    node = _node("top", "proj1")
    edge = _edge("e0", "top", None, resolved=False)
    IRGraphRepository(db_session).save(IRGraph(project_id="proj1", nodes={"top": node}, edges=[edge]))
    db_session.flush()

    queries = PostgresGraphQueries(db_session)
    results = queries.transitive_dependencies("proj1", "top")
    assert results == []


def test_scale_benchmark_wide_and_deep_graph(db_session):
    """Not a correctness test — a real measurement. Builds a synthetic
    graph shaped like a plausible large SoC hierarchy (branching factor 4,
    8 levels deep => ~87K nodes) and times a full transitive-dependency
    traversal from the root. Result printed, not just asserted, so the
    number is visible in test output — see this part's design doc for the
    actual measured value and what it means for the Neo4j decision."""
    branching = 4
    depth = 6  # 4+16+64+256+1024+4096 = 5,460 nodes — already larger than
    # most real RTL hierarchies; kept from going deeper (8 levels =~87K
    # nodes) purely because IRGraphRepository upserts row-by-row (Part 5's
    # design, correct for its actual incremental-scan write pattern) and
    # a test fixture isn't the place to pay a bulk-insert-shaped cost for
    # a one-off benchmark setup.

    node_ids = ["root"]
    edges: list[tuple[str, str]] = []
    frontier = ["root"]
    counter = 0
    for level in range(depth):
        next_frontier = []
        for parent in frontier:
            for _ in range(branching):
                counter += 1
                child = f"n{counter}"
                node_ids.append(child)
                edges.append((parent, child))
                next_frontier.append(child)
        frontier = next_frontier

    _seed_graph(db_session, "proj_scale", node_ids, edges)
    queries = PostgresGraphQueries(db_session)

    start = time.perf_counter()
    results = queries.transitive_dependencies("proj_scale", "root", max_depth=depth)
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"\n[scale benchmark] nodes={len(node_ids)} edges={len(edges)} "
          f"depth={depth} branching={branching} -> traversal took {elapsed_ms:.1f}ms, "
          f"returned {len(results)} descendants")

    assert len(results) == len(node_ids) - 1  # every node except root
    assert elapsed_ms < 30_000  # generous bound; real number logged above

    # A more representative query than "full closure from the root":
    # most real questions ("what does this module depend on") are
    # bounded to a handful of levels from an arbitrary mid-graph node,
    # not an unbounded walk of the entire hierarchy. Picked from early in
    # the level-2 block (breadth-first build order) so it's guaranteed to
    # have several levels of descendants below it, not a leaf.
    mid_node = node_ids[10]
    start = time.perf_counter()
    shallow_results = queries.transitive_dependencies("proj_scale", mid_node, max_depth=3)
    shallow_elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[shallow benchmark] bounded depth-3 query from a mid-graph node -> "
          f"{shallow_elapsed_ms:.1f}ms, {len(shallow_results)} results")
