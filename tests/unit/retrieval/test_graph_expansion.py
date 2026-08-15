import pytest

from semrag.ingestion.ir.models import IREdge, IREdgeKind, IRGraph, IRNode, IRNodeKind
from semrag.retrieval.graph_expansion import (
    DEFAULT_DEPENDENCIES_MAX_DEPTH,
    expand_with_graph_context,
)
from semrag.retrieval.intent import (
    GraphExpansionDirection,
    classify_graph_direction,
)
from semrag.retrieval.models import RetrievalResult
from semrag.storage.postgres.graph_queries import PostgresGraphQueries
from semrag.storage.postgres.ir_repository import IRGraphRepository
from semrag.storage.vector.models import VectorChunk
from semrag.storage.vector.pgvector_models import EMBEDDING_DIMENSION
from semrag.storage.vector.pgvector_store import PgvectorStore


def _node(node_id: str) -> IRNode:
    return IRNode(id=node_id, kind=IRNodeKind.SV_MODULE, project_id="proj1",
                  relative_path=f"RTL/{node_id}.sv", name=node_id)


def _chunk(chunk_id: str, source_node_id: str, text: str) -> VectorChunk:
    return VectorChunk(
        chunk_id=chunk_id, project_id="proj1", relative_path=f"RTL/{source_node_id}.sv",
        chunk_type="sv_module", text=text, embedding=[0.1] * EMBEDDING_DIMENSION,
        source_node_id=source_node_id,
    )


def _seed_chain(db_session, node_ids: list[str]) -> None:
    """top -> mid -> leaf linear chain plus a leaf-2 sibling off mid."""
    nodes = {nid: _node(nid) for nid in node_ids}
    edges = [
        IREdge(id=f"e{i}", kind=IREdgeKind.INSTANTIATES, source_id=a, target_name=b,
               target_id=b, resolved=True)
        for i, (a, b) in enumerate(zip(node_ids, node_ids[1:], strict=False))
    ]
    IRGraphRepository(db_session).save(IRGraph(project_id="proj1", nodes=nodes, edges=edges))
    db_session.flush()
    PgvectorStore(db_session).upsert_chunks([
        _chunk(f"c_{nid}", nid, f"{nid} implementation") for nid in node_ids
    ])
    db_session.flush()


def test_expansion_pulls_in_dependent_module_chunk(db_session):
    # top instantiates mid; mid is our "base" retrieval hit — expansion
    # should surface top's chunk as a dependent, since top depends on mid.
    nodes = {"top": _node("top"), "mid": _node("mid")}
    edge = IREdge(id="e1", kind=IREdgeKind.INSTANTIATES, source_id="top", target_name="mid",
                  target_id="mid", resolved=True)
    IRGraphRepository(db_session).save(IRGraph(project_id="proj1", nodes=nodes, edges=[edge]))
    db_session.flush()

    PgvectorStore(db_session).upsert_chunks([
        _chunk("c_top", "top", "top module instantiates mid"),
        _chunk("c_mid", "mid", "mid module implementation"),
    ])
    db_session.flush()

    base_result = RetrievalResult(
        chunk_id="c_mid", project_id="proj1", relative_path="RTL/mid.sv",
        chunk_type="sv_module", text="mid module implementation",
        source_node_id="mid", fused_score=1.0,
    )

    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session), [base_result], "proj1"
    )
    assert {e.chunk_id for e in expansions} == {"c_top"}


def test_expansion_skips_results_without_source_node_id(db_session):
    base_result = RetrievalResult(
        chunk_id="c_md", project_id="proj1", relative_path="docs/spec.md",
        chunk_type="markdown_section", text="some markdown content",
        source_node_id=None, fused_score=1.0,
    )
    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session), [base_result], "proj1"
    )
    assert expansions == []


def test_expansion_does_not_duplicate_existing_base_results(db_session):
    nodes = {"top": _node("top"), "mid": _node("mid")}
    edge = IREdge(id="e1", kind=IREdgeKind.INSTANTIATES, source_id="top", target_name="mid",
                  target_id="mid", resolved=True)
    IRGraphRepository(db_session).save(IRGraph(project_id="proj1", nodes=nodes, edges=[edge]))
    db_session.flush()

    PgvectorStore(db_session).upsert_chunks([
        _chunk("c_top", "top", "top text"),
        _chunk("c_mid", "mid", "mid text"),
    ])
    db_session.flush()

    # Both chunks already present as base results — expansion should add nothing new.
    base_results = [
        RetrievalResult(chunk_id="c_mid", project_id="proj1", relative_path="RTL/mid.sv",
                         chunk_type="sv_module", text="mid text", source_node_id="mid", fused_score=1.0),
        RetrievalResult(chunk_id="c_top", project_id="proj1", relative_path="RTL/top.sv",
                         chunk_type="sv_module", text="top text", source_node_id="top", fused_score=0.9),
    ]
    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session), base_results, "proj1"
    )
    assert expansions == []


def test_expansion_candidate_set_unchanged_by_seed_score_metadata(db_session):
    """Graph expansion must still produce the same candidate set as before
    the R2D-1 seed-relative scoring change — only the ``graph_seed_score``
    scoring metadata was added, never the candidate generation."""
    nodes = {"top": _node("top"), "mid": _node("mid")}
    edge = IREdge(id="e1", kind=IREdgeKind.INSTANTIATES, source_id="top", target_name="mid",
                  target_id="mid", resolved=True)
    IRGraphRepository(db_session).save(IRGraph(project_id="proj1", nodes=nodes, edges=[edge]))
    db_session.flush()

    PgvectorStore(db_session).upsert_chunks([
        _chunk("c_top", "top", "top module instantiates mid"),
        _chunk("c_mid", "mid", "mid module implementation"),
    ])
    db_session.flush()

    base_result = RetrievalResult(
        chunk_id="c_mid", project_id="proj1", relative_path="RTL/mid.sv",
        chunk_type="sv_module", text="mid module implementation",
        source_node_id="mid", fused_score=0.5,
    )

    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session), [base_result], "proj1"
    )
    assert {e.chunk_id for e in expansions} == {"c_top"}
    # The expansion must carry the producing seed's score as scoring metadata.
    exp = expansions[0]
    assert exp.signal_scores["graph_expansion"] == pytest.approx(0.5)  # depth=1
    assert exp.signal_scores["graph_seed_score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# R3A: direction-aware graph expansion
# ---------------------------------------------------------------------------


def _result(chunk_id: str, source_node_id: str | None = None, fused_score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, project_id="proj1", relative_path="RTL/x.sv",
        chunk_type="sv_module", text="module x; endmodule",
        source_node_id=source_node_id, fused_score=fused_score,
    )


def test_forward_dependency_expansion(db_session):
    # top -> mid -> leaf. A seed "top" walking DEPENDENCIES must surface
    # mid and leaf (downstream), not just mid's own text.
    _seed_chain(db_session, ["top", "mid", "leaf"])
    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session),
        [_result("c_top", "top")], "proj1",
        direction=GraphExpansionDirection.DEPENDENCIES,
        dependencies_max_depth=4,
    )
    assert {e.chunk_id for e in expansions} == {"c_mid", "c_leaf"}


def test_reverse_dependent_expansion_preserved(db_session):
    # top -> mid -> leaf. A seed "leaf" walking DEPENDENTS must surface
    # mid and top (who uses it), matching historical reverse behavior.
    _seed_chain(db_session, ["top", "mid", "leaf"])
    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session),
        [_result("c_leaf", "leaf")], "proj1",
        direction=GraphExpansionDirection.DEPENDENTS,
        max_depth=4,
    )
    assert {e.chunk_id for e in expansions} == {"c_top", "c_mid"}


def test_default_direction_remains_dependents(db_session):
    # Backward compatibility: no direction argument keeps the historical
    # reverse walk (proven by the existing tests above); this pins the
    # default enum value explicitly.
    assert GraphExpansionDirection.DEPENDENTS is not None
    assert classify_graph_direction("") == GraphExpansionDirection.DEPENDENTS


@pytest.mark.parametrize(
    "query,expected",
    [
        # forward / dependency phrasing
        ("trace the read from test through the driver", GraphExpansionDirection.DEPENDENCIES),
        ("how does i2c_master_read_test reach the driver?", GraphExpansionDirection.DEPENDENCIES),
        ("what does the driver use?", GraphExpansionDirection.DEPENDENCIES),
        ("path from the sequence to the DUT", GraphExpansionDirection.DEPENDENCIES),
        ("walk through the transaction flow", GraphExpansionDirection.DEPENDENCIES),
        # reverse / dependent phrasing
        ("who uses the driver?", GraphExpansionDirection.DEPENDENTS),
        ("what depends on the sequencer?", GraphExpansionDirection.DEPENDENTS),
        ("who calls the monitor?", GraphExpansionDirection.DEPENDENTS),
        ("which module references i2c_master_if?", GraphExpansionDirection.DEPENDENTS),
        ("instantiated by the environment", GraphExpansionDirection.DEPENDENTS),
        # bidirectional / relationship phrasing
        ("how is the driver connected to the sequencer?", GraphExpansionDirection.BIDIRECTIONAL),
        ("what is the relationship between agent and driver?", GraphExpansionDirection.BIDIRECTIONAL),
        ("related to the environment", GraphExpansionDirection.BIDIRECTIONAL),
    ],
)
def test_classify_graph_direction(query, expected):
    assert classify_graph_direction(query) == expected


def test_forward_multi_hop_traversal(db_session):
    # Chain of 5 nodes: walking forward with the default dependencies depth
    # (4) must reach the far end of the chain.
    chain = [f"n{i}" for i in range(5)]
    _seed_chain(db_session, chain)
    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session),
        [_result("c_n0", "n0")], "proj1",
        direction=GraphExpansionDirection.DEPENDENCIES,
        dependencies_max_depth=DEFAULT_DEPENDENCIES_MAX_DEPTH,
    )
    assert {e.chunk_id for e in expansions} == {f"c_{n}" for n in chain[1:]}


def test_forward_traversal_respects_bounded_depth(db_session):
    # Depth must stay bounded — a 10-hop chain with dependencies_max_depth=3
    # must not surface nodes beyond depth 3.
    chain = [f"n{i}" for i in range(10)]
    _seed_chain(db_session, chain)
    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session),
        [_result("c_n0", "n0")], "proj1",
        direction=GraphExpansionDirection.DEPENDENCIES,
        dependencies_max_depth=3,
    )
    assert {e.chunk_id for e in expansions} == {f"c_n{i}" for i in range(1, 4)}


def test_forward_deep_chain_node_survives_fan_out(db_session):
    # n0 -> a, b (d1); a -> c, d (d2); d -> e, f (d3); f -> g (d4). With the
    # default dependencies bounds, the depth-4 node g must survive even
    # though shallower levels fan out — this is the test->seq->sequencer->
    # driver->iface shape the forward direction exists for.
    nodes = {nid: _node(nid) for nid in ["n0", "a", "b", "c", "d", "e", "f", "g"]}
    edges = [
        IREdge(id="e1", kind=IREdgeKind.INSTANTIATES, source_id="n0", target_name="a", target_id="a", resolved=True),
        IREdge(id="e2", kind=IREdgeKind.INSTANTIATES, source_id="n0", target_name="b", target_id="b", resolved=True),
        IREdge(id="e3", kind=IREdgeKind.INSTANTIATES, source_id="a", target_name="c", target_id="c", resolved=True),
        IREdge(id="e4", kind=IREdgeKind.INSTANTIATES, source_id="a", target_name="d", target_id="d", resolved=True),
        IREdge(id="e5", kind=IREdgeKind.INSTANTIATES, source_id="d", target_name="e", target_id="e", resolved=True),
        IREdge(id="e6", kind=IREdgeKind.INSTANTIATES, source_id="d", target_name="f", target_id="f", resolved=True),
        IREdge(id="e7", kind=IREdgeKind.INSTANTIATES, source_id="f", target_name="g", target_id="g", resolved=True),
    ]
    IRGraphRepository(db_session).save(IRGraph(project_id="proj1", nodes=nodes, edges=edges))
    db_session.flush()
    PgvectorStore(db_session).upsert_chunks([_chunk(f"c_{n}", n, f"{n} text") for n in nodes])
    db_session.flush()

    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session),
        [_result("c_n0", "n0")], "proj1",
        direction=GraphExpansionDirection.DEPENDENCIES,
    )
    ids = {e.chunk_id for e in expansions}
    assert "c_g" in ids, f"depth-4 chain node must survive fan-out; got {ids}"
    # A tight cap must still truncate it — the cap is what the deep-node
    # guarantee is measured against.
    tight = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session),
        [_result("c_n0", "n0")], "proj1",
        direction=GraphExpansionDirection.DEPENDENCIES,
        dependencies_max_expansions_per_result=3,
    )
    assert "c_g" not in {e.chunk_id for e in tight}


def test_forward_expansion_scoring_is_seed_relative(db_session):
    _seed_chain(db_session, ["top", "mid", "leaf"])
    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session),
        [_result("c_top", "top", fused_score=0.4)], "proj1",
        direction=GraphExpansionDirection.DEPENDENCIES,
    )
    by_id = {e.chunk_id: e for e in expansions}
    assert by_id["c_mid"].signal_scores["graph_seed_score"] == pytest.approx(0.4)
    # depth-1 -> 0.5
    assert by_id["c_mid"].signal_scores["graph_expansion"] == pytest.approx(0.5)
    # depth-2 -> 1/3
    assert by_id["c_leaf"].signal_scores["graph_expansion"] == pytest.approx(1 / 3)


def test_bidirectional_dedupes_no_duplicate_candidates(db_session):
    # diamond: top -> a, top -> b, a -> shared, b -> shared. Walking both
    # directions from top must yield each chunk exactly once.
    nodes = {nid: _node(nid) for nid in ["top", "a", "b", "shared"]}
    edges = [
        IREdge(id="e1", kind=IREdgeKind.INSTANTIATES, source_id="top", target_name="a", target_id="a", resolved=True),
        IREdge(id="e2", kind=IREdgeKind.INSTANTIATES, source_id="top", target_name="b", target_id="b", resolved=True),
        IREdge(id="e3", kind=IREdgeKind.INSTANTIATES, source_id="a", target_name="shared", target_id="shared", resolved=True),
        IREdge(id="e4", kind=IREdgeKind.INSTANTIATES, source_id="b", target_name="shared", target_id="shared", resolved=True),
    ]
    IRGraphRepository(db_session).save(IRGraph(project_id="proj1", nodes=nodes, edges=edges))
    db_session.flush()
    PgvectorStore(db_session).upsert_chunks([
        _chunk("c_top", "top", "top text"),
        _chunk("c_a", "a", "a text"),
        _chunk("c_b", "b", "b text"),
        _chunk("c_shared", "shared", "shared text"),
    ])
    db_session.flush()

    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session),
        [_result("c_top", "top")], "proj1",
        direction=GraphExpansionDirection.BIDIRECTIONAL,
    )
    ids = [e.chunk_id for e in expansions]
    assert len(ids) == len(set(ids))
    assert set(ids) == {"c_a", "c_b", "c_shared"}


def test_no_fabricated_nodes_on_forward_expansion(db_session):
    # A seed whose node has no forward neighbors produces no expansion —
    # never a synthetic node.
    _seed_chain(db_session, ["top", "mid"])
    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session),
        [_result("c_mid", "mid")], "proj1",
        direction=GraphExpansionDirection.DEPENDENCIES,
    )
    assert expansions == []


def test_direction_metadata_on_expansion(db_session):
    _seed_chain(db_session, ["top", "mid", "leaf"])
    expansions = expand_with_graph_context(
        db_session, PostgresGraphQueries(db_session),
        [_result("c_top", "top")], "proj1",
        direction=GraphExpansionDirection.DEPENDENCIES,
    )
    assert all(
        e.signal_scores["graph_direction"] == GraphExpansionDirection.DEPENDENCIES.value
        for e in expansions
    )
