from semrag.ingestion.ir.models import IREdge, IREdgeKind, IRGraph, IRNode, IRNodeKind
from semrag.storage.postgres.ir_repository import IRGraphRepository


def _node(node_id: str, kind: IRNodeKind, path: str, name: str, parent_file_id: str | None = None) -> IRNode:
    return IRNode(
        id=node_id,
        kind=kind,
        project_id="proj1",
        relative_path=path,
        name=name,
        parent_file_id=parent_file_id,
        attributes={"ports": [{"name": "clk", "direction": "input"}]},
    )


def test_save_and_load_roundtrip(db_session):
    file_node = _node("proj1::RTL/i2c_master.sv::sv_file", IRNodeKind.SV_FILE, "RTL/i2c_master.sv", "i2c_master.sv")
    module_node = _node(
        "proj1::RTL/i2c_master.sv::sv_module::i2c_master",
        IRNodeKind.SV_MODULE,
        "RTL/i2c_master.sv",
        "i2c_master",
        parent_file_id=file_node.id,
    )
    edge = IREdge(
        id="edge-1",
        kind=IREdgeKind.INCLUDES,
        source_id=file_node.id,
        target_name="i2c_defines.svh",
        target_id=None,
        resolved=False,
    )

    graph = IRGraph(
        project_id="proj1",
        nodes={file_node.id: file_node, module_node.id: module_node},
        edges=[edge],
    )

    repo = IRGraphRepository(db_session)
    repo.save(graph)
    db_session.flush()

    loaded = repo.load("proj1")
    assert set(loaded.nodes.keys()) == {file_node.id, module_node.id}
    assert loaded.nodes[module_node.id].name == "i2c_master"
    assert loaded.nodes[module_node.id].attributes["ports"][0]["name"] == "clk"
    assert len(loaded.edges) == 1
    assert loaded.edges[0].target_name == "i2c_defines.svh"
    assert loaded.edges[0].resolved is False


def test_resaving_prunes_removed_nodes_and_edges(db_session):
    node_a = _node("n-a", IRNodeKind.SV_MODULE, "RTL/a.sv", "a")
    node_b = _node("n-b", IRNodeKind.SV_MODULE, "RTL/b.sv", "b")
    edge = IREdge(id="e-ab", kind=IREdgeKind.INSTANTIATES, source_id="n-a", target_name="b", target_id="n-b", resolved=True)

    repo = IRGraphRepository(db_session)
    repo.save(IRGraph(project_id="proj1", nodes={"n-a": node_a, "n-b": node_b}, edges=[edge]))
    db_session.flush()

    # Rebuild the project with node_b (and therefore the edge to it) gone —
    # simulating a module being deleted from the repo.
    repo.save(IRGraph(project_id="proj1", nodes={"n-a": node_a}, edges=[]))
    db_session.flush()

    loaded = repo.load("proj1")
    assert set(loaded.nodes.keys()) == {"n-a"}
    assert loaded.edges == []


def test_resaving_updates_existing_node_attributes(db_session):
    node = _node("n-a", IRNodeKind.SV_MODULE, "RTL/a.sv", "a")
    repo = IRGraphRepository(db_session)
    repo.save(IRGraph(project_id="proj1", nodes={"n-a": node}, edges=[]))
    db_session.flush()

    updated_node = node.model_copy(update={"attributes": {"ports": [], "note": "updated"}})
    repo.save(IRGraph(project_id="proj1", nodes={"n-a": updated_node}, edges=[]))
    db_session.flush()

    loaded = repo.load("proj1")
    assert loaded.nodes["n-a"].attributes["note"] == "updated"


def test_ambiguous_edge_preserves_candidate_ids(db_session):
    node_a = _node("n-a", IRNodeKind.SV_MODULE, "RTL/a.sv", "a")
    edge = IREdge(
        id="e-ambig",
        kind=IREdgeKind.INSTANTIATES,
        source_id="n-a",
        target_name="fifo",
        target_id=None,
        resolved=False,
        ambiguous=True,
        candidate_ids=["n-fifo-1", "n-fifo-2"],
    )
    repo = IRGraphRepository(db_session)
    repo.save(IRGraph(project_id="proj1", nodes={"n-a": node_a}, edges=[edge]))
    db_session.flush()

    loaded = repo.load("proj1")
    assert loaded.edges[0].ambiguous is True
    assert set(loaded.edges[0].candidate_ids) == {"n-fifo-1", "n-fifo-2"}


def test_extends_edge_roundtrips(db_session):
    """An EXTENDS edge (sv_class -> sv_class) must persist and load back
    with its resolved target — proving the extends edge survives storage."""
    parent = _node(
        "proj1::TB/base.svh::sv_class::i2c_master_base_seq",
        IRNodeKind.SV_CLASS,
        "TB/base.svh",
        "i2c_master_base_seq",
    )
    child = _node(
        "proj1::TB/child.svh::sv_class::i2c_master_read_seq",
        IRNodeKind.SV_CLASS,
        "TB/child.svh",
        "i2c_master_read_seq",
    )
    edge = IREdge(
        id="extends-read-seq->base-seq",
        kind=IREdgeKind.EXTENDS,
        source_id=child.id,
        target_name="i2c_master_base_seq",
        target_id=parent.id,
        resolved=True,
    )

    repo = IRGraphRepository(db_session)
    repo.save(IRGraph(project_id="proj1", nodes={parent.id: parent, child.id: child}, edges=[edge]))
    db_session.flush()

    loaded = repo.load("proj1")
    extends_edges = [e for e in loaded.edges if e.kind == IREdgeKind.EXTENDS]
    assert len(extends_edges) == 1
    assert extends_edges[0].source_id == child.id
    assert extends_edges[0].target_id == parent.id
    assert extends_edges[0].resolved is True
    assert extends_edges[0].target_name == "i2c_master_base_seq"


def test_unresolved_extends_edge_roundtrips(db_session):
    """An EXTENDS edge whose base class is external (uvm_driver) must
    persist as unresolved with no target_id."""
    child = _node(
        "proj1::TB/driver.svh::sv_class::i2c_master_driver",
        IRNodeKind.SV_CLASS,
        "TB/driver.svh",
        "i2c_master_driver",
    )
    edge = IREdge(
        id="extends-driver->uvm_driver",
        kind=IREdgeKind.EXTENDS,
        source_id=child.id,
        target_name="uvm_driver",
        target_id=None,
        resolved=False,
    )

    repo = IRGraphRepository(db_session)
    repo.save(IRGraph(project_id="proj1", nodes={child.id: child}, edges=[edge]))
    db_session.flush()

    loaded = repo.load("proj1")
    extends_edges = [e for e in loaded.edges if e.kind == IREdgeKind.EXTENDS]
    assert len(extends_edges) == 1
    assert extends_edges[0].target_id is None
    assert extends_edges[0].resolved is False


def test_references_edge_roundtrips(db_session):
    """A REFERENCES edge (sv_class -> sv_class) must persist and load back
    with its resolved target."""
    seq_item = _node(
        "proj1::TB/item.svh::sv_class::i2c_master_sequence_item",
        IRNodeKind.SV_CLASS,
        "TB/item.svh",
        "i2c_master_sequence_item",
    )
    read_seq = _node(
        "proj1::TB/read_seq.svh::sv_class::i2c_master_read_seq",
        IRNodeKind.SV_CLASS,
        "TB/read_seq.svh",
        "i2c_master_read_seq",
    )
    edge = IREdge(
        id="refs-read-seq->seq-item",
        kind=IREdgeKind.REFERENCES,
        source_id=read_seq.id,
        target_name="i2c_master_sequence_item",
        target_id=seq_item.id,
        resolved=True,
    )

    repo = IRGraphRepository(db_session)
    repo.save(IRGraph(project_id="proj1", nodes={seq_item.id: seq_item, read_seq.id: read_seq}, edges=[edge]))
    db_session.flush()

    loaded = repo.load("proj1")
    ref_edges = [e for e in loaded.edges if e.kind == IREdgeKind.REFERENCES]
    assert len(ref_edges) == 1
    assert ref_edges[0].source_id == read_seq.id
    assert ref_edges[0].target_id == seq_item.id
    assert ref_edges[0].resolved is True
    assert ref_edges[0].target_name == "i2c_master_sequence_item"


def test_unresolved_references_edge_roundtrips(db_session):
    child = _node(
        "proj1::TB/holder.svh::sv_class::holder",
        IRNodeKind.SV_CLASS,
        "TB/holder.svh",
        "holder",
    )
    edge = IREdge(
        id="refs-holder->uvm_driver",
        kind=IREdgeKind.REFERENCES,
        source_id=child.id,
        target_name="uvm_driver",
        target_id=None,
        resolved=False,
    )

    repo = IRGraphRepository(db_session)
    repo.save(IRGraph(project_id="proj1", nodes={child.id: child}, edges=[edge]))
    db_session.flush()

    loaded = repo.load("proj1")
    ref_edges = [e for e in loaded.edges if e.kind == IREdgeKind.REFERENCES]
    assert len(ref_edges) == 1
    assert ref_edges[0].target_id is None
    assert ref_edges[0].resolved is False


def test_two_projects_do_not_interfere(db_session):
    repo = IRGraphRepository(db_session)
    node_a = _node("proj_a::n", IRNodeKind.SV_MODULE, "RTL/a.sv", "a")
    node_a = node_a.model_copy(update={"project_id": "proj_a"})
    node_b = _node("proj_b::n", IRNodeKind.SV_MODULE, "RTL/b.sv", "b")
    node_b = node_b.model_copy(update={"project_id": "proj_b"})

    repo.save(IRGraph(project_id="proj_a", nodes={node_a.id: node_a}, edges=[]))
    repo.save(IRGraph(project_id="proj_b", nodes={node_b.id: node_b}, edges=[]))
    db_session.flush()

    loaded_a = repo.load("proj_a")
    loaded_b = repo.load("proj_b")
    assert set(loaded_a.nodes.keys()) == {node_a.id}
    assert set(loaded_b.nodes.keys()) == {node_b.id}


def test_calls_edge_roundtrips(db_session):
    """A resolved CALLS edge (sv_method -> sv_method) and an unresolved one
    (external UVM method) must both persist and load back."""
    run_phase = _node(
        "proj1::TB/driver.svh::sv_class::i2c_master_driver::run_phase",
        IRNodeKind.SV_METHOD,
        "TB/driver.svh",
        "run_phase",
    )
    drive = _node(
        "proj1::TB/driver.svh::sv_class::i2c_master_driver::drive",
        IRNodeKind.SV_METHOD,
        "TB/driver.svh",
        "drive",
    )
    resolved = IREdge(
        id="calls-run_phase->drive",
        kind=IREdgeKind.CALLS,
        source_id=run_phase.id,
        target_name="drive",
        target_id=drive.id,
        resolved=True,
    )
    unresolved = IREdge(
        id="calls-run_phase->get_next_item",
        kind=IREdgeKind.CALLS,
        source_id=run_phase.id,
        target_name="get_next_item",
        target_id=None,
        resolved=False,
    )

    repo = IRGraphRepository(db_session)
    repo.save(
        IRGraph(
            project_id="proj1",
            nodes={run_phase.id: run_phase, drive.id: drive},
            edges=[resolved, unresolved],
        )
    )
    db_session.flush()

    loaded = repo.load("proj1")
    calls = [e for e in loaded.edges if e.kind == IREdgeKind.CALLS]
    assert len(calls) == 2

    by_target = {e.target_name: e for e in calls}
    assert by_target["drive"].target_id == drive.id
    assert by_target["drive"].resolved is True
    assert by_target["get_next_item"].target_id is None
    assert by_target["get_next_item"].resolved is False


def test_references_edge_to_sv_interface_roundtrips(db_session):
    """R2E-D: a resolved REFERENCES edge (sv_class -> sv_interface), e.g.
    driver -> i2c_master_if, must persist and load back with its target."""
    driver = _node(
        "proj1::TB/driver.svh::sv_class::i2c_master_driver",
        IRNodeKind.SV_CLASS,
        "TB/driver.svh",
        "i2c_master_driver",
    )
    iface = _node(
        "proj1::TB/i2c_master_if.sv::sv_interface::i2c_master_if",
        IRNodeKind.SV_INTERFACE,
        "TB/i2c_master_if.sv",
        "i2c_master_if",
    )
    edge = IREdge(
        id="refs-driver->i2c_master_if",
        kind=IREdgeKind.REFERENCES,
        source_id=driver.id,
        target_name="i2c_master_if",
        target_id=iface.id,
        resolved=True,
    )

    repo = IRGraphRepository(db_session)
    repo.save(IRGraph(project_id="proj1", nodes={driver.id: driver, iface.id: iface}, edges=[edge]))
    db_session.flush()

    loaded = repo.load("proj1")
    ref_edges = [e for e in loaded.edges if e.kind == IREdgeKind.REFERENCES]
    assert len(ref_edges) == 1
    assert ref_edges[0].source_id == driver.id
    assert ref_edges[0].target_name == "i2c_master_if"
    assert ref_edges[0].target_id == iface.id
    assert ref_edges[0].resolved is True
