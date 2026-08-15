"""
End-to-end test: a real multi-file fixture project on disk, run through the
actual Scanner (Part 2), the actual parser registry (Part 3), and the IR
builder (Part 4) — proving the whole chain links correctly, not just each
part in isolation.
"""
from pathlib import Path

from semrag.ingestion.ir.models import IREdgeKind, IRNodeKind
from semrag.ingestion.ir.pipeline import scan_parse_and_build_ir
from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def build_fixture_project(root: Path) -> None:
    _write(
        root, "RTL/i2c_defines.svh",
        "`define I2C_WIDTH 8\n",
    )
    _write(
        root, "RTL/i2c_master.sv",
        '`include "i2c_defines.svh"\n'
        "module i2c_master(input logic clk, output logic scl); endmodule\n",
    )
    _write(
        root, "RTL/i2c_top.sv",
        "module i2c_top;\n"
        "  logic clk;\n"
        "  i2c_master u_master (.clk(clk));\n"
        "endmodule\n",
    )
    _write(
        root, "TB/i2c_pkg.sv",
        "package i2c_pkg;\n"
        "  typedef enum {IDLE, START} state_t;\n"
        "endpackage\n",
    )
    _write(
        root, "TB/uvm/i2c_driver.sv",
        "import i2c_pkg::*;\n"
        "class i2c_driver extends uvm_driver;\n"
        "  extern task run_phase(uvm_phase phase);\n"
        "  extern function void connect_phase(uvm_phase phase);\n"
        "endclass\n",
    )
    _write(
        root, "src/regression/runner.py",
        "from regression.helpers import setup\n"
        "def main():\n    pass\n",
    )
    _write(
        root, "src/regression/helpers.py",
        "def setup():\n    pass\n",
    )
    _write(
        root, "Scripts/build.tcl",
        "source common.tcl\n"
        "proc build_design {} { puts hi }\n",
    )
    _write(root, "Scripts/common.tcl", "proc common_setup {} { puts setup }\n")


def test_end_to_end_ir_graph(tmp_path: Path):
    build_fixture_project(tmp_path)
    store = InMemoryManifestStore()

    manifest, graph = scan_parse_and_build_ir("i2c_project", str(tmp_path), store)

    assert manifest.file_count > 0
    assert graph.project_id == "i2c_project"

    # --- SV module instantiation resolves across files ---
    top_module = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_MODULE) if n.name == "i2c_top")
    master_module = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_MODULE) if n.name == "i2c_master")

    inst_edges = [e for e in graph.edges_from(top_module.id) if e.kind == IREdgeKind.INSTANTIATES]
    assert len(inst_edges) == 1
    assert inst_edges[0].resolved is True
    assert inst_edges[0].target_id == master_module.id

    # --- `include resolves to the SV_FILE node ---
    master_file_node = next(
        n for n in graph.nodes_of_kind(IRNodeKind.SV_FILE) if n.relative_path == "RTL/i2c_master.sv"
    )
    defines_file_node = next(
        n for n in graph.nodes_of_kind(IRNodeKind.SV_FILE) if n.relative_path == "RTL/i2c_defines.svh"
    )
    include_edges = [e for e in graph.edges_from(master_file_node.id) if e.kind == IREdgeKind.INCLUDES]
    assert len(include_edges) == 1
    assert include_edges[0].target_id == defines_file_node.id

    # --- SV package import resolves ---
    driver_file_node = next(
        n for n in graph.nodes_of_kind(IRNodeKind.SV_FILE) if n.relative_path == "TB/uvm/i2c_driver.sv"
    )
    pkg_node = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_PACKAGE) if n.name == "i2c_pkg")
    import_edges = [e for e in graph.edges_from(driver_file_node.id) if e.kind == IREdgeKind.IMPORTS_PACKAGE]
    assert len(import_edges) == 1
    assert import_edges[0].target_id == pkg_node.id

    # --- UVM class + methods produce SV_CLASS / SV_METHOD nodes ---
    class_nodes = graph.nodes_of_kind(IRNodeKind.SV_CLASS)
    assert any(n.name == "i2c_driver" for n in class_nodes)
    driver_class = next(n for n in class_nodes if n.name == "i2c_driver")
    assert driver_class.parent_file_id == driver_file_node.id

    method_names = {n.name: n for n in graph.nodes_of_kind(IRNodeKind.SV_METHOD)}
    assert method_names["run_phase"].parent_file_id == driver_class.id
    assert method_names["connect_phase"].parent_file_id == driver_class.id

    # --- Python import resolves via dotted-suffix matching ---
    runner_node = next(n for n in graph.nodes_of_kind(IRNodeKind.PY_MODULE) if n.name == "runner")
    helpers_node = next(n for n in graph.nodes_of_kind(IRNodeKind.PY_MODULE) if n.name == "helpers")
    py_edges = [e for e in graph.edges_from(runner_node.id) if e.kind == IREdgeKind.PY_IMPORTS]
    assert len(py_edges) == 1
    assert py_edges[0].target_id == helpers_node.id

    # --- TCL source resolves ---
    build_node = next(n for n in graph.nodes_of_kind(IRNodeKind.TCL_SCRIPT) if n.name == "build.tcl")
    common_node = next(n for n in graph.nodes_of_kind(IRNodeKind.TCL_SCRIPT) if n.name == "common.tcl")
    tcl_edges = [e for e in graph.edges_from(build_node.id) if e.kind == IREdgeKind.TCL_SOURCES]
    assert len(tcl_edges) == 1
    assert tcl_edges[0].target_id == common_node.id


def test_unresolved_instantiation_target_is_marked_not_crashed(tmp_path: Path):
    _write(
        tmp_path, "RTL/i2c_top.sv",
        "module i2c_top;\n"
        "  external_ip_block u_blk (.a(1));\n"  # never defined in this project
        "endmodule\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    top_module = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_MODULE) if n.name == "i2c_top")
    inst_edges = [e for e in graph.edges_from(top_module.id) if e.kind == IREdgeKind.INSTANTIATES]
    assert len(inst_edges) == 1
    assert inst_edges[0].resolved is False
    assert inst_edges[0].ambiguous is False
    assert inst_edges[0].target_name == "external_ip_block"


def test_ambiguous_module_name_across_files_is_flagged(tmp_path: Path):
    _write(tmp_path, "RTL/fifo_v1.sv", "module fifo; endmodule\n")
    _write(tmp_path, "RTL/subsystem/fifo_v2.sv", "module fifo; endmodule\n")
    _write(
        tmp_path, "RTL/top.sv",
        "module top;\n  fifo u_fifo();\nendmodule\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    top_module = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_MODULE) if n.name == "top")
    inst_edges = [e for e in graph.edges_from(top_module.id) if e.kind == IREdgeKind.INSTANTIATES]
    assert len(inst_edges) == 1
    assert inst_edges[0].resolved is False
    assert inst_edges[0].ambiguous is True
    assert len(inst_edges[0].candidate_ids) == 2


def test_node_ids_are_stable_across_rebuilds(tmp_path: Path):
    build_fixture_project(tmp_path)
    store1 = InMemoryManifestStore()
    store2 = InMemoryManifestStore()

    _m1, graph1 = scan_parse_and_build_ir("i2c_project", str(tmp_path), store1)
    _m2, graph2 = scan_parse_and_build_ir("i2c_project", str(tmp_path), store2)

    assert set(graph1.nodes.keys()) == set(graph2.nodes.keys())
    edge_ids_1 = {e.id for e in graph1.edges}
    edge_ids_2 = {e.id for e in graph2.edges}
    assert edge_ids_1 == edge_ids_2


def test_package_scoped_uvm_classes_produce_sv_class_and_method_nodes(tmp_path: Path):
    """Classes declared inside a package must flow through to sv_class /
    sv_method IR nodes — the extraction fix, verified end to end."""
    _write(
        tmp_path, "TB/i2c_master_agent_pkg.sv",
        "package i2c_master_agent_pkg;\n"
        "    class i2c_master_driver;\n"
        "        extern function new(string name = \"i2c_master_driver\");\n"
        "        extern task run_phase(uvm_phase phase);\n"
        "        task drive(input i2c_master_sequence_item item);\n"
        "            vif.address <= item.address;\n"
        "        endtask : drive\n"
        "    endclass : i2c_master_driver\n"
        "endpackage : i2c_master_agent_pkg\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    class_nodes = graph.nodes_of_kind(IRNodeKind.SV_CLASS)
    assert [n.name for n in class_nodes] == ["i2c_master_driver"]
    driver = class_nodes[0]

    method_names = {n.name: n for n in graph.nodes_of_kind(IRNodeKind.SV_METHOD)}
    # extern prototype + inline body: each is a distinct method, no dup.
    assert set(method_names) == {"new", "run_phase", "drive"}
    for method in method_names.values():
        assert method.parent_file_id == driver.id


def test_class_extends_resolves_to_sv_class_node(tmp_path: Path):
    """class child extends parent; must produce a resolved EXTENDS edge from
    the child sv_class node to the parent sv_class node."""
    _write(
        tmp_path, "TB/i2c_master_base_seq.svh",
        "class i2c_master_base_seq;\n"
        "  extern task body();\n"
        "endclass : i2c_master_base_seq\n",
    )
    _write(
        tmp_path, "TB/i2c_master_read_seq.svh",
        "class i2c_master_read_seq extends i2c_master_base_seq;\n"
        "  extern task body();\n"
        "endclass : i2c_master_read_seq\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    read_seq = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_read_seq")
    base_seq = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_base_seq")

    edges = [e for e in graph.edges_from(read_seq.id) if e.kind == IREdgeKind.EXTENDS]
    assert len(edges) == 1
    assert edges[0].resolved is True
    assert edges[0].ambiguous is False
    assert edges[0].target_id == base_seq.id
    assert edges[0].target_name == "i2c_master_base_seq"


def test_class_extends_resolves_across_multiple_classes_in_one_file(tmp_path: Path):
    _write(
        tmp_path, "TB/multi.svh",
        "class base_a;\nendclass\n"
        "class base_b;\nendclass\n"
        "class child extends base_a;\nendclass\n"
        "class grandchild extends child;\nendclass\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    by_name = {n.name: n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS)}
    child = by_name["child"]
    grandchild = by_name["grandchild"]

    child_edges = [e for e in graph.edges_from(child.id) if e.kind == IREdgeKind.EXTENDS]
    assert len(child_edges) == 1
    assert child_edges[0].target_id == by_name["base_a"].id

    grandchild_edges = [e for e in graph.edges_from(grandchild.id) if e.kind == IREdgeKind.EXTENDS]
    assert len(grandchild_edges) == 1
    assert grandchild_edges[0].target_id == child.id


def test_class_extends_inside_package_resolves(tmp_path: Path):
    _write(
        tmp_path, "TB/i2c_master_agent_pkg.sv",
        "package i2c_master_agent_pkg;\n"
        "    class i2c_master_base_seq;\n"
        "    endclass\n"
        "    class i2c_master_read_seq extends i2c_master_base_seq;\n"
        "        extern task body();\n"
        "    endclass\n"
        "endpackage\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    by_name = {n.name: n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS)}
    read_seq = by_name["i2c_master_read_seq"]
    base_seq = by_name["i2c_master_base_seq"]

    edges = [e for e in graph.edges_from(read_seq.id) if e.kind == IREdgeKind.EXTENDS]
    assert len(edges) == 1
    assert edges[0].resolved is True
    assert edges[0].target_id == base_seq.id


def test_class_without_extends_produces_no_extends_edge(tmp_path: Path):
    _write(tmp_path, "TB/plain.svh", "class plain;\nendclass\n")
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    plain = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "plain")
    assert [e for e in graph.edges_from(plain.id) if e.kind == IREdgeKind.EXTENDS] == []


def test_unresolved_external_base_class_does_not_create_project_edge(tmp_path: Path):
    """extends uvm_driver (not indexed in this project) must produce an
    EXTENDS edge marked unresolved with no target_id — never a fabricated
    project-local node or a resolved edge to nothing."""
    _write(
        tmp_path, "TB/driver.svh",
        "class i2c_master_driver extends uvm_driver #(i2c_master_agent);\n"
        "  extern task run_phase(uvm_phase phase);\n"
        "endclass : i2c_master_driver\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    driver = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_driver")
    edges = [e for e in graph.edges_from(driver.id) if e.kind == IREdgeKind.EXTENDS]
    assert len(edges) == 1
    assert edges[0].target_name == "uvm_driver"  # parameterized base stripped to bare name
    assert edges[0].resolved is False
    assert edges[0].target_id is None
    assert edges[0].ambiguous is False
    # No invented sv_class node for uvm_driver.
    assert all(n.name != "uvm_driver" for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS))


def test_class_property_type_creates_resolved_reference_edge(tmp_path: Path):
    """class child; parent obj; endclass must create a resolved REFERENCES
    edge child -> parent when parent is a project-local sv_class."""
    _write(tmp_path, "TB/parent.svh", "class i2c_master_sequence_item;\nendclass\n")
    _write(
        tmp_path, "TB/child.svh",
        "class child;\n"
        "  i2c_master_sequence_item item;\n"
        "endclass\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    child = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "child")
    item = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_sequence_item")

    edges = [e for e in graph.edges_from(child.id) if e.kind == IREdgeKind.REFERENCES]
    assert len(edges) == 1
    assert edges[0].target_name == "i2c_master_sequence_item"
    assert edges[0].resolved is True
    assert edges[0].target_id == item.id
    assert edges[0].ambiguous is False


def test_parameterized_property_type_captures_bare_and_argument(tmp_path: Path):
    """some_class #(parent) obj; must resolve parent (the bare argument)
    to the project-local sv_class."""
    _write(tmp_path, "TB/parent.svh", "class i2c_master_sequence_item;\nendclass\n")
    _write(
        tmp_path, "TB/child.svh",
        "class child;\n"
        "  some_class #(i2c_master_sequence_item) obj;\n"
        "endclass\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    child = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "child")
    refs = [e for e in graph.edges_from(child.id) if e.kind == IREdgeKind.REFERENCES]

    # The bare parameterized type (some_class) is unresolved (not indexed),
    # but the type argument i2c_master_sequence_item resolves.
    resolved = [e for e in refs if e.resolved]
    assert len(resolved) == 1
    assert resolved[0].target_name == "i2c_master_sequence_item"
    item = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_sequence_item")
    assert resolved[0].target_id == item.id


def test_parameterized_extends_type_argument_reference(tmp_path: Path):
    """extends uvm_sequencer #(i2c_master_sequence_item) must capture the
    type argument i2c_master_sequence_item as a project-local reference,
    without fabricating a node for uvm_sequencer."""
    _write(tmp_path, "TB/item.svh", "class i2c_master_sequence_item;\nendclass\n")
    _write(
        tmp_path, "TB/sequencer.svh",
        "class i2c_master_sequencer extends uvm_sequencer #(i2c_master_sequence_item);\n"
        "endclass\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    seq = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_sequencer")
    refs = [e for e in graph.edges_from(seq.id) if e.kind == IREdgeKind.REFERENCES]
    resolved = [e for e in refs if e.resolved]
    assert len(resolved) == 1
    assert resolved[0].target_name == "i2c_master_sequence_item"
    # No fabricated uvm_sequencer sv_class node.
    assert all(n.name != "uvm_sequencer" for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS))


def test_multiple_project_local_references_in_one_class(tmp_path: Path):
    _write(tmp_path, "TB/item.svh", "class i2c_master_sequence_item;\nendclass\n")
    _write(tmp_path, "TB/cfg.svh", "class i2c_master_agent_config;\nendclass\n")
    _write(
        tmp_path, "TB/driver.svh",
        "class i2c_master_driver;\n"
        "  i2c_master_agent_config cfg;\n"
        "  i2c_master_sequence_item drv_item;\n"
        "endclass\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    driver = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_driver")
    refs = [e for e in graph.edges_from(driver.id) if e.kind == IREdgeKind.REFERENCES]
    assert len(refs) == 2
    assert {e.target_name for e in refs} == {"i2c_master_sequence_item", "i2c_master_agent_config"}
    assert all(e.resolved for e in refs)


def test_class_with_no_project_local_reference_creates_no_reference_edge(tmp_path: Path):
    _write(
        tmp_path, "TB/plain.svh",
        "class plain;\n"
        "  int counter;\n"
        "  logic en;\n"
        "endclass\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    plain = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "plain")
    assert [e for e in graph.edges_from(plain.id) if e.kind == IREdgeKind.REFERENCES] == []


def test_external_uvm_type_does_not_create_fabricated_reference(tmp_path: Path):
    """A property typed uvm_driver #(i2c_master_sequence_item) must not
    fabricate an edge to uvm_driver, but may resolve i2c_master_sequence_item
    when it is a project-local sv_class."""
    _write(tmp_path, "TB/item.svh", "class i2c_master_sequence_item;\nendclass\n")
    _write(
        tmp_path, "TB/holder.svh",
        "class holder;\n"
        "  uvm_driver #(i2c_master_sequence_item) drv;\n"
        "endclass\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    holder = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "holder")
    refs = [e for e in graph.edges_from(holder.id) if e.kind == IREdgeKind.REFERENCES]

    # uvm_driver is a REFERENCES candidate but unresolved; i2c_master_sequence_item resolves.
    assert any(e.target_name == "uvm_driver" and not e.resolved for e in refs)
    resolved = [e for e in refs if e.resolved]
    assert len(resolved) == 1
    assert resolved[0].target_name == "i2c_master_sequence_item"
    assert all(n.name != "uvm_driver" for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS))


def _method_node(graph, class_name: str, method_name: str):
    return next(
        n
        for n in graph.nodes_of_kind(IRNodeKind.SV_METHOD)
        if n.name == method_name and n.parent_file_id
        and graph.nodes[n.parent_file_id].name == class_name
    )


def test_project_local_method_call_creates_resolved_calls_edge(tmp_path: Path):
    """A method that calls another project-local method must produce a
    resolved CALLS edge sv_method -> sv_method."""
    _write(
        tmp_path, "TB/driver.svh",
        "class i2c_master_driver;\n"
        "  extern task run_phase(uvm_phase phase);\n"
        "  extern task drive(input i2c_master_sequence_item item);\n"
        "endclass\n"
        "task i2c_master_driver::run_phase(uvm_phase phase);\n"
        "  drive(item);\n"
        "endtask : run_phase\n"
        "task i2c_master_driver::drive(input i2c_master_sequence_item item);\n"
        "endtask : drive\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    run_phase = _method_node(graph, "i2c_master_driver", "run_phase")
    drive = _method_node(graph, "i2c_master_driver", "drive")

    calls = [e for e in graph.edges_from(run_phase.id) if e.kind == IREdgeKind.CALLS]
    assert len(calls) == 1
    assert calls[0].target_name == "drive"
    assert calls[0].resolved is True
    assert calls[0].target_id == drive.id
    assert calls[0].ambiguous is False


def test_unresolved_external_method_call_does_not_fabricate_node(tmp_path: Path):
    """A call to an external UVM method (get_next_item) must produce an
    unresolved CALLS edge with no target_id and no fabricated sv_method node."""
    _write(
        tmp_path, "TB/driver.svh",
        "class i2c_master_driver;\n"
        "  extern task run_phase(uvm_phase phase);\n"
        "endclass\n"
        "task i2c_master_driver::run_phase(uvm_phase phase);\n"
        "  seq_item_port.get_next_item(item);\n"
        "endtask : run_phase\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    run_phase = _method_node(graph, "i2c_master_driver", "run_phase")
    calls = [e for e in graph.edges_from(run_phase.id) if e.kind == IREdgeKind.CALLS]
    assert len(calls) == 1
    assert calls[0].target_name == "get_next_item"
    assert calls[0].resolved is False
    assert calls[0].target_id is None
    assert calls[0].ambiguous is False
    # No fabricated sv_method node for the external method.
    assert all(n.name != "get_next_item" for n in graph.nodes_of_kind(IRNodeKind.SV_METHOD))


def test_type_id_create_resolves_to_project_local_class(tmp_path: Path):
    """Class::type_id::create(...) must record a REFERENCES edge from the
    containing class to the created project-local class."""
    _write(tmp_path, "TB/item.svh", "class i2c_master_sequence_item;\nendclass\n")
    _write(
        tmp_path, "TB/seq.svh",
        "class i2c_master_read_seq;\n"
        "  extern task body();\n"
        "endclass\n"
        "task i2c_master_read_seq::body();\n"
        "  i2c_master_sequence_item::type_id::create(\"item\");\n"
        "endtask : body\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    seq = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_read_seq")
    refs = [e for e in graph.edges_from(seq.id) if e.kind == IREdgeKind.REFERENCES]
    resolved = [e for e in refs if e.resolved and e.target_name == "i2c_master_sequence_item"]
    assert len(resolved) == 1
    assert resolved[0].kind == IREdgeKind.REFERENCES


def test_existing_references_and_extends_edges_unchanged(tmp_path: Path):
    """Adding CALLS/type_id::create handling must not disturb existing
    EXTENDS and property-derived REFERENCES edges."""
    _write(tmp_path, "TB/parent.svh", "class i2c_master_base_seq;\nendclass\n")
    _write(tmp_path, "TB/item.svh", "class i2c_master_sequence_item;\nendclass\n")
    _write(
        tmp_path, "TB/seq.svh",
        "class i2c_master_read_seq extends i2c_master_base_seq;\n"
        "  i2c_master_sequence_item item;\n"
        "  extern task body();\n"
        "endclass\n"
        "task i2c_master_read_seq::body();\n"
        "  start_item(item);\n"
        "endtask : body\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    seq = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_read_seq")
    base_seq = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_base_seq")
    item = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_sequence_item")

    extends = [e for e in graph.edges_from(seq.id) if e.kind == IREdgeKind.EXTENDS]
    assert len(extends) == 1 and extends[0].resolved and extends[0].target_id == base_seq.id

    refs = [e for e in graph.edges_from(seq.id) if e.kind == IREdgeKind.REFERENCES]
    assert len(refs) == 1 and refs[0].resolved and refs[0].target_id == item.id


def test_multiple_calls_produce_expected_edges(tmp_path: Path):
    """A method with several call sites yields one CALLS edge per distinct
    target method name."""
    _write(
        tmp_path, "TB/seq.svh",
        "class i2c_master_read_seq;\n"
        "  extern task body();\n"
        "endclass\n"
        "task i2c_master_read_seq::body();\n"
        "  start_item(item);\n"
        "  finish_item(item);\n"
        "  start_item(item);\n"
        "endtask : body\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    body = _method_node(graph, "i2c_master_read_seq", "body")
    calls = [e for e in graph.edges_from(body.id) if e.kind == IREdgeKind.CALLS]
    # start_item deduplicated by the parser; two distinct external targets.
    assert {e.target_name for e in calls} == {"start_item", "finish_item"}
    assert all(not e.resolved for e in calls)


def test_type_id_create_does_not_fabricate_external_class(tmp_path: Path):
    """uvm_object::type_id::create(...) must not fabricate a project-local
    sv_class node for uvm_object."""
    _write(
        tmp_path, "TB/seq.svh",
        "class i2c_master_read_seq;\n"
        "  extern task body();\n"
        "endclass\n"
        "task i2c_master_read_seq::body();\n"
        "  uvm_object::type_id::create(\"obj\");\n"
        "endtask : body\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    seq = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_read_seq")
    refs = [e for e in graph.edges_from(seq.id) if e.kind == IREdgeKind.REFERENCES]
    uvm_obj_refs = [e for e in refs if e.target_name == "uvm_object"]
    assert len(uvm_obj_refs) == 1
    assert uvm_obj_refs[0].resolved is False
    assert uvm_obj_refs[0].target_id is None
    assert all(n.name != "uvm_object" for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS))


def test_agent_member_typed_references_resolve(tmp_path: Path):
    """R2E-A: each typed member of the agent (sqr/drv/mon) must produce a
    resolved REFERENCES edge agent -> member class. This relationship already
    exists; this guards the members structure."""
    _write(tmp_path, "TB/sequencer.svh", "class i2c_master_sequencer;\nendclass\n")
    _write(tmp_path, "TB/driver.svh", "class i2c_master_driver;\nendclass\n")
    _write(tmp_path, "TB/monitor.svh", "class i2c_master_monitor;\nendclass\n")
    _write(
        tmp_path, "TB/agent.svh",
        "class i2c_master_agent;\n"
        "  i2c_master_sequencer sqr;\n"
        "  i2c_master_driver drv;\n"
        "  i2c_master_monitor mon;\n"
        "endclass\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    agent = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_agent")
    refs = [e for e in graph.edges_from(agent.id) if e.kind == IREdgeKind.REFERENCES]
    assert {e.target_name for e in refs} == {
        "i2c_master_sequencer", "i2c_master_driver", "i2c_master_monitor"
    }
    assert all(e.resolved for e in refs)


def test_connect_creates_driver_to_sequencer_reference_edge(tmp_path: Path):
    """R2E-B: ``drv.seq_item_port.connect(sqr.seq_item_export);`` inside the
    agent's connect_phase must create a RESOLVED REFERENCES edge
    i2c_master_driver -> i2c_master_sequencer (the missing driver<->sequencer
    relationship)."""
    _write(tmp_path, "TB/sequencer.svh", "class i2c_master_sequencer;\nendclass\n")
    _write(tmp_path, "TB/driver.svh", "class i2c_master_driver;\nendclass\n")
    _write(
        tmp_path, "TB/agent.svh",
        "class i2c_master_agent;\n"
        "  i2c_master_driver drv;\n"
        "  i2c_master_sequencer sqr;\n"
        "  extern function void connect_phase(uvm_phase phase);\n"
        "endclass\n"
        "function void i2c_master_agent::connect_phase(uvm_phase phase);\n"
        "  drv.seq_item_port.connect(sqr.seq_item_export);\n"
        "endfunction : connect_phase\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    driver = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_driver")
    sequencer = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_sequencer")

    refs = [e for e in graph.edges_from(driver.id) if e.kind == IREdgeKind.REFERENCES]
    seq_refs = [e for e in refs if e.target_name == "i2c_master_sequencer"]
    assert len(seq_refs) == 1
    assert seq_refs[0].resolved is True
    assert seq_refs[0].target_id == sequencer.id
    assert seq_refs[0].ambiguous is False


def test_connect_to_external_uvm_target_fabricates_nothing(tmp_path: Path):
    """R2E-B guard: a connect whose receiver or target member is not a
    project-local sv_class must not fabricate any node or edge."""
    _write(
        tmp_path, "TB/env.svh",
        "class i2c_master_env;\n"
        "  extern function void connect_phase(uvm_phase phase);\n"
        "endclass\n"
        "function void i2c_master_env::connect_phase(uvm_phase phase);\n"
        "  this.analysis_port.connect(uvm_top.scb);\n"
        "  phase.raise_objection(this);\n"
        "endfunction : connect_phase\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    env = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_env")
    # No resolved connect edge to an external target.
    refs = [e for e in graph.edges_from(env.id) if e.kind == IREdgeKind.REFERENCES]
    assert all(not e.resolved for e in refs)
    # Nothing fabricated.
    assert all(n.name != "uvm_top" for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS))


def test_virtual_interface_reference_resolves(tmp_path: Path):
    """R2E-D: ``virtual i2c_master_if vif;`` must create a RESOLVED REFERENCES
    edge driver -> i2c_master_if targeting the SV_INTERFACE node."""
    _write(tmp_path, "TB/i2c_master_if.sv", "interface i2c_master_if(input logic clk); endinterface\n")
    _write(
        tmp_path, "TB/driver.svh",
        "class i2c_master_driver;\n"
        "  virtual i2c_master_if vif;\n"
        "endclass\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    driver = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_driver")
    iface = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_INTERFACE) if n.name == "i2c_master_if")

    refs = [e for e in graph.edges_from(driver.id) if e.kind == IREdgeKind.REFERENCES]
    iface_refs = [e for e in refs if e.target_name == "i2c_master_if"]
    assert len(iface_refs) == 1
    assert iface_refs[0].resolved is True
    assert iface_refs[0].target_id == iface.id
    assert iface_refs[0].ambiguous is False


def test_sequence_start_does_not_fabricate_method(tmp_path: Path):
    """R2E-E: ``rd_seq.start(mstr_env.mstr_agnt.sqr);`` must not fabricate a
    ``start`` sv_method node; the test -> sequence link exists via the member
    REFERENCES edge (read_test -> read_seq)."""
    _write(tmp_path, "TB/read_seq.svh", "class i2c_master_read_seq;\nendclass\n")
    _write(
        tmp_path, "TB/read_test.svh",
        "class i2c_master_read_test;\n"
        "  i2c_master_read_seq rd_seq;\n"
        "  extern task run_phase(uvm_phase phase);\n"
        "endclass\n"
        "task i2c_master_read_test::run_phase(uvm_phase phase);\n"
        "  rd_seq.start(mstr_env.mstr_agnt.sqr);\n"
        "endtask : run_phase\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    read_test = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_read_test")
    read_seq = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_read_seq")

    # test -> sequence via the member REFERENCES edge.
    refs = [e for e in graph.edges_from(read_test.id) if e.kind == IREdgeKind.REFERENCES]
    seq_refs = [e for e in refs if e.target_name == "i2c_master_read_seq"]
    assert len(seq_refs) == 1
    assert seq_refs[0].resolved is True
    assert seq_refs[0].target_id == read_seq.id

    # The external start() call produced only an unresolved CALLS edge; no node.
    run_phase = _method_node(graph, "i2c_master_read_test", "run_phase")
    calls = [e for e in graph.edges_from(run_phase.id) if e.kind == IREdgeKind.CALLS]
    start_calls = [e for e in calls if e.target_name == "start"]
    assert len(start_calls) == 1 and not start_calls[0].resolved
    assert all(n.name != "start" for n in graph.nodes_of_kind(IRNodeKind.SV_METHOD))


def test_same_field_name_in_unrelated_classes_does_not_cross_connect(tmp_path: Path):
    """R2E-A guard: two unrelated classes both declaring a member named ``sqr``
    must not cross-connect — member resolution is per-class, so the connect in
    each class resolves against its own member map."""
    _write(tmp_path, "TB/seq_a.svh", "class seq_a;\nendclass\n")
    _write(tmp_path, "TB/seq_b.svh", "class seq_b;\nendclass\n")
    _write(tmp_path, "TB/drv_a.svh", "class drv_a;\nendclass\n")
    _write(tmp_path, "TB/drv_b.svh", "class drv_b;\nendclass\n")
    _write(
        tmp_path, "TB/agents.svh",
        "class agent_a;\n"
        "  drv_a drv;\n"
        "  seq_a sqr;\n"
        "  extern function void connect_phase();\n"
        "endclass\n"
        "class agent_b;\n"
        "  drv_b drv;\n"
        "  seq_b sqr;\n"
        "  extern function void connect_phase();\n"
        "endclass\n"
        "function void agent_a::connect_phase();\n"
        "  drv.seq_item_port.connect(sqr.seq_item_export);\n"
        "endfunction : connect_phase\n"
        "function void agent_b::connect_phase();\n"
        "  drv.seq_item_port.connect(sqr.seq_item_export);\n"
        "endfunction : connect_phase\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    by_name = {n.name: n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS)}
    drv_a, drv_b = by_name["drv_a"], by_name["drv_b"]
    seq_a, seq_b = by_name["seq_a"], by_name["seq_b"]

    a_refs = [e for e in graph.edges_from(drv_a.id) if e.kind == IREdgeKind.REFERENCES]
    b_refs = [e for e in graph.edges_from(drv_b.id) if e.kind == IREdgeKind.REFERENCES]
    # drv_a's connect resolves to seq_a only; drv_b's to seq_b only.
    assert {e.target_id for e in a_refs if e.resolved} == {seq_a.id}
    assert {e.target_id for e in b_refs if e.resolved} == {seq_b.id}
    assert all(e.resolved for e in a_refs) and all(e.resolved for e in b_refs)


def test_duplicate_connect_edges_deduplicated(tmp_path: Path):
    """R2E-B: the same connect in both the extern prototype's merged body and
    an inline duplicate must produce a single edge, not two."""
    _write(tmp_path, "TB/sequencer.svh", "class i2c_master_sequencer;\nendclass\n")
    _write(tmp_path, "TB/driver.svh", "class i2c_master_driver;\nendclass\n")
    _write(
        tmp_path, "TB/agent.svh",
        "class i2c_master_agent;\n"
        "  i2c_master_driver drv;\n"
        "  i2c_master_sequencer sqr;\n"
        "  extern function void connect_phase(uvm_phase phase);\n"
        "  function void connect_phase(uvm_phase phase);\n"
        "    drv.seq_item_port.connect(sqr.seq_item_export);\n"
        "  endfunction : connect_phase\n"
        "endclass\n"
        "function void i2c_master_agent::connect_phase(uvm_phase phase);\n"
        "  drv.seq_item_port.connect(sqr.seq_item_export);\n"
        "endfunction : connect_phase\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    driver = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_driver")
    seq_refs = [
        e for e in graph.edges_from(driver.id)
        if e.kind == IREdgeKind.REFERENCES and e.target_name == "i2c_master_sequencer"
    ]
    assert len(seq_refs) == 1
    assert seq_refs[0].resolved is True


def test_connect_from_environment_to_scoreboard_and_coverage(tmp_path: Path):
    """R2E-B: environment's connect_phase wiring agent analysis ports to the
    scoreboard and coverage collector must create resolved edges
    i2c_master_agent -> i2c_master_scoreboard and i2c_master_agent ->
    i2c_master_coverage_collector."""
    _write(tmp_path, "TB/agent.svh", "class i2c_master_agent;\nendclass\n")
    _write(tmp_path, "TB/scoreboard.svh", "class i2c_master_scoreboard;\nendclass\n")
    _write(tmp_path, "TB/cov.svh", "class i2c_master_coverage_collector;\nendclass\n")
    _write(
        tmp_path, "TB/env.svh",
        "class i2c_master_env;\n"
        "  i2c_master_agent mstr_agnt;\n"
        "  i2c_master_scoreboard mstr_scb;\n"
        "  i2c_master_coverage_collector mstr_cov;\n"
        "  extern function void connect_phase(uvm_phase phase);\n"
        "endclass\n"
        "function void i2c_master_env::connect_phase(uvm_phase phase);\n"
        "  mstr_agnt.mon2agnt_port_2.connect(mstr_scb.scb_imp_addr);\n"
        "  mstr_agnt.mon2agnt_port_1.connect(mstr_cov.cov_imp_stat);\n"
        "endfunction : connect_phase\n",
    )
    store = InMemoryManifestStore()
    _manifest, graph = scan_parse_and_build_ir("proj1", str(tmp_path), store)

    agent = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_agent")
    scb = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_scoreboard")
    cov = next(n for n in graph.nodes_of_kind(IRNodeKind.SV_CLASS) if n.name == "i2c_master_coverage_collector")

    refs = [e for e in graph.edges_from(agent.id) if e.kind == IREdgeKind.REFERENCES]
    assert {e.target_id for e in refs if e.resolved} == {scb.id, cov.id}
