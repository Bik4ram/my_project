from semrag.ingestion.ir.models import IRNode, IRNodeKind
from semrag.ingestion.ir.symbol_tables import SymbolTables, _dotted_path_suffixes


def _module_node(project_id: str, path: str, name: str) -> IRNode:
    return IRNode(
        id=f"{project_id}::{path}::{name}",
        kind=IRNodeKind.SV_MODULE,
        project_id=project_id,
        relative_path=path,
        name=name,
    )


def test_unique_module_resolves():
    symbols = SymbolTables()
    node = _module_node("proj1", "RTL/i2c_master.sv", "i2c_master")
    symbols.index_node(node)

    resolved_id, ambiguous, candidates = symbols.resolve(symbols.sv_module_by_name, "i2c_master")
    assert resolved_id == node.id
    assert ambiguous is False


def test_duplicate_module_name_is_ambiguous_not_silently_picked():
    symbols = SymbolTables()
    node_a = _module_node("proj1", "RTL/fifo_v1.sv", "fifo")
    node_b = _module_node("proj1", "RTL/subsystem/fifo_v2.sv", "fifo")
    symbols.index_node(node_a)
    symbols.index_node(node_b)

    resolved_id, ambiguous, candidates = symbols.resolve(symbols.sv_module_by_name, "fifo")
    assert resolved_id is None
    assert ambiguous is True
    assert set(candidates) == {node_a.id, node_b.id}


def test_unknown_name_is_unresolved_but_not_ambiguous():
    symbols = SymbolTables()
    resolved_id, ambiguous, candidates = symbols.resolve(symbols.sv_module_by_name, "does_not_exist")
    assert resolved_id is None
    assert ambiguous is False
    assert candidates == []


def test_dotted_path_suffixes_strip_src_root():
    suffixes = _dotted_path_suffixes("src/semrag/ingestion/scanner/hashing.py")
    assert suffixes == [
        "semrag.ingestion.scanner.hashing",
        "ingestion.scanner.hashing",
        "scanner.hashing",
        "hashing",
    ]


def test_dotted_path_suffixes_without_src_root():
    suffixes = _dotted_path_suffixes("scripts/regression_runner.py")
    assert suffixes == ["scripts.regression_runner", "regression_runner"]
