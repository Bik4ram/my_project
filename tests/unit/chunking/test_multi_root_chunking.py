"""Multi-root chunking: chunk_project stamps source_type/source_root from
the ProjectPaths and parses files from their area roots via absolute_path."""
from __future__ import annotations

from pathlib import Path

from semrag.core.project_paths import ProjectPaths
from semrag.ingestion.chunking.pipeline import chunk_project
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore
from semrag.ingestion.scanner.orchestrator import Scanner


def _build(tmp_path: Path):
    rtl = tmp_path / "rtl_src"
    tb = tmp_path / "tb_src"
    docs = tmp_path / "docs"
    for p in (rtl, tb, docs):
        p.mkdir(exist_ok=True)
    (rtl / "i2c_master.sv").write_text(
        "module i2c_master(input logic clk);\n"
        "  logic start_condition;\n"
        "  always_ff @(posedge clk) start_condition <= 1'b0;\n"
        "endmodule\n"
    )
    uvm = tb / "uvm"
    uvm.mkdir(exist_ok=True)
    (uvm / "i2c_test.sv").write_text("class i2c_test extends uvm_test; endclass\n")
    (docs / "spec.md").write_text("# I2C Spec\n\n## Overview\n\nThis is the I2C spec.\n")

    area_roots = {"rtl": str(rtl), "tb": str(tb), "docs": str(docs)}
    store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=store)
    manifest, _ = scanner.scan_multi_root("chip", area_roots)

    parser_registry = build_default_registry()
    parse_results = {}
    for rel, rec in manifest.files.items():
        parsed = parser_registry.parse_file(Path(rec.absolute_path), rec)
        if parsed is not None:
            parse_results[rel] = parsed

    ir_graph = IRBuilder().build("chip", manifest.files, parse_results)
    paths = ProjectPaths(
        project_id="chip",
        rtl_root=rtl,
        tb_root=tb,
        docs_root=docs,
    )
    return chunk_project(
        area_roots["rtl"], manifest.files, parse_results, ir_graph, project_paths=paths
    )


def test_multi_root_chunks_stamp_source_area(tmp_path: Path):
    chunks = _build(tmp_path)
    by_path = {}
    for c in chunks:
        by_path.setdefault(c.relative_path, c)
    assert "rtl/i2c_master.sv" in by_path
    assert "tb/uvm/i2c_test.sv" in by_path
    assert "docs/spec.md" in by_path

    assert by_path["rtl/i2c_master.sv"].source_type == "rtl"
    assert by_path["tb/uvm/i2c_test.sv"].source_type == "tb"
    assert by_path["docs/spec.md"].source_type == "docs"


def test_multi_root_source_roots_match_area_roots(tmp_path: Path):
    chunks = _build(tmp_path)
    rtl = tmp_path / "rtl_src"
    tb = tmp_path / "tb_src"
    docs = tmp_path / "docs"
    by_path = {}
    for c in chunks:
        by_path.setdefault(c.relative_path, c)
    assert by_path["rtl/i2c_master.sv"].source_root == str(rtl)
    assert by_path["tb/uvm/i2c_test.sv"].source_root == str(tb)
    assert by_path["docs/spec.md"].source_root == str(docs)


def test_multi_root_chunks_are_deterministic(tmp_path: Path):
    first = _build(tmp_path)
    second = _build(tmp_path)
    assert len(first) == len(second)
    for a, b in zip(sorted(first, key=lambda c: c.chunk_id),
                    sorted(second, key=lambda c: c.chunk_id), strict=True):
        assert a.chunk_id == b.chunk_id
        assert a.source_type == b.source_type
        assert a.source_root == b.source_root