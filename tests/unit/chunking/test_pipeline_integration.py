from pathlib import Path

from semrag.ingestion.chunking.pipeline import chunk_project
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore
from semrag.ingestion.scanner.orchestrator import Scanner


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_full_pipeline_produces_chunks_across_languages(tmp_path: Path):
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, output logic scl);\n"
        "  assert property (@(posedge clk) scl == 1'b0) else $error(\"bad\");\n"
        "endmodule\n",
    )
    _write(
        tmp_path, "src/runner.py",
        '"""Regression runner."""\n'
        "import os\n"
        "\n"
        "\n"
        "def run():\n"
        "    pass\n",
    )
    _write(
        tmp_path, "docs/spec.md",
        "# I2C Spec\n\n## Overview\nThis describes the I2C master.\n",
    )
    _write(tmp_path, "Scripts/build.tcl", "proc build {} { puts hi }\n")
    _write(
        tmp_path, "sim/log/regression.log",
        "# UVM_INFO @ 0: reporter [RNTST] Running test top_test\n"
        "# UVM_ERROR @ 500: tb.uut [CHK] data mismatch\n"
        "#   property expected: 0x3f\n"
        "# UVM_INFO @ 600: reporter [DONE] Test finished\n",
    )

    manifest_store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=manifest_store)
    manifest, _ = scanner.scan("proj1", str(tmp_path))

    parser_registry = build_default_registry()
    parse_results = {}
    for relative_path, record in manifest.files.items():
        result = parser_registry.parse_file(tmp_path / relative_path, record)
        if result is not None:
            parse_results[relative_path] = result

    ir_graph = IRBuilder().build("proj1", manifest.files, parse_results)
    chunks = chunk_project(str(tmp_path), manifest.files, parse_results, ir_graph)

    chunks_by_path = {}
    for c in chunks:
        chunks_by_path.setdefault(c.relative_path, []).append(c)

    # SystemVerilog: structure-aware chunker used.
    sv_chunks = chunks_by_path["RTL/i2c_master.sv"]
    assert any(c.chunk_type == "sv_module" for c in sv_chunks)
    assert any(c.source_node_id is not None for c in sv_chunks)

    # Python: structure-aware chunker used.
    py_chunks = chunks_by_path["src/runner.py"]
    assert any(c.chunk_type == "py_function" for c in py_chunks)
    assert any(c.chunk_type == "py_module_overview" for c in py_chunks)

    # Markdown: section-aware chunker used, no IR node needed.
    md_chunks = chunks_by_path["docs/spec.md"]
    assert any(c.chunk_type == "markdown_section" for c in md_chunks)
    assert all(c.source_node_id is None for c in md_chunks)

    # TCL: no bespoke chunker registered — falls back to generic window.
    tcl_chunks = chunks_by_path["Scripts/build.tcl"]
    assert all(c.chunk_type == "generic_window" for c in tcl_chunks)

    # Logs: event-aware chunker used — the ERROR + continuation stays one chunk.
    log_chunks = chunks_by_path["sim/log/regression.log"]
    assert all(c.chunk_type == "log" for c in log_chunks)
    assert any("data mismatch" in c.text and "property expected" in c.text for c in log_chunks)

    # Every chunk_id across the whole project must be globally unique —
    # a collision here would mean Part 6's upsert silently overwrites one
    # chunk with another's embedding.
    all_ids = [c.chunk_id for c in chunks]
    assert len(all_ids) == len(set(all_ids))


def test_chunking_is_idempotent_across_runs(tmp_path: Path):
    _write(tmp_path, "RTL/top.sv", "module top(input logic clk); endmodule\n")

    manifest_store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=manifest_store)
    parser_registry = build_default_registry()

    def run():
        manifest, _ = scanner.scan("proj1", str(tmp_path))
        parse_results = {
            rel: r for rel, rec in manifest.files.items()
            if (r := parser_registry.parse_file(tmp_path / rel, rec)) is not None
        }
        ir_graph = IRBuilder().build("proj1", manifest.files, parse_results)
        return chunk_project(str(tmp_path), manifest.files, parse_results, ir_graph)

    chunks_1 = run()
    chunks_2 = run()

    assert {c.chunk_id for c in chunks_1} == {c.chunk_id for c in chunks_2}
