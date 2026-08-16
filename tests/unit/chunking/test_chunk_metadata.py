"""RI-06 chunk metadata: the enrichment contract, one test per requirement.

Covers language/category/parser_name/chunker_name/symbol_name assignment,
the derived ``is_generated`` property, determinism, existing-chunk
backward compatibility, and the no-symbol rule for generic text.
"""
from pathlib import Path

from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.chunking.pipeline import chunk_project
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore
from semrag.ingestion.scanner.orchestrator import Scanner


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _chunks_for(tmp_path: Path, files: dict[str, str]) -> list[Chunk]:
    for rel, content in files.items():
        _write(tmp_path, rel, content)

    manifest_store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=manifest_store)
    manifest, _ = scanner.scan("proj1", str(tmp_path))

    parser_registry = build_default_registry()
    parse_results = {
        rel: r for rel, rec in manifest.files.items()
        if (r := parser_registry.parse_file(tmp_path / rel, rec)) is not None
    }
    ir_graph = IRBuilder().build("proj1", manifest.files, parse_results)
    return chunk_project(str(tmp_path), manifest.files, parse_results, ir_graph)


def _make_sv_chunks(tmp_path: Path) -> list[Chunk]:
    return _chunks_for(
        tmp_path,
        {
            "RTL/i2c_master.sv": (
                "module i2c_master(input logic clk, output logic scl);\n"
                "  logic start_condition;\n"
                "  always_ff @(posedge clk) start_condition <= 1'b0;\n"
                "endmodule\n"
            )
        },
    )


def test_sv_metadata(tmp_path: Path):
    chunks = _make_sv_chunks(tmp_path)
    assert chunks
    for c in chunks:
        assert c.language == "systemverilog"
        assert c.category == "rtl"  # Source Authority classify_document reuse
        assert c.parser_name is not None
        assert c.chunker_name == "systemverilog"
        assert c.symbol_name == "i2c_master"
        assert c.source_node_id is not None
        assert c.line_start is not None and c.line_end is not None
        assert c.is_generated is False


def test_generated_category_drives_is_generated(tmp_path: Path):
    chunks = _chunks_for(
        tmp_path,
        {
            "generated/gen_axi.sv": "module gen_axi; endmodule\n",
        },
    )
    assert chunks
    for c in chunks:
        assert c.category == "generated"
        assert c.is_generated is True


def test_non_generated_category_yields_false(tmp_path: Path):
    chunks = _make_sv_chunks(tmp_path)
    for c in chunks:
        assert c.category != "generated"
        assert c.is_generated is False


def test_python_metadata(tmp_path: Path):
    chunks = _chunks_for(
        tmp_path,
        {
            "src/runner.py": (
                '"""Regression runner."""\n'
                "import os\n"
                "\n"
                "\n"
                "def run_all():\n"
                "    pass\n"
            )
        },
    )
    assert chunks
    by_type = {c.chunk_type: c for c in chunks}
    assert by_type["py_module_overview"].language == "python"
    assert by_type["py_module_overview"].category == "python"
    assert by_type["py_module_overview"].symbol_name == "runner"
    assert by_type["py_function"].symbol_name == "run_all"
    assert by_type["py_function"].chunker_name == "python"
    assert by_type["py_function"].parser_name is not None


def test_markdown_metadata_no_symbol(tmp_path: Path):
    chunks = _chunks_for(
        tmp_path,
        {"README.md": "# I2C Spec\n\n## Overview\nThis describes the I2C master.\n"},
    )
    assert chunks
    for c in chunks:
        assert c.language == "markdown"
        assert c.category == "markdown"
        assert c.chunker_name == "markdown"
        assert c.symbol_name is None  # no invented symbols for prose
        assert c.parser_name is not None


def test_markdown_path_rule_beats_chunk_type_rule(tmp_path: Path):
    """Source Authority semantics: a path rule (docs/ -> design_documentation)
    wins over the chunk-type rule — the category is whatever the canonical
    classifier says, not a copy of the chunk_type."""
    chunks = _chunks_for(
        tmp_path,
        {"docs/spec.md": "# I2C Spec\n\n## Overview\nThis describes the I2C master.\n"},
    )
    assert chunks
    for c in chunks:
        assert c.category == "design_documentation"


def test_tcl_fallback_generic_chunker(tmp_path: Path):
    chunks = _chunks_for(tmp_path, {"Scripts/build.tcl": "proc build {} { puts hi }\n"})
    assert chunks
    for c in chunks:
        assert c.language == "tcl"
        assert c.category == "other"  # no source-authority rule matches
        assert c.chunker_name == "generic_window"  # fallback chunker stamped
        assert c.symbol_name is None  # generic chunker has no symbol


def test_log_metadata(tmp_path: Path):
    chunks = _chunks_for(
        tmp_path,
        {
            "sim/log/regression.log": (
                "# UVM_INFO @ 0: reporter [RNTST] Running test top_test\n"
                "# UVM_ERROR @ 500: tb.uut [CHK] data mismatch\n"
                "#   property expected: 0x3f\n"
                "# UVM_INFO @ 600: reporter [DONE] Test finished\n"
            )
        },
    )
    assert chunks
    for c in chunks:
        assert c.language == "log"
        assert c.category == "simulation_log"
        assert c.chunker_name == "log"
        assert c.symbol_name is None
        assert c.source_node_id is None


def test_line_metadata_remains_intact(tmp_path: Path):
    chunks = _make_sv_chunks(tmp_path)
    for c in chunks:
        assert c.line_start >= 1
        assert c.line_end >= c.line_start
        assert c.line_end - c.line_start + 1 == len(c.text.splitlines())


def test_metadata_is_deterministic_across_runs(tmp_path: Path):
    first = _make_sv_chunks(tmp_path)
    second = _make_sv_chunks(tmp_path)
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert a.chunk_id == b.chunk_id
        assert a.language == b.language
        assert a.category == b.category
        assert a.parser_name == b.parser_name
        assert a.chunker_name == b.chunker_name
        assert a.symbol_name == b.symbol_name


def test_existing_chunks_constructed_without_new_fields_remain_compatible():
    legacy = Chunk(
        chunk_id="c1",
        project_id="p1",
        relative_path="RTL/legacy.sv",
        chunk_type="sv_module",
        text="module legacy; endmodule",
    )
    assert legacy.language == ""
    assert legacy.category == ""
    assert legacy.parser_name is None
    assert legacy.chunker_name == ""
    assert legacy.symbol_name is None
    assert legacy.is_generated is False  # derived from empty category


def test_explicitly_generated_category_without_extra_flag():
    chunk = Chunk(
        chunk_id="c1",
        project_id="p1",
        relative_path="gen/thing.sv",
        chunk_type="sv_module",
        text="module thing; endmodule",
        category="generated",
    )
    assert chunk.is_generated is True
    # The property reads the category — no second flag to keep in sync.
    assert "generated" not in chunk.model_fields_set or chunk.model_dump().get("is_generated") is not True
