from pathlib import Path

from semrag.ingestion.chunking.systemverilog_chunker import SystemVerilogChunker
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.systemverilog.parser import SystemVerilogParser
from semrag.ingestion.scanner.models import FileCategory, FileRecord
from datetime import datetime, timezone


def _record(path: Path, category=FileCategory.RTL_VERILOG) -> FileRecord:
    return FileRecord(
        project_id="proj1", absolute_path=str(path), relative_path=f"RTL/{path.name}",
        size_bytes=path.stat().st_size, mtime=datetime.now(timezone.utc), content_hash="x",
        category=category, extension=path.suffix, is_binary=False,
    )


def _build_ir_nodes(path: Path, record: FileRecord):
    parse_result = SystemVerilogParser().parse(path, record)
    graph = IRBuilder().build("proj1", {record.relative_path: record}, {record.relative_path: parse_result})
    return [n for n in graph.nodes.values() if n.relative_path == record.relative_path]


def test_small_module_produces_one_whole_chunk(tmp_path: Path):
    f = tmp_path / "i2c_master.sv"
    f.write_text(
        "module i2c_master(input logic clk, output logic scl);\n"
        "  logic start_condition;\n"
        "  always_ff @(posedge clk) start_condition <= 1'b0;\n"
        "endmodule\n"
    )
    record = _record(f)
    ir_nodes = _build_ir_nodes(f, record)
    chunks = SystemVerilogChunker().chunk(record, None, f.read_text(), ir_nodes)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "sv_module"
    assert "i2c_master" in chunks[0].text
    assert "endmodule" in chunks[0].text


def test_large_module_produces_signature_assertion_and_window_chunks(tmp_path: Path):
    lines = ["module big_module(input logic clk, output logic scl);"]
    lines.append("  logic start_condition;")
    # Pad the body well past the 150-line small-module threshold.
    for i in range(180):
        lines.append(f"  // filler line {i}")
    lines.append(
        "  assert property (@(posedge clk) start_condition |-> scl == 1'b0) else $error(\"failed\");"
    )
    lines.append("endmodule")
    f = tmp_path / "big_module.sv"
    f.write_text("\n".join(lines) + "\n")

    record = _record(f)
    ir_nodes = _build_ir_nodes(f, record)
    chunks = SystemVerilogChunker().chunk(record, None, f.read_text(), ir_nodes)

    chunk_types = {c.chunk_type for c in chunks}
    assert "sv_module_signature" in chunk_types
    assert "sv_assertion" in chunk_types
    assert "sv_body_window" in chunk_types

    signature_chunk = next(c for c in chunks if c.chunk_type == "sv_module_signature")
    assert "big_module" in signature_chunk.text

    assertion_chunk = next(c for c in chunks if c.chunk_type == "sv_assertion")
    assert "assert property" in assertion_chunk.text

    # Body windows must cover the whole module span between them (allowing
    # for overlap) — spot check first window starts at module start and
    # last window reaches module end.
    window_chunks = sorted((c for c in chunks if c.chunk_type == "sv_body_window"), key=lambda c: c.line_start)
    assert window_chunks[0].line_start == 1
    assert window_chunks[-1].line_end == len(lines)


def test_chunk_ids_are_deterministic(tmp_path: Path):
    f = tmp_path / "i2c_master.sv"
    f.write_text("module i2c_master(input logic clk); endmodule\n")
    record = _record(f)
    ir_nodes = _build_ir_nodes(f, record)

    chunks_1 = SystemVerilogChunker().chunk(record, None, f.read_text(), ir_nodes)
    chunks_2 = SystemVerilogChunker().chunk(record, None, f.read_text(), ir_nodes)

    assert [c.chunk_id for c in chunks_1] == [c.chunk_id for c in chunks_2]


def test_package_produces_whole_chunk(tmp_path: Path):
    f = tmp_path / "i2c_pkg.sv"
    f.write_text("package i2c_pkg;\n  typedef enum {IDLE, START} state_t;\nendpackage\n")
    record = _record(f)
    ir_nodes = _build_ir_nodes(f, record)
    chunks = SystemVerilogChunker().chunk(record, None, f.read_text(), ir_nodes)

    assert len(chunks) == 1
    assert chunks[0].chunk_type == "sv_package"
    assert "i2c_pkg" in chunks[0].text


def test_class_produces_one_chunk_linked_to_sv_class_node(tmp_path: Path):
    f = tmp_path / "i2c_master_read_seq.svh"
    f.write_text(
        "`ifndef I2C_MASTER_READ_SEQ_SVH\n"
        "`define I2C_MASTER_READ_SEQ_SVH\n"
        "\n"
        "class i2c_master_read_seq extends i2c_master_base_seq;\n"
        "    `uvm_object_utils(i2c_master_read_seq)\n"
        "    i2c_master_sequence_item item;\n"
        "    extern function new(string name = \"i2c_master_read_seq\");\n"
        "    extern task body();\n"
        "endclass :i2c_master_read_seq\n"
        "\n"
        "function i2c_master_read_seq::new(string name = \"i2c_master_read_seq\");\n"
        "    super.new(name);\n"
        "endfunction :new\n"
        "\n"
        "task i2c_master_read_seq::body();\n"
        "    item = i2c_master_sequence_item::type_id::create(\"item\");\n"
        "    start_item(item);\n"
        "    finish_item(item);\n"
        "endtask :body\n"
        "\n"
        "`endif\n"
    )
    record = _record(f, category=FileCategory.TESTBENCH_UVM)
    ir_nodes = _build_ir_nodes(f, record)
    chunks = SystemVerilogChunker().chunk(record, None, f.read_text(), ir_nodes)

    assert len(chunks) == 1
    class_chunk = chunks[0]
    assert class_chunk.chunk_type == "sv_class"
    assert class_chunk.symbol_name == "i2c_master_read_seq"
    assert class_chunk.source_node_id is not None

    class_node = next(n for n in ir_nodes if n.kind.name == "SV_CLASS")
    assert class_chunk.source_node_id == class_node.id
    assert class_chunk.relative_path == record.relative_path
    assert class_chunk.line_start == 4
    assert class_chunk.line_end == len(f.read_text().splitlines())
    assert "class i2c_master_read_seq" in class_chunk.text
    assert "endclass" in class_chunk.text
    assert "endtask :body" in class_chunk.text


def test_class_chunk_includes_extern_method_bodies_after_endclass(tmp_path: Path):
    """The class node span covers only class...endclass; the chunk must
    extend to EOF so out-of-class extern method bodies are not lost."""
    f = tmp_path / "seq.svh"
    f.write_text(
        "class seq extends base_seq;\n"
        "    extern task body();\n"
        "endclass :seq\n"
        "\n"
        "task seq::body();\n"
        "    start_item(item);\n"
        "    finish_item(item);\n"
        "endtask :body\n"
    )
    record = _record(f, category=FileCategory.TESTBENCH_UVM)
    ir_nodes = _build_ir_nodes(f, record)
    chunks = SystemVerilogChunker().chunk(record, None, f.read_text(), ir_nodes)

    assert len(chunks) == 1
    assert "start_item(item)" in chunks[0].text
    assert "endtask :body" in chunks[0].text


def test_class_chunk_stops_at_next_top_level_construct(tmp_path: Path):
    """A class chunk must not swallow a subsequent class/module — it ends
    the line before the next top-level construct."""
    f = tmp_path / "two_classes.svh"
    f.write_text(
        "class first;\n"
        "    extern task run();\n"
        "endclass :first\n"
        "\n"
        "task first::run();\n"
        "endtask :run\n"
        "\n"
        "class second;\n"
        "endclass :second\n"
    )
    record = _record(f, category=FileCategory.TESTBENCH_UVM)
    ir_nodes = _build_ir_nodes(f, record)
    chunks = SystemVerilogChunker().chunk(record, None, f.read_text(), ir_nodes)

    assert len(chunks) == 2
    first_chunk = next(c for c in chunks if c.symbol_name == "first")
    assert "task first::run()" in first_chunk.text
    assert "endtask :run" in first_chunk.text
    assert "class second" not in first_chunk.text
    assert first_chunk.line_end == 7

    second_chunk = next(c for c in chunks if c.symbol_name == "second")
    assert "class second" in second_chunk.text
    assert second_chunk.line_start == 8


def test_no_ir_nodes_produces_no_chunks(tmp_path: Path):
    f = tmp_path / "empty.sv"
    f.write_text("// just a comment, no module\n")
    record = _record(f)
    chunks = SystemVerilogChunker().chunk(record, None, f.read_text(), [])
    assert chunks == []
