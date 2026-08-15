from datetime import datetime, timezone
from pathlib import Path

from semrag.ingestion.chunking.python_chunker import PythonChunker
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.python_tcl.python_parser import PythonParser
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _record(path: Path) -> FileRecord:
    return FileRecord(
        project_id="proj1", absolute_path=str(path), relative_path=f"src/{path.name}",
        size_bytes=path.stat().st_size, mtime=datetime.now(timezone.utc), content_hash="x",
        category=FileCategory.PYTHON, extension=".py", is_binary=False,
    )


def _build_ir_nodes(path: Path, record: FileRecord):
    parse_result = PythonParser().parse(path, record)
    graph = IRBuilder().build("proj1", {record.relative_path: record}, {record.relative_path: parse_result})
    return [n for n in graph.nodes.values() if n.relative_path == record.relative_path]


def test_module_overview_and_function_chunks(tmp_path: Path):
    f = tmp_path / "runner.py"
    f.write_text(
        '"""Runs the regression suite."""\n'
        "import os\n"
        "import sys\n"
        "\n"
        "\n"
        "def run_regression(seed):\n"
        '    """Runs one test."""\n'
        "    return True\n"
    )
    record = _record(f)
    ir_nodes = _build_ir_nodes(f, record)
    chunks = PythonChunker().chunk(record, None, f.read_text(), ir_nodes)

    chunk_types = {c.chunk_type: c for c in chunks}
    assert "py_module_overview" in chunk_types
    assert "import os" in chunk_types["py_module_overview"].text

    assert "py_function" in chunk_types
    assert "def run_regression" in chunk_types["py_function"].text


def test_small_class_produces_one_chunk(tmp_path: Path):
    f = tmp_path / "driver.py"
    f.write_text(
        "class I2CDriver:\n"
        "    def __init__(self):\n"
        "        pass\n"
        "\n"
        "    def send(self, data):\n"
        "        pass\n"
    )
    record = _record(f)
    ir_nodes = _build_ir_nodes(f, record)
    chunks = PythonChunker().chunk(record, None, f.read_text(), ir_nodes)

    class_chunks = [c for c in chunks if c.chunk_type == "py_class"]
    assert len(class_chunks) == 1
    assert "class I2CDriver" in class_chunks[0].text
    assert "def send" in class_chunks[0].text  # whole class, both methods included


def test_large_class_splits_into_signature_and_methods(tmp_path: Path):
    lines = ["class BigDriver:"]
    lines.append("    def __init__(self):")
    for i in range(120):
        lines.append(f"        # filler {i}")
        if i == 0:
            lines.append("        pass")
    lines.append("")
    lines.append("    def send(self, data):")
    lines.append("        pass")
    f = tmp_path / "big_driver.py"
    f.write_text("\n".join(lines) + "\n")

    record = _record(f)
    ir_nodes = _build_ir_nodes(f, record)
    chunks = PythonChunker().chunk(record, None, f.read_text(), ir_nodes)

    chunk_types = {c.chunk_type for c in chunks}
    assert "py_class_signature" in chunk_types
    method_chunks = [c for c in chunks if c.chunk_type == "py_method"]
    assert len(method_chunks) == 2
    method_names = {c.chunk_id.split("::")[-1] for c in method_chunks}
    assert any("__init__" in n for n in method_names)
    assert any("send" in n for n in method_names)


def test_no_overview_chunk_when_no_preamble_content(tmp_path: Path):
    f = tmp_path / "immediate.py"
    f.write_text("def foo():\n    pass\n")
    record = _record(f)
    ir_nodes = _build_ir_nodes(f, record)
    chunks = PythonChunker().chunk(record, None, f.read_text(), ir_nodes)

    assert all(c.chunk_type != "py_module_overview" for c in chunks)
