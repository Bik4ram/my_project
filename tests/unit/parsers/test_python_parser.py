from datetime import datetime, timezone
from pathlib import Path

from semrag.ingestion.parsers.python_tcl.python_parser import PythonParser
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _make_record(path: Path) -> FileRecord:
    return FileRecord(
        project_id="proj1",
        absolute_path=str(path),
        relative_path=f"Scripts/{path.name}",
        size_bytes=path.stat().st_size,
        mtime=datetime.now(timezone.utc),
        content_hash="deadbeef",
        category=FileCategory.PYTHON,
        extension=".py",
        is_binary=False,
    )


def test_extracts_imports_functions_classes(tmp_path: Path):
    f = tmp_path / "regression_runner.py"
    f.write_text(
        '''
"""Runs the I2C regression suite."""
import os
from pathlib import Path
from typing import Optional


def run_regression(seed: int, test_name: str) -> bool:
    """Runs one regression test."""
    return True


class RegressionRunner:
    """Orchestrates a batch of regression runs."""

    def __init__(self, config):
        self.config = config

    async def run_all(self):
        pass
'''
    )
    parser = PythonParser()
    result = parser.parse(f, _make_record(f))

    assert result.success is True
    payload = result.payload

    assert payload["module_docstring"] == "Runs the I2C regression suite."

    modules_imported = {i["module"] for i in payload["imports"]}
    assert "os" in modules_imported
    assert "pathlib" in modules_imported
    assert "typing" in modules_imported

    from_import = next(i for i in payload["imports"] if i["module"] == "pathlib")
    assert from_import["names"] == ["Path"]

    assert len(payload["functions"]) == 1
    func = payload["functions"][0]
    assert func["name"] == "run_regression"
    assert func["args"] == ["seed", "test_name"]
    assert func["docstring"] == "Runs one regression test."

    assert len(payload["classes"]) == 1
    cls = payload["classes"][0]
    assert cls["name"] == "RegressionRunner"
    method_names = {m["name"] for m in cls["methods"]}
    assert method_names == {"__init__", "run_all"}
    run_all = next(m for m in cls["methods"] if m["name"] == "run_all")
    assert run_all["is_async"] is True


def test_extracts_base_classes_and_decorators(tmp_path: Path):
    f = tmp_path / "driver.py"
    f.write_text(
        """
import cocotb

class I2CDriver(cocotb.handle.SimHandleBase):
    @property
    def sda(self):
        return None
"""
    )
    parser = PythonParser()
    result = parser.parse(f, _make_record(f))
    cls = result.payload["classes"][0]
    assert cls["base_classes"] == ["cocotb.handle.SimHandleBase"]
    method = cls["methods"][0]
    assert method["decorators"] == ["property"]


def test_syntax_error_fails_gracefully(tmp_path: Path):
    f = tmp_path / "broken.py"
    f.write_text("def broken(:\n    pass\n")
    parser = PythonParser()
    result = parser.parse(f, _make_record(f))
    assert result.success is False
    assert result.has_errors is True
    assert result.diagnostics[0].line is not None
