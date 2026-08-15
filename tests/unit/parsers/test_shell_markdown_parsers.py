from datetime import datetime, timezone
from pathlib import Path

from semrag.ingestion.parsers.documents.markdown_parser import MarkdownParser
from semrag.ingestion.parsers.python_tcl.shell_parser import ShellParser
from semrag.ingestion.scanner.models import FileCategory, FileRecord


def _record(path: Path, category: FileCategory) -> FileRecord:
    return FileRecord(
        project_id="proj1",
        absolute_path=str(path),
        relative_path=path.name,
        size_bytes=path.stat().st_size,
        mtime=datetime.now(timezone.utc),
        content_hash="deadbeef",
        category=category,
        extension=path.suffix,
        is_binary=False,
    )


def test_shell_extracts_functions_and_source_refs(tmp_path: Path):
    f = tmp_path / "run_regression.sh"
    f.write_text(
        """#!/bin/bash
        source env_setup.sh
        . common_functions.sh

        run_test() {
            echo "running $1"
        }

        function cleanup() {
            echo "cleanup"
        }
        """
    )
    result = ShellParser().parse(f, _record(f, FileCategory.SHELL))
    assert result.success is True
    func_names = {fn["name"] for fn in result.payload["functions"]}
    assert func_names == {"run_test", "cleanup"}
    source_paths = {s["path"] for s in result.payload["source_refs"]}
    assert source_paths == {"env_setup.sh", "common_functions.sh"}


def test_markdown_extracts_heading_hierarchy(tmp_path: Path):
    f = tmp_path / "spec.md"
    f.write_text(
        """# I2C Master Specification

## Overview

Some text.

### Timing Requirements

More text.

## Register Map

```systemverilog
module foo;
endmodule
```
"""
    )
    result = MarkdownParser().parse(f, _record(f, FileCategory.MARKDOWN))
    assert result.success is True
    headings = result.payload["headings"]
    assert [(h["level"], h["text"]) for h in headings] == [
        (1, "I2C Master Specification"),
        (2, "Overview"),
        (3, "Timing Requirements"),
        (2, "Register Map"),
    ]
    assert result.payload["code_block_languages"] == ["systemverilog"]
