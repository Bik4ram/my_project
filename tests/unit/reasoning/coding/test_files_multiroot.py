"""Multi-root write/read safety in the coding files layer.

The docs root is read-only; writes may only target rtl/tb (or the legacy
single root). Area-prefixed relative paths resolve against the matching
area root; anything else falls back to the legacy root."""
from __future__ import annotations

from pathlib import Path

import pytest

from semrag.core.project_paths import ProjectPaths
from semrag.reasoning.coding.files import (
    ProjectFileError,
    read_file,
    safe_resolve_target,
    write_file,
)


@pytest.fixture
def paths(tmp_path: Path) -> ProjectPaths:
    rtl = tmp_path / "rtl_src"
    tb = tmp_path / "tb_src"
    docs = tmp_path / "docs"
    for p in (rtl, tb, docs):
        p.mkdir()
    return ProjectPaths(project_id="chip", rtl_root=rtl, tb_root=tb, docs_root=docs)


def test_write_to_rtl_area_allowed(paths: ProjectPaths):
    result = write_file(paths.infer_project_root(), "rtl/top.sv", "module top; endmodule\n", project_paths=paths)
    assert result["created"] is True
    assert (paths.rtl_root / "top.sv").read_text() == "module top; endmodule\n"


def test_write_to_tb_area_allowed(paths: ProjectPaths):
    write_file(paths.infer_project_root(), "tb/uvm/top_test.sv", "class top_test; endclass\n", project_paths=paths)
    assert (paths.tb_root / "uvm" / "top_test.sv").is_file()


def test_write_to_docs_area_rejected(paths: ProjectPaths):
    with pytest.raises(ProjectFileError) as excinfo:
        write_file(paths.infer_project_root(), "docs/spec.md", "# spec\n", project_paths=paths)
    assert "read-only" in str(excinfo.value)


def test_write_to_unknown_area_rejected(paths: ProjectPaths):
    with pytest.raises(ProjectFileError):
        write_file(paths.infer_project_root(), "firmware/boot.s", "x\n", project_paths=paths)


def test_read_from_docs_area_allowed(paths: ProjectPaths):
    (paths.docs_root / "spec.md").write_text("# spec\n")
    assert read_file(paths.infer_project_root(), "docs/spec.md", project_paths=paths) == "# spec\n"


def test_read_rejects_unknown_area(paths: ProjectPaths):
    with pytest.raises(ProjectFileError):
        read_file(paths.infer_project_root(), "firmware/boot.s", project_paths=paths)


def test_write_area_prefixed_path_does_not_escape_area(paths: ProjectPaths):
    # A prefixed path is anchored to its area root; traversal within is rejected.
    with pytest.raises(ProjectFileError):
        safe_resolve_target(paths.infer_project_root(), "rtl/../docs/escape.md", project_paths=paths)


def test_unprefixed_path_falls_back_to_legacy_root(tmp_path: Path):
    root = tmp_path / "legacy"
    root.mkdir()
    paths = ProjectPaths(project_id="p", root_path=root)
    result = write_file(root, "x.sv", "module x; endmodule\n", project_paths=paths)
    assert result["created"] is True
    assert (root / "x.sv").is_file()


def test_write_rejects_absolute_and_dotdot_even_with_paths(paths: ProjectPaths):
    with pytest.raises(ProjectFileError):
        write_file(paths.infer_project_root(), "/etc/passwd", "x\n", project_paths=paths)
    with pytest.raises(ProjectFileError):
        write_file(paths.infer_project_root(), "../x.sv", "x\n", project_paths=paths)