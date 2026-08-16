"""Multi-root validation targets: rtl:/tb:/docs: prefixes and bare
first-segment mapping resolve against the matching area root."""
from __future__ import annotations

from pathlib import Path

import pytest

from semrag.core.project_paths import ProjectPaths
from semrag.validation.config import ValidationConfig
from semrag.validation.runner import build_argv


@pytest.fixture
def multi(tmp_path: Path) -> tuple[ProjectPaths, Path]:
    rtl = tmp_path / "rtl_src"
    tb = tmp_path / "tb_src"
    docs = tmp_path / "docs"
    for p in (rtl, tb, docs):
        p.mkdir()
    (rtl / "i2c_master.sv").write_text("module i2c_master; endmodule\n")
    uvm = tb / "uvm"
    uvm.mkdir()
    (uvm / "i2c_test.sv").write_text("class i2c_test; endclass\n")
    paths = ProjectPaths(project_id="chip", rtl_root=rtl, tb_root=tb, docs_root=docs)
    return paths, tmp_path


def _cfg(files, incdirs=None):
    return ValidationConfig(
        project_id="chip",
        tool="verilator",
        targets={"t": {"tool_args": ["--lint-only"], "incdirs": incdirs or [], "files": files}},
    )


def test_explicit_area_prefix_resolves_against_area_root(multi):
    paths, _ = multi
    argv = build_argv(paths.infer_project_root(), _cfg(["rtl:i2c_master.sv"], ["rtl:."]), "t", project_paths=paths)
    assert str(paths.rtl_root / "i2c_master.sv") in argv
    assert f"+incdir+{paths.rtl_root}" in argv


def test_bare_first_segment_mapping(multi):
    paths, _ = multi
    argv = build_argv(paths.infer_project_root(), _cfg(["tb/uvm/i2c_test.sv"], ["tb/uvm"]), "t", project_paths=paths)
    assert str(paths.tb_root / "uvm" / "i2c_test.sv") in argv
    assert f"+incdir+{paths.tb_root / 'uvm'}" in argv


def test_area_prefix_escaping_area_root_rejected(multi):
    paths, _ = multi
    with pytest.raises(ValueError):
        build_argv(paths.infer_project_root(), _cfg(["rtl:../docs/spec.md"]), "t", project_paths=paths)


def test_multi_root_target_without_project_paths_uses_legacy_fallback(multi, tmp_path):
    # Without project_paths the resolver falls back to the working root —
    # the pre-refactor single-root behaviour.
    paths, _ = multi
    with pytest.raises(FileNotFoundError):
        build_argv(tmp_path, _cfg(["rtl/i2c_master.sv"]), "t")


def test_unprefixed_path_resolves_via_legacy_root_path(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "x.sv").write_text("module x; endmodule\n")
    paths = ProjectPaths(project_id="p", root_path=root)
    argv = build_argv(root, _cfg(["x.sv"]), "t", project_paths=paths)
    assert str(root / "x.sv") in argv