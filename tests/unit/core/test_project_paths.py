"""ProjectPaths: the portable multi-root path algebra and cwd detection.

One test per contract:

- area-prefixed relative paths resolve against the matching area root
- unprefixed paths fall back to the legacy ``root_path``
- a path with no covering root raises (never guesses)
- ``relativize`` emits portable ``rtl/<rel>`` / ``tb/<rel>`` / ``docs/<rel>``
- docs is read-only; rtl/tb (and the legacy root) are writable
- write/read safety rejects symlink and ``..`` escapes via ``resolve()``
- detection order: explicit config beats layout discovery; unconfigured
  backend with no matching layout returns None (never invents a project)
- explicit area roots may have no common parent (the portability point)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from semrag.core.project_paths import (
    ActiveProject,
    ProjectPathError,
    ProjectPaths,
    detect_active_project,
)


def _settings(**overrides):
    class _S:
        rtl_root = None
        tb_root = None
        docs_root = None
        project_id = None

    s = _S()
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


@pytest.fixture
def areas(tmp_path: Path):
    # Deliberately unrelated directories sharing NO parent: rtl under
    # tmp/a, tb under tmp/b, docs under tmp/c.
    rtl = tmp_path / "a" / "rtl_src"
    tb = tmp_path / "b" / "tb_src"
    docs = tmp_path / "c" / "docs"
    for p in (rtl, tb, docs):
        p.mkdir(parents=True)
    (rtl / "top.sv").write_text("module top; endmodule\n")
    (tb / "top_test.sv").write_text("class top_test; endclass\n")
    (docs / "spec.md").write_text("# spec\n")
    return ProjectPaths(
        project_id="chip",
        rtl_root=rtl,
        tb_root=tb,
        docs_root=docs,
    )


def test_resolve_area_prefixed_relative(areas: ProjectPaths):
    assert areas.resolve_relative("rtl/top.sv") == areas.rtl_root / "top.sv"
    assert areas.resolve_relative("tb/top_test.sv") == areas.tb_root / "top_test.sv"
    assert areas.resolve_relative("docs/spec.md") == areas.docs_root / "spec.md"


def test_resolve_unprefixed_falls_back_to_root_path(tmp_path: Path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "x.sv").write_text("module x; endmodule\n")
    paths = ProjectPaths(project_id="p", root_path=root)
    assert paths.resolve_relative("x.sv") == root / "x.sv"


def test_resolve_with_no_covering_root_raises(areas: ProjectPaths):
    with pytest.raises(ProjectPathError):
        areas.resolve_relative("firmware/boot.s")


def test_resolve_rejects_absolute_and_escape(areas: ProjectPaths):
    with pytest.raises(ProjectPathError):
        areas.resolve_relative("/etc/passwd")
    with pytest.raises(ProjectPathError):
        areas.resolve_relative("rtl/../tb/evil.sv")


def test_resolve_area_rejects_unknown_area(areas: ProjectPaths):
    with pytest.raises(ProjectPathError):
        areas.resolve_area("firmware", "x.sv")


def test_resolve_area_prefix_missing_area_root(tmp_path: Path):
    paths = ProjectPaths(project_id="p", rtl_root=tmp_path / "rtl")
    with pytest.raises(ProjectPathError):
        paths.resolve_relative("tb/x.sv")


def test_relativize_produces_area_prefixed_paths(areas: ProjectPaths):
    assert areas.relativize(areas.rtl_root / "top.sv") == "rtl/top.sv"
    assert areas.relativize(areas.tb_root / "top_test.sv") == "tb/top_test.sv"
    assert areas.relativize(areas.docs_root / "spec.md") == "docs/spec.md"
    assert areas.relativize(Path("/elsewhere/nope.sv")) is None


def test_area_for_assigns_area(areas: ProjectPaths):
    assert areas.area_for(areas.rtl_root / "top.sv") == "rtl"
    assert areas.area_for(areas.tb_root / "top_test.sv") == "tb"
    assert areas.area_for(areas.docs_root / "spec.md") == "docs"
    assert areas.area_for(Path("/elsewhere")) is None


def test_docs_is_read_only(areas: ProjectPaths):
    assert areas.is_write_allowed(areas.rtl_root / "new.sv") is True
    assert areas.is_write_allowed(areas.tb_root / "new_test.sv") is True
    assert areas.is_write_allowed(areas.docs_root / "new.md") is False
    assert areas.is_read_allowed(areas.docs_root / "new.md") is True


def test_legacy_root_is_writable(tmp_path: Path):
    root = tmp_path / "legacy"
    root.mkdir()
    paths = ProjectPaths(project_id="p", root_path=root)
    assert paths.is_write_allowed(root / "x.sv") is True
    assert paths.is_read_allowed(root / "x.sv") is True


def test_write_safety_rejects_outside_every_root(areas: ProjectPaths):
    assert areas.is_write_allowed(Path("/etc/passwd")) is False
    assert areas.is_read_allowed(Path("/etc/passwd")) is False


def test_infer_project_root_uses_root_path_when_present(tmp_path: Path):
    root = tmp_path / "legacy"
    paths = ProjectPaths(project_id="p", root_path=root)
    assert paths.infer_project_root() == root


def test_infer_project_root_with_unrelated_areas(tmp_path: Path):
    rtl = tmp_path / "a" / "rtl"
    tb = tmp_path / "b" / "tb"
    paths = ProjectPaths(project_id="p", rtl_root=rtl, tb_root=tb)
    assert paths.infer_project_root() == tmp_path


def test_detect_active_project_from_config(tmp_path: Path):
    rtl = tmp_path / "rtl"
    rtl.mkdir()
    settings = _settings(rtl_root=str(rtl), project_id="chip")
    # cwd is INSIDE the configured rtl root → the active area is rtl.
    active = detect_active_project(rtl, settings)
    assert active is not None
    assert active.project_id == "chip"
    assert active.area == "rtl"
    assert active.source == "config"


def test_detect_active_project_area_is_none_outside_roots(tmp_path: Path):
    rtl = tmp_path / "rtl"
    rtl.mkdir()
    settings = _settings(rtl_root=str(rtl), project_id="chip")
    active = detect_active_project(Path("/tmp"), settings)
    assert active is not None
    assert active.project_id == "chip"
    assert active.area is None


def test_detect_active_project_project_id_derived_when_unset(tmp_path: Path):
    rtl = tmp_path / "rtl"
    rtl.mkdir()
    settings = _settings(rtl_root=str(rtl))
    active = detect_active_project(rtl, settings)
    assert active is not None
    # Single configured root → the deepest common ancestor IS that root,
    # so the derived id is the root directory's name.
    assert active.project_id == "rtl"


def test_detect_active_project_falls_back_to_layout_discovery(tmp_path: Path):
    # No env roots configured, but cwd is inside a directory with rtl/ + tb/.
    proj = tmp_path / "my_chip"
    (proj / "rtl").mkdir(parents=True)
    (proj / "tb").mkdir()
    sub = proj / "tb" / "uvm"
    sub.mkdir()

    active = detect_active_project(sub, _settings())
    assert active is not None
    assert active.project_id == "my_chip"
    assert active.source == "discovery"
    assert active.paths.is_write_allowed(proj / "rtl" / "new.sv") is True


def test_detect_active_project_returns_none_when_unconfigured_and_no_layout(tmp_path: Path):
    # Nothing configured, no rtl/ + tb/ ancestor anywhere — must be None,
    # never a fabricated project.
    lone = tmp_path / "lone"
    lone.mkdir()
    assert detect_active_project(lone, _settings()) is None


def test_from_settings_returns_none_when_no_area_roots():
    assert ProjectPaths.from_settings(_settings()) is None


def test_from_single_root_shim(tmp_path: Path):
    root = tmp_path / "legacy"
    root.mkdir()
    paths = ProjectPaths.from_single_root("p", root)
    assert paths.project_id == "p"
    assert paths.root_path == root
    assert paths.has_areas is False


def test_to_dict_round_trip(areas: ProjectPaths):
    d = areas.to_dict()
    assert d["project_id"] == "chip"
    assert d["rtl_root"] == str(areas.rtl_root)
    assert d["tb_root"] == str(areas.tb_root)
    assert d["docs_root"] == str(areas.docs_root)
    assert d["root_path"] is None


def test_active_project_to_dict(areas: ProjectPaths):
    active = ActiveProject(project_id="chip", paths=areas, area="tb")
    d = active.to_dict()
    assert d["project_id"] == "chip"
    assert d["area"] == "tb"
    assert d["source"] == "config"
