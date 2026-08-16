"""GET /v1/projects/resolve — portable context resolution.

Tests the ``resolve_project`` route handler directly (no HTTP/auth/Postgres
needed): detection order is config → layout discovery → registered
projects, and an unresolvable cwd must 404 rather than invent a project.

The route itself is exercised end-to-end by the test suite's live-DB API
tests (which are skipped when no test database is available, matching the
repo convention).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from semrag.api.api_key import APIKey
from semrag.api.routers.projects import resolve_project


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


def _fake_db(rows):
    class _Scalars:
        def all(self):
            return rows

    class _Result:
        def scalars(self):
            return _Scalars()

    class _Db:
        def execute(self, *args, **kwargs):
            return _Result()

    return _Db()


def _fake_key():
    return APIKey(key="x", name="test")


def _project_row(project_id, root_path=None, rtl_root=None, tb_root=None, docs_root=None):
    return SimpleNamespace(
        project_id=project_id,
        root_path=root_path,
        rtl_root=rtl_root,
        tb_root=tb_root,
        docs_root=docs_root,
    )


def test_resolve_via_config_roots(tmp_path):
    rtl = tmp_path / "rtl_src"
    rtl.mkdir()
    settings = _settings(rtl_root=str(rtl), project_id="chip")

    result = resolve_project(cwd=str(rtl), db=_fake_db([]), api_key=_fake_key(), settings=settings)

    assert result["project_id"] == "chip"
    assert result["source"] == "config"
    assert result["area"] == "rtl"


def test_resolve_via_registered_project_fallback(tmp_path):
    rtl = tmp_path / "rtl_src"
    tb = tmp_path / "tb_src"
    rtl.mkdir()
    tb.mkdir()
    rows = [
        _project_row("chip", root_path=str(rtl), rtl_root=str(rtl), tb_root=str(tb)),
    ]

    # cwd inside the tb root → not resolvable by config (none set) and not
    # by layout discovery (no rtl/ + tb/ ancestor), so the registered
    # project wins.
    result = resolve_project(cwd=str(tb), db=_fake_db(rows), api_key=_fake_key(), settings=_settings())

    assert result["project_id"] == "chip"
    assert result["source"] == "registered"
    assert result["area"] == "tb"


def test_resolve_config_wins_over_registered(tmp_path):
    rtl = tmp_path / "rtl_src"
    rtl.mkdir()
    rows = [_project_row("registered", root_path=str(rtl), rtl_root=str(rtl))]

    result = resolve_project(
        cwd=str(rtl),
        db=_fake_db(rows),
        api_key=_fake_key(),
        settings=_settings(rtl_root=str(rtl), project_id="chip"),
    )

    assert result["project_id"] == "chip"
    assert result["source"] == "config"


def test_resolve_unresolvable_cwd_raises_404(tmp_path):
    lone = tmp_path / "lone"
    lone.mkdir()

    with pytest.raises(HTTPException) as excinfo:
        resolve_project(cwd=str(lone), db=_fake_db([]), api_key=_fake_key(), settings=_settings())

    assert excinfo.value.status_code == 404
    assert "No project resolved" in excinfo.value.detail


def test_resolve_default_cwd_is_os_getcwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    rtl = tmp_path / "rtl_src"
    rtl.mkdir()

    result = resolve_project(cwd=None, db=_fake_db([]), api_key=_fake_key(), settings=_settings(rtl_root=str(rtl), project_id="chip"))

    assert result["project_id"] == "chip"