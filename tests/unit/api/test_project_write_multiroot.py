"""Multi-root write safety at the API handler level.

The write/patch handlers must reject writes into the read-only docs root
and accept area-prefixed paths against the matching writable area root.
Handlers are exercised directly with a stub db (no HTTP/auth/Postgres)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from semrag.api.api_key import APIKey
from semrag.api.routers.projects import patch_file_content, write_file_content
from semrag.api.schemas import PatchFileRequest, WriteFileRequest


def _stub_db(row):
    class _Db:
        def execute(self, query, *args, **kwargs):
            if row is None:
                return SimpleNamespace(scalar_one_or_none=lambda: None)
            return SimpleNamespace(scalar_one_or_none=lambda: row)

    return _Db()


def _project_row(project_id="chip", root_path=None, rtl_root=None, tb_root=None, docs_root=None):
    return SimpleNamespace(
        project_id=project_id,
        root_path=root_path,
        rtl_root=rtl_root,
        tb_root=tb_root,
        docs_root=docs_root,
    )


def _fake_key():
    return APIKey(key="x", name="test")


@pytest.fixture
def rows(tmp_path: Path):
    rtl = tmp_path / "rtl_src"
    tb = tmp_path / "tb_src"
    docs = tmp_path / "docs"
    for p in (rtl, tb, docs):
        p.mkdir()
    return _project_row(
        project_id="chip",
        root_path=str(rtl),
        rtl_root=str(rtl),
        tb_root=str(tb),
        docs_root=str(docs),
    )


def test_write_area_prefixed_path_to_rtl(rows, tmp_path):
    resp = write_file_content(
        project_id="chip",
        request=WriteFileRequest(file_path="rtl/top.sv", content="module top; endmodule\n"),
        db=_stub_db(rows),
        api_key=_fake_key(),
        _rate_limit=None,
    )
    assert resp.created is True
    assert resp.relative_path == "rtl/top.sv"
    assert (tmp_path / "rtl_src" / "top.sv").is_file()


def test_write_to_docs_area_rejected_400(rows):
    with pytest.raises(HTTPException) as excinfo:
        write_file_content(
            project_id="chip",
            request=WriteFileRequest(file_path="docs/spec.md", content="# spec\n"),
            db=_stub_db(rows),
            api_key=_fake_key(),
            _rate_limit=None,
        )
    assert excinfo.value.status_code == 400
    assert "docs is read-only" in excinfo.value.detail


def test_write_unprefixed_path_falls_back_to_legacy_root(rows, tmp_path):
    # With a legacy root_path present, unprefixed paths resolve against it —
    # the single-root compatibility behaviour, unchanged.
    resp = write_file_content(
        project_id="chip",
        request=WriteFileRequest(file_path="firmware/boot.s", content="x\n"),
        db=_stub_db(rows),
        api_key=_fake_key(),
        _rate_limit=None,
    )
    assert resp.created is True
    assert (tmp_path / "rtl_src" / "firmware" / "boot.s").is_file()


def test_write_absolute_and_traversal_rejected(rows):
    for bad in ("/etc/passwd", "../escape.sv"):
        with pytest.raises(HTTPException) as excinfo:
            write_file_content(
                project_id="chip",
                request=WriteFileRequest(file_path=bad, content="x\n"),
                db=_stub_db(rows),
                api_key=_fake_key(),
                _rate_limit=None,
            )
        assert excinfo.value.status_code == 400


def test_patch_to_docs_area_rejected_400(rows):
    with pytest.raises(HTTPException) as excinfo:
        patch_file_content(
            project_id="chip",
            request=PatchFileRequest(file_path="docs/spec.md", old_text="a", new_text="b"),
            db=_stub_db(rows),
            api_key=_fake_key(),
            _rate_limit=None,
        )
    assert excinfo.value.status_code == 400


def test_write_unknown_project_404(rows):
    with pytest.raises(HTTPException) as excinfo:
        write_file_content(
            project_id="nope",
            request=WriteFileRequest(file_path="rtl/x.sv", content="x\n"),
            db=_stub_db(None),
            api_key=_fake_key(),
            _rate_limit=None,
        )
    assert excinfo.value.status_code == 404