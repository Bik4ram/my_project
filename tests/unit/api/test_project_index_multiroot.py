"""Multi-root index registration at the handler level.

``index_project`` must accept area roots, require at least one root, and
pass the roots through to ``job_manager.create_job``."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Response

from semrag.api.api_key import APIKey
from semrag.api.routers.projects import index_project
from semrag.api.schemas import IndexProjectRequest


def _fake_key():
    return APIKey(key="x", name="test")


def _args_for(tmp_path):
    rtl = tmp_path / "rtl_src"
    tb = tmp_path / "tb_src"
    rtl.mkdir()
    tb.mkdir()
    return rtl, tb


def test_index_requires_at_least_one_root(tmp_path):
    job_manager = MagicMock()
    with pytest.raises(HTTPException) as excinfo:
        index_project(
            project_id="chip",
            request=IndexProjectRequest(),
            response=Response(),
            db=MagicMock(),
            api_key=_fake_key(),
            _rate_limit=None,
            job_manager=job_manager,
            session_factory=MagicMock(),
            settings=MagicMock(),
            query_cache=MagicMock(),
        )
    assert excinfo.value.status_code == 400


def test_index_passes_area_roots_to_create_job(tmp_path):
    rtl, tb = _args_for(tmp_path)
    job = MagicMock()
    job.job_id = "job-1"
    job_manager = MagicMock()
    job_manager.create_job.return_value = (job, True)

    index_project(
        project_id="chip",
        request=IndexProjectRequest(rtl_root=str(rtl), tb_root=str(tb)),
        response=Response(),
        db=MagicMock(),
        api_key=_fake_key(),
        _rate_limit=None,
        job_manager=job_manager,
        session_factory=MagicMock(),
        settings=MagicMock(),
        query_cache=MagicMock(),
    )

    job_manager.create_job.assert_called_once()
    kwargs = job_manager.create_job.call_args.kwargs
    assert kwargs["project_id"] == "chip"
    assert kwargs["rtl_root"] == str(rtl)
    assert kwargs["tb_root"] == str(tb)
    assert kwargs["docs_root"] is None


def test_index_multi_root_plus_display_root(tmp_path):
    rtl, tb = _args_for(tmp_path)
    job = MagicMock()
    job.job_id = "job-2"
    job_manager = MagicMock()
    job_manager.create_job.return_value = (job, True)

    index_project(
        project_id="chip",
        request=IndexProjectRequest(root_path=str(tmp_path), rtl_root=str(rtl), tb_root=str(tb)),
        response=Response(),
        db=MagicMock(),
        api_key=_fake_key(),
        _rate_limit=None,
        job_manager=job_manager,
        session_factory=MagicMock(),
        settings=MagicMock(),
        query_cache=MagicMock(),
    )

    kwargs = job_manager.create_job.call_args.kwargs
    assert kwargs["root_path"] == str(tmp_path)
    assert kwargs["rtl_root"] == str(rtl)
    assert kwargs["tb_root"] == str(tb)


def test_index_rejects_nonexistent_area_root(tmp_path):
    job_manager = MagicMock()
    with pytest.raises(HTTPException) as excinfo:
        index_project(
            project_id="chip",
            request=IndexProjectRequest(rtl_root=str(tmp_path / "does-not-exist")),
            response=Response(),
            db=MagicMock(),
            api_key=_fake_key(),
            _rate_limit=None,
            job_manager=job_manager,
            session_factory=MagicMock(),
            settings=MagicMock(),
            query_cache=MagicMock(),
        )
    assert excinfo.value.status_code == 400
    job_manager.create_job.assert_not_called()