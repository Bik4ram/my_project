"""Unit tests for the SemiRAG MCP server tools.

The MCP server module lives at the repository root (it is not part of the
``semrag`` package), so its test doubles the HTTP layer via monkeypatching
and drives the tool coroutines directly with asyncio.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # noqa: E402

import semrag_mcp_server  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _completed_job(job_id, extra=None):
    payload = {
        "job_id": job_id,
        "project_id": "rtl_uvm",
        "state": "completed",
        "progress": 100,
        "current_stage": "persisting",
        "duration": 3.2,
        "error": None,
        "statistics": {"files_indexed": 5},
    }
    if extra:
        payload.update(extra)
    return payload


def test_semrag_index_reports_completed_job(monkeypatch):
    async def fake_request(method, path, **kwargs):
        if path == "/v1/projects/rtl_uvm/info":
            return {"project_id": "rtl_uvm", "root_path": "/home/x/RTL_UVM"}
        if path.endswith("/jobs/job-1"):
            return _completed_job("job-1")
        raise AssertionError(f"unexpected _request path: {path}")

    async def fake_request_index(method, path, **kwargs):
        assert method == "POST"
        assert path == "/v1/projects/rtl_uvm/index"
        assert kwargs["json"] == {"root_path": "/home/x/RTL_UVM"}
        return {"project_id": "rtl_uvm", "job_id": "job-1"}

    monkeypatch.setattr(semrag_mcp_server, "_request", fake_request)
    monkeypatch.setattr(semrag_mcp_server, "_request_index", fake_request_index)

    payload = json.loads(_run(semrag_mcp_server.semrag_index("rtl_uvm")))

    assert payload["project_id"] == "rtl_uvm"
    assert payload["job_id"] == "job-1"
    assert payload["status"] == "completed"
    assert payload["statistics"]["files_indexed"] == 5


def test_semrag_index_reports_failed_job(monkeypatch):
    async def fake_request(method, path, **kwargs):
        if path == "/v1/projects/rtl_uvm/info":
            return {"project_id": "rtl_uvm", "root_path": "/home/x/RTL_UVM"}
        if path.endswith("/jobs/job-2"):
            return {
                "job_id": "job-2",
                "project_id": "rtl_uvm",
                "state": "failed",
                "progress": 40,
                "current_stage": "chunking",
                "error": "boom",
                "statistics": {},
            }
        raise AssertionError(f"unexpected _request path: {path}")

    async def fake_request_index(method, path, **kwargs):
        return {"project_id": "rtl_uvm", "job_id": "job-2"}

    monkeypatch.setattr(semrag_mcp_server, "_request", fake_request)
    monkeypatch.setattr(semrag_mcp_server, "_request_index", fake_request_index)

    payload = json.loads(_run(semrag_mcp_server.semrag_index("rtl_uvm")))

    assert payload["status"] == "failed"
    assert payload["error"] == "boom"


def test_semrag_index_polls_until_terminal_state(monkeypatch):
    async def fake_sleep(_seconds):
        return None

    calls = {"n": 0}

    async def fake_request(method, path, **kwargs):
        if path == "/v1/projects/rtl_uvm/info":
            return {"project_id": "rtl_uvm", "root_path": "/home/x/RTL_UVM"}
        if path.endswith("/jobs/job-3"):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "job_id": "job-3",
                    "project_id": "rtl_uvm",
                    "state": "queued",
                    "progress": 0,
                }
            return _completed_job("job-3")
        raise AssertionError(f"unexpected _request path: {path}")

    async def fake_request_index(method, path, **kwargs):
        return {"project_id": "rtl_uvm", "job_id": "job-3"}

    monkeypatch.setattr(semrag_mcp_server, "_request", fake_request)
    monkeypatch.setattr(semrag_mcp_server, "_request_index", fake_request_index)
    monkeypatch.setattr(semrag_mcp_server.asyncio, "sleep", fake_sleep)

    payload = json.loads(_run(semrag_mcp_server.semrag_index("rtl_uvm")))

    assert calls["n"] == 2
    assert payload["status"] == "completed"


def test_semrag_index_propagates_info_error(monkeypatch):
    async def fake_request(method, path, **kwargs):
        return "SemiRAG API error (401): nope"

    async def fake_request_index(method, path, **kwargs):
        raise AssertionError("index should not be called when info failed")

    monkeypatch.setattr(semrag_mcp_server, "_request", fake_request)
    monkeypatch.setattr(semrag_mcp_server, "_request_index", fake_request_index)

    result = _run(semrag_mcp_server.semrag_index("rtl_uvm"))

    assert result == "SemiRAG API error (401): nope"


def test_semrag_index_empty_project_id(monkeypatch):
    result = _run(semrag_mcp_server.semrag_index("   "))
    assert result == "Error: project_id cannot be empty."