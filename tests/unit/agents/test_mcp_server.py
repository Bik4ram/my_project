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

    payload = json.loads(_run(semrag_mcp_server.index("rtl_uvm")))

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

    payload = json.loads(_run(semrag_mcp_server.index("rtl_uvm")))

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

    payload = json.loads(_run(semrag_mcp_server.index("rtl_uvm")))

    assert calls["n"] == 2
    assert payload["status"] == "completed"


def test_semrag_index_propagates_info_error(monkeypatch):
    async def fake_request(method, path, **kwargs):
        return "SemiRAG API error (401): nope"

    async def fake_request_index(method, path, **kwargs):
        raise AssertionError("index should not be called when info failed")

    monkeypatch.setattr(semrag_mcp_server, "_request", fake_request)
    monkeypatch.setattr(semrag_mcp_server, "_request_index", fake_request_index)

    result = _run(semrag_mcp_server.index("rtl_uvm"))

    assert result == "SemiRAG API error (401): nope"


def test_semrag_index_empty_project_id(monkeypatch):
    """An omitted/blank project_id is resolved from SEMRAG_CWD via the
    /resolve endpoint rather than rejected — the portable-context contract."""
    async def fake_request(method, path, **kwargs):
        if path == "/v1/projects/resolve":
            return {"project_id": "resolved-proj", "root_path": "/home/x/RTL_UVM"}
        if path == "/v1/projects/resolved-proj/info":
            return {"project_id": "resolved-proj", "root_path": "/home/x/RTL_UVM"}
        if path.endswith("/jobs/job-r"):
            return _completed_job("job-r", {"project_id": "resolved-proj"})
        raise AssertionError(f"unexpected _request path: {path}")

    async def fake_request_index(method, path, **kwargs):
        assert method == "POST"
        assert path == "/v1/projects/resolved-proj/index"
        assert kwargs["json"] == {"root_path": "/home/x/RTL_UVM"}
        return {"project_id": "resolved-proj", "job_id": "job-r"}

    monkeypatch.setattr(semrag_mcp_server, "_request", fake_request)
    monkeypatch.setattr(semrag_mcp_server, "_request_index", fake_request_index)

    payload = json.loads(_run(semrag_mcp_server.index("   ")))

    assert payload["project_id"] == "resolved-proj"
    assert payload["status"] == "completed"


def test_semrag_index_resolve_error_propagates(monkeypatch):
    """When /resolve fails and no project_id was supplied, the tool surfaces
    the resolution error instead of fabricating a project."""
    async def fake_request(method, path, **kwargs):
        return "SemiRAG API error (404): no project for this cwd"

    async def fake_request_index(method, path, **kwargs):
        raise AssertionError("index should not be called when resolve failed")

    monkeypatch.setattr(semrag_mcp_server, "_request", fake_request)
    monkeypatch.setattr(semrag_mcp_server, "_request_index", fake_request_index)

    result = _run(semrag_mcp_server.index("   "))

    assert "Error: no project_id supplied and /resolve failed" in result
    assert "no project for this cwd" in result

def test_semrag_index_sends_area_roots_for_multi_root_project(monkeypatch):
    """A multi-root project's /info roots are forwarded to the index
    payload, not just root_path."""
    async def fake_request(method, path, **kwargs):
        if path == "/v1/projects/chip/info":
            return {
                "project_id": "chip",
                "root_path": None,
                "rtl_root": "/srv/rtl",
                "tb_root": "/srv/tb",
                "docs_root": "/srv/docs",
            }
        if path.endswith("/jobs/job-4"):
            return _completed_job("job-4", {"project_id": "chip"})
        raise AssertionError(f"unexpected _request path: {path}")

    async def fake_request_index(method, path, **kwargs):
        assert method == "POST"
        assert path == "/v1/projects/chip/index"
        assert kwargs["json"] == {
            "rtl_root": "/srv/rtl",
            "tb_root": "/srv/tb",
            "docs_root": "/srv/docs",
        }
        return {"project_id": "chip", "job_id": "job-4"}

    monkeypatch.setattr(semrag_mcp_server, "_request", fake_request)
    monkeypatch.setattr(semrag_mcp_server, "_request_index", fake_request_index)

    payload = json.loads(_run(semrag_mcp_server.index("chip")))

    assert payload["project_id"] == "chip"
    assert payload["status"] == "completed"


def test_semrag_index_errors_when_no_root_available(monkeypatch):
    """If /info reports neither root_path nor any area root, index refuses
    to invent a path."""
    async def fake_request(method, path, **kwargs):
        if path == "/v1/projects/empty/info":
            return {"project_id": "empty", "root_path": None}
        raise AssertionError(f"unexpected _request path: {path}")

    async def fake_request_index(method, path, **kwargs):
        raise AssertionError("index should not be called without a root")

    monkeypatch.setattr(semrag_mcp_server, "_request", fake_request)
    monkeypatch.setattr(semrag_mcp_server, "_request_index", fake_request_index)

    result = _run(semrag_mcp_server.index("empty"))

    assert "cannot determine any project root" in result


def test_mcp_timeout_defaults_and_overrides(monkeypatch):
    import importlib

    monkeypatch.delenv("SEMRAG_MCP_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SEMRAG_MCP_INDEX_MAX_WAIT_SECONDS", raising=False)
    mod = importlib.reload(semrag_mcp_server)
    assert mod.SEMRAG_MCP_TIMEOUT_SECONDS == 7200.0
    assert mod.SEMRAG_MCP_INDEX_MAX_WAIT_SECONDS == 1800.0

    monkeypatch.setenv("SEMRAG_MCP_TIMEOUT_SECONDS", "3600")
    monkeypatch.setenv("SEMRAG_MCP_INDEX_MAX_WAIT_SECONDS", "900")
    mod = importlib.reload(semrag_mcp_server)
    assert mod.SEMRAG_MCP_TIMEOUT_SECONDS == 3600.0
    assert mod.SEMRAG_MCP_INDEX_MAX_WAIT_SECONDS == 900.0
