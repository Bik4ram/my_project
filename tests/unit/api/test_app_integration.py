"""
Full API integration tests - real FastAPI app, real HTTP-shaped requests
(via TestClient), real Postgres. This is what proves Parts 2-11 are
actually reachable through the API layer, not just callable as Python
functions in a test file.
"""
from pathlib import Path

from fastapi.testclient import TestClient

from semrag.api.app import create_app
from semrag.api.dependencies import get_db_session
from semrag.api.rate_limit import MemoryRateLimiter, get_rate_limiter


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _index_sample_project(api_client, auth_headers, tmp_path: Path, project_id: str = "proj1") -> None:
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, output logic scl);\n"
        "  assert property (@(posedge clk) scl == 1'b0)\n"
        "    else $error(\"start condition assertion failed\");\n"
        "endmodule\n",
    )
    response = api_client.post(
        f"/v1/projects/{project_id}/index", json={"root_path": str(tmp_path)}, headers=auth_headers
    )
    assert response.status_code == 200, response.text


def test_health_check_requires_no_auth(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_protected_endpoint_rejects_missing_auth(api_client):
    response = api_client.get("/v1/projects/proj1/files")
    assert response.status_code == 401


def test_protected_endpoint_rejects_wrong_key(api_client):
    response = api_client.get("/v1/projects/proj1/files", headers={"Authorization": "Bearer wrong-key"})
    assert response.status_code == 401


def test_index_project_returns_file_and_chunk_counts(api_client, auth_headers, tmp_path: Path):
    _index_sample_project(api_client, auth_headers, tmp_path)
    response = api_client.post(
        "/v1/projects/proj1/index", json={"root_path": str(tmp_path)}, headers=auth_headers
    )
    body = response.json()
    assert body["project_id"] == "proj1"
    assert body["file_count"] >= 1
    assert body["chunk_count"] >= 1


def test_list_files_after_indexing(api_client, auth_headers, tmp_path: Path):
    _index_sample_project(api_client, auth_headers, tmp_path)
    response = api_client.get("/v1/projects/proj1/files", headers=auth_headers)
    assert response.status_code == 200
    files = response.json()["files"]
    assert any(f["relative_path"] == "RTL/i2c_master.sv" for f in files)


def test_query_returns_grounded_messages_with_citations(api_client, auth_headers, tmp_path: Path):
    _index_sample_project(api_client, auth_headers, tmp_path)
    response = api_client.post(
        "/v1/projects/proj1/query",
        json={"query": "why does the start condition assertion fail"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["from_cache"] is False
    final_message = body["messages"][-1]["content"]
    assert "[Source: RTL/i2c_master.sv" in final_message
    assert len(body["included_chunks"]) > 0


def test_repeated_query_hits_cache(api_client, auth_headers, tmp_path: Path):
    _index_sample_project(api_client, auth_headers, tmp_path)
    query_payload = {"query": "why does the start condition assertion fail"}

    first = api_client.post("/v1/projects/proj1/query", json=query_payload, headers=auth_headers)
    assert first.json()["from_cache"] is False

    second = api_client.post("/v1/projects/proj1/query", json=query_payload, headers=auth_headers)
    assert second.json()["from_cache"] is True
    assert second.json()["messages"] == first.json()["messages"]


def test_reindexing_invalidates_cache(api_client, auth_headers, tmp_path: Path):
    _index_sample_project(api_client, auth_headers, tmp_path)
    query_payload = {"query": "why does the start condition assertion fail"}

    first = api_client.post("/v1/projects/proj1/query", json=query_payload, headers=auth_headers)
    assert first.json()["from_cache"] is False
    cached = api_client.post("/v1/projects/proj1/query", json=query_payload, headers=auth_headers)
    assert cached.json()["from_cache"] is True

    api_client.post("/v1/projects/proj1/index", json={"root_path": str(tmp_path)}, headers=auth_headers)

    after_reindex = api_client.post("/v1/projects/proj1/query", json=query_payload, headers=auth_headers)
    assert after_reindex.json()["from_cache"] is False


def test_streaming_query_emits_stage_events_then_result(api_client, auth_headers, tmp_path: Path):
    _index_sample_project(api_client, auth_headers, tmp_path)

    with api_client.stream(
        "POST", "/v1/projects/proj1/query/stream",
        json={"query": "why does the start condition assertion fail"},
        headers=auth_headers,
    ) as response:
        assert response.status_code == 200
        raw_events = "".join(response.iter_text())

    assert "event: stage" in raw_events
    assert "event: result" in raw_events
    assert "start condition assertion failed" in raw_events


def test_query_request_validates_empty_query(api_client, auth_headers, tmp_path: Path):
    _index_sample_project(api_client, auth_headers, tmp_path)
    response = api_client.post("/v1/projects/proj1/query", json={"query": ""}, headers=auth_headers)
    assert response.status_code == 422


def test_rate_limit_enforced_via_dependency_override(db_session, auth_headers, monkeypatch, tmp_path: Path):
    """Uses its own app instance with get_rate_limiter overridden to a
    deliberately tiny limit, rather than relying on the module-level
    singleton (whose limit is fixed at import time from the environment)
    - a clean, real way to test the enforcement wiring without fighting
    import order."""
    monkeypatch.setenv("SEMRAG_API_KEYS", auth_headers["Authorization"].removeprefix("Bearer "))

    def _override_db():
        yield db_session

    # A single shared instance, not a factory that builds a fresh one per
    # call — a rate limiter whose state resets on every dependency
    # resolution isn't a rate limiter. Caught by this test actually
    # failing against the first (lambda-per-call) version: the second
    # request got a brand-new, empty-history limiter instead of the same
    # one the first request had already used up.
    tiny_limiter = MemoryRateLimiter(requests_per_window=1)

    app = create_app()
    app.dependency_overrides[get_db_session] = _override_db
    app.dependency_overrides[get_rate_limiter] = lambda: tiny_limiter

    with TestClient(app) as limited_client:
        first = limited_client.get("/v1/projects/proj1/files", headers=auth_headers)
        second = limited_client.get("/v1/projects/proj1/files", headers=auth_headers)

    assert first.status_code == 200
    assert second.status_code == 429
