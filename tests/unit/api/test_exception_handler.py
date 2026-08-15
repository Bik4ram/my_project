"""
Tests for the exception handler.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from semrag.api.exception_handler import add_exception_handlers


def test_http_exception_handler():
    """Test that the HTTP exception handler returns the expected format."""
    app = FastAPI()
    add_exception_handlers(app)

    @app.get("/test")
    async def test_route():
        raise StarletteHTTPException(status_code=404, detail="Not found")

    client = TestClient(app)
    response = client.get("/test")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "HTTP_404"
    assert data["error"]["message"] == "Not found"
    assert "request_id" in data["error"]
    assert data["error"]["details"] == {}


def test_validation_exception_handler():
    """Test that the validation exception handler returns the expected format."""
    app = FastAPI()
    add_exception_handlers(app)

    @app.get("/test")
    async def test_route(item_id: int):
        return {"item_id": item_id}

    client = TestClient(app)
    response = client.get("/test?item_id=not-an-integer")
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"
    assert data["error"]["message"] == "Invalid request parameters"
    assert "request_id" in data["error"]
    assert "errors" in data["error"]["details"]


def test_general_exception_handler():
    """Test that the general exception handler returns the expected format."""
    app = FastAPI()
    add_exception_handlers(app)

    @app.get("/test")
    async def test_route():
        raise ValueError("Something went wrong")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/test")
    assert response.status_code == 500
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert data["error"]["message"] == "An internal server error occurred"
    assert "request_id" in data["error"]
    assert data["error"]["details"] == {}