"""
Tests for middleware.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient

from semrag.api.middleware import (
    RequestIDMiddleware,
    LoggingMiddleware,
    SecurityHeadersMiddleware,
    setup_middleware,
    MaxBodySizeMiddleware,
)
from semrag.core.settings import get_settings


def test_request_id_middleware():
    """Test that the RequestIDMiddleware adds a request ID header."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    async def test_route():
        return {"message": "ok"}

    client = TestClient(app)
    response = client.get("/test")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    # Check that it's a UUID4
    import uuid
    try:
        uuid.UUID(response.headers["X-Request-ID"], version=4)
    except ValueError:
        pytest.fail("X-Request-ID is not a valid UUID4")


def test_security_headers_middleware():
    """Test that the SecurityHeadersMiddleware adds security headers."""
    app = FastAPI()
    # We need to set the settings for security headers
    from semrag.core.settings import get_settings
    settings = get_settings()
    settings.security_headers = {"X-Frame-Options": "DENY", "X-Content-Type-Options": "nosniff"}
    app.add_middleware(SecurityHeadersMiddleware)

    @app.get("/test")
    async def test_route():
        return {"message": "ok"}

    client = TestClient(app)
    response = client.get("/test")
    assert response.status_code == 200
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"


def test_setup_middleware():
    """Test that setup_middleware adds all the expected middleware."""
    app = FastAPI()
    setup_middleware(app)
    # Check that the middleware is added
    # We can't easily check the middleware, but we can test the effects
    # We'll test with a request and see if we get the expected headers and behavior
    client = TestClient(app)
    response = client.get("/test")  # This will 404, but we can still see the middleware effects
    # We expect to see the request ID header
    assert "X-Request-ID" in response.headers
    # We expect to see the security headers if they are set in settings
    settings = get_settings()
    for header, value in settings.security_headers.items():
        assert response.headers.get(header) == value
    # We expect to see the GZip middleware if enabled (it's not by default in the test)
    # We expect to see the trusted host middleware if set (it's not by default)
    # We expect to see the CORS middleware if set (it's not by default)


def test_logging_middleware():
    """Test that the LoggingMiddleware logs requests and audit events."""
    # We'll mock the audit logger to capture audit events
    with patch("semrag.audit.logger") as mock_audit_logger, patch("semrag.api.middleware.logger") as mock_logger:
        app = FastAPI()
        app.add_middleware(LoggingMiddleware)

        @app.get("/protected")
        async def protected_route(api_key = None):
            if api_key is None:
                from fastapi import HTTPException
                raise HTTPException(status_code=401, detail="Missing or invalid API key")
            return {"message": "ok"}

        client = TestClient(app)
        response = client.get("/protected")  # No Authorization header -> 401
        assert response.status_code == 401
        # Check that the middleware logger was called for the request
        assert mock_logger.info.called

        # The middleware logs an audit event for authentication failures
        found = False
        for call in mock_audit_logger.info.call_args_list:
            args, kwargs = call
            # stdlib `logger.info` accepts user fields only via `extra=...`.
            extra = kwargs.get("extra", {})
            if extra.get("event_type") == "authentication_failure":
                found = True
                break
        assert found, "Expected authentication failure audit event not found"


def test_max_body_size_middleware():
    """Test that the MaxBodySizeMiddleware rejects large requests."""
    app = FastAPI()
    # We need to set the max body size in settings
    from semrag.core.settings import get_settings
    settings = get_settings()
    settings.max_body_size = 10  # 10 bytes
    app.add_middleware(MaxBodySizeMiddleware)

    @app.post("/test")
    async def test_route():
        return {"message": "ok"}

    client = TestClient(app)
    # A request with a small body should succeed
    response = client.post("/test", data="hello")
    assert response.status_code == 200
    # A request with a large body should be rejected
    response = client.post("/test", data="x" * 20)
    assert response.status_code == 413
    assert "Request body too large" in response.text


def test_cors_middleware():
    """Test that the CORS middleware is added when settings.cors_origins is set."""
    app = FastAPI()
    # Set the CORS origins in settings
    from semrag.core.settings import get_settings
    settings = get_settings()
    settings.cors_origins = ["http://example.com"]
    setup_middleware(app)

    client = TestClient(app)
    # Make a request with an Origin header
    response = client.options(
        "/test",
        headers={"Origin": "http://example.com"},
    )
    # We expect the request to succeed (200 or 204) and to have the CORS headers
    # Note: the OPTIONS request might not be defined, but the TestClient will still send it
    # We'll just check that the response does not raise an exception and that we get some response
    # We can check for the Access-Control-Allow-Origin header
    # Actually, the TestClient will return a response for the OPTIONS request even if the route is not defined
    # because of the way the TestClient works.
    # We'll check that the Access-Control-Allow-Origin header is present
    assert response.headers.get("access-control-allow-origin") == "http://example.com"


def test_trusted_host_middleware():
    """Test that the TrustedHostMiddleware rejects requests with invalid hosts."""
    app = FastAPI()
    # Set the trusted hosts in settings
    from semrag.core.settings import get_settings
    settings = get_settings()
    settings.trusted_hosts = ["example.com", "localhost"]
    setup_middleware(app)

    client = TestClient(app)
    # Make a request with a trusted host
    response = client.get(
        "/test",
        headers={"Host": "example.com"},
    )
    # We expect the request to be processed (it will 404 because the route doesn't exist, but we don't get a 400 from the middleware)
    # Actually, the TrustedHostMiddleware will return a 400 if the host is not in the allowed list.
    # We'll check that the response is not a 400 for a trusted host.
    assert response.status_code != 400
    # Make a request with an untrusted host
    response = client.get(
        "/test",
        headers={"Host": "evil.com"},
    )
    # We expect a 400 Bad Request
    assert response.status_code == 400