"""
Tests for audit logging.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import Request

from semrag.audit import log_audit_event


def test_log_audit_event():
    """Test the log_audit_event function."""
    with patch("semrag.audit.logger") as mock_logger:
        # Create a mock request object
        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [[b"user-agent", b"test-agent"]],
                "query_string": b"",
            }
        )
        # We need to set the request state for the request ID
        request.state.request_id = "test-request-id"

        log_audit_event(
            event_type="test_event",
            request=request,
            project_id="proj1",
            api_key_id="key1",
            success=True,
            error_message=None,
            custom_field="custom_value",
        )

        # Check that the logger was called
        assert mock_logger.info.called
        # Get the arguments of the call
        args, kwargs = mock_logger.info.call_args
        # The first argument is the message, the second is the kwargs
        assert args[0] == "Audit event"
        # stdlib `logger.info` accepts user fields only via `extra=...`,
        # so audit fields land under kwargs["extra"].
        extra = kwargs.get("extra", {})
        assert extra.get("event_type") == "test_event"
        assert extra.get("project_id") == "proj1"
        assert extra.get("api_key_id") == "key1"
        assert extra.get("success") is True
        assert extra.get("error_message") is None
        assert extra.get("request_id") == "test-request-id"
        assert extra.get("method") == "GET"
        assert extra.get("path") == "/test"
        assert extra.get("user_agent") == "test-agent"
        assert extra.get("custom_field") == "custom_value"