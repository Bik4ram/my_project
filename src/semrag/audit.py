"""
Audit logging for security events.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import Request
from semrag.core.settings import get_settings
from semrag.core.observability import get_logger

logger = get_logger(__name__)


def log_audit_event(
    event_type: str,
    request: Request,
    project_id: Optional[str] = None,
    api_key_id: Optional[str] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    **kwargs,
) -> None:
    """Log an audit event.

    Args:
        event_type: Type of event (e.g., "authentication", "authorization", "project_indexing")
        request: The FastAPI request object
        project_id: The project ID involved (if any)
        api_key_id: The API key identifier (e.g., the key name or a hash)
        success: Whether the event was successful
        error_message: Error message if the event failed
        **kwargs: Additional fields to include in the audit log
    """
    settings = get_settings()
    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "event_type": event_type,
        "project_id": project_id,
        "api_key_id": api_key_id,
        "success": success,
        "error_message": error_message,
        "request_id": getattr(request.state, "request_id", None),
        "method": request.method,
        "path": request.url.path,
        "user_agent": request.headers.get("user-agent"),
        "latency": None,  # We'll fill this in the middleware if needed
        **kwargs,
    }
    # Remove None values to keep the log clean
    audit_entry = {k: v for k, v in audit_entry.items() if v is not None}
    # stdlib `logger.info` accepts user fields only via `extra=...` — passing
    # them as plain kwargs raises TypeError("Logger._log() got an unexpected
    # keyword argument") on every request.
    logger.info("Audit event", extra=audit_entry)