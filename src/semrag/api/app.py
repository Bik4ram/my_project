"""FastAPI application assembly."""
from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from semrag.api.dependencies import shutdown_resources
from semrag.api.exception_handler import add_exception_handlers
from semrag.api.health import router as health_router
from semrag.api.middleware import setup_middleware
from semrag.api.routers import answer, code, projects, query, validation
from semrag.api.startup_validation import validate_runtime_config


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Process-wide startup/shutdown.

    Startup validates the runtime configuration (fail-fast on a bad
    deploy) and configures logging from Settings. Shutdown drains the
    job executor and disposes the SQLAlchemy engine so a SIGTERM (k8s
    preStop hook, docker stop, systemd) gives every long-lived
    subsystem a chance to clean up instead of being killed mid-request.

    The lifespan runs once per worker process, so multi-worker uvicorn
    workers each open their own pool / executor — same behaviour as the
    module-level singletons they replace.

    When ``api_runs_worker`` is true (default), the API process also
    starts an in-process ``WorkerService`` that polls the persistent
    job store and runs jobs on a background thread. Production multi-
    replica deployments set this to false and run a dedicated worker
    (``python -m semrag.worker``).
    """
    settings = await validate_runtime_config()
    from semrag.core.observability import configure_logging
    from semrag.core.tracing import configure_tracing

    configure_logging(settings)
    configure_tracing(settings)

    worker = None
    worker_thread = None
    if settings.api_runs_worker:
        from semrag.worker import build_worker_service

        worker = build_worker_service()
        worker_thread = threading.Thread(
            target=worker.run_forever,
            name="semrag-inproc-worker",
            daemon=True,
        )
        worker_thread.start()

    try:
        yield
    finally:
        if worker is not None:
            worker.stop()
        if worker_thread is not None:
            worker_thread.join(timeout=5.0)
        shutdown_resources()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Semiconductor RAG Platform API",
        version="0.1.0",
        description="Retrieval and prompt-construction API over an indexed semiconductor project. "
        "Does not call an LLM (Part 13) — returns assembled, citation-grounded prompts ready to send to one.",
        lifespan=_lifespan,
    )

    # Set up middleware
    setup_middleware(app)

    # Set up exception handlers
    add_exception_handlers(app)

    app.include_router(projects.router)
    app.include_router(query.router)
    app.include_router(answer.router)
    app.include_router(code.router)
    app.include_router(validation.router)
    app.include_router(health_router)

    # Backwards-compatible alias for /health — earlier docker HEALTHCHECK
    # directives and any operator muscle memory depend on it. Same
    # semantics as /livez.
    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()