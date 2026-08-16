"""
Project indexing, file listing, and MCP-navigation endpoints.

POST /v1/projects/{project_id}/index runs the full Parts 2-9 pipeline
asynchronously via a job queue. This endpoint returns immediately with a job ID.
The actual indexing happens in the background.

GET /v1/projects/{project_id}/jobs/{job_id} returns the status of a job.
DELETE /v1/projects/{project_id}/jobs/{job_id} requests cancellation of a job.

GET /v1/projects/{project_id}/files              list indexed files
GET /v1/projects/{project_id}/files/content      read a single file (MCP navigation)
GET /v1/projects/{project_id}/symbols            search the IR for a symbol (MCP navigation)
GET /v1/projects/{project_id}/info               project summary stats (MCP navigation)

The file-content / symbols / info routes exist to back the SemiRAG MCP
server's cheap navigation tools — they intentionally do NOT call an LLM
and reuse the same FileRow / IRNodeRow tables the rest of the system reads
from. No new storage or retrieval implementation.

The embedding provider is determined by the SEM RAG_EMBEDDING_PROVIDER
setting. For the tfidf provider, the provider is fit on the corpus and
saved to disk for later use by the query endpoint. For other providers
(Ollama, OpenAI-compatible), no fitting or saving is required.
Re-indexing a project changes the vector space for tfidf (and thus
invalidates the query cache), while for neural providers the vector space
is fixed and re-indexing does not invalidate the query cache due to
embedding changes (though it may still invalidate due to new/changed
chunks).
"""
from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response, status
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from semrag.api.api_key import APIKey
from semrag.api.auth import require_api_key
from semrag.api.dependencies import (
    get_db_session,
    get_job_manager,
    get_session_factory,
    get_settings,
)
from semrag.api.project_auth import get_project_authorizer
from semrag.api.query_cache import QueryCache, get_query_cache
from semrag.api.rate_limit import enforce_rate_limit
from semrag.api.schemas import (
    DeleteFileResponse,
    FileSummary,
    IndexProjectRequest,
    IndexProjectResponse,
    ListFilesResponse,
    PatchFileRequest,
    PatchFileResponse,
    WriteFileRequest,
    WriteFileResponse,
)
from semrag.core.project_paths import (
    ActiveProject,
    ProjectPathError,
    ProjectPaths,
    detect_active_project,
)
from semrag.jobs.models import JobState
from semrag.storage.postgres.models import FileRow, IRNodeRow, Project, ScanManifestRow

router = APIRouter(prefix="/v1/projects", tags=["projects"])


def validate_project_root(root_path: str) -> Path:
    """Validate that the root path is within allowed project roots.
    Returns the Path object if valid, otherwise raises HTTPException.
    """
    settings = get_settings()
    root = Path(root_path).resolve()
    # If no project roots are configured, allow any path. Symlink/traversal
    # checks are unnecessary here because ``resolve()`` already returns the
    # absolute, real path.
    if not settings.project_roots:
        return root
    # Check for symlinks and path traversal by ensuring the resolved path is within one of the allowed roots.
    for allowed_root in settings.project_roots:
        allowed_path = Path(allowed_root).resolve()
        try:
            # Check if root is a subpath of allowed_path
            root.relative_to(allowed_path)
            break
        except ValueError:
            continue
    else:
        # If we didn't break, then the root is not within any allowed root.
        raise HTTPException(
            status_code=400,
            detail=f"Project root '{root_path}' is not within any allowed project root.",
        )
    # Additionally, we can check that the path does not contain any symlink tricks by using resolve() again?
    # We already used resolve() on the root and the allowed roots, so we are safe.
    return root


@router.post("/{project_id}/index", response_model=IndexProjectResponse)
def index_project(
    project_id: str,
    request: IndexProjectRequest,
    response: Response,
    db: Session = Depends(get_db_session),
    api_key: APIKey = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
    job_manager: JobManager = Depends(get_job_manager),
    session_factory: Callable[[], Session] = Depends(get_session_factory),
    settings: Settings = Depends(get_settings),
    query_cache: QueryCache = Depends(get_query_cache),
) -> IndexProjectResponse:
    """Enqueue an indexing job and return immediately with job ID.

    The job is persisted to the configured ``job_store_backend``. A
    worker process (or the in-process worker started by the API
    lifespan when ``api_runs_worker`` is true) polls the store and
    runs the job. The endpoint does NOT execute the job in-process
    anymore — that path is what made multi-replica deployments
    double-run or silently drop jobs across restarts.
    """
    # Accept either the legacy single ``root_path`` or the portable
    # multi-root form (rtl_root / tb_root / docs_root). Both may be given
    # together — the area roots are authoritative for resolution, the
    # ``root_path`` remains the display/validation working root.
    raw_roots = {
        "rtl_root": request.rtl_root,
        "tb_root": request.tb_root,
        "docs_root": request.docs_root,
    }
    provided_areas = {name for name, value in raw_roots.items() if value}
    has_root = bool(request.root_path and request.root_path.strip())
    if not has_root and not provided_areas:
        raise HTTPException(
            status_code=400,
            detail=(
                "IndexProjectRequest requires at least one of root_path, "
                "rtl_root, tb_root, docs_root."
            ),
        )

    roots: dict[str, Path] = {}
    for name, value in raw_roots.items():
        if not value:
            continue
        resolved = validate_project_root(value)
        if not resolved.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Project {name} does not exist: {resolved}",
            )
        roots[name.replace("_root", "")] = resolved

    root: Path | None = None
    if has_root:
        root = validate_project_root(request.root_path)
        if not root.is_dir():
            raise HTTPException(
                status_code=400,
                detail=f"Project root does not exist: {root}",
            )

    # Create the job or get existing one
    job, created = job_manager.create_job(
        project_id=project_id,
        root_path=str(root) if root else None,
        rtl_root=str(roots.get("rtl")) if "rtl" in roots else None,
        tb_root=str(roots.get("tb")) if "tb" in roots else None,
        docs_root=str(roots.get("docs")) if "docs" in roots else None,
    )

    # If the job was just queued, invalidate any cached query results
    # for this project. We invalidate on QUEUE (not on completion) so
    # that any concurrent query that hits the same project gets a
    # cache miss rather than stale results while the re-index is in
    # flight. The cost is a few cache hits lost; the alternative is
    # silently serving pre-reindex results.
    if created:
        query_cache.invalidate_project(project_id)

    # Prepare response
    # We always return the job's current state, but note that if the job is already running,
    # we don't submit a new one.
    # We'll return a placeholder scan_id for now; the real scan_id will be available in the job statistics after completion.
    placeholder_scan_id = "00000000-0000-0000-0000-000000000000"

    # Determine status code: 202 if we created a new job (and submitted it), 200 if we returned an existing job
    response.status_code = status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK

    return IndexProjectResponse(
        project_id=project_id,
        job_id=job.job_id,
        file_count=0,  # will be updated in job statistics
        chunk_count=0,
        scan_id=placeholder_scan_id,
    )


@router.get("/{project_id}/jobs/{job_id}")
def get_job(
    project_id: str,
    job_id: str,
    api_key: APIKey = Depends(get_project_authorizer()),
    job_manager: JobManager = Depends(get_job_manager),
):
    """Get the status of an indexing job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.project_id != project_id:
        raise HTTPException(status_code=403, detail="Job does not belong to this project")
    return job.to_dict()


@router.delete("/{project_id}/jobs/{job_id}")
def cancel_job(
    project_id: str,
    job_id: str,
    api_key: APIKey = Depends(get_project_authorizer()),
    job_manager: JobManager = Depends(get_job_manager),
):
    """Request cancellation of an indexing job."""
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.project_id != project_id:
        raise HTTPException(status_code=403, detail="Job does not belong to this project")
    if job.state not in (JobState.QUEUED, JobState.RUNNING):
        raise HTTPException(
            status_code=400,
            detail=f"Job cannot be cancelled because it is in state {job.state.name}",
        )
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cancel job")
    return {"status": "cancellation requested"}


# The existing file listing endpoint remains unchanged
@router.get("/{project_id}/files", response_model=ListFilesResponse)
def list_files(
    project_id: str,
    db: Session = Depends(get_db_session),
    api_key: APIKey = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> ListFilesResponse:
    rows = db.execute(
        select(FileRow.relative_path, FileRow.category, FileRow.size_bytes)
        .where(FileRow.project_id == project_id)
        .order_by(FileRow.relative_path)
    ).all()

    return ListFilesResponse(
        project_id=project_id,
        files=[
            FileSummary(relative_path=r.relative_path, category=r.category, size_bytes=r.size_bytes)
            for r in rows
        ],
    )


# ---------------------------------------------------------------------------
# MCP navigation routes — backing the SemiRAG MCP server's read-only tools.
# No LLM, no embedding call, no second retrieval implementation. These
# just expose the rows already in the project's database.
# ---------------------------------------------------------------------------

_MAX_FILE_BYTES = 2 * 1024 * 1024  # 2 MiB cap on /files/content reads


@router.get("/{project_id}/files/content")
def read_file_content(
    project_id: str,
    file_path: str = Query(..., description="Project-relative path, as returned by /files."),
    start_line: int | None = Query(None, ge=1),
    end_line: int | None = Query(None, ge=1),
    db: Session = Depends(get_db_session),
    api_key: APIKey = Depends(get_project_authorizer()),
) -> dict:
    """Read the text content of an indexed file.

    Only files present in the project's ``files`` table can be read — the
    table is the canonical "what was indexed" list, so this endpoint
    cannot be turned into a generic ``cat`` of arbitrary host paths.
    Binary files are rejected up-front rather than returned as a
    latin-1 blob.
    """
    file_row = db.execute(
        select(FileRow)
        .where(FileRow.project_id == project_id, FileRow.relative_path == file_path)
    ).scalar_one_or_none()

    if file_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"File not found in project '{project_id}': {file_path}",
        )

    if file_row.is_binary:
        raise HTTPException(
            status_code=400,
            detail=f"File is binary and cannot be returned as text: {file_path}",
        )

    if file_row.size_bytes > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large to read in one request ({file_row.size_bytes} bytes > {_MAX_FILE_BYTES}): {file_path}",
        )

    abs_path = Path(file_row.absolute_path)
    try:
        text = abs_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read file from disk: {exc}",
        ) from exc

    total_lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    selected_start = start_line or 1
    selected_end = end_line or total_lines
    if selected_end < selected_start:
        raise HTTPException(
            status_code=400,
            detail=f"end_line ({selected_end}) is before start_line ({selected_start})",
        )

    if start_line is None and end_line is None:
        snippet = text
        returned_start = 1
        returned_end = total_lines
    else:
        lines = text.splitlines()
        # 1-indexed inclusive slice
        snippet = "\n".join(lines[selected_start - 1 : selected_end])
        returned_start = selected_start
        returned_end = min(selected_end, total_lines)

    return {
        "project_id": project_id,
        "relative_path": file_path,
        "category": file_row.category,
        "size_bytes": file_row.size_bytes,
        "total_lines": total_lines,
        "start_line": returned_start,
        "end_line": returned_end,
        "truncated": (start_line is not None or end_line is not None) and (
            returned_start != selected_start or returned_end != selected_end
        ),
        "content": snippet,
    }


# ---------------------------------------------------------------------------
# Phase 1: controlled file writing (MCP-backed agent code generation).
#
# Intentionally NOT gated on the indexed-files table: the agent needs to
# write NEW files (a freshly-generated testcase is not in `files` yet).
# All filesystem writes are constrained to a project's registered root
# path (from the `projects` table), and `file_path` is interpreted as a
# project-relative path. Absolute paths, `..` traversal, and symlink
# escapes are all rejected. Re-indexing is the caller's responsibility
# — the response simply flags that re-indexing is required.
# ---------------------------------------------------------------------------

_MAX_WRITE_FILE_BYTES = 2 * 1024 * 1024  # 2 MiB cap on a single write


def _project_paths_from_row(row) -> ProjectPaths:
    """Build a ``ProjectPaths`` from a ``Project`` row (no DB access)."""
    return ProjectPaths(
        project_id=row.project_id,
        root_path=Path(row.root_path).resolve() if row.root_path else None,
        rtl_root=Path(row.rtl_root).resolve() if row.rtl_root else None,
        tb_root=Path(row.tb_root).resolve() if row.tb_root else None,
        docs_root=Path(row.docs_root).resolve() if row.docs_root else None,
    )


def _resolve_project_paths(db: Session, project_id: str) -> ProjectPaths:
    """Look up the project, return its roots as a ``ProjectPaths``.

    Centralises the project-existence check so both the read and write
    routes error consistently on a bad project_id. A legacy single-root
    project populates ``root_path`` only; a multi-root project populates
    the area roots (and possibly ``root_path`` as its display root).
    """
    project_row = db.execute(
        select(Project).where(Project.project_id == project_id)
    ).scalar_one_or_none()
    if project_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Project not found: {project_id}",
        )
    return _project_paths_from_row(project_row)


@router.get("/resolve")
def resolve_project(
    cwd: str | None = Query(
        default=None,
        description=(
            "Working directory to resolve the active project for. Defaults "
            "to the API process's current working directory."
        ),
    ),
    db: Session = Depends(get_db_session),
    api_key: APIKey = Depends(require_api_key),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Resolve the active project context for a working directory.

    This is the single source of truth the MCP layer (and any other
    context-aware caller) uses to turn a bare ``cwd`` into a project.
    Detection order (documented in docs/PROJECT_ROOT_CONFIGURATION.md):

    1. Explicit configuration: ``SEMRAG_RTL_ROOT`` / ``SEMRAG_TB_ROOT`` /
       ``SEMRAG_DOCS_ROOT`` (plus ``SEMRAG_PROJECT_ID``).
    2. Safe layout discovery: an ancestor of ``cwd`` containing both
       ``rtl/`` and ``tb/`` subdirectories (legacy single-root).
    3. Registered projects: any previously-registered project whose
       configured roots contain ``cwd``.

    Uses ``require_api_key`` (not the per-project authorizer) because the
    path carries no ``{project_id}`` — the whole point is to resolve it.
    """
    cwd_path = Path(cwd or os.getcwd()).resolve()

    active = detect_active_project(cwd_path, settings)
    if active is not None:
        return active.to_dict()

    for row in db.execute(select(Project)).scalars().all():
        paths = _project_paths_from_row(row)
        if paths.is_read_allowed(cwd_path):
            return ActiveProject(
                project_id=row.project_id,
                paths=paths,
                area=paths.area_for(cwd_path),
                source="registered",
            ).to_dict()

    raise HTTPException(
        status_code=404,
        detail=(
            f"No project resolved for working directory {cwd_path}. "
            "Configure SEMRAG_RTL_ROOT / SEMRAG_TB_ROOT / SEMRAG_DOCS_ROOT "
            "or register the project first."
        ),
    )


def _safe_resolve_write_target(
    project_id: str,
    file_path: str,
    paths: ProjectPaths,
) -> Path:
    """Resolve ``file_path`` (project-relative) to an absolute path and
    confirm it stays inside a writable project root. Rejects:

    - absolute paths (``/etc/passwd``, ``C:\\foo``)
    - ``..`` traversal segments
    - symlink escapes (resolved target must remain under a project root)
    - writes under the read-only docs root (multi-root projects)
    - empty / whitespace-only / NUL-byte paths
    """
    if not file_path or not file_path.strip():
        raise HTTPException(
            status_code=400,
            detail="file_path must not be empty.",
        )
    if "\x00" in file_path:
        raise HTTPException(
            status_code=400,
            detail="file_path must not contain NUL bytes.",
        )

    raw = Path(file_path)
    if raw.is_absolute():
        raise HTTPException(
            status_code=400,
            detail=(
                f"file_path must be relative to the project root, "
                f"got absolute path: {file_path}"
            ),
        )
    if any(part == ".." for part in raw.parts):
        raise HTTPException(
            status_code=400,
            detail=(f"file_path must not contain '..' traversal: {file_path}"),
        )

    try:
        target = paths.resolve_relative(file_path)
    except ProjectPathError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    if not paths.is_write_allowed(target):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Resolved file path is outside a writable project root "
                f"(docs is read-only): {file_path} (project_id={project_id})"
            ),
        )

    return target


@router.post(
    "/{project_id}/files/content",
    response_model=WriteFileResponse,
)
def write_file_content(
    project_id: str,
    request: WriteFileRequest = Body(...),
    db: Session = Depends(get_db_session),
    api_key: APIKey = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> WriteFileResponse:
    """Create or replace a text file inside the project root.

    This is the Phase-1 write surface for the SemiRAG coding agent.
    It is intentionally narrow:

    - target is constrained to the project's registered root path
    - ``file_path`` must be project-relative
    - ``..`` and absolute paths are rejected with 400
    - symlink escapes are rejected with 400 (resolved target must
      remain under the project root)
    - binary writes are not supported (UTF-8 only)
    - the request is capped at ``_MAX_WRITE_FILE_BYTES`` bytes
    - parent directories are created on demand
    - the project is NOT auto-reindexed; the response signals that
      the caller (MCP tool, agent loop) should re-index

    The endpoint does not differentiate "create" from "replace" in
    the HTTP status: both succeed with 200 and a ``created`` boolean
    in the body. Atomicity is a single ``os.replace`` of a temp file
    in the same directory, so a writer crash mid-flush cannot leave a
    half-written file at the target path.
    """
    paths = _resolve_project_paths(db, project_id)
    target = _safe_resolve_write_target(project_id, request.file_path, paths)

    content_bytes = request.content.encode("utf-8")
    if len(content_bytes) > _MAX_WRITE_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Content too large to write in one request "
                f"({len(content_bytes)} bytes > {_MAX_WRITE_FILE_BYTES})"
            ),
        )

    existed = target.exists()
    created = not existed

    target.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: stage in a sibling temp file then os.replace().
    # Same-directory staging keeps the replace() a single rename(2)
    # on POSIX, so a concurrent reader can never observe a partial
    # file at the target path.
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stage write for {request.file_path}: {exc}",
        ) from exc

    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content_bytes)
        os.replace(tmp_path, target)
    except OSError as exc:
        # Best-effort cleanup of the staging file before bubbling up.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write {request.file_path}: {exc}",
        ) from exc

    lines_written = request.content.count("\n") + (
        0 if request.content.endswith("\n") or request.content == "" else 1
    )

    return WriteFileResponse(
        project_id=project_id,
        relative_path=request.file_path,
        bytes_written=len(content_bytes),
        lines_written=lines_written,
        created=created,
        requires_reindex=True,
    )


@router.post(
    "/{project_id}/files/patch",
    response_model=PatchFileResponse,
)
def patch_file_content(
    project_id: str,
    request: PatchFileRequest = Body(...),
    db: Session = Depends(get_db_session),
    api_key: APIKey = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> PatchFileResponse:
    """Apply a single exact-text replacement to an existing project file.

    Companion to the Phase-1 write surface for the SemiRAG coding agent.
    It is intentionally strict:

    - the target must already exist (a patch modifies, it does not create)
    - the target is constrained to the project's registered root path,
      with the same absolute-path / ``..`` / symlink-escape rejection as
      the write endpoint
    - only UTF-8 text files are supported
    - ``old_text`` must match EXACTLY ONCE in the file. Zero matches or
      more than one match both fail with 409 and leave the file untouched
      — no partial or ambiguous edits are ever applied.
    - the combined result is capped at ``_MAX_WRITE_FILE_BYTES`` bytes
    - the write itself is atomic (temp file + ``os.replace``), the same
      mechanism used by the write endpoint

    Re-indexing is NOT triggered automatically; the response signals that
    the caller (MCP tool, agent loop) should re-index.
    """
    paths = _resolve_project_paths(db, project_id)
    target = _safe_resolve_write_target(project_id, request.file_path, paths)

    if not target.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"File does not exist: {request.file_path}",
        )

    try:
        current = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"File is not valid UTF-8 text: {request.file_path}",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read {request.file_path}: {exc}",
        ) from exc

    match_count = current.count(request.old_text)
    if match_count == 0:
        raise HTTPException(
            status_code=409,
            detail=(
                f"old_text not found in {request.file_path} "
                f"({match_count} matches); no change made."
            ),
        )
    if match_count > 1:
        raise HTTPException(
            status_code=409,
            detail=(
                f"old_text is ambiguous in {request.file_path}: "
                f"{match_count} matches found; no change made."
            ),
        )

    updated = current.replace(request.old_text, request.new_text)

    content_bytes = updated.encode("utf-8")
    if len(content_bytes) > _MAX_WRITE_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Patched content too large to write in one request "
                f"({len(content_bytes)} bytes > {_MAX_WRITE_FILE_BYTES})"
            ),
        )

    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stage write for {request.file_path}: {exc}",
        ) from exc

    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(content_bytes)
        os.replace(tmp_path, target)
    except OSError as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write {request.file_path}: {exc}",
        ) from exc

    return PatchFileResponse(
        project_id=project_id,
        relative_path=request.file_path,
        replacements=1,
        bytes_written=len(content_bytes),
        requires_reindex=True,
    )


@router.delete(
    "/{project_id}/files/content",
    response_model=DeleteFileResponse,
)
def delete_file_content(
    project_id: str,
    file_path: str = Query(..., description="Project-relative path of the file to delete."),
    db: Session = Depends(get_db_session),
    api_key: APIKey = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
    query_cache: QueryCache = Depends(get_query_cache),
) -> DeleteFileResponse:
    """Delete a text file inside the project root.

    This is the companion to the write/patch surface for the SemiRAG
    coding agent. It is intentionally strict:

    - the target must already exist (a delete removes, it does not create)
    - the target is constrained to the project's registered root path,
      with the same absolute-path / ``..`` / symlink-escape rejection as
      the write endpoint
    - only files under writable roots (rtl/tb) can be deleted; docs is read-only
    - the filesystem file is removed
    - the corresponding database/index state is cleaned up:
      * FileRow is deleted from the files table
      * IR nodes for the file are deleted
      * Chunks for the file are deleted from the vector store
    - re-indexing is NOT triggered automatically; the response signals that
      the caller (MCP tool, agent loop) should re-index

    The operation is transactional: all database cleanup happens in a
    single transaction that commits only if the filesystem delete succeeds.
    """
    paths = _resolve_project_paths(db, project_id)
    target = _safe_resolve_write_target(project_id, file_path, paths)

    if not target.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"File does not exist: {file_path}",
        )

    # Delete from filesystem first (outside transaction) so a failure
    # here doesn't leave the database in an inconsistent state with a
    # ghost file that no longer exists on disk.
    try:
        target.unlink()
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file from filesystem: {exc}",
        ) from exc

    # Now clean up database state in a transaction
    try:
        # CRITICAL ORDERING: Delete chunks FIRST (FK constraint:
        # chunks.source_node_id -> ir_nodes.id). Then delete IR nodes,
        # then FileRow.
        from semrag.storage.vector.pgvector_store import PgvectorStore
        PgvectorStore(db).delete_by_paths(project_id, [file_path])

        # Delete IR nodes for this file
        db.execute(
            delete(IRNodeRow).where(
                IRNodeRow.project_id == project_id,
                IRNodeRow.relative_path == file_path,
            )
        )

        # Delete FileRow
        db.execute(
            delete(FileRow).where(
                FileRow.project_id == project_id,
                FileRow.relative_path == file_path,
            )
        )

        db.commit()

        # Invalidate query cache so subsequent searches don't return stale results
        query_cache.invalidate_project(project_id)

    except Exception as exc:
        db.rollback()
        # Best-effort: try to restore the file? Not safe; just report failure.
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clean up database state after filesystem delete: {exc}",
        ) from exc

    return DeleteFileResponse(
        project_id=project_id,
        relative_path=file_path,
        deleted=True,
        requires_reindex=True,
    )


@router.get("/{project_id}/symbols")
def find_symbols(
    project_id: str,
    symbol: str = Query(..., min_length=1, description="Symbol name to look up."),
    kind: str | None = Query(None, description="Optional IRNode kind filter, e.g. 'sv_module'."),
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db_session),
    api_key: APIKey = Depends(get_project_authorizer()),
) -> dict:
    """Find IR nodes (modules, packages, files) by name within a project.

    The IR is the parser-built structural graph the rest of SemiRAG
    already uses (chunking, graph expansion). Reusing the same table
    keeps this tool consistent with what /query and /answer retrieve.
    """
    stmt = (
        select(IRNodeRow.id, IRNodeRow.kind, IRNodeRow.name, IRNodeRow.relative_path, IRNodeRow.attributes)
        .where(IRNodeRow.project_id == project_id)
        .where(IRNodeRow.name == symbol)
    )
    if kind:
        stmt = stmt.where(IRNodeRow.kind == kind)
    stmt = stmt.order_by(IRNodeRow.relative_path).limit(limit)

    rows = db.execute(stmt).all()

    results = []
    for row in rows:
        attrs = row.attributes or {}
        line_info = attrs.get("line_start") or attrs.get("line")
        if line_info is None and isinstance(attrs.get("span"), dict):
            line_info = attrs["span"].get("start_line")
        results.append(
            {
                "id": row.id,
                "kind": row.kind,
                "name": row.name,
                "relative_path": row.relative_path,
                "line": line_info,
                "attributes": {k: v for k, v in attrs.items() if k != "ports"},
            }
        )

    return {
        "project_id": project_id,
        "symbol": symbol,
        "kind_filter": kind,
        "count": len(results),
        "results": results,
    }


@router.get("/{project_id}/info")
def project_info(
    project_id: str,
    db: Session = Depends(get_db_session),
    api_key: APIKey = Depends(get_project_authorizer()),
) -> dict:
    """Return summary statistics about an indexed project.

    No fields are invented: every value comes from a real table that
    ingestion populates.
    """
    project_row = db.execute(
        select(Project).where(Project.project_id == project_id)
    ).scalar_one_or_none()
    if project_row is None:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    file_count = db.execute(
        select(func.count()).select_from(FileRow).where(FileRow.project_id == project_id)
    ).scalar_one()
    total_bytes = db.execute(
        select(func.coalesce(func.sum(FileRow.size_bytes), 0))
        .where(FileRow.project_id == project_id)
    ).scalar_one()
    node_count = db.execute(
        select(func.count())
        .select_from(IRNodeRow)
        .where(IRNodeRow.project_id == project_id)
    ).scalar_one()
    last_scan = db.execute(
        select(ScanManifestRow.completed_at, ScanManifestRow.file_count, ScanManifestRow.total_bytes)
        .where(ScanManifestRow.project_id == project_id)
        .order_by(ScanManifestRow.started_at.desc())
        .limit(1)
    ).first()

    # Category breakdown (real column, indexed).
    category_rows = db.execute(
        select(FileRow.category, func.count())
        .where(FileRow.project_id == project_id)
        .group_by(FileRow.category)
        .order_by(func.count().desc())
    ).all()

    return {
        "project_id": project_id,
        "root_path": project_row.root_path,
        "rtl_root": project_row.rtl_root,
        "tb_root": project_row.tb_root,
        "docs_root": project_row.docs_root,
        "created_at": project_row.created_at.isoformat() if project_row.created_at else None,
        "file_count": int(file_count),
        "total_bytes": int(total_bytes),
        "ir_node_count": int(node_count),
        "last_scan": (
            {
                "completed_at": last_scan.completed_at.isoformat() if last_scan.completed_at else None,
                "file_count": int(last_scan.file_count) if last_scan.file_count is not None else None,
                "total_bytes": int(last_scan.total_bytes) if last_scan.total_bytes is not None else None,
            }
            if last_scan
            else None
        ),
        "categories": [{"category": cat, "count": int(count)} for cat, count in category_rows],
    }