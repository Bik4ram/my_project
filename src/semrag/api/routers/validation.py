"""Project validation endpoint.

POST /v1/projects/{project_id}/validate runs a named validation target
(Verilator lint) for a registered project and returns a structured result
(status, exit code, stdout, stderr, command, duration).

Safety model:

- ``project_id`` must resolve to a registered project (from the DB), whose
  root becomes the working directory for the command.
- The command is NOT supplied by the caller. The request body only names a
  target from the project's validation config (``configs/validation/``).
- The runner builds the argv list from that config, with ``shell=False``,
  a hard timeout, and every file/incdir path verified to stay under the
  project root.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from semrag.api.api_key import APIKey
from semrag.api.dependencies import get_db_session
from semrag.api.project_auth import get_project_authorizer
from semrag.api.rate_limit import enforce_rate_limit
from semrag.core.project_paths import ProjectPaths
from semrag.storage.postgres.models import Project
from semrag.validation import load_validation_config, run_validation

router = APIRouter(prefix="/v1/projects", tags=["validation"])


@router.post("/{project_id}/validate")
def validate_project(
    project_id: str,
    request: dict = Body(...),
    db: Session = Depends(get_db_session),
    api_key: APIKey = Depends(get_project_authorizer()),
    _rate_limit: None = Depends(enforce_rate_limit),
) -> dict:
    """Run a named validation target for a project.

    Body: ``{"target": "rtl_lint"}``. ``target`` defaults to ``rtl_lint``
    if omitted.
    """
    target = (request or {}).get("target", "rtl_lint")

    root, project_paths = _resolve_project_roots(db, project_id)
    config = load_validation_config(project_id)
    if config is None:
        raise HTTPException(
            status_code=404,
            detail=f"No validation config registered for project: {project_id}",
        )
    if target not in config.targets:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown validation target {target!r}. "
                f"Available targets: {sorted(config.targets)}"
            ),
        )

    try:
        result = run_validation(root, config, target, project_paths=project_paths)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Validation config error: {exc}",
        ) from exc

    if result.status == "blocked":
        # A blocked result (missing dependency/toolchain) is a legitimate,
        # explicit outcome — return it as 200 so the caller can read the
        # structured payload instead of an opaque 5xx.
        return result.to_dict()
    return result.to_dict()


def _resolve_project_roots(db: Session, project_id: str) -> tuple[Path, ProjectPaths | None]:
    """Look up the project and return its roots.

    Returns ``(working_dir, project_paths)`` where ``working_dir`` is the
    validation cwd (the legacy ``root_path`` when present, else the deepest
    common ancestor of the area roots) and ``project_paths`` carries the
    multi-root layout for config-path resolution (None for legacy
    single-root projects)."""
    project_row = db.execute(
        select(Project).where(Project.project_id == project_id)
    ).scalar_one_or_none()
    if project_row is None:
        raise HTTPException(
            status_code=404,
            detail=f"Project not found: {project_id}",
        )

    paths = ProjectPaths(
        project_id=project_id,
        root_path=Path(project_row.root_path).resolve() if project_row.root_path else None,
        rtl_root=Path(project_row.rtl_root).resolve() if project_row.rtl_root else None,
        tb_root=Path(project_row.tb_root).resolve() if project_row.tb_root else None,
        docs_root=Path(project_row.docs_root).resolve() if project_row.docs_root else None,
    )
    working = paths.infer_project_root()
    if working is None:
        raise HTTPException(
            status_code=404,
            detail=f"Project has no resolvable root: {project_id}",
        )
    if not paths.has_areas:
        return working, None
    return working, paths