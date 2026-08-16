"""Safe execution of a project validation target.

The runner takes a validated ``ValidationConfig`` and a target name, builds
the exact argv list declared in the config, and executes it with:

- ``shell=False`` (no shell, no operator chaining)
- ``cwd`` pinned to the resolved project root (no arbitrary working dirs)
- a hard timeout
- all file / incdir paths resolved and checked to stay under the project
  root before the process starts

The result is classified as ``passed`` (exit 0), ``blocked`` (missing
dependency or toolchain), or ``failed`` (a code problem surfaced by the
tool). Blocked is never faked into a pass.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from semrag.core.project_paths import AREA_ORDER, ProjectPathError
from semrag.validation.config import (
    TrustedExternalDep,
    ValidationConfig,
    validation_config_dir,
)

logger = logging.getLogger(__name__)

# Markers in combined tool output that indicate a missing environment
# dependency (toolchain / library) rather than a defect in the code being
# validated.
_BLOCKED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("uvm_pkg", "UVM library (uvm_pkg / uvm_macros.svh)"),
    ("uvm_macros", "UVM library (uvm_pkg / uvm_macros.svh)"),
    ("cannot find include file", "missing include file referenced by the testbench"),
)

# Prefix that marks an incdir/file entry as a reference into a trusted
# external dependency. Format: ``ext:<dep_name>/<relative/subpath>``.
_EXTERNAL_PREFIX = "ext:"


@dataclass
class ValidationResult:
    status: str  # "passed" | "failed" | "blocked"
    exit_code: int | None
    stdout: str
    stderr: str
    command: str
    duration_seconds: float
    timed_out: bool = False
    blocked_reason: str | None = None
    target: str = ""
    log_lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "command": self.command,
            "duration_seconds": round(self.duration_seconds, 3),
            "timed_out": self.timed_out,
            "blocked_reason": self.blocked_reason,
            "target": self.target,
        }


def _require_within_root(root: Path, rel: str) -> Path:
    """Resolve a project-relative path and verify it stays under root."""
    raw = Path(rel)
    if raw.is_absolute() or any(part == ".." for part in raw.parts):
        raise ValueError(f"validation path escapes project root: {rel}")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError(f"validation path escapes project root: {rel}") from None
    return resolved


def _require_within_project(root: Path, project_paths, rel: str) -> Path:
    """Resolve a project-relative path against a ``ProjectPaths``.

    Supports three forms, so validation configs stay portable across
    project shapes:

    - Explicit area prefix: ``rtl:...`` / ``tb:...`` / ``docs:...``.
    - Bare first-segment mapping: ``rtl/...`` / ``tb/...`` / ``docs/...``
      resolve against the matching area root (the exact form stored in the
      manifest for multi-root projects).
    - Anything else resolves against the legacy ``root_path`` (or the
      passed ``root`` working directory when the project has no
      ``root_path``) — the pre-refactor single-root behaviour, unchanged.
    """
    if project_paths is None:
        return _require_within_root(root, rel)

    for area in AREA_ORDER:
        if rel.startswith(f"{area}:"):
            body = rel[len(f"{area}:"):]
            try:
                return project_paths.resolve_area(area, body)
            except ProjectPathError as exc:
                raise ValueError(str(exc)) from None

    first, _, rest = rel.partition("/")
    if first in AREA_ORDER and rest:
        try:
            return project_paths.resolve_area(first, rest)
        except ProjectPathError as exc:
            raise ValueError(str(exc)) from None

    if project_paths.root_path is not None:
        return _require_within_root(project_paths.root_path, rel)
    return _require_within_root(root, rel)


def _resolve_path(
    root: Path,
    deps: dict[str, TrustedExternalDep],
    entry: str,
    *,
    require_dir: bool,
    config_dir: Path,
    project_paths=None,
) -> Path:
    """Resolve a single incdir/file entry.

    Normal project entries stay under ``root`` via ``_require_within_root``.
    Entries prefixed with ``ext:`` resolve against a named trusted external
    dep: ``ext:<name>/<subpath>``. The dep name must be in ``deps``; the
    subpath must be relative and contain no ``..`` segments. An
    unconfigured external dep is rejected — external paths must be
    declared in the validation config, not supplied via the request body.

    A dep's declared path may be:
      - an absolute directory (unchanged),
      - an absolute directory that uses ``$VAR`` / ``${VAR}`` environment
        expansion (e.g. ``${SEMRAG_UVM_ROOT}`` so the UVM checkout stays
        portable between machines), or
      - a relative directory, resolved against the validation config
        directory (e.g. ``stubs/rtl_uvm`` -> ``configs/validation/stubs/
        rtl_uvm``). This keeps in-repo helper files portable without any
        environment variables.
    """
    if entry.startswith(_EXTERNAL_PREFIX):
        body = entry[len(_EXTERNAL_PREFIX):]
        dep_name, _, sub = body.partition("/")
        dep = deps.get(dep_name)
        if dep is None:
            raise ValueError(
                f"unconfigured external dependency {dep_name!r} "
                f"referenced by {entry!r}"
            )
        sub_path = Path(sub) if sub else Path(".")
        if sub_path.is_absolute() or any(part == ".." for part in sub_path.parts):
            raise ValueError(
                f"external dep subpath escapes dep root: {entry!r}"
            )
        expanded = os.path.expandvars(dep.path)
        dep_root = Path(expanded)
        if not dep_root.is_absolute():
            dep_root = (config_dir / expanded).resolve()
        resolved = (dep_root / sub_path).resolve()
        try:
            resolved.relative_to(dep_root)
        except ValueError:
            raise ValueError(
                f"external dep subpath escapes dep root: {entry!r}"
            ) from None
    else:
        resolved = _require_within_project(root, project_paths, entry)

    if require_dir:
        if not resolved.is_dir():
            raise FileNotFoundError(
                f"validation incdir does not exist: {entry}"
            )
    else:
        if not resolved.is_file():
            raise FileNotFoundError(
                f"validation file does not exist: {entry}"
            )
    return resolved


def build_argv(
    root: Path,
    config: ValidationConfig,
    target_name: str,
    project_paths=None,
) -> list[str]:
    """Assemble the exact argv for a target from its config (no user input)."""
    target = config.targets[target_name]

    deps = {d.name: d for d in config.trusted_external_deps}
    config_dir = validation_config_dir()

    argv = [config.tool, *target.tool_args]

    for incdir in target.incdirs:
        resolved = _resolve_path(
            root, deps, incdir, require_dir=True, config_dir=config_dir, project_paths=project_paths
        )
        argv.append(f"+incdir+{resolved}")

    for define in target.defines:
        argv.append(f"-D{define}")

    for f in target.files:
        resolved = _resolve_path(
            root, deps, f, require_dir=False, config_dir=config_dir, project_paths=project_paths
        )
        argv.append(str(resolved))

    return argv


def _classify(result: subprocess.CompletedProcess) -> tuple[str, str | None]:
    """Map a finished subprocess to (status, blocked_reason).

    ``passed`` if exit 0. If exit != 0 and the output matches a known
    missing-dependency marker, ``blocked`` with the exact dependency.
    Otherwise ``failed`` (the tool reported a code problem).
    """
    if result.returncode == 0:
        return "passed", None

    combined = f"{result.stdout}\n{result.stderr}".lower()
    for marker, dependency in _BLOCKED_PATTERNS:
        if marker in combined:
            return "blocked", dependency
    return "failed", None


def run_validation(
    root: Path,
    config: ValidationConfig,
    target_name: str,
    log_output: bool = True,
    project_paths=None,
) -> ValidationResult:
    """Execute one validation target under ``root`` and classify the result.

    ``project_paths`` (a ``core.project_paths.ProjectPaths``) is optional;
    when provided, config incdirs/files are resolved against the area roots
    (``rtl:``/``tb:``/``docs:`` prefixes and bare first-segment mapping)
    instead of only against ``root``."""
    if target_name not in config.targets:
        raise KeyError(f"unknown validation target: {target_name}")

    argv = build_argv(root, config, target_name, project_paths=project_paths)
    command = " ".join(str(a) for a in argv)

    if shutil.which(config.tool) is None:
        return ValidationResult(
            status="blocked",
            exit_code=None,
            stdout="",
            stderr="",
            command=command,
            duration_seconds=0.0,
            blocked_reason=f"toolchain missing: {config.tool!r} not found on PATH",
            target=target_name,
        )

    start = time.monotonic()
    try:
        result: subprocess.CompletedProcess | None = subprocess.run(
            argv,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=config.timeout_seconds,
            shell=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        result = None
    duration = time.monotonic() - start

    if timed_out:
        status = "blocked"
        blocked_reason = (
            f"validation timed out after {config.timeout_seconds}s "
            f"(command: {command})"
        )
        stdout = ""
        stderr = ""
        exit_code: int | None = None
    else:
        assert result is not None
        status, blocked_reason = _classify(result)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        exit_code = result.returncode

    if log_output:
        logger.info(
            "validation target %s finished status=%s exit=%s duration=%.2fs",
            target_name,
            status,
            exit_code,
            duration,
            extra={"project": config.project_id, "target": target_name},
        )

    return ValidationResult(
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        command=command,
        duration_seconds=duration,
        timed_out=timed_out,
        blocked_reason=blocked_reason,
        target=target_name,
    )