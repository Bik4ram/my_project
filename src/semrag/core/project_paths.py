"""
Portable project path resolution.

The backend is fully decoupled from *where* a user keeps their RTL,
testbench, and documentation. A project is described by up to three
authoritative roots (``SEMRAG_RTL_ROOT``, ``SEMRAG_TB_ROOT``,
``SEMRAG_DOCS_ROOT``) plus an optional legacy single ``root_path``.

Why three roots instead of one project root:

- RTL, testbench, and docs frequently live in unrelated directories that
  do not share a parent (sibling checkouts, a shared VIP tree, a separate
  spec repo). Requiring a common parent would silently exclude content.
- The write-safety rule is per-area: the coding agent may write under the
  RTL and testbench roots but the docs root is read-only by default.
- ``relative_path`` (the portable, stored identity of every file/chunk/IR
  node) is expressed as ``rtl/<rel>`` / ``tb/<rel>`` / ``docs/<rel>`` —
  the exact shape the old single-root layout produced naturally when a
  project root contained ``rtl/`` and ``tb/`` subdirectories.

Backward compatibility: a project registered with only ``root_path`` (the
historic contract) keeps every existing behaviour bit-for-bit. The
multi-root paths are strictly additive; ``ProjectPaths.from_single_root``
is the shim that keeps old callers working unchanged.

This module owns no filesystem writes and never reads the database — it is
pure path algebra + the documented detection order, so it is unit-testable
without PostgreSQL or Ollama.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Deterministic iteration order for multi-root scans / validation argv.
AREA_ORDER: tuple[str, ...] = ("rtl", "tb", "docs")

# Areas the coding agent / write endpoints are allowed to modify.
WRITABLE_AREAS: frozenset[str] = frozenset({"rtl", "tb"})


class ProjectPathError(ValueError):
    """A path could not be resolved against the configured project roots."""


def _resolve_opt(raw: str | None) -> Path | None:
    if not raw:
        return None
    return Path(raw).resolve()


def _derive_project_id(
    rtl: Path | None, tb: Path | None, docs: Path | None
) -> str:
    """Derive a stable project id from configured roots (used only when
    ``SEMRAG_PROJECT_ID`` is unset). Picks the deepest common ancestor of
    the configured roots and uses its name."""
    roots = [p for p in (rtl, tb, docs) if p is not None]
    if not roots:
        return "project"
    common = Path(os.path.commonpath([str(r) for r in roots]))
    return common.name or "project"


@dataclass(frozen=True)
class ProjectPaths:
    """The authoritative root set for one project.

    ``project_id`` is the logical id every stored row is keyed by.
    ``rtl_root`` / ``tb_root`` / ``docs_root`` are the optional area roots.
    ``root_path`` is the legacy single project root (kept for backward
    compatibility; still required for single-root registration).
    """

    project_id: str
    rtl_root: Path | None = None
    tb_root: Path | None = None
    docs_root: Path | None = None
    root_path: Path | None = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_settings(cls, settings) -> ProjectPaths | None:
        """Build from a ``Settings`` instance. Returns ``None`` when no
        area root is configured (a fully unconfigured backend has no
        project context to detect)."""
        rtl = _resolve_opt(getattr(settings, "rtl_root", None))
        tb = _resolve_opt(getattr(settings, "tb_root", None))
        docs = _resolve_opt(getattr(settings, "docs_root", None))
        if rtl is None and tb is None and docs is None:
            return None
        project_id = getattr(settings, "project_id", None) or _derive_project_id(
            rtl, tb, docs
        )
        return cls(
            project_id=project_id,
            rtl_root=rtl,
            tb_root=tb,
            docs_root=docs,
        )

    @classmethod
    def from_single_root(cls, project_id: str, root_path: str | Path) -> ProjectPaths:
        """Legacy single-root project (the pre-refactor contract)."""
        return cls(project_id=project_id, root_path=Path(root_path).resolve())

    @classmethod
    def from_mapping(cls, project_id: str, mapping: dict) -> ProjectPaths:
        """Build from a dict of ``{"rtl_root": ..., "tb_root": ...}``
        (used by the API request schema)."""
        return cls(
            project_id=project_id,
            rtl_root=_resolve_opt(mapping.get("rtl_root")),
            tb_root=_resolve_opt(mapping.get("tb_root")),
            docs_root=_resolve_opt(mapping.get("docs_root")),
            root_path=_resolve_opt(mapping.get("root_path")),
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def roots(self) -> dict[str, Path]:
        """Configured area roots in deterministic order."""
        out: dict[str, Path] = {}
        for area in AREA_ORDER:
            root = getattr(self, f"{area}_root")
            if root is not None:
                out[area] = root
        return out

    @property
    def has_areas(self) -> bool:
        return bool(self.roots())

    def area_for(self, path: str | Path) -> str | None:
        """Return the area ('rtl'/'tb'/'docs') that contains ``path``, or
        ``None`` when the path is outside every configured root."""
        p = Path(path).resolve()
        for area, root in self.roots().items():
            try:
                p.relative_to(root)
                return area
            except ValueError:
                continue
        return None

    def infer_project_root(self) -> Path | None:
        """The closest thing to a single project root: the legacy
        ``root_path`` when present, else the deepest common ancestor of the
        configured area roots. Used as the working directory for validation
        and as a display root — never for file resolution (which always
        goes through a specific area root)."""
        if self.root_path is not None:
            return self.root_path
        roots = list(self.roots().values())
        if not roots:
            return None
        return Path(os.path.commonpath([str(r) for r in roots]))

    def resolve_relative(self, rel: str) -> Path:
        """Map a project-relative path to an absolute path.

        ``rtl/...`` / ``tb/...`` / ``docs/...`` resolve against the
        matching area root. Anything else resolves against the legacy
        ``root_path``. Raises ``ProjectPathError`` when no root covers the
        path (never guesses)."""
        if not rel or rel.startswith("/"):
            raise ProjectPathError(f"invalid relative path: {rel!r}")
        parts = rel.split("/", 1)
        if len(parts) == 2 and parts[0] in AREA_ORDER:
            area, sub = parts
            root = self.roots().get(area)
            if root is None:
                raise ProjectPathError(
                    f"project {self.project_id!r} has no configured {area} root"
                )
            target = (root / sub).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                raise ProjectPathError(
                    f"path escapes {area} root: {rel!r}"
                ) from None
            return target
        if self.root_path is None:
            raise ProjectPathError(
                f"cannot resolve {rel!r}: project has no root that covers it"
            )
        target = (self.root_path / rel).resolve()
        try:
            target.relative_to(self.root_path)
        except ValueError:
            raise ProjectPathError(f"path escapes project root: {rel!r}") from None
        return target

    def resolve_area(self, area: str, rel: str) -> Path:
        """Resolve a path relative to one named area root."""
        if area not in AREA_ORDER:
            raise ProjectPathError(f"unknown area: {area!r}")
        return self.resolve_relative(f"{area}/{rel}")

    def relativize(self, abs_path: str | Path) -> str | None:
        """Convert an absolute path into the portable project-relative form
        (``rtl/<rel>``, ``tb/<rel>``, ``docs/<rel>``, or ``<rel>`` under a
        legacy root). Returns ``None`` when the path is outside every
        root."""
        p = Path(abs_path).resolve()
        for area, root in self.roots().items():
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            return f"{area}/{rel.as_posix()}"
        if self.root_path is not None:
            try:
                return p.relative_to(self.root_path).as_posix()
            except ValueError:
                pass
        return None

    # ------------------------------------------------------------------
    # Write / read safety
    # ------------------------------------------------------------------
    def is_write_allowed(self, abs_path: str | Path) -> bool:
        """True only under a writable area root (rtl/tb) or a legacy
        ``root_path``. The docs root is read-only."""
        p = Path(abs_path).resolve()
        for area in WRITABLE_AREAS:
            root = self.roots().get(area)
            if root is None:
                continue
            try:
                p.relative_to(root)
                return True
            except ValueError:
                continue
        if self.root_path is not None:
            try:
                p.relative_to(self.root_path)
                return True
            except ValueError:
                pass
        return False

    def is_read_allowed(self, abs_path: str | Path) -> bool:
        """True under any configured root (including docs)."""
        p = Path(abs_path).resolve()
        for root in self.roots().values():
            try:
                p.relative_to(root)
                return True
            except ValueError:
                continue
        if self.root_path is not None:
            try:
                p.relative_to(self.root_path)
                return True
            except ValueError:
                pass
        return False

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "rtl_root": str(self.rtl_root) if self.rtl_root else None,
            "tb_root": str(self.tb_root) if self.tb_root else None,
            "docs_root": str(self.docs_root) if self.docs_root else None,
            "root_path": str(self.root_path) if self.root_path else None,
        }


@dataclass(frozen=True)
class ActiveProject:
    """A project context resolved from a working directory. ``area`` is the
    configured root that contains the cwd (when exactly one does), used by
    the MCP layer to disambiguate which part of the project a request
    refers to."""

    project_id: str
    paths: ProjectPaths
    area: str | None = None
    source: str = "config"  # 'config' | 'discovery'

    def to_dict(self) -> dict:
        return {**self.paths.to_dict(), "area": self.area, "source": self.source}


def _detect_by_layout(cwd: Path) -> ActiveProject | None:
    """Conservative fallback: walk ``cwd`` and its ancestors for a
    directory that has both ``rtl/`` and ``tb/`` subdirectories — a strong
    single-root signal. Never guesses unrelated directories, never invents
    roots."""
    for d in (cwd, *cwd.parents):
        if (d / "rtl").is_dir() and (d / "tb").is_dir():
            return ActiveProject(
                project_id=d.name or "project",
                paths=ProjectPaths.from_single_root(d.name or "project", d),
                area=None,
                source="discovery",
            )
    return None


def detect_active_project(
    cwd: str | Path, settings
) -> ActiveProject | None:
    """Resolve the active project for a working directory.

    Order (documented in docs/PROJECT_ROOT_CONFIGURATION.md):

    1. Explicit configuration: when any of ``rtl_root`` / ``tb_root`` /
       ``docs_root`` is set on ``settings``, that IS the project context.
       ``area`` is set when the cwd falls inside exactly one configured
       root.
    2. Safe layout discovery: nearest ancestor of ``cwd`` containing both
       an ``rtl/`` and a ``tb/`` directory (legacy single-root projects).
    3. Otherwise ``None`` — an unconfigured backend must not fabricate a
       project.
    """
    cwd = Path(cwd).resolve()
    configured = ProjectPaths.from_settings(settings)
    if configured is not None:
        area = configured.area_for(cwd)
        return ActiveProject(
            project_id=configured.project_id,
            paths=configured,
            area=area,
            source="config",
        )
    return _detect_by_layout(cwd)
