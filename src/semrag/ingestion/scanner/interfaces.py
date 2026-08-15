"""
Public interface of the scanner subsystem.

Every other plane (workers, API, future Part 4 IR builder) depends on
these Protocols, never on `orchestrator.Scanner` or `models.*` internals
directly, matching the module-boundary rule set in Part 1 Section 2: any
plane must be swappable behind its interface without touching callers.

`ManifestStore` is defined here (not in storage/postgres/) as a Protocol
the scanner subsystem depends on but does not implement — Part 6 provides
the real PostgreSQL-backed implementation; `InMemoryManifestStore` in
this package exists only so the scanner is independently testable and
runnable before Part 6 exists.
"""
from __future__ import annotations

from typing import Protocol

from semrag.ingestion.scanner.models import ChangeSet, ScanManifest


class ManifestStore(Protocol):
    """Persistence boundary for scan manifests. The scanner subsystem only
    ever needs the *latest* manifest for a project to diff against, plus
    the ability to persist a newly completed one."""

    def get_latest_manifest(self, project_id: str) -> ScanManifest | None: ...

    def save_manifest(self, manifest: ScanManifest) -> None: ...


class ScannerInterface(Protocol):
    """What the rest of the system is allowed to depend on. Downstream
    consumers (Celery workers in Part 2's `workers/` entrypoint, the future
    IR builder in Part 4) import this, not `Scanner` directly, so the
    concrete scanner implementation can be replaced (e.g. a distributed
    variant for very large multi-repo deployments) without touching
    callers."""

    def scan(self, project_id: str, root_path: str) -> tuple[ScanManifest, ChangeSet]: ...
