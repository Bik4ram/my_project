"""
In-memory / JSON-file ManifestStore implementations.

Neither of these is the production store — Part 6 defines the real
PostgreSQL-backed `ManifestStore` implementation (manifests as a table with
a `files` JSONB column, indexed by `project_id` + `completed_at`). These
exist so:

  1. The scanner subsystem has zero hard dependency on Part 6 being built
     yet — you can run and test the scanner today, standalone.
  2. `JSONFileManifestStore` gives you a genuinely usable persistence
     mechanism for a single-project, single-workstation setup without
     standing up Postgres at all, which may be the right call for initial
     local development regardless of what Part 6 ultimately recommends for
     the multi-project production deployment.
"""
from __future__ import annotations

import json
from pathlib import Path

from semrag.ingestion.scanner.interfaces import ManifestStore
from semrag.ingestion.scanner.models import ScanManifest


class InMemoryManifestStore(ManifestStore):
    """Process-lifetime only. Used in unit tests and short-lived scripts."""

    def __init__(self) -> None:
        self._manifests: dict[str, ScanManifest] = {}

    def get_latest_manifest(self, project_id: str) -> ScanManifest | None:
        return self._manifests.get(project_id)

    def save_manifest(self, manifest: ScanManifest) -> None:
        self._manifests[manifest.project_id] = manifest


class JSONFileManifestStore(ManifestStore):
    """Persists one JSON file per project under `store_dir`. Not safe for
    concurrent writers (no locking) — fine for a single-scanner-process
    local setup, not fine once Part 2's Celery workers run scans
    concurrently across projects on the same store, at which point Part 6
    is the correct store to switch to."""

    def __init__(self, store_dir: Path) -> None:
        self._store_dir = store_dir
        self._store_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, project_id: str) -> Path:
        safe_name = project_id.replace("/", "_")
        return self._store_dir / f"{safe_name}.manifest.json"

    def get_latest_manifest(self, project_id: str) -> ScanManifest | None:
        path = self._path_for(project_id)
        if not path.exists():
            return None
        return ScanManifest.model_validate(json.loads(path.read_text()))

    def save_manifest(self, manifest: ScanManifest) -> None:
        path = self._path_for(manifest.project_id)
        path.write_text(manifest.model_dump_json(indent=2))
