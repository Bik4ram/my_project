"""
Diagnostics persistence.

Simpler than the file/IR sync above: diagnostics are write-once, append-
only per scan (each scan's diagnostics are a new batch, not an update to a
previous batch), so this is a straight insert, not an upsert. Old
diagnostics for a file are superseded by a new scan's diagnostics
implicitly (queries filter to the latest `scan_id` per project — see
`latest_diagnostics_for_project`), rather than being deleted; keeping the
history is cheap and occasionally useful (did this file's diagnostics get
worse over time), unlike the files/IR tables where stale rows are actively
misleading if kept.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from semrag.ingestion.parsers.models import ParseResult
from semrag.storage.postgres.models import ParseDiagnosticRow, ScanManifestRow


class DiagnosticsRepository:
    def __init__(self, session: Session):
        self._session = session

    def record(self, project_id: str, scan_id: str, results: list[ParseResult]) -> None:
        rows = [
            ParseDiagnosticRow(
                project_id=project_id,
                relative_path=result.relative_path,
                parser_name=result.parser_name,
                severity=diag.severity.value,
                message=diag.message,
                line=diag.line,
                column=diag.column,
                scan_id=scan_id,
            )
            for result in results
            for diag in result.diagnostics
        ]
        if rows:
            self._session.add_all(rows)

    def latest_diagnostics_for_project(self, project_id: str) -> list[ParseDiagnosticRow]:
        latest_scan_id = self._session.execute(
            select(ScanManifestRow.scan_id)
            .where(ScanManifestRow.project_id == project_id)
            .order_by(ScanManifestRow.started_at.desc())
            .limit(1)
        ).scalar_one_or_none()

        if latest_scan_id is None:
            return []

        return list(
            self._session.execute(
                select(ParseDiagnosticRow).where(
                    ParseDiagnosticRow.project_id == project_id,
                    ParseDiagnosticRow.scan_id == latest_scan_id,
                )
            ).scalars().all()
        )
