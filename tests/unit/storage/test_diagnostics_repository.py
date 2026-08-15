from datetime import datetime, timezone

from semrag.ingestion.parsers.models import DiagnosticSeverity, ParseDiagnostic, ParseResult
from semrag.ingestion.scanner.models import GitRepositoryMetadata, ScanManifest
from semrag.storage.postgres.diagnostics_repository import DiagnosticsRepository
from semrag.storage.postgres.manifest_store import PostgresManifestStore
from semrag.storage.postgres.models import Project


def _seed_scan(db_session, project_id: str, scan_id: str) -> None:
    """Diagnostics reference a scan_id FK, and latest_diagnostics_for_project
    looks up the latest scan_manifests row — both require a real scan row
    to exist first, not just a Project row."""
    manifest = ScanManifest(
        scan_id=scan_id,
        project_id=project_id,
        root_path="/repo",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        git_metadata=GitRepositoryMetadata(is_git_repo=False),
        file_count=0,
        total_bytes=0,
        files={},
    )
    PostgresManifestStore(db_session).save_manifest(manifest)


def test_record_and_query_diagnostics(db_session):
    _seed_scan(db_session, "proj1", "scan-1")
    repo = DiagnosticsRepository(db_session)

    results = [
        ParseResult(
            project_id="proj1",
            relative_path="RTL/i2c_master.sv",
            parser_name="pyslang_syntax",
            success=True,
            diagnostics=[
                ParseDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    message="CouldNotOpenIncludeFile",
                    line=1,
                    column=1,
                )
            ],
        ),
        ParseResult(
            project_id="proj1",
            relative_path="RTL/i2c_slave.sv",
            parser_name="pyslang_syntax",
            success=True,
            diagnostics=[],
        ),
    ]
    repo.record("proj1", "scan-1", results)
    db_session.flush()

    diagnostics = repo.latest_diagnostics_for_project("proj1")
    assert len(diagnostics) == 1
    assert diagnostics[0].relative_path == "RTL/i2c_master.sv"
    assert diagnostics[0].severity == "error"
    assert diagnostics[0].message == "CouldNotOpenIncludeFile"


def test_no_diagnostics_when_no_scan_exists(db_session):
    repo = DiagnosticsRepository(db_session)
    assert repo.latest_diagnostics_for_project("nonexistent") == []


def test_only_latest_scans_diagnostics_are_returned(db_session):
    _seed_scan(db_session, "proj1", "scan-1")
    repo = DiagnosticsRepository(db_session)
    repo.record(
        "proj1",
        "scan-1",
        [
            ParseResult(
                project_id="proj1",
                relative_path="RTL/old.sv",
                parser_name="pyslang_syntax",
                success=False,
                diagnostics=[ParseDiagnostic(severity=DiagnosticSeverity.ERROR, message="old error")],
            )
        ],
    )
    db_session.flush()

    _seed_scan(db_session, "proj1", "scan-2")
    repo.record(
        "proj1",
        "scan-2",
        [
            ParseResult(
                project_id="proj1",
                relative_path="RTL/new.sv",
                parser_name="pyslang_syntax",
                success=False,
                diagnostics=[ParseDiagnostic(severity=DiagnosticSeverity.ERROR, message="new error")],
            )
        ],
    )
    db_session.flush()

    diagnostics = repo.latest_diagnostics_for_project("proj1")
    assert len(diagnostics) == 1
    assert diagnostics[0].message == "new error"
