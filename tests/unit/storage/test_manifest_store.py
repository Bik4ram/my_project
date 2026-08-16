from datetime import datetime, timezone

from semrag.ingestion.scanner.models import (
    FileCategory,
    FileRecord,
    GitBlobInfo,
    GitRepositoryMetadata,
    ScanManifest,
)
from semrag.storage.postgres.manifest_store import PostgresManifestStore


def _file_record(relative_path: str, content_hash: str = "abc123") -> FileRecord:
    return FileRecord(
        project_id="proj1",
        absolute_path=f"/repo/{relative_path}",
        relative_path=relative_path,
        size_bytes=100,
        mtime=datetime.now(timezone.utc),
        content_hash=content_hash,
        category=FileCategory.RTL_VERILOG,
        extension=".sv",
        is_binary=False,
        git_info=GitBlobInfo(blob_sha="deadbeef"),
    )


def _manifest(scan_id: str, files: dict[str, FileRecord]) -> ScanManifest:
    return ScanManifest(
        scan_id=scan_id,
        project_id="proj1",
        root_path="/repo",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        git_metadata=GitRepositoryMetadata(is_git_repo=True, current_branch="main", head_commit_sha="aaa111"),
        file_count=len(files),
        total_bytes=sum(f.size_bytes for f in files.values()),
        files=files,
    )


def test_returns_none_when_no_manifest_exists(db_session):
    store = PostgresManifestStore(db_session)
    assert store.get_latest_manifest("nonexistent_project") is None


def test_save_and_retrieve_roundtrip(db_session):
    store = PostgresManifestStore(db_session)
    files = {
        "RTL/i2c_master.sv": _file_record("RTL/i2c_master.sv"),
        "RTL/i2c_slave.sv": _file_record("RTL/i2c_slave.sv"),
    }
    manifest = _manifest("scan-1", files)

    store.save_manifest(manifest)
    db_session.flush()

    retrieved = store.get_latest_manifest("proj1")
    assert retrieved is not None
    assert retrieved.scan_id == "scan-1"
    assert set(retrieved.files.keys()) == {"RTL/i2c_master.sv", "RTL/i2c_slave.sv"}
    assert retrieved.files["RTL/i2c_master.sv"].content_hash == "abc123"
    assert retrieved.files["RTL/i2c_master.sv"].git_info.blob_sha == "deadbeef"
    assert retrieved.git_metadata.current_branch == "main"


def test_area_roots_roundtrip_through_manifest(db_session):
    store = PostgresManifestStore(db_session)
    manifest = _manifest("scan-1", {"rtl/i2c_master.sv": _file_record("rtl/i2c_master.sv")})
    manifest.rtl_root = "/srv/rtl"
    manifest.tb_root = "/srv/tb"
    manifest.docs_root = "/srv/docs"

    store.save_manifest(manifest)
    db_session.flush()

    retrieved = store.get_latest_manifest("proj1")
    assert retrieved.rtl_root == "/srv/rtl"
    assert retrieved.tb_root == "/srv/tb"
    assert retrieved.docs_root == "/srv/docs"


def test_legacy_save_does_not_clobber_area_roots(db_session):
    store = PostgresManifestStore(db_session)
    multi = _manifest("scan-1", {"rtl/top.sv": _file_record("rtl/top.sv")})
    multi.rtl_root = "/srv/rtl"
    store.save_manifest(multi)
    db_session.flush()

    # A later legacy single-root save (area roots None) must not erase them.
    legacy = _manifest("scan-2", {"RTL/top.sv": _file_record("RTL/top.sv")})
    store.save_manifest(legacy)
    db_session.flush()

    retrieved = store.get_latest_manifest("proj1")
    assert retrieved.rtl_root == "/srv/rtl"


def test_second_scan_updates_modified_file(db_session):
    store = PostgresManifestStore(db_session)
    files_v1 = {"RTL/i2c_master.sv": _file_record("RTL/i2c_master.sv", content_hash="hash_v1")}
    store.save_manifest(_manifest("scan-1", files_v1))
    db_session.flush()

    files_v2 = {"RTL/i2c_master.sv": _file_record("RTL/i2c_master.sv", content_hash="hash_v2")}
    store.save_manifest(_manifest("scan-2", files_v2))
    db_session.flush()

    retrieved = store.get_latest_manifest("proj1")
    assert retrieved.scan_id == "scan-2"
    assert retrieved.files["RTL/i2c_master.sv"].content_hash == "hash_v2"


def test_deleted_file_is_removed_from_current_state(db_session):
    store = PostgresManifestStore(db_session)
    files_v1 = {
        "RTL/i2c_master.sv": _file_record("RTL/i2c_master.sv"),
        "RTL/i2c_slave.sv": _file_record("RTL/i2c_slave.sv"),
    }
    store.save_manifest(_manifest("scan-1", files_v1))
    db_session.flush()

    # i2c_slave.sv removed in the next scan
    files_v2 = {"RTL/i2c_master.sv": _file_record("RTL/i2c_master.sv")}
    store.save_manifest(_manifest("scan-2", files_v2))
    db_session.flush()

    retrieved = store.get_latest_manifest("proj1")
    assert set(retrieved.files.keys()) == {"RTL/i2c_master.sv"}


def test_two_projects_do_not_interfere(db_session):
    store = PostgresManifestStore(db_session)

    manifest_a = _manifest("scan-a", {"RTL/a.sv": _file_record("RTL/a.sv")})
    manifest_a.project_id = "proj_a"
    manifest_a.files["RTL/a.sv"].project_id = "proj_a"

    manifest_b = _manifest("scan-b", {"RTL/b.sv": _file_record("RTL/b.sv")})
    manifest_b.project_id = "proj_b"
    manifest_b.files["RTL/b.sv"].project_id = "proj_b"

    store.save_manifest(manifest_a)
    store.save_manifest(manifest_b)
    db_session.flush()

    retrieved_a = store.get_latest_manifest("proj_a")
    retrieved_b = store.get_latest_manifest("proj_b")
    assert set(retrieved_a.files.keys()) == {"RTL/a.sv"}
    assert set(retrieved_b.files.keys()) == {"RTL/b.sv"}
