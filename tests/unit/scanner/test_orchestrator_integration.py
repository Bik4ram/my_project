"""
End-to-end test of the scanner against a *real* git repository built in a
tmp_path fixture (via actual `git` subprocess calls, not mocks) — this is
the test that would catch a regression in the bulk git ls-tree/log logic
that a pure-mock test would happily pass despite being wrong.
"""
import subprocess
from pathlib import Path

import pytest

from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore
from semrag.ingestion.scanner.models import FileCategory
from semrag.ingestion.scanner.orchestrator import Scanner


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    repo = tmp_path / "i2c_project"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "dv@example.com")
    _git(repo, "config", "user.name", "DV Engineer")

    (repo / "RTL").mkdir()
    (repo / "TB" / "uvm").mkdir(parents=True)
    (repo / "Waveforms").mkdir()

    (repo / "RTL" / "i2c_master.sv").write_text(
        '`include "i2c_defines.svh"\nmodule i2c_master; endmodule\n'
    )
    (repo / "TB" / "uvm" / "i2c_driver.sv").write_text(
        "import i2c_pkg::*;\nclass i2c_driver extends uvm_driver; endclass\n"
    )
    (repo / "Waveforms" / "dump.vcd").write_bytes(b"\x00fake vcd content")

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "initial commit")

    return repo


def test_first_scan_marks_everything_added(fake_project: Path):
    store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=store)

    manifest, changes = scanner.scan("i2c_project", str(fake_project))

    # Waveforms/ is pruned by the walker regardless of git tracking status.
    assert "Waveforms/dump.vcd" not in manifest.files
    assert "RTL/i2c_master.sv" in manifest.files
    assert "TB/uvm/i2c_driver.sv" in manifest.files

    assert changes.previous_scan_id is None
    assert len(changes.added) == manifest.file_count
    assert changes.modified == []
    assert changes.deleted_paths == []

    rtl_record = manifest.files["RTL/i2c_master.sv"]
    assert rtl_record.category == FileCategory.RTL_VERILOG
    assert rtl_record.raw_dependency_refs == ["i2c_defines.svh"]
    assert rtl_record.git_info is not None
    assert rtl_record.git_info.blob_sha

    tb_record = manifest.files["TB/uvm/i2c_driver.sv"]
    assert tb_record.category == FileCategory.TESTBENCH_UVM
    assert "i2c_pkg" in tb_record.raw_dependency_refs


def test_incremental_scan_detects_unchanged_modified_added_deleted(fake_project: Path):
    store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=store)

    manifest_1, _ = scanner.scan("i2c_project", str(fake_project))
    original_rtl_hash = manifest_1.files["RTL/i2c_master.sv"].content_hash

    # Modify one file, add a new one, delete another, commit.
    (fake_project / "RTL" / "i2c_master.sv").write_text(
        '`include "i2c_defines.svh"\nmodule i2c_master; // added a comment\nendmodule\n'
    )
    (fake_project / "RTL" / "i2c_slave.sv").write_text("module i2c_slave; endmodule\n")
    (fake_project / "TB" / "uvm" / "i2c_driver.sv").unlink()

    _git(fake_project, "add", "-A")
    _git(fake_project, "commit", "-q", "-m", "modify master, add slave, remove driver")

    manifest_2, changes = scanner.scan("i2c_project", str(fake_project))

    assert changes.previous_scan_id == manifest_1.scan_id
    assert "RTL/i2c_slave.sv" in [r.relative_path for r in changes.added]
    assert "RTL/i2c_master.sv" in [r.relative_path for r in changes.modified]
    assert "TB/uvm/i2c_driver.sv" in changes.deleted_paths

    modified_record = manifest_2.files["RTL/i2c_master.sv"]
    assert modified_record.content_hash != original_rtl_hash

    assert "TB/uvm/i2c_driver.sv" not in manifest_2.files


def test_second_scan_with_no_changes_reuses_hashes_via_git_blob_sha(fake_project: Path):
    store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=store)

    manifest_1, _ = scanner.scan("i2c_project", str(fake_project))
    manifest_2, changes = scanner.scan("i2c_project", str(fake_project))

    assert changes.has_changes is False
    assert changes.unchanged_count == manifest_1.file_count

    # Content hashes should be byte-identical across scans since nothing
    # changed (this also proves the "reuse previous record" fast path
    # didn't silently drop or corrupt fields).
    for path, record in manifest_1.files.items():
        assert manifest_2.files[path].content_hash == record.content_hash


def test_git_repository_metadata_captured(fake_project: Path):
    store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=store)

    manifest, _ = scanner.scan("i2c_project", str(fake_project))

    assert manifest.git_metadata.is_git_repo is True
    assert manifest.git_metadata.head_commit_sha is not None
    assert manifest.git_metadata.is_dirty is False
