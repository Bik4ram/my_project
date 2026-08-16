"""Multi-root scanning: each area root is walked independently and every
file's relative_path is area-prefixed into one shared manifest."""
from __future__ import annotations

from pathlib import Path

from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore
from semrag.ingestion.scanner.models import FileCategory
from semrag.ingestion.scanner.orchestrator import Scanner


def _build_areas(tmp_path: Path):
    rtl = tmp_path / "rtl_src"
    tb = tmp_path / "tb_src"
    docs = tmp_path / "docs"
    for p in (rtl, tb, docs):
        p.mkdir()
    (rtl / "i2c_master.sv").write_text("module i2c_master; endmodule\n")
    (rtl / "i2c_pkg.sv").write_text("package i2c_pkg; endpackage\n")
    uvm = tb / "uvm"
    uvm.mkdir()
    (uvm / "i2c_test.sv").write_text("class i2c_test; endclass\n")
    (docs / "README.md").write_text("# I2C\n")
    return {
        "rtl": str(rtl),
        "tb": str(tb),
        "docs": str(docs),
    }


def test_scan_multi_root_prefixes_all_areas(tmp_path: Path):
    store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=store)
    area_roots = _build_areas(tmp_path)

    manifest, changes = scanner.scan_multi_root("chip", area_roots)

    assert manifest.project_id == "chip"
    assert manifest.rtl_root == area_roots["rtl"]
    assert manifest.tb_root == area_roots["tb"]
    assert manifest.docs_root == area_roots["docs"]

    paths = set(manifest.files)
    assert paths == {
        "rtl/i2c_master.sv",
        "rtl/i2c_pkg.sv",
        "tb/uvm/i2c_test.sv",
        "docs/README.md",
    }
    assert manifest.files["rtl/i2c_master.sv"].category == FileCategory.RTL_VERILOG
    assert manifest.files["tb/uvm/i2c_test.sv"].category == FileCategory.TESTBENCH_UVM
    assert manifest.files["docs/README.md"].category == FileCategory.MARKDOWN

    assert manifest.file_count == 4
    assert len(changes.added) == 4
    assert manifest.files["rtl/i2c_master.sv"].absolute_path == str(
        Path(area_roots["rtl"]) / "i2c_master.sv"
    )


def test_scan_multi_root_ignores_missing_area(tmp_path: Path):
    store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=store)
    rtl = tmp_path / "rtl"
    rtl.mkdir()
    (rtl / "top.sv").write_text("module top; endmodule\n")

    manifest, _ = scanner.scan_multi_root("chip", {"rtl": str(rtl), "tb": None})

    assert set(manifest.files) == {"rtl/top.sv"}
    assert manifest.tb_root is None


def test_scan_multi_root_second_pass_reuses_git_blobs(tmp_path: Path):
    # Non-git, second scan: no change is detected and hashes are stable.
    store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=store)
    area_roots = _build_areas(tmp_path)

    manifest_1, _ = scanner.scan_multi_root("chip", area_roots)
    manifest_2, changes = scanner.scan_multi_root("chip", area_roots)

    assert changes.has_changes is False
    assert changes.unchanged_count == 4
    for path, record in manifest_1.files.items():
        assert manifest_2.files[path].content_hash == record.content_hash


def test_scan_multi_root_detects_deleted_and_added(tmp_path: Path):
    store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=store)
    area_roots = _build_areas(tmp_path)

    scanner.scan_multi_root("chip", area_roots)

    (Path(area_roots["tb"]) / "uvm" / "i2c_test.sv").unlink()
    (Path(area_roots["rtl"]) / "i2c_slave.sv").write_text("module i2c_slave; endmodule\n")

    _, changes = scanner.scan_multi_root("chip", area_roots)

    assert "tb/uvm/i2c_test.sv" in changes.deleted_paths
    assert "rtl/i2c_slave.sv" in [r.relative_path for r in changes.added]
