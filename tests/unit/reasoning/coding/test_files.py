from pathlib import Path

import pytest

from semrag.reasoning.coding.files import (
    ProjectFileError,
    file_exists,
    patch_file,
    read_file,
    snapshot_paths,
    write_file,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


def test_write_creates_new_file(root: Path):
    result = write_file(root, "rtl/i2c.v", "module i2c; endmodule\n")
    assert result["created"] is True
    assert (root / "rtl" / "i2c.v").read_text() == "module i2c; endmodule\n"


def test_write_replaces_existing_file(root: Path):
    write_file(root, "f.txt", "old")
    result = write_file(root, "f.txt", "new")
    assert result["created"] is False
    assert (root / "f.txt").read_text() == "new"


def test_write_rejects_absolute_path(root: Path):
    with pytest.raises(ProjectFileError) as exc:
        write_file(root, "/etc/passwd", "x")
    assert exc.value.kind == "path"


def test_write_rejects_traversal(root: Path):
    with pytest.raises(ProjectFileError) as exc:
        write_file(root, "../escape.txt", "x")
    assert exc.value.kind == "path"


def test_write_rejects_symlink_escape(root: Path):
    outside = root.parent / "outside.txt"
    outside.write_text("secret")
    link = root / "link.txt"
    link.symlink_to(outside)
    with pytest.raises(ProjectFileError) as exc:
        write_file(root, "link.txt", "x")
    assert exc.value.kind == "path"


def test_patch_exact_once(root: Path):
    write_file(root, "f.sv", "a\nb\nc\n")
    result = patch_file(root, "f.sv", "b", "B")
    assert result["replaced_count"] == 1
    assert read_file(root, "f.sv") == "a\nB\nc\n"


def test_patch_zero_matches_fails(root: Path):
    write_file(root, "f.sv", "a\nb\n")
    with pytest.raises(ProjectFileError) as exc:
        patch_file(root, "f.sv", "zzz", "yyy")
    assert exc.value.kind == "mismatch"
    assert read_file(root, "f.sv") == "a\nb\n"


def test_patch_ambiguous_multi_match_fails(root: Path):
    write_file(root, "f.sv", "dup\ndup\n")
    with pytest.raises(ProjectFileError) as exc:
        patch_file(root, "f.sv", "dup", "DUP")
    assert exc.value.kind == "mismatch"
    assert read_file(root, "f.sv") == "dup\ndup\n"


def test_patch_nonexistent_file_fails(root: Path):
    with pytest.raises(ProjectFileError) as exc:
        patch_file(root, "missing.sv", "a", "b")
    assert exc.value.kind == "not_found"


def test_read_file_line_range(root: Path):
    write_file(root, "f.sv", "l1\nl2\nl3\nl4\n")
    assert read_file(root, "f.sv", start=2, end=3) == "l2\nl3\n"


def test_read_nonexistent_fails(root: Path):
    with pytest.raises(ProjectFileError):
        read_file(root, "nope.sv")


def test_file_exists(root: Path):
    write_file(root, "a.sv", "x")
    assert file_exists(root, "a.sv") is True
    assert file_exists(root, "b.sv") is False
    assert file_exists(root, "../b.sv") is False


def test_snapshot_paths(root: Path):
    write_file(root, "a.sv", "x")
    write_file(root, "sub/b.sv", "y")
    assert snapshot_paths(root) == {"a.sv", "sub/b.sv"}