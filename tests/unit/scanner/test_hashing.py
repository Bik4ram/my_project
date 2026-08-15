from pathlib import Path

from semrag.ingestion.scanner import hashing


def test_hash_file_is_deterministic(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello semiconductor world")
    h1 = hashing.hash_file(f)
    h2 = hashing.hash_file(f)
    assert h1 == h2
    assert len(h1) > 0


def test_hash_file_changes_with_content(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("version one")
    h1 = hashing.hash_file(f)
    f.write_text("version two")
    h2 = hashing.hash_file(f)
    assert h1 != h2


def test_hash_bytes_matches_hash_file(tmp_path: Path):
    f = tmp_path / "a.txt"
    content = b"module top; endmodule"
    f.write_bytes(content)
    assert hashing.hash_file(f) == hashing.hash_bytes(content)


def test_hash_large_file_streams_correctly(tmp_path: Path):
    f = tmp_path / "big.bin"
    # Larger than the internal chunk size to exercise the streaming loop.
    chunk = b"x" * (1024 * 1024)
    with f.open("wb") as fh:
        for _ in range(6):  # ~6MB, > 4MB chunk size
            fh.write(chunk)
    h = hashing.hash_file(f)
    assert len(h) > 0
