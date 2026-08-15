"""
Content hashing for change detection.

Why blake3, not sha256: at the file counts and file sizes this system deals
with (waveform dumps and netlists can be gigabytes; a full repo scan may
hash tens of GB), blake3 is roughly an order of magnitude faster than
sha256 on modern CPUs due to internal parallelism (it's a tree hash), while
providing equivalent collision resistance for our purposes (change
detection, not cryptographic signing of untrusted input). sha256 is kept as
an automatic fallback if the `blake3` package isn't installed in a given
environment, so the system degrades rather than hard-fails — but blake3
should be a pinned dependency in production (Part 2 pyproject.toml), not
left to chance.

Streaming, not `path.read_bytes()`: files here can be far larger than
comfortably fits in memory when many are hashed concurrently (see
orchestrator.py's thread pool). Reading in fixed-size chunks keeps peak
memory bounded regardless of file size or concurrency level.
"""
from __future__ import annotations

from pathlib import Path

_CHUNK_SIZE = 4 * 1024 * 1024  # 4MB — large enough to amortize syscall
                                 # overhead, small enough to keep memory
                                 # bounded under concurrent hashing.

try:
    import blake3

    _ALGORITHM = "blake3"

    def _new_hasher():
        return blake3.blake3()

except ImportError:  # pragma: no cover - exercised only when blake3 absent
    import hashlib

    _ALGORITHM = "sha256"

    def _new_hasher():
        return hashlib.sha256()


def algorithm_name() -> str:
    return _ALGORITHM


def hash_file(path: Path) -> str:
    """Compute the content hash of a file, streaming so memory use is
    independent of file size. Raises FileNotFoundError / PermissionError
    to the caller rather than swallowing them — the orchestrator decides
    how to handle an unreadable file (log + skip), not this function."""
    hasher = _new_hasher()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_bytes(data: bytes) -> str:
    """Used by the change detector's tests and by any in-memory content
    (e.g. content pulled from a git blob without touching the working
    tree) rather than a file on disk."""
    hasher = _new_hasher()
    hasher.update(data)
    return hasher.hexdigest()
