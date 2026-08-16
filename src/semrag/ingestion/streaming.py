"""
Streaming ingestion pipeline helpers.

The default `chunk_project` returns a fully-materialised list of chunks.
For very large projects, downstream stages (embedding, persistence) can
start as soon as the first batch of chunks is ready rather than waiting
for the entire file set to finish chunking. The streaming helpers here
yield chunks in windowed batches so callers can plug a sliding-window
embedder/persister on top without rewriting the chunker.

Worth keeping in mind: the IR builder must complete before chunking
starts (it reads parse_results), and persistence must complete after
embedding (chunks reference embeddings), so the streaming boundary is
"chunk -> embed -> persist", not earlier. That's still the hot path on
big projects because both chunking and embedding are independently
expensive.
"""
from __future__ import annotations

from collections.abc import Iterator

from semrag.ingestion.chunking.models import Chunk


def stream_chunks_in_windows(
    chunks: list[Chunk],
    window_size: int,
) -> Iterator[list[Chunk]]:
    """Yield fixed-size windows of chunks. ``window_size <= 0`` returns
    the entire list as a single window."""
    if window_size <= 0 or len(chunks) <= window_size:
        yield list(chunks)
        return
    for start in range(0, len(chunks), window_size):
        yield chunks[start : start + window_size]
