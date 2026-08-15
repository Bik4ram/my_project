"""
Embedding pipeline.

Cache-aware batching: every chunk's text is hashed first; only chunks
whose hash misses the cache are actually sent to `provider.embed_texts`,
and the cache is populated with every new result before returning — this
what makes re-running ingestion on a project where only a handful of
files changed cheap (Part 1's "incremental generation" requirement),
regardless of which `EmbeddingProvider` is configured. The provider
doesn't need to know about caching at all; it's a pure
text-in/vectors-out function, and caching is layered on top uniformly for
every provider.

Batches (not one `embed_texts` call per chunk, not one call for the whole
project) because both providers benefit from batching for different
reasons: `TfidfEmbeddingProvider`'s transform is vectorized and batching
avoids per-call overhead; `OpenAICompatibleEmbeddingProvider` batching
reduces HTTP round-trips against a local model server. `batch_size` is a
parameter, not hardcoded, since the right batch size trades off
differently for a local in-process provider vs. a network call to a
model server.
"""
from __future__ import annotations

from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.embedding.cache import text_hash
from semrag.ingestion.embedding.interfaces import EmbeddingCache, EmbeddingProvider
from semrag.storage.vector.models import VectorChunk

# Increased default batch size to reduce the number of batches, especially
# for network-based embedding providers (e.g., Ollama, OpenAI-compatible).
# The optimal batch size depends on the provider and the environment.
DEFAULT_BATCH_SIZE = 64


def embed_chunks(
    chunks: list[Chunk],
    provider: EmbeddingProvider,
    cache: EmbeddingCache,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[VectorChunk]:
    if not chunks:
        return []

    from semrag.core.observability import timer, count

    hashes = [text_hash(c.text) for c in chunks]

    # Resolve by unique hash, not by chunk — two chunks can share identical
    # text (e.g. a boilerplate header chunked separately per file), and
    # each unique hash should be embedded at most once per call, not once
    # per chunk that happens to produce it. Checking only the persistent
    # cache before batching (an earlier version's bug) missed this: two
    # chunks with the same brand-new text both looked like cache misses at
    # the time each was checked, so both got sent to the provider. Fixed
    # by deduplicating against an in-call map that's checked *and updated*
    # as results come back, not just checked once against the cache.
    resolved: dict[str, list[float]] = {}
    unresolved_hashes: list[str] = []
    seen_hashes: set[str] = set()

    # One batched cache lookup rather than N individual reads — on the
    # disk-backed cache (JSONFileEmbeddingCache) this turns N file reads
    # into a single dict scan and avoids per-call serialization overhead.
    unique_hashes: list[str] = []
    for h in hashes:
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_hashes.append(h)
    batch_hits = cache.get_many(unique_hashes)
    cache_hits = 0
    for h in unique_hashes:
        if h in batch_hits:
            resolved[h] = batch_hits[h]
            cache_hits += 1
        else:
            unresolved_hashes.append(h)

    count("embedding.cache.hit", cache_hits, model=provider.model_name)
    count("embedding.cache.miss", len(unresolved_hashes), model=provider.model_name)

    # Build a single map of hash -> text once, outside the batch loop —
    # the prior implementation rebuilt it per batch which was O(N) per
    # batch and O(N²) overall on large projects.
    text_by_hash = {h: next(c.text for c, ch in zip(chunks, hashes) if ch == h) for h in (unresolved_hashes or seen_hashes)}

    with timer("embedding.run.duration_ms", model=provider.model_name, total_chunks=len(chunks)):
        for batch_start in range(0, len(unresolved_hashes), batch_size):
            batch_hashes = unresolved_hashes[batch_start : batch_start + batch_size]
            batch_texts = [text_by_hash[h] for h in batch_hashes]
            batch_embeddings = provider.embed_texts(batch_texts)

            new_entries = dict(zip(batch_hashes, batch_embeddings))
            resolved.update(new_entries)
            cache.set_many(new_entries)
        # Persist any deferred writes once at the end of the run rather
        # than per-batch — the JSONFileEmbeddingCache batches its disk
        # flushes for the same reason.
        cache.flush()

    return [
        VectorChunk(
            chunk_id=chunk.chunk_id,
            project_id=chunk.project_id,
            relative_path=chunk.relative_path,
            chunk_type=chunk.chunk_type,
            source_node_id=chunk.source_node_id,
            text=chunk.text,
            embedding=resolved[h],
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            language=chunk.language,
            category=chunk.category,
            parser_name=chunk.parser_name,
            chunker_name=chunk.chunker_name,
            symbol_name=chunk.symbol_name,
        )
        for chunk, h in zip(chunks, hashes)
    ]