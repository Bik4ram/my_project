from semrag.ingestion.chunking.models import Chunk
from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.ingestion.embedding.pipeline import embed_chunks


class _CountingFakeProvider:
    """A deterministic fake provider (one float per text, its length) that
    counts how many texts it was actually asked to embed — this is what
    lets the cache-hit-skips-provider-call behavior be verified precisely,
    without needing a real (or even the TF-IDF) provider involved."""

    def __init__(self, dimension: int = 4):
        self.dimension = dimension
        self.model_name = "counting-fake"
        self.call_count = 0
        self.total_texts_embedded = 0

    def embed_texts(self, texts):
        self.call_count += 1
        self.total_texts_embedded += len(texts)
        return [[float(len(t))] * self.dimension for t in texts]


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, project_id="proj1", relative_path="RTL/a.sv",
        chunk_type="sv_module", text=text,
    )


def test_embed_chunks_produces_vector_chunks_with_correct_fields():
    provider = _CountingFakeProvider(dimension=4)
    cache = InMemoryEmbeddingCache()
    chunks = [_chunk("c1", "hello"), _chunk("c2", "world!!")]

    result = embed_chunks(chunks, provider, cache)

    assert len(result) == 2
    assert result[0].chunk_id == "c1"
    assert result[0].embedding == [5.0, 5.0, 5.0, 5.0]  # len("hello") == 5
    assert result[1].embedding == [7.0, 7.0, 7.0, 7.0]  # len("world!!") == 7
    assert result[0].project_id == "proj1"
    assert result[0].chunk_type == "sv_module"


def test_repeated_text_hits_cache_and_skips_provider_call():
    provider = _CountingFakeProvider()
    cache = InMemoryEmbeddingCache()
    chunks = [_chunk("c1", "same text"), _chunk("c2", "same text")]  # identical text, different chunk_id

    embed_chunks(chunks, provider, cache)

    assert provider.total_texts_embedded == 1  # only embedded once despite two chunks


def test_second_pipeline_run_with_unchanged_text_hits_cache():
    provider = _CountingFakeProvider()
    cache = InMemoryEmbeddingCache()
    chunks = [_chunk("c1", "unchanged content")]

    embed_chunks(chunks, provider, cache)
    assert provider.total_texts_embedded == 1

    embed_chunks(chunks, provider, cache)  # re-run, e.g. next ingestion pass
    assert provider.total_texts_embedded == 1  # no new calls — cache hit


def test_changed_text_is_re_embedded():
    provider = _CountingFakeProvider()
    cache = InMemoryEmbeddingCache()

    embed_chunks([_chunk("c1", "version one")], provider, cache)
    embed_chunks([_chunk("c1", "version two")], provider, cache)  # same chunk_id, different text

    assert provider.total_texts_embedded == 2


def test_batching_respects_batch_size():
    provider = _CountingFakeProvider()
    cache = InMemoryEmbeddingCache()
    chunks = [_chunk(f"c{i}", f"text number {i}") for i in range(10)]

    embed_chunks(chunks, provider, cache, batch_size=3)

    assert provider.call_count == 4  # ceil(10/3) = 4 batches
    assert provider.total_texts_embedded == 10


def test_empty_chunk_list_produces_no_calls():
    provider = _CountingFakeProvider()
    cache = InMemoryEmbeddingCache()
    result = embed_chunks([], provider, cache)
    assert result == []
    assert provider.call_count == 0


def test_ri06_metadata_propagates_to_vector_chunk_unchanged():
    provider = _CountingFakeProvider(dimension=4)
    cache = InMemoryEmbeddingCache()
    chunk = Chunk(
        chunk_id="c1",
        project_id="proj1",
        relative_path="RTL/i2c_master.sv",
        chunk_type="sv_module",
        source_node_id="ir-1",
        text="module i2c_master; endmodule",
        line_start=1,
        line_end=2,
        language="systemverilog",
        category="rtl",
        parser_name="pyslang_syntax",
        chunker_name="systemverilog",
        symbol_name="i2c_master",
    )

    result = embed_chunks([chunk], provider, cache)[0]

    assert result.language == "systemverilog"
    assert result.category == "rtl"
    assert result.parser_name == "pyslang_syntax"
    assert result.chunker_name == "systemverilog"
    assert result.symbol_name == "i2c_master"
    assert result.text == "module i2c_master; endmodule"  # embedding text untouched
    assert result.chunk_id == "c1"
    assert result.line_start == 1
    assert result.line_end == 2
