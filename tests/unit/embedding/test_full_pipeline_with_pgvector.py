"""
The real proof: scan -> parse -> IR -> chunk -> embed (TF-IDF, since it's
the only embedding path actually runnable in this sandbox) -> store in
live pgvector -> search, and get back the chunk whose content actually
matches the query. This is the test that would catch a shape mismatch
between Part 8's Chunk, Part 9's embedding step, and Part 6's
VectorChunk/PgvectorStore that no single part's own tests would catch.
"""
from pathlib import Path

from semrag.ingestion.chunking.pipeline import chunk_project
from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.ingestion.embedding.pipeline import embed_chunks
from semrag.ingestion.embedding.tfidf_provider import TfidfEmbeddingProvider
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore
from semrag.ingestion.scanner.orchestrator import Scanner
from semrag.storage.postgres.ir_repository import IRGraphRepository
from semrag.storage.vector.models import VectorSearchFilter
from semrag.storage.vector.pgvector_models import EMBEDDING_DIMENSION
from semrag.storage.vector.pgvector_store import PgvectorStore


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_end_to_end_chunk_embed_store_search(tmp_path: Path, db_session):
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, output logic scl, output logic sda);\n"
        "  // I2C master controller implements serial clock generation sda scl bus arbitration\n"
        "  // start condition stop condition acknowledge bit clock stretching protocol handshake\n"
        "endmodule\n",
    )
    _write(
        tmp_path, "src/regression_runner.py",
        '"""Nightly regression harness collects simulation results coverage database."""\n'
        "import subprocess\n"
        "\n"
        "\n"
        "def run_all_tests():\n"
        "    subprocess.run(['make', 'regress', 'coverage', 'report', 'seed', 'testlist'])\n",
    )

    manifest_store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=manifest_store)
    manifest, _ = scanner.scan("proj1", str(tmp_path))

    parser_registry = build_default_registry()
    parse_results = {
        rel: r for rel, rec in manifest.files.items()
        if (r := parser_registry.parse_file(tmp_path / rel, rec)) is not None
    }
    ir_graph = IRBuilder().build("proj1", manifest.files, parse_results)
    chunks = chunk_project(str(tmp_path), manifest.files, parse_results, ir_graph)
    assert len(chunks) >= 2

    provider = TfidfEmbeddingProvider(dimension=EMBEDDING_DIMENSION)
    provider.fit([c.text for c in chunks])
    vector_chunks = embed_chunks(chunks, provider, InMemoryEmbeddingCache())

    # Chunks reference ir_nodes via a foreign key (source_node_id) — the
    # IR graph (Part 5) must be persisted before chunks referencing it
    # (Part 6) can be upserted. This ordering requirement is real, not a
    # test-only workaround: it's exactly the order a production ingestion
    # worker must follow too, caught here by a live foreign-key violation
    # on the first run of this test, not reasoned about in advance.
    IRGraphRepository(db_session).save(ir_graph)
    db_session.flush()

    store = PgvectorStore(db_session)
    store.upsert_chunks(vector_chunks)
    db_session.flush()

    # Query with vocabulary that heavily overlaps the I2C module's text
    # specifically — TF-IDF is a term-overlap method, not a semantic one
    # (documented explicitly in tfidf_provider.py), so this is a fair test
    # of what it can actually do, not asking it to infer meaning from
    # vocabulary it has never seen relate.
    query_embedding = provider.embed_texts(
        ["I2C master clock sda scl start condition acknowledge protocol"]
    )[0]
    results = store.search(query_embedding, top_k=2, filters=VectorSearchFilter(project_id="proj1"))

    assert len(results) >= 1
    assert results[0].relative_path == "RTL/i2c_master.sv"
