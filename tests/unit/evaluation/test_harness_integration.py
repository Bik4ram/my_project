"""
The real proof of Part 15: index a small real project, build a genuine
golden dataset (the actual chunk_ids the real pipeline produced, so the
"ground truth" is honest rather than guessed), and run the harness
against real live retrieval - checking the metrics computed match what
we already know should be true (the assertion query should find the
assertion chunk with perfect or near-perfect precision).
"""
from pathlib import Path

from semrag.evaluation.harness import EvaluationHarness
from semrag.evaluation.models import GoldenQuery
from semrag.ingestion.chunking.pipeline import chunk_project
from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.ingestion.embedding.pipeline import embed_chunks
from semrag.ingestion.embedding.tfidf_provider import TfidfEmbeddingProvider
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore
from semrag.ingestion.scanner.orchestrator import Scanner
from semrag.retrieval.pipeline import HybridRetriever
from semrag.storage.postgres.ir_repository import IRGraphRepository
from semrag.storage.vector.pgvector_models import EMBEDDING_DIMENSION
from semrag.storage.vector.pgvector_store import PgvectorStore


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_harness_end_to_end_against_real_pipeline(tmp_path: Path, db_session):
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, output logic scl);\n"
        "  assert property (@(posedge clk) scl == 1'b0)\n"
        "    else $error(\"start condition assertion failed\");\n"
        "endmodule\n",
    )
    _write(
        tmp_path, "src/regression_runner.py",
        '"""Nightly regression harness collects coverage reports."""\n'
        "import subprocess\n\n\n"
        "def run_all_tests():\n    subprocess.run(['make', 'regress'])\n",
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

    provider = TfidfEmbeddingProvider(dimension=EMBEDDING_DIMENSION)
    provider.fit([c.text for c in chunks])
    vector_chunks = embed_chunks(chunks, provider, InMemoryEmbeddingCache())

    IRGraphRepository(db_session).save(ir_graph)
    db_session.flush()
    PgvectorStore(db_session).upsert_chunks(vector_chunks)
    db_session.flush()

    i2c_chunk_ids = {c.chunk_id for c in vector_chunks if c.relative_path == "RTL/i2c_master.sv"}
    golden = [
        GoldenQuery(
            query_id="q1", project_id="proj1",
            query_text="why does the start condition assertion fail on the i2c master",
            relevant_chunk_ids=i2c_chunk_ids,
        ),
    ]

    retriever = HybridRetriever(db_session, reranker=None, enable_graph_expansion=False)
    harness = EvaluationHarness(retriever, embed_query_fn=lambda q: provider.embed_texts([q])[0], k=5)

    report = harness.evaluate_all(golden)

    assert report.query_count == 1
    assert report.mean_precision_at_k > 0
    assert report.mean_mrr > 0
    assert report.latency_p50_ms >= 0
    assert report.per_query_results[0].estimated_total_tokens > 0


def test_harness_reports_zero_precision_for_irrelevant_ground_truth(tmp_path: Path, db_session):
    """Sanity check on the harness itself: if the golden dataset points
    at chunk_ids that don't exist in the project at all, precision/recall
    must genuinely be zero, not silently pass."""
    _write(tmp_path, "RTL/top.sv", "module top; endmodule\n")

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

    provider = TfidfEmbeddingProvider(dimension=EMBEDDING_DIMENSION)
    provider.fit([c.text for c in chunks])
    vector_chunks = embed_chunks(chunks, provider, InMemoryEmbeddingCache())

    IRGraphRepository(db_session).save(ir_graph)
    db_session.flush()
    PgvectorStore(db_session).upsert_chunks(vector_chunks)
    db_session.flush()

    golden = [
        GoldenQuery(
            query_id="q1", project_id="proj1", query_text="top module",
            relevant_chunk_ids={"proj1::nonexistent::sv_module::fake"},
        )
    ]
    retriever = HybridRetriever(db_session, reranker=None, enable_graph_expansion=False)
    harness = EvaluationHarness(retriever, embed_query_fn=lambda q: provider.embed_texts([q])[0], k=5)

    report = harness.evaluate_all(golden)
    assert report.mean_precision_at_k == 0.0
    assert report.mean_mrr == 0.0
