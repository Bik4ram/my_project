"""
Tests for the retrieval benchmark runner.

Covers:
- chunk categorization (LOG/RTL/SPEC) and category counting
- per-query metric computation (top-1/3/5 recall, MRR, average rank)
- aggregation across queries (mean_average_rank excludes no-hit queries)
- report serialization (strict-JSON safe, human summary format)
- report persistence to retrieval_benchmark.json
- an end-to-end run against the real pipeline + a small indexed project
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from semrag.evaluation.benchmark_runner import (
    RetrievalBenchmark,
    categorize_chunk,
    count_categories,
    format_human_summary,
    to_json_dict,
    write_benchmark_reports,
)
from semrag.evaluation.models import ChunkCategory, GoldenQuery
from semrag.retrieval.models import RetrievalResult


def _result(
    chunk_id: str,
    *,
    chunk_type: str,
    relative_path: str,
    fused_score: float = 0.5,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="proj1",
        relative_path=relative_path,
        chunk_type=chunk_type,
        text="",
        fused_score=fused_score,
        signal_scores={},
    )


class _StubRetriever:
    """Canned-results retriever; records the queries it was given."""

    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]):
        self._results = results_by_query
        self.queries: list[str] = []

    def retrieve(self, query_text, query_embedding, project_id, top_k=10, filters=None):
        self.queries.append(query_text)
        return self._results.get(query_text, [])[:top_k]


# ── Categorization ──────────────────────────────────────────────────────


class TestCategorizeChunk:
    def test_log_by_chunk_type(self):
        r = _result("c1", chunk_type="log", relative_path="sim/log/run.log")
        assert categorize_chunk(r) is ChunkCategory.LOG

    def test_log_by_path_when_chunk_type_is_generic(self):
        # .log files are chunked as generic_window; the path carries the signal.
        r = _result("c2", chunk_type="generic_window", relative_path="sim/log/run.log")
        assert categorize_chunk(r) is ChunkCategory.LOG
        r2 = _result("c3", chunk_type="generic_window", relative_path="output/trace.log")
        assert categorize_chunk(r2) is ChunkCategory.LOG

    def test_rtl_by_chunk_type(self):
        r = _result("c4", chunk_type="sv_module", relative_path="src/i2c_master.sv")
        assert categorize_chunk(r) is ChunkCategory.RTL

    def test_rtl_by_path(self):
        r = _result("c5", chunk_type="generic_window", relative_path="rtl/top.sv")
        assert categorize_chunk(r) is ChunkCategory.RTL

    def test_spec_by_chunk_type(self):
        r = _result("c6", chunk_type="markdown_section", relative_path="docs/architecture/i2c.md")
        assert categorize_chunk(r) is ChunkCategory.SPEC

    def test_spec_by_path(self):
        r = _result("c7", chunk_type="generic_window", relative_path="specs/interface.txt")
        assert categorize_chunk(r) is ChunkCategory.SPEC

    def test_uncategorized_returns_none(self):
        r = _result("c8", chunk_type="py_function", relative_path="src/tools.py")
        assert categorize_chunk(r) is None

    def test_log_substring_does_not_bleed_into_paths(self):
        # "catalog" contains "log" as a substring but is not a log dir.
        r = _result("c9", chunk_type="sv_module", relative_path="src/catalog/i2c.sv")
        assert categorize_chunk(r) is ChunkCategory.RTL

    def test_category_counts(self):
        results = [
            _result("a", chunk_type="log", relative_path="sim/log/a.log"),
            _result("b", chunk_type="sv_module", relative_path="rtl/b.sv"),
            _result("c", chunk_type="markdown_section", relative_path="docs/c.md"),
            _result("d", chunk_type="py_function", relative_path="src/d.py"),
        ]
        counts = count_categories(results)
        assert counts == {ChunkCategory.LOG: 1, ChunkCategory.RTL: 1, ChunkCategory.SPEC: 1}


# ── Per-query metrics ───────────────────────────────────────────────────


class TestPerQueryMetrics:
    def test_perfect_first_position(self):
        rtl1 = _result("rtl1", chunk_type="sv_module", relative_path="rtl/a.sv")
        rtl2 = _result("rtl2", chunk_type="sv_module", relative_path="rtl/b.sv")
        log = _result("log1", chunk_type="generic_window", relative_path="sim/log/run.log")
        results = [rtl1, rtl2, log]

        golden = [GoldenQuery(
            query_id="q1", project_id="proj1", query_text="what is rtl a",
            relevant_chunk_ids={rtl1.chunk_id},
        )]
        report = RetrievalBenchmark(_StubRetriever({"what is rtl a": results}), lambda q: [], top_k=5).run(golden)

        q = report.per_query_results[0]
        assert q.top1_recall == 1.0
        assert q.top3_recall == 1.0
        assert q.top5_recall == 1.0
        assert q.mrr == 1.0
        assert q.average_rank == 1.0
        assert q.rtl_count == 2
        assert q.log_count == 1
        assert q.spec_count == 0

    def test_relevant_at_third_rank(self):
        results = [
            _result("a", chunk_type="generic_window", relative_path="sim/log/a.log"),
            _result("b", chunk_type="generic_window", relative_path="sim/log/b.log"),
            _result("target", chunk_type="sv_module", relative_path="rtl/target.sv"),
        ]
        golden = [GoldenQuery(
            query_id="q2", project_id="proj1", query_text="target query",
            relevant_chunk_ids={"target"},
        )]
        report = RetrievalBenchmark(_StubRetriever({"target query": results}), lambda q: [], top_k=5).run(golden)

        q = report.per_query_results[0]
        assert q.top1_recall == 0.0
        assert q.top3_recall == 1.0
        assert q.mrr == pytest.approx(1 / 3)
        assert q.average_rank == pytest.approx(3.0)
        assert q.log_count == 2
        assert q.rtl_count == 1

    def test_no_relevant_hit_marks_infinite_rank(self):
        results = [_result("a", chunk_type="py_function", relative_path="src/a.py")]
        golden = [GoldenQuery(
            query_id="q3", project_id="proj1", query_text="no match",
            relevant_chunk_ids={"missing"},
        )]
        report = RetrievalBenchmark(_StubRetriever({"no match": results}), lambda q: [], top_k=5).run(golden)

        q = report.per_query_results[0]
        assert q.mrr == 0.0
        assert q.average_rank == float("inf")
        assert q.top1_recall == 0.0


# ── Aggregation ─────────────────────────────────────────────────────────


class TestAggregation:
    def test_means_over_queries(self):
        a = _result("a", chunk_type="sv_module", relative_path="rtl/a.sv")
        b = _result("b", chunk_type="sv_module", relative_path="rtl/b.sv")
        results_a_first = [a, b]
        results_b_first = [b, a]

        golden = [
            GoldenQuery(query_id="q1", project_id="proj1", query_text="rtl one", relevant_chunk_ids={a.chunk_id}),
            GoldenQuery(query_id="q2", project_id="proj1", query_text="rtl two", relevant_chunk_ids={a.chunk_id}),
        ]
        stub = _StubRetriever({"rtl one": results_a_first, "rtl two": results_b_first})
        report = RetrievalBenchmark(stub, lambda q: [], top_k=5).run(golden)

        # q1: a at rank 1. q2: a at rank 2.
        assert report.query_count == 2
        assert report.top1_recall == pytest.approx(0.5)
        assert report.top3_recall == pytest.approx(1.0)
        assert report.top5_recall == pytest.approx(1.0)
        assert report.mean_mrr == pytest.approx(0.75)
        assert report.mean_average_rank == pytest.approx(1.5)
        assert report.mean_rtl_count == pytest.approx(2.0)

    def test_mean_average_rank_skips_no_hit_queries(self):
        a = _result("a", chunk_type="sv_module", relative_path="rtl/a.sv")
        results = [a]
        golden = [
            GoldenQuery(query_id="q1", project_id="proj1", query_text="hit", relevant_chunk_ids={a.chunk_id}),
            GoldenQuery(query_id="q2", project_id="proj1", query_text="miss", relevant_chunk_ids={"missing"}),
        ]
        stub = _StubRetriever({"hit": results, "miss": results})
        report = RetrievalBenchmark(stub, lambda q: [], top_k=5).run(golden)

        # Only q1 contributes a finite rank (1.0); q2's inf is excluded.
        assert report.mean_average_rank == pytest.approx(1.0)
        assert report.mean_mrr == pytest.approx(0.5)

    def test_empty_benchmark_is_zeroed(self):
        report = RetrievalBenchmark(_StubRetriever({}), lambda q: [], top_k=5).run([])
        assert report.query_count == 0
        assert report.top1_recall == 0.0
        assert report.mean_mrr == 0.0
        assert report.per_query_results == []


# ── Reporting ───────────────────────────────────────────────────────────


class TestReporting:
    def _report(self):
        a = _result("a", chunk_type="sv_module", relative_path="rtl/a.sv")
        results = [a]
        golden = [GoldenQuery(
            query_id="q1", project_id="proj1", query_text="q1",
            relevant_chunk_ids={a.chunk_id},
        )]
        return RetrievalBenchmark(_StubRetriever({"q1": results}), lambda q: [], top_k=10).run(golden)

    def test_to_json_dict_is_strict_json(self):
        report = self._report()
        data = to_json_dict(report)
        dumped = json.dumps(data)
        assert "top1_recall" in dumped
        parsed = json.loads(dumped)
        assert parsed["query_count"] == 1
        assert parsed["top1_recall"] == 1.0

    def test_infinite_rank_serializes_as_null(self):
        results = [_result("a", chunk_type="log", relative_path="sim/log/a.log")]
        golden = [GoldenQuery(
            query_id="q1", project_id="proj1", query_text="q1",
            relevant_chunk_ids={"missing"},
        )]
        report = RetrievalBenchmark(_StubRetriever({"q1": results}), lambda q: [], top_k=5).run(golden)
        data = to_json_dict(report)
        # Strict JSON cannot carry inf; it must come out as null.
        assert data["per_query_results"][0]["average_rank"] is None
        json.loads(json.dumps(data))

    def test_human_summary_contains_labels(self):
        text = format_human_summary(self._report())
        assert "Retrieval Benchmark" in text
        assert "Top1 Recall" in text
        assert "Top3 Recall" in text
        assert "Top5 Recall" in text
        assert "Average Rank" in text
        assert "Average Logs" in text
        assert "Average RTL" in text
        assert "Average Specs" in text

    def test_human_summary_percentage_format(self):
        report = self._report()
        # top1_recall == 1.0 -> "100%"
        assert "Top1 Recall ........ 100%" in format_human_summary(report)

    def test_write_benchmark_reports_writes_both_files(self, tmp_path: Path):
        report = self._report()
        json_path, summary_path = write_benchmark_reports(
            report,
            tmp_path / "retrieval_benchmark.json",
            tmp_path / "retrieval_benchmark.txt",
        )
        assert json_path.exists()
        assert summary_path is not None and summary_path.exists()

        loaded = json.loads(json_path.read_text())
        assert loaded["query_count"] == 1
        assert loaded["per_query_results"][0]["query_id"] == "q1"

        summary_text = summary_path.read_text()
        assert "Retrieval Benchmark" in summary_text
        assert "Top1 Recall" in summary_text


# ── Integration: real pipeline against an indexed project ──────────────


def test_benchmark_end_to_end_against_real_pipeline(tmp_path: Path, db_session):
    """Index a small project with RTL + log + spec material and run the
    benchmark through the real pipeline; the top-k category composition
    must reflect what was indexed."""
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, input logic sda, output logic scl);\n"
        "  logic start_detected;\n"
        "  always_ff @(posedge clk) begin\n"
        "    start_detected <= (scl === 1'b1) && (sda === 1'b0);\n"
        "  end\n"
        "  assert property (@(posedge clk) scl == 1'b0)\n"
        "    else $error(\"start condition assertion failed\");\n"
        "endmodule\n",
    )
    _write(
        tmp_path, "sim/log/run.log",
        "[00:00:00] uvm_error: start condition timeout on i2c master\n"
        "[00:00:01] uvm_error: scl stuck high detected\n"
        "[00:00:02] simulation timeout after 1000 cycles\n"
        "[00:00:03] uvm_error: fifo overflow in i2c master\n",
    )
    _write(
        tmp_path, "docs/architecture/i2c_spec.md",
        "# i2c master interface\n\n"
        "The i2c master drives the scl and sda lines and detects the start "
        "condition. The start condition asserts when scl is high and sda "
        "falls low.\n",
    )

    provider, vector_chunks = _index(tmp_path, db_session)
    i2c_ids = {c.chunk_id for c in vector_chunks if c.relative_path == "RTL/i2c_master.sv"}
    log_ids = {c.chunk_id for c in vector_chunks if c.relative_path == "sim/log/run.log"}

    golden = [
        GoldenQuery(
            query_id="rtl", project_id="proj1",
            query_text="what module detects the start condition on the i2c sda line",
            relevant_chunk_ids=i2c_ids,
        ),
        GoldenQuery(
            query_id="debug", project_id="proj1",
            query_text="why does the i2c simulation show a uvm_error timeout",
            relevant_chunk_ids=log_ids,
        ),
    ]

    retriever = _build_retriever(db_session)
    runner = RetrievalBenchmark(
        retriever,
        embed_query_fn=lambda q: provider.embed_texts([q])[0],
        top_k=5,
    )
    report = runner.run(golden)

    assert report.query_count == 2
    assert report.top5_recall >= report.top1_recall
    assert report.mean_rtl_count > 0
    assert report.mean_log_count > 0

    by_id = {r.query_id: r for r in report.per_query_results}
    assert by_id["rtl"].rtl_count > 0
    assert by_id["debug"].log_count > 0
    assert by_id["rtl"].top1_recall > 0

    # The report serializes cleanly.
    json.loads(json.dumps(to_json_dict(report)))


# ── Shared helpers ──────────────────────────────────────────────────────


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _index(tmp_path: Path, db_session):
    from semrag.ingestion.chunking.pipeline import chunk_project
    from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
    from semrag.ingestion.embedding.pipeline import embed_chunks
    from semrag.ingestion.embedding.tfidf_provider import TfidfEmbeddingProvider
    from semrag.ingestion.ir.builder import IRBuilder
    from semrag.ingestion.parsers.pipeline import build_default_registry
    from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore
    from semrag.ingestion.scanner.orchestrator import Scanner
    from semrag.storage.postgres.ir_repository import IRGraphRepository
    from semrag.storage.vector.pgvector_models import EMBEDDING_DIMENSION
    from semrag.storage.vector.pgvector_store import PgvectorStore

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
    return provider, vector_chunks


def _build_retriever(db_session):
    from semrag.retrieval.pipeline import HybridRetriever

    return HybridRetriever(db_session, reranker=None, enable_graph_expansion=False)
