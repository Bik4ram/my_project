"""
The real proof of Part 10: a multi-file project run through the entire
pipeline built across Parts 2-10 - scan, parse, IR, chunk, embed, persist
(IR graph + chunks), then hybrid retrieval (vector + lexical + graph
expansion + rerank) against a query, checked against what an engineer
would actually expect back.
"""
from pathlib import Path

import pytest

from semrag.ingestion.chunking.pipeline import chunk_project
from semrag.ingestion.embedding.cache import InMemoryEmbeddingCache
from semrag.ingestion.embedding.pipeline import embed_chunks
from semrag.ingestion.embedding.tfidf_provider import TfidfEmbeddingProvider
from semrag.ingestion.ir.builder import IRBuilder
from semrag.ingestion.parsers.pipeline import build_default_registry
from semrag.ingestion.scanner.manifest_store import InMemoryManifestStore
from semrag.ingestion.scanner.orchestrator import Scanner
from semrag.retrieval.pipeline import HybridRetriever
from semrag.retrieval.reranking import LexicalOverlapReranker
from semrag.storage.postgres.ir_repository import IRGraphRepository
from semrag.storage.vector.models import VectorSearchFilter
from semrag.storage.vector.pgvector_models import EMBEDDING_DIMENSION
from semrag.storage.vector.pgvector_store import PgvectorStore


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _run_full_pipeline(tmp_path: Path, db_session, project_id: str):
    manifest_store = InMemoryManifestStore()
    scanner = Scanner(manifest_store=manifest_store)
    manifest, _ = scanner.scan(project_id, str(tmp_path))

    parser_registry = build_default_registry()
    parse_results = {
        rel: r for rel, rec in manifest.files.items()
        if (r := parser_registry.parse_file(tmp_path / rel, rec)) is not None
    }
    ir_graph = IRBuilder().build(project_id, manifest.files, parse_results)
    chunks = chunk_project(str(tmp_path), manifest.files, parse_results, ir_graph)

    provider = TfidfEmbeddingProvider(dimension=EMBEDDING_DIMENSION)
    provider.fit([c.text for c in chunks])
    vector_chunks = embed_chunks(chunks, provider, InMemoryEmbeddingCache())

    IRGraphRepository(db_session).save(ir_graph)
    db_session.flush()
    PgvectorStore(db_session).upsert_chunks(vector_chunks)
    db_session.flush()

    return provider


def test_hybrid_retrieval_end_to_end(tmp_path: Path, db_session):
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, output logic scl, output logic sda);\n"
        "  logic start_condition;\n"
        "  always_ff @(posedge clk) start_condition <= 1'b0;\n"
        "  assert property (@(posedge clk) start_condition |-> scl == 1'b0)\n"
        "    else $error(\"start condition assertion failed\");\n"
        "endmodule\n",
    )
    _write(
        tmp_path, "RTL/i2c_top.sv",
        "module i2c_top;\n"
        "  logic clk;\n"
        "  i2c_master u_master (.clk(clk));\n"
        "endmodule\n",
    )
    _write(
        tmp_path, "src/regression_runner.py",
        '"""Nightly regression harness collects simulation coverage reports."""\n'
        "import subprocess\n"
        "\n"
        "\n"
        "def run_all_tests():\n"
        "    subprocess.run(['make', 'regress', 'coverage'])\n",
    )

    provider = _run_full_pipeline(tmp_path, db_session, "proj1")

    retriever = HybridRetriever(db_session, reranker=LexicalOverlapReranker(), enable_graph_expansion=True)
    query_text = "why does the start condition assertion fail on the i2c master"
    query_embedding = provider.embed_texts([query_text])[0]

    results, trace = retriever.retrieve(
        query_text, query_embedding, project_id="proj1", top_k=5,
        filters=VectorSearchFilter(project_id="proj1"), return_trace=True,
    )

    assert len(results) > 0
    # The assertion chunk itself (from i2c_master.sv) should be the top hit
    # - it's the most specific, most directly-relevant unit for this query.
    assert results[0].relative_path == "RTL/i2c_master.sv"

    result_paths = {r.relative_path for r in results}
    assert "RTL/i2c_top.sv" in result_paths or len(results) < 5  # allow for top_k truncation

    # Graph expansion scoring metadata: every graph-expanded chunk must
    # record both its depth weight and the producing seed's RRF score, so
    # the failure analyzer can reconstruct the GRAPH stage ordering.
    for ct in trace.chunks():
        if ct.graph_expansion_score is not None:
            assert ct.graph_seed_score is not None


def test_project_filter_isolates_results_across_projects(tmp_path: Path, db_session):
    proj_a_dir = tmp_path / "a"
    proj_b_dir = tmp_path / "b"
    _write(proj_a_dir, "RTL/only_in_a.sv", "module only_in_a; endmodule\n")
    _write(proj_b_dir, "RTL/only_in_b.sv", "module only_in_b; endmodule\n")

    provider_a = _run_full_pipeline(proj_a_dir, db_session, "proj_a")
    _run_full_pipeline(proj_b_dir, db_session, "proj_b")

    retriever = HybridRetriever(db_session, reranker=None, enable_graph_expansion=False)
    query_embedding = provider_a.embed_texts(["module"])[0]
    results = retriever.retrieve(
        "module", query_embedding, project_id="proj_a", top_k=10,
        filters=VectorSearchFilter(project_id="proj_a"),
    )

    assert all(r.project_id == "proj_a" for r in results)
    assert all("only_in_b" not in r.text for r in results)


def test_noisy_source_penalty_scales_fused_score(tmp_path: Path, db_session):
    """The noisy-source penalty multiplicatively scales the fused score of
    log-file chunks, without affecting non-log chunks."""
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master;\n"
        "  logic addr; logic data;\n"
        "  logic monitor_sample;\n"
        "endmodule\n",
    )
    log_lines = []
    for i in range(60):
        log_lines.append(f"  [line {i}] addr={i} data={i} monitor sampled sda word={i}")
    _write(tmp_path, "sim/log/sim.log", "\n".join(log_lines) + "\n")

    provider = _run_full_pipeline(tmp_path, db_session, "proj_log")
    query_text = "how does the monitor sample addr and data"
    query_embedding = provider.embed_texts([query_text])[0]

    # Verify score scaling directly via trace
    retriever_off = HybridRetriever(
        db_session, reranker=LexicalOverlapReranker(), enable_graph_expansion=False,
        noisy_source_penalty=0.0,
    )
    retriever_on = HybridRetriever(
        db_session, reranker=LexicalOverlapReranker(), enable_graph_expansion=False,
        noisy_source_path_terms=("/log/",), noisy_source_penalty=0.5,
    )

    for retriever, name in [(retriever_off, "off"), (retriever_on, "on")]:
        results, trace = retriever.retrieve(
            query_text, query_embedding, project_id="proj_log", top_k=5,
            filters=VectorSearchFilter(project_id="proj_log"), return_trace=True,
        )
        log_chunks = [c for c in trace.chunks() if "sim/log/sim.log" in c.relative_path]
        rtl_chunks = [c for c in trace.chunks() if c.relative_path == "RTL/i2c_master.sv"]
        if name == "off":
            off_log_score = log_chunks[0].final_score
            off_rtl_score = rtl_chunks[0].final_score
        else:
            on_log_score = log_chunks[0].final_score
            on_rtl_score = rtl_chunks[0].final_score

    # Log score should be scaled by (1 - penalty) = 0.5
    assert on_log_score == pytest.approx(off_log_score * 0.5, rel=1e-6)
    # RTL score should be unchanged
    assert on_rtl_score == pytest.approx(off_rtl_score, rel=1e-6)


def test_noisy_source_penalty_improves_log_vs_source_rank(tmp_path: Path, db_session):
    """The penalty reduces the rank of log-file chunks relative to real
    source chunks, by downweighting their fused score."""
    _write(
        tmp_path, "RTL/i2c_monitor.sv",
        "module i2c_monitor;\n"
        "  logic [6:0] addr; logic [7:0] data;\n"
        "  logic sda; logic scl;\n"
        "  always_ff @(posedge scl) addr <= sda;\n"
        "endmodule\n",
    )
    # Verbose log file: many lines of natural-language text matching the query.
    log_lines = []
    for i in range(120):
        log_lines.append(
            f"  [cycle {i}] monitor samples address {i} from sda line; "
            f"data check passed for word {i}; address_check updated"
        )
    _write(tmp_path, "sim/log/sim.log", "\n".join(log_lines) + "\n")

    provider = _run_full_pipeline(tmp_path, db_session, "proj_crowd")
    query_text = "how does the monitor build address_check and data_check from sampled sda"
    query_embedding = provider.embed_texts([query_text])[0]

    retriever_off = HybridRetriever(
        db_session, reranker=LexicalOverlapReranker(), enable_graph_expansion=False,
        noisy_source_penalty=0.0,
    )
    retriever_on = HybridRetriever(
        db_session, reranker=LexicalOverlapReranker(), enable_graph_expansion=False,
        noisy_source_path_terms=("/log/",), noisy_source_penalty=0.5,
    )

    results_off, trace_off = retriever_off.retrieve(
        query_text, query_embedding, project_id="proj_crowd", top_k=10,
        filters=VectorSearchFilter(project_id="proj_crowd"), return_trace=True,
    )
    results_on, trace_on = retriever_on.retrieve(
        query_text, query_embedding, project_id="proj_crowd", top_k=10,
        filters=VectorSearchFilter(project_id="proj_crowd"), return_trace=True,
    )

    # Get ranks from trace (final_rank is None for dropped chunks)
    def rank_of(trace, path):
        for c in trace.chunks():
            if c.relative_path == path and c.final_rank is not None:
                return c.final_rank
        return None

    off_rtl = rank_of(trace_off, "RTL/i2c_monitor.sv")
    on_rtl = rank_of(trace_on, "RTL/i2c_monitor.sv")
    off_log = rank_of(trace_off, "sim/log/sim.log")
    on_log = rank_of(trace_on, "sim/log/sim.log")

    # Penalty should improve RTL rank (lower number = better) and
    # worsen log rank (higher number = worse)
    assert off_rtl is not None and on_rtl is not None
    assert on_rtl <= off_rtl  # RTL rank improves or stays same
    # Log rank worsens (or stays if it was already out)
    if on_log is not None and off_log is not None:
        assert on_log >= off_log


def test_ri06_metadata_propagates_to_retrieval_result_and_trace(tmp_path: Path, db_session):
    """RI-06: metadata set at ingestion survives the full storage/retrieval
    path — VectorChunk → chunks table → VectorSearchResult → RetrievalResult
    → ChunkTrace. Also covers the generated category on a log-style path."""
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, output logic scl);\n"
        "  logic start_condition;\n"
        "  always_ff @(posedge clk) start_condition <= 1'b0;\n"
        "endmodule\n",
    )
    _write(
        tmp_path, "generated/gen_axi.sv",
        "module gen_axi(input logic clk);\n  always_ff @(posedge clk) q <= 1'b0;\nendmodule\n",
    )

    provider = _run_full_pipeline(tmp_path, db_session, "proj_meta")

    retriever = HybridRetriever(db_session, reranker=None, enable_graph_expansion=False)
    query_text = "start condition"
    query_embedding = provider.embed_texts([query_text])[0]
    results, trace = retriever.retrieve(
        query_text, query_embedding, project_id="proj_meta", top_k=10,
        filters=VectorSearchFilter(project_id="proj_meta"), return_trace=True,
    )

    assert results
    for r in results:
        # Every result carries metadata (empty for unenriched rows only).
        assert isinstance(r.language, str)
        assert isinstance(r.category, str)
        assert r.chunker_name != "" or r.category == ""
        if r.relative_path.endswith(".sv"):
            assert r.language == "systemverilog"
            assert r.chunker_name == "systemverilog"

    # The generated module must carry the derived flag through to results.
    gen = next((r for r in results if "gen_axi" in r.relative_path), None)
    if gen is not None:
        assert gen.category == "generated"
        assert gen.is_generated is True

    # Trace exposes the same metadata for every traced chunk.
    for ct in trace.chunks():
        assert isinstance(ct.language, str)
        assert isinstance(ct.category, str)
        if ct.relative_path.endswith(".sv"):
            assert ct.language == "systemverilog"


def test_lexical_search_preserves_metadata(tmp_path: Path, db_session):
    """Lexical path constructs RetrievalResult directly from SQL — metadata
    columns must be selected and carried, not dropped."""
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, output logic scl);\n"
        "  logic start_condition;\n"
        "  always_ff @(posedge clk) start_condition <= 1'b0;\n"
        "endmodule\n",
    )

    _run_full_pipeline(tmp_path, db_session, "proj_lex_meta")

    from semrag.retrieval.lexical_search import PostgresLexicalSearch
    from semrag.retrieval.models import RetrievalResult

    results = PostgresLexicalSearch(db_session).search(
        "i2c_master", top_k=5, filters=VectorSearchFilter(project_id="proj_lex_meta")
    )
    assert results
    for r in results:
        assert isinstance(r, RetrievalResult)
        assert r.language == "systemverilog"
        assert r.category == "rtl"
        assert r.chunker_name == "systemverilog"
        assert r.symbol_name == "i2c_master"
        assert r.parser_name is not None


def test_reranker_failure_is_fail_open(tmp_path: Path, db_session):
    """RI-07 fail-open contract: a configured reranker that raises must
    never abort retrieval — the pre-reranker ranking is used, the failure
    is recorded on the trace, and results are still returned."""
    _write(
        tmp_path, "RTL/i2c_master.sv",
        "module i2c_master(input logic clk, output logic scl);\n"
        "  logic start_condition;\n"
        "  always_ff @(posedge clk) start_condition <= 1'b0;\n"
        "endmodule\n",
    )

    provider = _run_full_pipeline(tmp_path, db_session, "proj_failopen")
    query_text = "start condition"
    query_embedding = provider.embed_texts([query_text])[0]

    class _FailingReranker:
        def rerank(self, query, results, top_k):  # noqa: ARG002 — contract signature
            raise RuntimeError("reranker endpoint down")

    retriever = HybridRetriever(db_session, reranker=_FailingReranker(), enable_graph_expansion=True)
    results, trace = retriever.retrieve(
        query_text, query_embedding, project_id="proj_failopen", top_k=5,
        filters=VectorSearchFilter(project_id="proj_failopen"), return_trace=True,
    )

    assert results, "retrieval must continue even when the reranker fails"
    assert len(results) <= 5
    assert trace.rerank_failed is True
