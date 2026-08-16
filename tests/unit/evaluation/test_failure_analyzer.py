"""
Tests for the retrieval failure analyzer.

Covers:
- per-stage ordering reconstruction from recorded trace signals
- failure classification (every FailureCategory, including the
  expansion-only and change-detection edge cases)
- PROMPT_CUTOFF detection via the prompt builder
- aggregate counts across queries
- deterministic JSON + markdown reports
- report persistence to retrieval_failure_analysis.json/.md
- the analyzer is read-only over the trace (backward compatibility)
- an end-to-end run against the real pipeline + a small indexed project
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from semrag.evaluation.failure_analyzer import (
    RetrievalFailureAnalyzer,
    build_stage_scores,
    classify_failure,
    format_markdown,
    order_stages,
    to_json_dict,
    write_failure_reports,
)
from semrag.evaluation.models import FailureCategory, GoldenQuery
from semrag.reasoning.prompting.pipeline import PromptBuilder
from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.pipeline import ChunkTrace, RankingTrace


def _trace(specs: list[dict]) -> RankingTrace:
    trace = RankingTrace(query="q")
    for spec in specs:
        ct = trace.get_or_create(spec["chunk_id"])
        for key, value in spec.items():
            if key != "chunk_id":
                setattr(ct, key, value)
    return trace


def _orders(chunks: list[ChunkTrace], **kwargs) -> dict[str, list[str]]:
    return order_stages(build_stage_scores(chunks, **kwargs))


def _result(chunk_id: str, *, relative_path: str = "", fused_score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="proj1",
        relative_path=relative_path,
        chunk_type="sv_module",
        text="",
        fused_score=fused_score,
        signal_scores={},
    )


class _StubRetriever:
    """Canned (results, trace) retriever; records the queries it was
    given and whether the trace was requested."""

    def __init__(self, store: dict[str, tuple[list[RetrievalResult], RankingTrace]]):
        self._store = store
        self.queries: list[str] = []
        self.requested_trace: list[bool] = []

    def retrieve(self, query_text, query_embedding, project_id, top_k=10, filters=None, return_trace=False):
        self.queries.append(query_text)
        self.requested_trace.append(return_trace)
        results, trace = self._store[query_text]
        if return_trace:
            return results[:top_k], trace
        return results[:top_k]


# ── Stage-score reconstruction ──────────────────────────────────────────


class TestBuildStageScores:
    def test_graph_expansion_scored_seed_relative(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015),
            ChunkTrace(chunk_id="E", graph_expansion_score=0.5, graph_seed_score=0.02),
        ]
        scores = build_stage_scores(chunks)
        # Expansion chunks carry no rrf_score but join at the GRAPH stage,
        # scored seed_relative * graph_expansion_score.
        assert "E" not in scores["RRF"]
        assert scores["GRAPH"]["E"] == pytest.approx(0.02 * 0.5)

    def test_graph_expansion_falls_back_to_max_base_without_seed_score(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015),
            ChunkTrace(chunk_id="E", graph_expansion_score=0.5),
        ]
        scores = build_stage_scores(chunks)
        assert scores["GRAPH"]["E"] == pytest.approx(0.02 * 0.5)

    def test_noisy_source_penalty_only_applies_to_matching_paths(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02, relative_path="rtl/a.sv"),
            ChunkTrace(chunk_id="L", rrf_score=0.012, relative_path="sim/log/run.log"),
        ]
        scores = build_stage_scores(chunks, noisy_penalty=0.5)
        assert scores["RRF"]["L"] == pytest.approx(0.012)
        assert scores["SOURCE"]["A"] == pytest.approx(0.02)
        assert scores["SOURCE"]["L"] == pytest.approx(0.012 * 0.5)

    def test_intent_uses_recorded_post_intent_score_for_boosted_chunks(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02),
            ChunkTrace(chunk_id="C", rrf_score=0.01, intent_boost=0.3, post_intent_score=0.013),
        ]
        scores = build_stage_scores(chunks)
        assert scores["INTENT"]["A"] == pytest.approx(0.02)
        assert scores["INTENT"]["C"] == pytest.approx(0.013)

    def test_doc_prior_multiplies_intent_score(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02),
            ChunkTrace(chunk_id="C", rrf_score=0.01, doc_type_prior=-0.5),
        ]
        scores = build_stage_scores(chunks)
        assert scores["DOC_PRIOR"]["C"] == pytest.approx(0.01 * 0.5)

    def test_rerank_uses_final_score_with_doc_score_fallback(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02, final_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015, final_score=0.03),
            ChunkTrace(chunk_id="C", rrf_score=0.01, doc_type_prior=0.2),
        ]
        scores = build_stage_scores(chunks)
        assert scores["RERANK"]["B"] == pytest.approx(0.03)
        # No recorded final_score -> the pre-rerank (doc-prior) score.
        assert scores["RERANK"]["C"] == pytest.approx(0.01 * 1.2)


# ── Failure classification ──────────────────────────────────────────────


class TestClassifyFailure:
    def test_missing_entry_is_not_retrieved(self):
        assert classify_failure("missing", None, {}, 3) is FailureCategory.NOT_RETRIEVED

    def test_reached_chunk_is_not_a_failure(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02, reached_llm=True),
            ChunkTrace(chunk_id="B", rrf_score=0.015, reached_llm=True),
            ChunkTrace(chunk_id="C", rrf_score=0.01, reached_llm=True),
        ]
        assert classify_failure("A", chunks[0], _orders(chunks), 3) is None

    def test_ranked_too_low_when_never_in_rrf_cut(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015),
            ChunkTrace(chunk_id="C", rrf_score=0.01),
            ChunkTrace(chunk_id="X", rrf_score=0.001),
        ]
        assert classify_failure("X", chunks[3], _orders(chunks), 3) is FailureCategory.RANKED_TOO_LOW

    def test_graph_expansion_demotes_base_chunk(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015),
            ChunkTrace(chunk_id="C", rrf_score=0.008),
            ChunkTrace(chunk_id="E", graph_expansion_score=1.0),  # max_base * 1.0 = 0.02 > 0.008
        ]
        assert classify_failure("C", chunks[2], _orders(chunks), 3) is FailureCategory.GRAPH_EXPANSION

    def test_intent_boost_demotes_unboosted_chunk(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015),
            ChunkTrace(chunk_id="C", rrf_score=0.01, intent_boost=0.5, post_intent_score=0.015),
            ChunkTrace(chunk_id="D", rrf_score=0.012),
        ]
        assert classify_failure("D", chunks[3], _orders(chunks), 3) is FailureCategory.INTENT

    def test_doc_type_prior_is_source_prior(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015),
            ChunkTrace(chunk_id="C", rrf_score=0.012, doc_type_prior=-0.5),
            ChunkTrace(chunk_id="D", rrf_score=0.01),
        ]
        assert classify_failure("C", chunks[2], _orders(chunks), 3) is FailureCategory.SOURCE_PRIOR

    def test_noisy_source_penalty_is_source_prior(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02, relative_path="rtl/a.sv"),
            ChunkTrace(chunk_id="B", rrf_score=0.015, relative_path="rtl/b.sv"),
            ChunkTrace(chunk_id="L", rrf_score=0.012, relative_path="sim/log/run.log"),
            ChunkTrace(chunk_id="C", rrf_score=0.01, relative_path="rtl/c.sv"),
        ]
        orders = _orders(chunks, noisy_penalty=0.5)
        assert classify_failure("L", chunks[2], orders, 3) is FailureCategory.SOURCE_PRIOR

    def test_reranker_demotes_chunk(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02, final_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015, final_score=0.015),
            ChunkTrace(chunk_id="C", rrf_score=0.01, final_score=0.02, reranker_score=0.8),
            ChunkTrace(chunk_id="D", rrf_score=0.012, final_score=0.009),
        ]
        assert classify_failure("D", chunks[3], _orders(chunks), 3) is FailureCategory.RERANKER

    def test_expansion_chunk_demoted_by_intent(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015),
            ChunkTrace(chunk_id="C", rrf_score=0.009, intent_boost=0.3, post_intent_score=0.0117),
            ChunkTrace(chunk_id="E", graph_expansion_score=1.0, graph_seed_score=0.01),  # 0.01 * 1.0 = 0.01
        ]
        assert classify_failure("E", chunks[3], _orders(chunks), 3) is FailureCategory.INTENT

    def test_unknown_when_in_every_cut_but_not_reached(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02, final_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015, final_score=0.015),
            ChunkTrace(chunk_id="C", rrf_score=0.01, final_score=0.01),
        ]
        # Inside every reconstructed top-3 yet not marked reached: the
        # trace cannot explain it — fall back to UNKNOWN.
        assert classify_failure("A", chunks[0], _orders(chunks), 3) is FailureCategory.UNKNOWN


# ── Diversity attribution (RI-03) ───────────────────────────────────────


class TestDiversityAttribution:
    def test_diversity_stage_absent_without_diversity_columns(self):
        """Legacy traces (no diversity recorded) reconstruct the exact
        pre-RI-03 stage order — no DIVERSITY stage, UNKNOWN preserved."""
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02, final_score=0.02),
            ChunkTrace(chunk_id="B", rrf_score=0.015, final_score=0.015),
            ChunkTrace(chunk_id="C", rrf_score=0.01, final_score=0.01),
        ]
        orders = _orders(chunks)
        assert "DIVERSITY" not in orders
        assert classify_failure("A", chunks[0], orders, 3) is FailureCategory.UNKNOWN

    def test_diversity_demotion_attributed(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02, final_score=0.02,
                       reached_llm=True, final_rank=1),
            ChunkTrace(chunk_id="B", rrf_score=0.015, final_score=0.015,
                       reached_llm=True, final_rank=2),
            ChunkTrace(chunk_id="C", rrf_score=0.01, final_score=0.01,
                       diversity=0.0, diversity_reason="file_limit"),
            ChunkTrace(chunk_id="D", rrf_score=0.008, final_score=0.008,
                       diversity=1.0, diversity_reason="selected",
                       reached_llm=True, final_rank=3),
        ]
        orders = _orders(chunks)
        assert "DIVERSITY" in orders
        # C was inside the RERANK top-3 but not in the diversity output.
        assert classify_failure("C", chunks[2], orders, 3) is FailureCategory.DIVERSITY

    def test_diversity_drop_has_primary_cause(self):
        trace = _trace([
            {"chunk_id": "A", "rrf_score": 0.02, "final_score": 0.02, "final_rank": 1,
             "reached_llm": True, "relative_path": "rtl/a.sv", "chunk_type": "sv_module"},
            {"chunk_id": "B", "rrf_score": 0.015, "final_score": 0.015, "final_rank": 2,
             "reached_llm": True, "relative_path": "rtl/a.sv", "chunk_type": "sv_module"},
            {"chunk_id": "C", "rrf_score": 0.012, "final_score": 0.012,
             "diversity": 0.0, "diversity_reason": "file_limit",
             "relative_path": "rtl/a.sv", "chunk_type": "sv_module"},
            {"chunk_id": "D", "rrf_score": 0.01, "final_score": 0.01,
             "diversity": 1.0, "diversity_reason": "selected",
             "final_rank": 3, "reached_llm": True,
             "relative_path": "rtl/b.sv", "chunk_type": "sv_module"},
        ])
        results = [
            _result("A", relative_path="rtl/a.sv"),
            _result("B", relative_path="rtl/a.sv"),
            _result("D", relative_path="rtl/b.sv"),
        ]
        stub = _StubRetriever({"q": (results, trace)})
        analyzer = RetrievalFailureAnalyzer(stub, lambda q: [], top_k=3)
        report = analyzer.analyze([GoldenQuery(
            query_id="q", project_id="proj1", query_text="q", relevant_chunk_ids={"C"}
        )])

        failure = report.per_query[0].failures[0]
        assert failure.failure_category is FailureCategory.DIVERSITY
        assert failure.reached_llm is False
        assert "file_limit" in failure.primary_cause
        assert failure.recommendation == "inspect diversity-aware selection — check the per-file / per-category limits and neighbour window in DiversityConfig"


# ── PROMPT_CUTOFF ───────────────────────────────────────────────────────


class _PromptBuilderCapsAtOne(PromptBuilder):
    """Forces max_chunks=1 so a retrieved-but-excluded chunk is easy to
    construct deterministically for the PROMPT_CUTOFF path."""

    def build(self, query, retrieval_results, conversation_history=None, file_index=None, max_chunks=None):
        return super().build(
            query, retrieval_results, conversation_history, file_index, max_chunks=1
        )


class TestPromptCutoff:
    def test_retrieved_but_excluded_by_prompt_budget(self):
        trace = _trace([
            {"chunk_id": "A", "rrf_score": 0.02, "final_score": 0.02, "final_rank": 1,
             "reached_llm": True, "relative_path": "rtl/a.sv", "chunk_type": "sv_module",
             "vector_rank": 1},
            {"chunk_id": "B", "rrf_score": 0.015, "final_score": 0.015, "final_rank": 2,
             "reached_llm": True, "relative_path": "rtl/b.sv", "chunk_type": "sv_module"},
        ])
        results = [_result("A", relative_path="rtl/a.sv"), _result("B", relative_path="rtl/b.sv")]
        stub = _StubRetriever({"q": (results, trace)})
        analyzer = RetrievalFailureAnalyzer(
            stub, lambda q: [], top_k=2, prompt_builder=_PromptBuilderCapsAtOne()
        )
        golden = [GoldenQuery(
            query_id="q", project_id="proj1", query_text="q", relevant_chunk_ids={"B"}
        )]
        report = analyzer.analyze(golden)

        failure = report.per_query[0].failures[0]
        assert failure.failure_category is FailureCategory.PROMPT_CUTOFF
        assert failure.reached_llm is True
        assert failure.final_rank == 2
        assert "excluded by the prompt context budget" in failure.primary_cause

    def test_reached_chunk_included_in_prompt_is_not_a_failure(self):
        trace = _trace([
            {"chunk_id": "A", "rrf_score": 0.02, "final_score": 0.02, "final_rank": 1,
             "reached_llm": True, "relative_path": "rtl/a.sv", "chunk_type": "sv_module"},
        ])
        results = [_result("A", relative_path="rtl/a.sv")]
        stub = _StubRetriever({"q": (results, trace)})
        analyzer = RetrievalFailureAnalyzer(stub, lambda q: [], top_k=2)
        golden = [GoldenQuery(
            query_id="q", project_id="proj1", query_text="q", relevant_chunk_ids={"A"}
        )]
        report = analyzer.analyze(golden)

        assert report.per_query[0].failure_count == 0
        assert report.per_query[0].failures[0].failure_category is None


# ── Aggregation ─────────────────────────────────────────────────────────


class TestAggregation:
    def test_counts_aggregate_across_queries(self):
        trace1 = _trace([
            {"chunk_id": "A", "rrf_score": 0.02, "final_score": 0.02, "final_rank": 1,
             "reached_llm": True, "relative_path": "rtl/a.sv", "chunk_type": "sv_module"},
            {"chunk_id": "B", "rrf_score": 0.015, "final_score": 0.015, "final_rank": 2,
             "reached_llm": True},
            {"chunk_id": "C", "rrf_score": 0.01, "final_score": 0.01, "final_rank": 3,
             "reached_llm": True},
            {"chunk_id": "X", "rrf_score": 0.001},
        ])
        trace2 = _trace([
            {"chunk_id": "A", "rrf_score": 0.02, "final_score": 0.02, "final_rank": 1,
             "reached_llm": True},
            {"chunk_id": "B", "rrf_score": 0.015, "final_score": 0.015, "final_rank": 2,
             "reached_llm": True},
            {"chunk_id": "C", "rrf_score": 0.008},
            {"chunk_id": "E", "graph_expansion_score": 1.0, "graph_seed_score": 0.01,
             "final_score": 0.01, "final_rank": 3, "reached_llm": True},
        ])
        results1 = [
            _result("A", relative_path="rtl/a.sv"),
            _result("B", relative_path="rtl/b.sv"),
            _result("C", relative_path="rtl/c.sv"),
        ]
        results2 = [
            _result("A", relative_path="rtl/a.sv"),
            _result("B", relative_path="rtl/b.sv"),
            _result("E", relative_path="rtl/e.sv"),
        ]
        golden = [
            GoldenQuery(query_id="q1", project_id="proj1", query_text="q1", relevant_chunk_ids={"X"}),
            GoldenQuery(query_id="q2", project_id="proj1", query_text="q2", relevant_chunk_ids={"C"}),
        ]
        stub = _StubRetriever({"q1": (results1, trace1), "q2": (results2, trace2)})
        report = RetrievalFailureAnalyzer(stub, lambda q: [], top_k=3).analyze(golden)

        assert report.query_count == 2
        assert report.expected_chunk_count == 2
        assert report.failure_count == 2
        assert report.aggregate.total_failures == 2
        counts = report.aggregate.counts
        assert counts[FailureCategory.RANKED_TOO_LOW.value] == 1
        assert counts[FailureCategory.GRAPH_EXPANSION.value] == 1

    def test_empty_query_set_is_zeroed(self):
        report = RetrievalFailureAnalyzer(_StubRetriever({}), lambda q: []).analyze([])
        assert report.query_count == 0
        assert report.failure_count == 0
        assert report.aggregate.total_failures == 0
        assert report.per_query == []


# ── Reporting ───────────────────────────────────────────────────────────


class TestReporting:
    def _report(self):
        trace = _trace([
            {"chunk_id": "A", "rrf_score": 0.02, "final_score": 0.02, "final_rank": 1,
             "reached_llm": True, "relative_path": "rtl/a.sv", "chunk_type": "sv_module",
             "vector_rank": 1},
            {"chunk_id": "B", "rrf_score": 0.015, "final_score": 0.015, "final_rank": 2,
             "reached_llm": True},
            {"chunk_id": "C", "rrf_score": 0.01, "final_score": 0.01, "final_rank": 3,
             "reached_llm": True},
            {"chunk_id": "X", "rrf_score": 0.001},
        ])
        results = [
            _result("A", relative_path="rtl/a.sv"),
            _result("B", relative_path="rtl/b.sv"),
            _result("C", relative_path="rtl/c.sv"),
        ]
        stub = _StubRetriever({"q1": (results, trace)})
        golden = [GoldenQuery(
            query_id="q1", project_id="proj1", query_text="q1", relevant_chunk_ids={"X"}
        )]
        return RetrievalFailureAnalyzer(stub, lambda q: [], top_k=3).analyze(golden)

    def test_reports_are_deterministic(self):
        first = self._report()
        second = self._report()
        assert first.model_dump() == second.model_dump()
        assert format_markdown(first) == format_markdown(second)

    def test_to_json_dict_is_strict_json_with_string_categories(self):
        data = to_json_dict(self._report())
        dumped = json.dumps(data)
        parsed = json.loads(dumped)
        assert parsed["failure_count"] == 1
        failure = parsed["per_query"][0]["failures"][0]
        assert failure["failure_category"] == "RANKED_TOO_LOW"
        assert failure["chunk_id"] == "X"

    def test_markdown_contains_summary_and_breakdown(self):
        text = format_markdown(self._report())
        assert "# Retrieval Failure Analysis" in text
        assert "## Failure Summary" in text
        assert "RANKED_TOO_LOW" in text
        assert "Total failures: 1" in text
        assert "## q1" in text
        assert "Ranking breakdown" in text
        assert "stage ranks: RRF=4" in text

    def test_write_failure_reports_writes_both_files(self, tmp_path: Path):
        report = self._report()
        json_path, md_path = write_failure_reports(
            report,
            tmp_path / "retrieval_failure_analysis.json",
            tmp_path / "retrieval_failure_analysis.md",
        )
        assert json_path.exists()
        assert md_path is not None and md_path.exists()

        loaded = json.loads(json_path.read_text())
        assert loaded["failure_count"] == 1
        assert loaded["per_query"][0]["query_id"] == "q1"

        md_text = md_path.read_text()
        assert "Failure Summary" in md_text
        assert "RANKED_TOO_LOW" in md_text

    def test_query_requests_the_ranking_trace(self):
        trace = _trace([
            {"chunk_id": "A", "rrf_score": 0.02, "reached_llm": True},
        ])
        results = [_result("A", relative_path="rtl/a.sv")]
        stub = _StubRetriever({"q": (results, trace)})
        analyzer = RetrievalFailureAnalyzer(stub, lambda q: [], top_k=3)
        analyzer.analyze([GoldenQuery(
            query_id="q", project_id="proj1", query_text="q", relevant_chunk_ids={"A"}
        )])
        assert stub.requested_trace == [True]

    def test_analyzer_is_read_only_over_the_trace(self):
        chunks = [
            ChunkTrace(chunk_id="A", rrf_score=0.02, reached_llm=True),
            ChunkTrace(chunk_id="X", rrf_score=0.001),
        ]
        trace = _trace([
            {"chunk_id": "A", "rrf_score": 0.02, "reached_llm": True},
            {"chunk_id": "X", "rrf_score": 0.001},
        ])
        snapshot = {c.chunk_id: c.to_dict() for c in chunks}
        results = [_result("A", relative_path="rtl/a.sv")]
        stub = _StubRetriever({"q": (results, trace)})
        analyzer = RetrievalFailureAnalyzer(stub, lambda q: [], top_k=3)
        analyzer.analyze([GoldenQuery(
            query_id="q", project_id="proj1", query_text="q", relevant_chunk_ids={"X"}
        )])
        assert {c.chunk_id: c.to_dict() for c in trace.chunks()} == snapshot


# ── Integration: real pipeline against an indexed project ──────────────


def test_failure_analysis_end_to_end_against_real_pipeline(tmp_path: Path, db_session):
    """Index a small project and classify an expected chunk that exists
    (retrieved) and one that does not (NOT_RETRIEVED)."""
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

    provider, vector_chunks = _index(tmp_path, db_session)
    i2c_id = next(iter({c.chunk_id for c in vector_chunks if c.relative_path == "RTL/i2c_master.sv"}))

    golden = [
        GoldenQuery(
            query_id="rtl", project_id="proj1",
            query_text="what module detects the start condition on the i2c sda line",
            relevant_chunk_ids={i2c_id, "does-not-exist"},
        ),
    ]
    retriever = _build_retriever(db_session)
    analyzer = RetrievalFailureAnalyzer(
        retriever, embed_query_fn=lambda q: provider.embed_texts([q])[0], top_k=5
    )
    report = analyzer.analyze(golden)

    assert report.query_count == 1
    assert report.expected_chunk_count == 2
    assert report.aggregate.counts[FailureCategory.NOT_RETRIEVED.value] == 1

    by_id = {f.chunk_id: f for f in report.per_query[0].failures}
    assert by_id["does-not-exist"].failure_category is FailureCategory.NOT_RETRIEVED
    assert by_id[i2c_id].retrieved is True
    assert by_id[i2c_id].relative_path == "RTL/i2c_master.sv"

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
