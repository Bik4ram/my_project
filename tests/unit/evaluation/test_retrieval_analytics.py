"""
Tests for the retrieval analytics dashboard (RB-03).

Covers:
- query-intent classification (deterministic keyword rules)
- document-category classification
- stage responsibility attribution
- overall health aggregation
- failure distribution (counts + percentages)
- stage attribution (counts + percentages)
- document analysis incl. unclassified failures
- intent analysis (health, failure rank, most common failure type)
- recommendation rules (each rule, thresholds, diagnostic-only wording)
- deterministic outputs
- JSON serialization + markdown + report persistence
- an end-to-end run: RB-01 benchmark + RB-02 analyzer -> analytics
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from semrag.evaluation.benchmark_runner import RetrievalBenchmark
from semrag.evaluation.failure_analyzer import RetrievalFailureAnalyzer
from semrag.evaluation.models import (
    DocumentCategory,
    ExpectedChunkAnalysis,
    FailureAggregate,
    FailureCategory,
    GoldenQuery,
    IntentCategory,
    QueryFailureAnalysis,
    RetrievalBenchmarkQueryResult,
    RetrievalBenchmarkReport,
    RetrievalFailureReport,
    StageContributions,
)
from semrag.evaluation.retrieval_analytics import (
    RetrievalAnalytics,
    attribute_responsibility,
    categorize_document,
    classify_query_intent,
    format_markdown,
    to_json_dict,
    write_analytics_reports,
)


def _failure(
    category,
    *,
    query_id="q1",
    chunk_id=None,
    relative_path="",
    chunk_type="",
    final_rank=None,
    vector_rank=None,
    lexical_rank=None,
    graph_expansion_score=None,
) -> ExpectedChunkAnalysis:
    return ExpectedChunkAnalysis(
        query_id=query_id,
        chunk_id=chunk_id or category.value,
        relative_path=relative_path,
        chunk_type=chunk_type,
        retrieved=final_rank is not None,
        final_rank=final_rank,
        reached_llm=False,
        failure_category=category,
        contributions=StageContributions(
            vector_rank=vector_rank,
            lexical_rank=lexical_rank,
            graph_expansion_score=graph_expansion_score,
        ),
    )


def _query(query_id: str, query_text: str, failures: list[ExpectedChunkAnalysis]) -> QueryFailureAnalysis:
    failures = list(failures)
    return QueryFailureAnalysis(
        query_id=query_id,
        query_text=query_text,
        top_k=10,
        expected_chunk_count=len(failures),
        failure_count=sum(1 for f in failures if f.failure_category is not None),
        failures=failures,
    )


def _failure_report(queries: list[QueryFailureAnalysis]) -> RetrievalFailureReport:
    all_failures = [f for q in queries for f in q.failures if f.failure_category is not None]
    counts = {category.value: 0 for category in FailureCategory}
    for failure in all_failures:
        counts[failure.failure_category.value] += 1
    return RetrievalFailureReport(
        query_count=len(queries),
        expected_chunk_count=sum(len(q.failures) for q in queries),
        failure_count=len(all_failures),
        aggregate=FailureAggregate(total_failures=len(all_failures), counts=counts),
        per_query=queries,
    )


def _bench_result(query_id: str, top1_recall: float, mrr: float = 0.0, average_rank: float = 0.0) -> RetrievalBenchmarkQueryResult:
    return RetrievalBenchmarkQueryResult(
        query_id=query_id, top_k=10, top1_recall=top1_recall, mrr=mrr, average_rank=average_rank
    )


def _benchmark(queries: list[RetrievalBenchmarkQueryResult]) -> RetrievalBenchmarkReport:
    n = len(queries)
    if n == 0:
        return RetrievalBenchmarkReport(query_count=0)
    return RetrievalBenchmarkReport(
        query_count=n,
        top_k=10,
        top1_recall=sum(q.top1_recall for q in queries) / n,
        top3_recall=sum(q.top3_recall for q in queries) / n,
        top5_recall=sum(q.top5_recall for q in queries) / n,
        mean_mrr=sum(q.mrr for q in queries) / n,
        mean_average_rank=sum(q.average_rank for q in queries) / n,
        per_query_results=queries,
    )


# ── Query intent classification ─────────────────────────────────────────


class TestClassifyQueryIntent:
    def test_debug_by_why(self):
        assert classify_query_intent("why does the simulation fail") is IntentCategory.DEBUG

    def test_debug_by_error_keyword(self):
        assert classify_query_intent("uvm_error assertion fires at reset") is IntentCategory.DEBUG

    def test_specification_beats_explain_keywords(self):
        # "what is" is an explain keyword but "axi" is a spec keyword and
        # SPECIFICATION has higher priority than EXPLAIN.
        assert classify_query_intent("what is the axi protocol") is IntentCategory.SPECIFICATION

    def test_navigation(self):
        assert classify_query_intent("which module instantiates the uvm driver") is IntentCategory.NAVIGATION

    def test_explain(self):
        assert classify_query_intent("how does the i2c master work") is IntentCategory.EXPLAIN

    def test_general(self):
        assert classify_query_intent("give me a summary of the design") is IntentCategory.GENERAL

    def test_unknown(self):
        assert classify_query_intent("gobbledygook zebra") is IntentCategory.UNKNOWN

    def test_empty_text_is_unknown(self):
        assert classify_query_intent("") is IntentCategory.UNKNOWN

    def test_case_insensitive(self):
        assert classify_query_intent("WHY IS IT FAILING") is IntentCategory.DEBUG


# ── Document category classification ────────────────────────────────────


class TestCategorizeDocument:
    def test_logs_by_chunk_type_and_path(self):
        assert categorize_document("log", "sim/log/run.log") is DocumentCategory.LOGS
        assert categorize_document("generic_window", "output/trace.log") is DocumentCategory.LOGS

    def test_rtl_by_chunk_type_and_path(self):
        assert categorize_document("sv_module", "rtl/i2c_master.sv") is DocumentCategory.RTL
        assert categorize_document("generic_window", "design/top.v") is DocumentCategory.RTL

    def test_uvm_beats_rtl(self):
        # A .sv under a UVM/verification dir is a testbench, not RTL.
        assert categorize_document("sv_module", "verification/uvm/seq.sv") is DocumentCategory.UVM
        assert categorize_document("sv_module", "tb/i2c_tb.sv") is DocumentCategory.UVM

    def test_generated_beats_rtl(self):
        assert categorize_document("sv_module", "generated/rtl/top.sv") is DocumentCategory.GENERATED

    def test_specification_markdown_in_docs(self):
        assert (
            categorize_document("markdown_section", "docs/architecture/i2c.md")
            is DocumentCategory.SPECIFICATION
        )

    def test_markdown_elsewhere(self):
        assert categorize_document("markdown_section", "README.md") is DocumentCategory.MARKDOWN

    def test_other_fallback(self):
        assert categorize_document("py_function", "src/tools.py") is DocumentCategory.OTHER

    def test_log_substring_does_not_bleed_into_paths(self):
        assert categorize_document("sv_module", "src/catalog/i2c.sv") is DocumentCategory.RTL


# ── Stage responsibility attribution ────────────────────────────────────


class TestAttributeResponsibility:
    def test_every_failure_category_maps_to_a_bucket(self):
        assert attribute_responsibility(_failure(FailureCategory.NOT_RETRIEVED)) == "base_retrieval"
        assert attribute_responsibility(_failure(FailureCategory.GRAPH_EXPANSION)) == "graph"
        assert attribute_responsibility(_failure(FailureCategory.SOURCE_PRIOR)) == "source"
        assert attribute_responsibility(_failure(FailureCategory.INTENT)) == "intent"
        assert attribute_responsibility(_failure(FailureCategory.RERANKER)) == "reranker"
        assert attribute_responsibility(_failure(FailureCategory.PROMPT_CUTOFF)) == "prompt_cutoff"
        assert attribute_responsibility(_failure(FailureCategory.UNKNOWN)) == "unknown"

    def test_diversity_maps_to_diversity_bucket(self):
        assert attribute_responsibility(_failure(FailureCategory.DIVERSITY)) == "diversity"

    def test_ranked_too_low_split_by_surfacing_signal(self):
        assert attribute_responsibility(_failure(FailureCategory.RANKED_TOO_LOW, vector_rank=1)) == "vector"
        assert attribute_responsibility(_failure(FailureCategory.RANKED_TOO_LOW, lexical_rank=2)) == "lexical"
        assert (
            attribute_responsibility(
                _failure(FailureCategory.RANKED_TOO_LOW, vector_rank=1, lexical_rank=2)
            )
            == "base_fusion"
        )
        assert (
            attribute_responsibility(
                _failure(FailureCategory.RANKED_TOO_LOW, graph_expansion_score=0.9)
            )
            == "graph"
        )


# ── Overall health ──────────────────────────────────────────────────────


class TestOverallHealth:
    def test_health_reflects_benchmark_aggregates(self):
        bench = _benchmark([
            _bench_result("q1", top1_recall=1.0, mrr=1.0, average_rank=1.0),
            _bench_result("q2", top1_recall=0.0, mrr=0.5, average_rank=3.0),
        ])
        report = RetrievalAnalytics().analyze(bench, _failure_report([]))
        assert report.query_count == 2
        assert report.top1_recall == pytest.approx(0.5)
        assert report.top3_recall == pytest.approx(0.0)
        assert report.mrr == pytest.approx(0.75)
        assert report.average_rank == pytest.approx(2.0)

    def test_empty_benchmark_zeroed(self):
        report = RetrievalAnalytics().analyze(_benchmark([]), _failure_report([]))
        assert report.query_count == 0
        assert report.top1_recall == 0.0
        assert report.mrr == 0.0


# ── Failure distribution ────────────────────────────────────────────────


class TestFailureDistribution:
    def test_counts_and_percentages(self):
        failures = [
            _failure(FailureCategory.NOT_RETRIEVED),
            _failure(FailureCategory.NOT_RETRIEVED),
            _failure(FailureCategory.SOURCE_PRIOR, relative_path="rtl/a.sv", chunk_type="sv_module"),
        ]
        report = RetrievalAnalytics().analyze(
            _benchmark([]), _failure_report([_query("q1", "why does it fail", failures)])
        )
        dist = {bucket.label: bucket for bucket in report.failure_distribution}
        assert report.total_failures == 3
        assert dist["NOT_RETRIEVED"].count == 2
        assert dist["NOT_RETRIEVED"].percentage == pytest.approx(66.66666666)
        assert dist["SOURCE_PRIOR"].count == 1
        assert dist["SOURCE_PRIOR"].percentage == pytest.approx(33.33333333)
        assert dist["RERANKER"].count == 0
        assert dist["RERANKER"].percentage == pytest.approx(0.0)

    def test_distribution_has_one_bucket_per_category(self):
        report = RetrievalAnalytics().analyze(_benchmark([]), _failure_report([]))
        labels = [bucket.label for bucket in report.failure_distribution]
        assert labels == [category.value for category in FailureCategory]
        assert all(bucket.count == 0 for bucket in report.failure_distribution)


# ── Stage attribution ───────────────────────────────────────────────────


class TestStageAttribution:
    def test_counts_and_percentages(self):
        failures = [
            _failure(FailureCategory.NOT_RETRIEVED),
            _failure(FailureCategory.RANKED_TOO_LOW, vector_rank=1),
            _failure(FailureCategory.RANKED_TOO_LOW, lexical_rank=2),
            _failure(FailureCategory.RANKED_TOO_LOW, vector_rank=1, lexical_rank=2),
            _failure(FailureCategory.GRAPH_EXPANSION),
            _failure(FailureCategory.INTENT),
            _failure(FailureCategory.SOURCE_PRIOR),
            _failure(FailureCategory.RERANKER),
            _failure(FailureCategory.PROMPT_CUTOFF),
        ]
        report = RetrievalAnalytics().analyze(
            _benchmark([]), _failure_report([_query("q1", "q1", failures)])
        )
        buckets = {bucket.label: bucket for bucket in report.stage_attribution}
        assert buckets["vector"].count == 1
        assert buckets["lexical"].count == 1
        assert buckets["base_fusion"].count == 1
        assert buckets["base_retrieval"].count == 1
        assert buckets["graph"].count == 1
        assert buckets["intent"].count == 1
        assert buckets["source"].count == 1
        assert buckets["reranker"].count == 1
        assert buckets["prompt_cutoff"].count == 1
        assert buckets["unknown"].count == 0
        assert buckets["vector"].percentage == pytest.approx(100.0 / 9)

    def test_successful_chunks_are_excluded(self):
        success = ExpectedChunkAnalysis(
            query_id="q1", chunk_id="ok", relative_path="rtl/ok.sv",
            chunk_type="sv_module", final_rank=1, reached_llm=True, failure_category=None,
        )
        fail = _failure(FailureCategory.INTENT)
        report = RetrievalAnalytics().analyze(
            _benchmark([]), _failure_report([_query("q1", "q1", [success, fail])])
        )
        buckets = {bucket.label: bucket for bucket in report.stage_attribution}
        assert buckets["intent"].count == 1
        assert sum(bucket.count for bucket in report.stage_attribution) == 1

    def test_diversity_failure_lands_in_diversity_bucket(self):
        failures = [_failure(FailureCategory.DIVERSITY)]
        report = RetrievalAnalytics().analyze(
            _benchmark([]), _failure_report([_query("q1", "q1", failures)])
        )
        buckets = {bucket.label: bucket for bucket in report.stage_attribution}
        assert buckets["diversity"].count == 1
        assert buckets["diversity"].percentage == pytest.approx(100.0)


# ── Document analysis ───────────────────────────────────────────────────


class TestDocumentAnalysis:
    def test_categories_and_unclassified(self):
        failures = [
            _failure(FailureCategory.NOT_RETRIEVED),
            _failure(FailureCategory.INTENT, relative_path="rtl/i2c.sv", chunk_type="sv_module"),
            _failure(FailureCategory.INTENT, relative_path="sim/log/run.log", chunk_type="generic_window"),
            _failure(FailureCategory.INTENT, relative_path="docs/spec/i2c.md", chunk_type="markdown_section"),
            _failure(FailureCategory.INTENT, relative_path="README.md", chunk_type="markdown_section"),
            _failure(FailureCategory.INTENT, relative_path="verification/uvm/seq.sv", chunk_type="sv_module"),
            _failure(FailureCategory.INTENT, relative_path="generated/rtl/top.sv", chunk_type="sv_module"),
            _failure(FailureCategory.INTENT, relative_path="src/tools.py", chunk_type="py_function"),
        ]
        report = RetrievalAnalytics().analyze(
            _benchmark([]), _failure_report([_query("q1", "q1", failures)])
        )
        docs = {doc.category: doc for doc in report.document_analysis}
        assert report.document_unclassified == 1
        assert docs[DocumentCategory.RTL].count == 1
        assert docs[DocumentCategory.LOGS].count == 1
        assert docs[DocumentCategory.SPECIFICATION].count == 1
        assert docs[DocumentCategory.MARKDOWN].count == 1
        assert docs[DocumentCategory.UVM].count == 1
        assert docs[DocumentCategory.GENERATED].count == 1
        assert docs[DocumentCategory.OTHER].count == 1
        # One of the 8 failures has no metadata; the others are each 1/8.
        assert docs[DocumentCategory.RTL].percentage == pytest.approx(12.5)


# ── Intent analysis ─────────────────────────────────────────────────────


class TestIntentAnalysis:
    def test_health_and_failure_profile_per_intent(self):
        bench = _benchmark([
            _bench_result("dbg1", top1_recall=0.0, mrr=0.0),
            _bench_result("dbg2", top1_recall=0.0, mrr=0.0),
            _bench_result("exp1", top1_recall=1.0, mrr=1.0),
        ])
        failures = [
            _failure(FailureCategory.NOT_RETRIEVED, query_id="dbg1"),
            _failure(FailureCategory.INTENT, query_id="dbg2", final_rank=5),
            _failure(FailureCategory.INTENT, query_id="dbg2", final_rank=8),
            _failure(FailureCategory.RANKED_TOO_LOW, query_id="exp1", final_rank=4, vector_rank=1),
        ]
        queries = [
            _query("dbg1", "why does the simulation fail", [failures[0]]),
            _query("dbg2", "why is the fifo stuck", [failures[1], failures[2]]),
            _query("exp1", "how does the i2c master work", [failures[3]]),
        ]
        report = RetrievalAnalytics().analyze(bench, _failure_report(queries))
        by_intent = {analysis.intent: analysis for analysis in report.intent_analysis}

        debug = by_intent[IntentCategory.DEBUG]
        assert debug.query_count == 2
        assert debug.top1_recall == pytest.approx(0.0)
        assert debug.mrr == pytest.approx(0.0)
        assert debug.failure_count == 3
        # NOT_RETRIEVED never ranked and is excluded from the average.
        assert debug.average_failure_rank == pytest.approx(6.5)
        assert debug.most_common_failure_type == "INTENT"

        explain = by_intent[IntentCategory.EXPLAIN]
        assert explain.query_count == 1
        assert explain.top1_recall == pytest.approx(1.0)
        assert explain.failure_count == 1
        assert explain.average_failure_rank == pytest.approx(4.0)
        assert explain.most_common_failure_type == "RANKED_TOO_LOW"

        assert by_intent[IntentCategory.UNKNOWN].query_count == 0
        assert by_intent[IntentCategory.UNKNOWN].failure_count == 0

    def test_all_six_intents_present_even_when_empty(self):
        report = RetrievalAnalytics().analyze(_benchmark([]), _failure_report([]))
        assert len(report.intent_analysis) == 6
        assert [analysis.intent for analysis in report.intent_analysis] == list(IntentCategory)
        assert all(analysis.query_count == 0 for analysis in report.intent_analysis)

    def test_most_common_failure_type_ties_break_to_enum_order(self):
        failures = [
            _failure(FailureCategory.INTENT, final_rank=2),
            _failure(FailureCategory.RERANKER, final_rank=3),
        ]
        bench = _benchmark([_bench_result("q1", top1_recall=0.0)])
        report = RetrievalAnalytics().analyze(
            bench, _failure_report([_query("q1", "why does it fail", failures)])
        )
        debug = {analysis.intent: analysis for analysis in report.intent_analysis}[IntentCategory.DEBUG]
        # INTENT and RERANKER tie at 1; INTENT comes first in enum order.
        assert debug.most_common_failure_type == "INTENT"


# ── Recommendation engine ───────────────────────────────────────────────


def _report_with(category_counts: dict[FailureCategory, int]) -> RetrievalFailureReport:
    failures = []
    for category, count in category_counts.items():
        for index in range(count):
            failures.append(
                _failure(
                    category,
                    chunk_id=f"{category.value}-{index}",
                    relative_path="rtl/x.sv",
                    chunk_type="sv_module",
                )
            )
    return _failure_report([_query("q1", "why does it fail", failures)])


class TestRecommendations:
    def test_no_failures(self):
        report = RetrievalAnalytics().analyze(_benchmark([]), _failure_report([]))
        rules = [rec.rule for rec in report.recommendations]
        assert rules == ["no_failures"]

    def test_not_retrieved_dominance_fires(self):
        frep = _report_with({
            FailureCategory.NOT_RETRIEVED: 5,
            FailureCategory.INTENT: 4,
            FailureCategory.SOURCE_PRIOR: 1,
        })
        report = RetrievalAnalytics().analyze(_benchmark([]), frep)
        rules = [rec.rule for rec in report.recommendations]
        assert "not_retrieved_dominance" in rules

    def test_dominance_requires_threshold(self):
        frep = _report_with({FailureCategory.NOT_RETRIEVED: 3, FailureCategory.INTENT: 7})
        report = RetrievalAnalytics().analyze(_benchmark([]), frep)
        rules = [rec.rule for rec in report.recommendations]
        assert "not_retrieved_dominance" not in rules

    def test_prompt_cutoff_dominance_fires(self):
        frep = _report_with({FailureCategory.PROMPT_CUTOFF: 5, FailureCategory.INTENT: 5})
        report = RetrievalAnalytics().analyze(_benchmark([]), frep)
        rules = [rec.rule for rec in report.recommendations]
        assert "prompt_cutoff_dominance" in rules

    def test_source_prior_dominance_fires(self):
        frep = _report_with({FailureCategory.SOURCE_PRIOR: 5, FailureCategory.INTENT: 5})
        report = RetrievalAnalytics().analyze(_benchmark([]), frep)
        rules = [rec.rule for rec in report.recommendations]
        assert "source_prior_dominance" in rules

    def test_stage_dominance_rules(self):
        for category, rule in (
            (FailureCategory.GRAPH_EXPANSION, "graph_expansion_dominance"),
            (FailureCategory.INTENT, "intent_dominance"),
            (FailureCategory.RERANKER, "reranker_dominance"),
        ):
            frep = _report_with({category: 6, FailureCategory.SOURCE_PRIOR: 4})
            report = RetrievalAnalytics().analyze(_benchmark([]), frep)
            rules = [rec.rule for rec in report.recommendations]
            assert rule in rules, f"expected {rule} in {rules}"

    def test_base_retrieval_dominance_fires(self):
        frep = _report_with({
            FailureCategory.NOT_RETRIEVED: 3,
            FailureCategory.RANKED_TOO_LOW: 3,
            FailureCategory.INTENT: 4,
        })
        report = RetrievalAnalytics().analyze(_benchmark([]), frep)
        rules = [rec.rule for rec in report.recommendations]
        assert "base_retrieval_dominance" in rules
        # NOT_RETRIEVED alone is only 30% — the combined rule fires, not R1.
        assert "not_retrieved_dominance" not in rules

    def test_log_documents_dominance_fires(self):
        failures = [
            _failure(FailureCategory.INTENT, relative_path="sim/log/run.log", chunk_type="generic_window")
            for _ in range(6)
        ] + [
            _failure(FailureCategory.INTENT, relative_path="rtl/a.sv", chunk_type="sv_module")
            for _ in range(4)
        ]
        frep = _failure_report([_query("q1", "why does it fail", failures)])
        report = RetrievalAnalytics().analyze(_benchmark([]), frep)
        rules = [rec.rule for rec in report.recommendations]
        assert "log_documents_dominance" in rules

    def test_worst_intent_fires(self):
        bench = _benchmark([
            _bench_result("dbg1", 0.0), _bench_result("dbg2", 0.0), _bench_result("dbg3", 0.1),
            _bench_result("exp1", 1.0), _bench_result("exp2", 1.0),
        ])
        frep = _failure_report([
            _query("dbg1", "why does it fail", []),
            _query("dbg2", "why is it stuck", []),
            _query("dbg3", "why the error", []),
            _query("exp1", "how does it work", []),
            _query("exp2", "what is it for", []),
        ])
        report = RetrievalAnalytics().analyze(bench, frep)
        rec = next((r for r in report.recommendations if r.rule == "worst_intent"), None)
        assert rec is not None
        assert "debug" in rec.condition

    def test_low_overall_health_fires(self):
        bench = _benchmark([_bench_result("q1", 0.0), _bench_result("q2", 0.0)])
        frep = _failure_report([
            _query("q1", "q1", [_failure(FailureCategory.INTENT, final_rank=3)]),
            _query("q2", "q2", []),
        ])
        report = RetrievalAnalytics().analyze(bench, frep)
        rules = [rec.rule for rec in report.recommendations]
        assert "low_overall_health" in rules

    def test_recommendations_are_diagnostic_only(self):
        frep = _report_with({
            FailureCategory.NOT_RETRIEVED: 5,
            FailureCategory.PROMPT_CUTOFF: 2,
            FailureCategory.SOURCE_PRIOR: 2,
            FailureCategory.INTENT: 1,
        })
        report = RetrievalAnalytics().analyze(_benchmark([]), frep)
        assert report.recommendations
        for rec in report.recommendations:
            assert rec.recommendation.startswith("Investigate") or rec.recommendation.startswith("No retrieval")


# ── Reporting ───────────────────────────────────────────────────────────


class TestReporting:
    def _report(self):
        bench = _benchmark([
            _bench_result("dbg1", 0.0), _bench_result("dbg2", 0.0), _bench_result("exp1", 1.0),
        ])
        failures = [
            _failure(FailureCategory.NOT_RETRIEVED, query_id="dbg1"),
            _failure(FailureCategory.INTENT, query_id="dbg2", final_rank=5, relative_path="rtl/i2c.sv", chunk_type="sv_module"),
            _failure(FailureCategory.RANKED_TOO_LOW, query_id="exp1", final_rank=4, relative_path="sim/log/run.log", chunk_type="generic_window", vector_rank=1),
        ]
        queries = [
            _query("dbg1", "why does the simulation fail", [failures[0]]),
            _query("dbg2", "why is the fifo stuck", [failures[1]]),
            _query("exp1", "how does the i2c master work", [failures[2]]),
        ]
        return RetrievalAnalytics().analyze(bench, _failure_report(queries))

    def test_reports_are_deterministic(self):
        first = self._report()
        second = self._report()
        assert first.model_dump() == second.model_dump()
        assert format_markdown(first) == format_markdown(second)

    def test_to_json_dict_is_strict_json(self):
        data = to_json_dict(self._report())
        dumped = json.dumps(data)
        parsed = json.loads(dumped)
        assert parsed["query_count"] == 3
        assert parsed["total_failures"] == 3
        assert parsed["failure_distribution"][0]["label"] == "NOT_RETRIEVED"
        assert parsed["document_analysis"][0]["category"] == "rtl"
        assert parsed["intent_analysis"][0]["intent"] == "explain"

    def test_markdown_contains_all_sections(self):
        text = format_markdown(self._report())
        for section in (
            "## Executive Summary",
            "## Overall Health",
            "## Failure Distribution",
            "## Stage Attribution",
            "## Document Analysis",
            "## Intent Analysis",
            "## Engineering Recommendations",
        ):
            assert section in text
        assert "NOT_RETRIEVED" in text
        assert "| vector |" in text
        assert "| explain |" in text

    def test_markdown_shows_unclassified_row(self):
        failures = [
            _failure(FailureCategory.NOT_RETRIEVED),
            _failure(FailureCategory.INTENT, relative_path="rtl/a.sv", chunk_type="sv_module"),
        ]
        report = RetrievalAnalytics().analyze(
            _benchmark([]), _failure_report([_query("q1", "q1", failures)])
        )
        text = format_markdown(report)
        assert "unclassified (no metadata)" in text

    def test_write_analytics_reports_writes_both_files(self, tmp_path: Path):
        report = self._report()
        json_path, md_path = write_analytics_reports(
            report,
            tmp_path / "retrieval_analytics.json",
            tmp_path / "retrieval_analytics.md",
        )
        assert json_path.exists()
        assert md_path is not None and md_path.exists()

        loaded = json.loads(json_path.read_text())
        assert loaded["query_count"] == 3
        assert loaded["failure_distribution"][0]["label"] == "NOT_RETRIEVED"

        md_text = md_path.read_text()
        assert "## Executive Summary" in md_text
        assert "## Engineering Recommendations" in md_text


# ── End to end: RB-01 + RB-02 -> analytics ─────────────────────────────


def test_analytics_end_to_end_against_real_pipeline(tmp_path: Path, db_session):
    """Index a small project and run the full chain: benchmark -> failure
    analysis -> analytics. An expected-but-missing chunk is guaranteed
    NOT_RETRIEVED."""
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
        "[00:00:02] simulation timeout after 1000 cycles\n",
    )

    provider, vector_chunks = _index(tmp_path, db_session)
    i2c_id = next(iter({c.chunk_id for c in vector_chunks if c.relative_path == "RTL/i2c_master.sv"}))

    golden = [
        GoldenQuery(
            query_id="rtl", project_id="proj1",
            query_text="what module detects the start condition on the i2c sda line",
            relevant_chunk_ids={i2c_id, "does-not-exist"},
        ),
        GoldenQuery(
            query_id="debug", project_id="proj1",
            query_text="why does the i2c simulation show a uvm_error timeout",
            relevant_chunk_ids={i2c_id},
        ),
    ]
    retriever = _build_retriever(db_session)
    embed = lambda q: provider.embed_texts([q])[0]  # noqa: E731

    benchmark_report = RetrievalBenchmark(retriever, embed_query_fn=embed, top_k=5).run(golden)
    failure_report = RetrievalFailureAnalyzer(retriever, embed_query_fn=embed, top_k=5).analyze(golden)
    analytics = RetrievalAnalytics().analyze(benchmark_report, failure_report, golden_queries=golden)

    assert analytics.query_count == 2
    assert analytics.total_failures >= 1
    not_retrieved = next(
        (bucket for bucket in analytics.failure_distribution if bucket.label == "NOT_RETRIEVED"),
        None,
    )
    assert not_retrieved is not None
    assert not_retrieved.count == 1
    assert analytics.stage_attribution[0].count >= 0

    json.loads(json.dumps(to_json_dict(analytics)))
    text = format_markdown(analytics)
    assert "## Executive Summary" in text
    assert "## Failure Distribution" in text


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
