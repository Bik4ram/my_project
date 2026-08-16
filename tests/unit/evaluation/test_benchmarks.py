"""
Tests for the retrieval evaluation benchmark framework.

Validates that:
  - Benchmark data is well-formed
  - All benchmark entries have required fields
  - Metrics produce correct values for textbook cases
  - The harness correctly aggregates results
"""

from __future__ import annotations

import pytest

from semrag.evaluation.benchmarks import get_benchmarks, get_benchmarks_by_area, get_benchmarks_by_intent
from semrag.evaluation.metrics import (
    average_rank,
    file_recall,
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from semrag.evaluation.models import ExpectedIntent, GoldenQuery, RepoArea, RetrievalMetrics


class TestBenchmarkData:
    """Benchmark entries must be well-formed."""

    def test_all_benchmarks_have_required_fields(self):
        for bm in get_benchmarks():
            assert bm.query_id, f"Missing query_id in {bm}"
            assert bm.query_text, f"Missing query_text in {bm}"
            assert bm.project_id, f"Missing project_id in {bm}"
            assert bm.expected_intent in ExpectedIntent, f"Invalid intent in {bm.query_id}"
            assert bm.repo_area in RepoArea, f"Invalid repo area in {bm.query_id}"
            assert bm.description, f"Missing description in {bm.query_id}"

    def test_all_benchmarks_have_unique_ids(self):
        ids = [bm.query_id for bm in get_benchmarks()]
        assert len(ids) == len(set(ids)), "Benchmark query IDs must be unique"

    def test_benchmarks_filter_by_area(self):
        retrieval = get_benchmarks_by_area(RepoArea.RETRIEVAL)
        for bm in retrieval:
            assert bm.repo_area == RepoArea.RETRIEVAL

    def test_benchmarks_filter_by_intent(self):
        arch = get_benchmarks_by_intent(ExpectedIntent.ARCHITECTURE)
        for bm in arch:
            assert bm.expected_intent == ExpectedIntent.ARCHITECTURE

    def test_repo_area_coverage(self):
        """All RepoArea values except GENERAL should have at least one
        benchmark (GENERAL is a fallback, not a real area)."""
        covered = {bm.repo_area for bm in get_benchmarks()}
        for area in RepoArea:
            if area != RepoArea.GENERAL:
                assert area in covered, f"No benchmarks for repo area {area}"

    def test_intent_coverage(self):
        """Architecture, verification, config, and API intents should
        have benchmarks. RTL and DEBUG need project-specific RTL data
        and are intentionally absent in this harness."""
        covered = {bm.expected_intent for bm in get_benchmarks()}
        required = {ExpectedIntent.ARCHITECTURE, ExpectedIntent.VERIFICATION,
                     ExpectedIntent.CONFIG, ExpectedIntent.API}
        for intent in required:
            assert intent in covered, f"No benchmarks for intent {intent}"


class TestMetrics:
    """IR metrics computed correctly against textbook examples."""

    def test_precision_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}
        assert precision_at_k(retrieved, relevant, 5) == pytest.approx(0.4)

    def test_precision_at_k_zero(self):
        assert precision_at_k(["a", "b"], {"c"}, 2) == 0.0

    def test_precision_at_k_empty_retrieved(self):
        assert precision_at_k([], {"a"}, 5) == 0.0

    def test_recall_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "f"}
        assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)

    def test_recall_at_k_no_relevant(self):
        assert recall_at_k(["a", "b"], set(), 2) == 0.0

    def test_reciprocal_rank(self):
        retrieved = ["x", "y", "a", "b"]
        relevant = {"a"}
        assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)

    def test_reciprocal_rank_no_match(self):
        assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    def test_reciprocal_rank_first_match(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == pytest.approx(1.0)

    def test_ndcg_at_k(self):
        # All relevant items in top positions — perfect NDCG
        retrieved = ["a", "b", "c"]
        relevant = {"a", "b"}
        assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)

    def test_ndcg_at_k_no_relevant(self):
        assert ndcg_at_k(["a", "b"], set(), 3) == 0.0

    def test_ndcg_at_k_with_misses(self):
        retrieved = ["c", "a", "b"]
        relevant = {"a", "b"}
        # DCG = 0/log2(1+2) + 1/log2(1+2) + 1/log2(1+3)
        #     = 0 + 1/log2(3) + 1/log2(4)
        #     = 0 + 0.6309 + 0.5 = 1.1309
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
        # NDCG = 1.1309 / 1.6309 = 0.6934
        ndcg = ndcg_at_k(retrieved, relevant, 3)
        assert ndcg == pytest.approx(0.6934, abs=1e-3)

    def test_hit_rate(self):
        assert hit_rate(["a", "b", "c"], {"a"}) == 1.0
        assert hit_rate(["a", "b"], {"c"}) == 0.0

    def test_average_rank(self):
        retrieved = ["x", "a", "y", "b"]
        relevant = {"a", "b"}
        avr = average_rank(retrieved, relevant)
        # "a" is at rank 2, "b" is at rank 4: (2 + 4) / 2 = 3.0
        assert avr == pytest.approx(3.0)

    def test_average_rank_inf(self):
        assert average_rank(["a", "b"], {"c"}) == float("inf")

    def test_file_recall(self):
        assert file_recall({"a.sv", "b.sv"}, {"a.sv", "c.sv"}) == pytest.approx(0.5)
        assert file_recall({"a.sv"}, {"a.sv"}) == 1.0
        assert file_recall(set(), {"a.sv"}) == 0.0
        assert file_recall({"a.sv"}, set()) == 1.0


class TestGoldenQuery:
    """GoldenQuery model validation."""

    def test_default_factories(self):
        gq = GoldenQuery(query_id="q1", project_id="p1", query_text="test query")
        assert gq.relevant_chunk_ids == set()
        assert gq.primary_files == set()
        assert gq.secondary_files == set()
        assert gq.expected_intent == ExpectedIntent.GENERAL

    def test_retrieval_metrics_defaults(self):
        rm = RetrievalMetrics(
            precision_at_k=0.0, recall_at_k=0.0,
            reciprocal_rank=0.0, ndcg_at_k=0.0, k=10,
        )
        assert rm.primary_file_recall == 0.0
        assert rm.hit_rate == 0.0
        assert rm.average_rank == 0.0
        assert rm.primary_files_found == []
        assert rm.primary_files_missed == []
