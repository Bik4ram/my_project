"""Diversity-aware selection (RI-03).

Covers the RI-03 acceptance criteria:

- duplicate log chunks capped per file (the 6-logs-top-k problem)
- duplicate RTL chunks from one file capped per file
- neighbouring-chunk suppression (adjacent windows of the same file)
- mixed-repository composition: multiple document categories coexist
- the highest-scoring chunk is never removed
- relevance preserved (score order) within the limits
- deterministic ordering (score desc, chunk_id tiebreak)
- explainability: ``signal_scores["diversity"]`` + selection reasons
- config validation + disabled config preserves the input ranking
- candidates beyond top_k are never evaluated
- end-to-end through the real pipeline stage 6.5 (stubbed stores):
  trace records ``diversity``/``diversity_reason``; disabled stage
  reproduces the pre-diversity ranking; the reranker pool is oversized
  when diversity is enabled.
- R3D: the reranker candidate pool (``top_k * pool_multiplier``) bounds
  what diversity can select. A narrow pool underfills ``top_k`` (the
  forward-chain candidate is lost); the shipped default pool width fills
  it.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.ranking.diversity import (
    DEFAULT_DIVERSITY,
    DiversityConfig,
    apply_diversity,
)
from semrag.retrieval.source_authority import classify_document
from semrag.storage.vector.models import VectorSearchFilter


def _result(
    chunk_id: str,
    *,
    relative_path: str = "rtl/a.sv",
    chunk_type: str = "sv_module",
    fused_score: float = 0.5,
    line_start: int | None = None,
    line_end: int | None = None,
    text: str = "",
    graph_expanded: bool = False,
    source_node_id: str | None = None,
) -> RetrievalResult:
    signal_scores = {}
    if graph_expanded:
        signal_scores["graph_expansion"] = 0.5
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="proj1",
        relative_path=relative_path,
        chunk_type=chunk_type,
        text=text,
        fused_score=fused_score,
        line_start=line_start,
        line_end=line_end,
        signal_scores=signal_scores,
        source_node_id=source_node_id,
    )


class TestDuplicateLogChunks:
    def test_top_scoring_chunk_never_removed(self):
        results = [
            _result(f"log{i}", relative_path="sim/log/run.log",
                    chunk_type="generic_window", fused_score=0.5 - i * 0.01)
            for i in range(6)
        ]
        out, _ = apply_diversity(results, top_k=3)
        assert out[0].chunk_id == "log0"
        assert [r.chunk_id for r in out] == ["log0", "log1", "log2"]

    def test_duplicate_log_chunks_capped_per_file(self):
        results = [
            _result(f"log{i}", relative_path="sim/log/run.log",
                    chunk_type="generic_window", fused_score=0.5 - i * 0.01)
            for i in range(6)
        ]
        results.append(
            _result("rtl", relative_path="rtl/i2c_master.sv",
                    chunk_type="sv_module", fused_score=0.40)
        )
        out, decisions = apply_diversity(results, top_k=10)
        ids = [r.chunk_id for r in out]
        assert sum(chunk_id.startswith("log") for chunk_id in ids) == 3
        assert "rtl" in ids
        assert decisions["log3"] == "file_limit"
        assert decisions["rtl"] == "selected"

    def test_duplicate_rtl_chunks_from_one_file_capped(self):
        results = [
            _result(f"rtl{i}", relative_path="rtl/i2c_master.sv",
                    chunk_type="sv_body_window", fused_score=0.5 - i * 0.01)
            for i in range(5)
        ]
        results.append(
            _result("tb", relative_path="tb/i2c_tb.svh",
                    chunk_type="generic_window", fused_score=0.35)
        )
        out, _ = apply_diversity(results, top_k=10)
        ids = [r.chunk_id for r in out]
        assert sum(chunk_id.startswith("rtl") for chunk_id in ids) == 3
        assert "tb" in ids


class TestCategoryLimit:
    def test_category_limit_caps_across_files(self):
        results = [
            _result(f"log{f}_{i}", relative_path=f"sim/log/run{f}.log",
                    chunk_type="generic_window", fused_score=0.5 - (f * 3 + i) * 0.01)
            for f in range(4)
            for i in range(3)
        ]
        out, decisions = apply_diversity(results, top_k=10)
        assert len(out) == 6  # max_per_category=6, all chunks are simulation_log
        assert all(classify_document(r.chunk_type, r.relative_path) == "simulation_log" for r in out)
        # file0 (3) + file1 (3) fill the category; the next chunk is rejected.
        assert decisions["log2_0"] == "category_limit"


class TestGraphExpansionCategoryExemption:
    """R2D-2: a small bounded number of graph-expanded candidates may exceed
    ``max_per_category``, while ordinary candidates always obey the cap."""

    def _uvm(self, chunk_id: str, fused_score: float, graph_expanded: bool = False):
        return _result(
            chunk_id,
            relative_path=f"tb/i2c_master_agent/{chunk_id}.svh",
            chunk_type="uvm_class",
            fused_score=fused_score,
            graph_expanded=graph_expanded,
        )

    def test_normal_category_limit_unchanged(self):
        """Six ordinary candidates in one category still enforce the cap;
        a 7th ordinary candidate is rejected with category_limit."""
        results = [self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(7)]
        out, decisions = apply_diversity(results, top_k=10)
        assert len(out) == 6
        assert decisions["normal6"] == "category_limit"
        assert all(d == "selected" for k, d in decisions.items() if k != "normal6")

    def test_one_graph_candidate_can_bypass_category_limit(self):
        """One graph-expanded candidate may exceed the category cap."""
        normal = [self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(6)]
        graph = self._uvm("graph_agent", 0.49, graph_expanded=True)
        out, decisions = apply_diversity(normal + [graph], top_k=10)
        ids = [r.chunk_id for r in out]
        assert len(out) == 7
        assert "graph_agent" in ids
        assert decisions["graph_agent"] == "selected"

    def test_more_than_quota_graph_candidates_cannot_bypass(self):
        """With ``graph_expansion_quota=1``, only one graph candidate may
        exceed the cap; the next one is rejected with category_limit."""
        normal = [self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(6)]
        graph1 = self._uvm("graph_a", 0.49, graph_expanded=True)
        graph2 = self._uvm("graph_b", 0.48, graph_expanded=True)
        cfg = DiversityConfig(graph_expansion_quota=1)
        out, decisions = apply_diversity(normal + [graph1, graph2], top_k=10, config=cfg)
        ids = [r.chunk_id for r in out]
        assert len(out) == 7
        assert "graph_a" in ids
        assert "graph_b" not in ids
        assert decisions["graph_b"] == "category_limit"

    def test_quota_three_allows_up_to_three_graph_candidates(self):
        """The selected quota (3) lets three graph candidates exceed the cap;
        the 4th is rejected with category_limit — the exemption stays
        bounded."""
        normal = [self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(6)]
        graphs = [
            self._uvm(f"graph_{i}", 0.49 - i * 0.001, graph_expanded=True)
            for i in range(4)
        ]
        cfg = DiversityConfig(graph_expansion_quota=3)
        out, decisions = apply_diversity(normal + graphs, top_k=12, config=cfg)
        ids = [r.chunk_id for r in out]
        assert len(out) == 9  # 6 normal + 3 exempted graph
        assert all(f"graph_{i}" in ids for i in range(3))
        assert "graph_3" not in ids
        assert decisions["graph_3"] == "category_limit"

    def test_default_quota_allows_three_graph_candidates(self):
        """The shipped default quota (3) lets three graph candidates bypass
        the category cap, keeping the forward multi-hop chain reachable."""
        normal = [self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(6)]
        graphs = [
            self._uvm(f"graph_{i}", 0.49 - i * 0.001, graph_expanded=True)
            for i in range(3)
        ]
        out, decisions = apply_diversity(normal + graphs, top_k=12)
        ids = [r.chunk_id for r in out]
        assert len(out) == 9
        assert all(f"graph_{i}" in ids for i in range(3))
        assert all(decisions[f"graph_{i}"] == "selected" for i in range(3))

    def test_graph_candidate_still_obeys_max_per_file(self):
        """The exemption never overrides ``max_per_file``: a graph candidate
        from an already-saturated file is rejected with file_limit."""
        results = [
            _result(f"f{i}", relative_path="tb/pkg.sv", chunk_type="uvm_package",
                    fused_score=0.50 - i * 0.001)
            for i in range(3)
        ]
        graph = _result("graph_same_file", relative_path="tb/pkg.sv",
                        chunk_type="uvm_package", fused_score=0.40,
                        graph_expanded=True)
        out, decisions = apply_diversity(results + [graph], top_k=10)
        assert "graph_same_file" not in {r.chunk_id for r in out}
        assert decisions["graph_same_file"] == "file_limit"

    def test_strong_normal_candidate_not_displaced(self):
        """A strong ordinary candidate in an open slot is never displaced by
        a graph candidate — the exemption only applies past the category cap."""
        normal = [self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(5)]
        graph = self._uvm("graph_agent", 0.49, graph_expanded=True)
        out, _ = apply_diversity(normal + [graph], top_k=10)
        ids = [r.chunk_id for r in out]
        # Both fit: 5 normal (under cap) + 1 graph = 6; order by score.
        assert ids[0] == "normal0"
        assert "graph_agent" in ids

    def test_graph_candidate_below_score_cutoff_not_promoted(self):
        """A graph candidate beyond the top-k is never evaluated — the
        exemption does not promote candidates score order would not reach."""
        top = [self._uvm(f"top{i}", 0.50 - i * 0.001) for i in range(8)]
        graph = self._uvm("graph_late", 0.30, graph_expanded=True)
        out, decisions = apply_diversity(top + [graph], top_k=6)
        assert "graph_late" not in {r.chunk_id for r in out}
        assert "graph_late" not in decisions  # never evaluated

    def test_deterministic_with_exemption(self):
        results = [
            self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(6)
        ] + [self._uvm("graph_a", 0.49, graph_expanded=True)]
        first, first_dec = apply_diversity(results, top_k=10)
        second, second_dec = apply_diversity(
            [r.model_copy(deep=True) for r in results], top_k=10
        )
        assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
        assert first_dec == second_dec

    def test_non_graph_categories_unaffected(self):
        """A graph candidate in one category never loosens the cap for a
        different category's ordinary candidates."""
        uvm_normals = [self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(6)]
        graph = self._uvm("graph_agent", 0.49, graph_expanded=True)
        spec = [
            _result(f"spec{i}", relative_path=f"docs/spec/s{i}.md",
                    chunk_type="markdown_section", fused_score=0.48 - i * 0.001)
            for i in range(6)
        ]
        out, decisions = apply_diversity(uvm_normals + [graph] + spec, top_k=15)
        cat = {r.chunk_id: classify_document(r.chunk_type, r.relative_path) for r in out}
        spec_selected = [cid for cid in cat if cat[cid] == "specification"]
        uvm_selected = [cid for cid in cat if cat[cid] == "uvm"]
        assert len(spec_selected) == 6  # spec cap untouched
        assert len(uvm_selected) == 7  # 6 normal + 1 exempted graph

    def test_exempted_graph_candidate_marked_selected(self):
        """An accepted exemption must record diversity=1.0 and reason
        ``"selected"`` — never category_limit."""
        normal = [self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(6)]
        graph = self._uvm("graph_agent", 0.49, graph_expanded=True)
        out, decisions = apply_diversity(normal + [graph], top_k=10)
        g = next(r for r in out if r.chunk_id == "graph_agent")
        assert g.signal_scores["diversity"] == 1.0
        assert decisions["graph_agent"] == "selected"
        assert decisions["graph_agent"] != "category_limit"

    def test_quota_zero_disables_exemption(self):
        normal = [self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(6)]
        graph = self._uvm("graph_agent", 0.49, graph_expanded=True)
        cfg = DiversityConfig(graph_expansion_quota=0)
        out, decisions = apply_diversity(normal + [graph], top_k=10, config=cfg)
        assert "graph_agent" not in {r.chunk_id for r in out}
        assert decisions["graph_agent"] == "category_limit"

    def test_quota_validation(self):
        with pytest.raises(ValueError):
            DiversityConfig(graph_expansion_quota=-1)


class TestNeighbourSuppression:
    def test_adjacent_window_suppressed_but_spaced_chunk_kept(self):
        results = [
            _result("a", relative_path="rtl/i2c.sv", chunk_type="sv_body_window",
                    fused_score=0.5, line_start=0, line_end=79),
            _result("b", relative_path="rtl/i2c.sv", chunk_type="sv_body_window",
                    fused_score=0.48, line_start=80, line_end=159),
            _result("c", relative_path="rtl/i2c.sv", chunk_type="sv_body_window",
                    fused_score=0.46, line_start=160, line_end=239),
        ]
        out, decisions = apply_diversity(results, top_k=3)
        ids = [r.chunk_id for r in out]
        assert "a" in ids and "c" in ids
        assert "b" not in ids
        assert decisions["b"] == "neighbour"

    def test_overlapping_windows_are_neighbours(self):
        results = [
            _result("a", relative_path="sim/log/run.log", chunk_type="generic_window",
                    fused_score=0.5, line_start=0, line_end=79),
            _result("b", relative_path="sim/log/run.log", chunk_type="generic_window",
                    fused_score=0.49, line_start=0, line_end=79),
        ]
        out, decisions = apply_diversity(results, top_k=2)
        assert [r.chunk_id for r in out] == ["a"]
        assert decisions["b"] == "neighbour"

    def test_metadata_free_chunks_never_neighbours(self):
        results = [
            _result("a", relative_path="sim/log/run.log", chunk_type="generic_window",
                    fused_score=0.5),
            _result("b", relative_path="sim/log/run.log", chunk_type="generic_window",
                    fused_score=0.49),
        ]
        out, _ = apply_diversity(results, top_k=2)
        # No line metadata -> no neighbour suppression; only the file cap bites.
        assert [r.chunk_id for r in out] == ["a", "b"]


class TestMixedRepository:
    def test_mixed_repository_composition(self):
        results = [
            _result(f"log{i}", relative_path="sim/log/run.log",
                    chunk_type="generic_window", fused_score=0.50 - i * 0.01)
            for i in range(8)
        ]
        results += [
            _result("rtl1", relative_path="rtl/i2c_master.sv",
                    chunk_type="sv_module", fused_score=0.42),
            _result("rtl2", relative_path="rtl/i2c_top.sv",
                    chunk_type="sv_module", fused_score=0.41),
            _result("spec", relative_path="docs/spec/i2c.md",
                    chunk_type="markdown_section", fused_score=0.40),
            _result("readme", relative_path="README.md",
                    chunk_type="markdown_section", fused_score=0.39),
        ]
        out, _ = apply_diversity(results, top_k=10)
        categories = {classify_document(r.chunk_type, r.relative_path) for r in out}
        assert {"simulation_log", "rtl", "specification", "markdown"} <= categories
        assert sum(classify_document(r.chunk_type, r.relative_path) == "simulation_log" for r in out) == 3


class TestRelevanceAndDeterminism:
    def test_score_order_preserved_within_limits(self):
        results = [
            _result("low", relative_path="rtl/a.sv", fused_score=0.3),
            _result("high", relative_path="rtl/b.sv", fused_score=0.9),
        ]
        out, _ = apply_diversity(results, top_k=2)
        assert [r.chunk_id for r in out] == ["high", "low"]

    def test_deterministic_across_runs_and_copies(self):
        results = [
            _result("log0", relative_path="sim/log/run.log", chunk_type="generic_window", fused_score=0.5),
            _result("log1", relative_path="sim/log/run.log", chunk_type="generic_window", fused_score=0.5),
            _result("rtl", relative_path="rtl/a.sv", fused_score=0.4),
            _result("spec", relative_path="docs/spec/x.md", chunk_type="markdown_section", fused_score=0.3),
        ]
        first, first_decisions = apply_diversity(results, top_k=3)
        second, second_decisions = apply_diversity(
            [r.model_copy(deep=True) for r in results], top_k=3
        )
        assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
        assert first_decisions == second_decisions

    def test_top_k_never_exceeded(self):
        results = [
            _result(f"r{i}", relative_path=f"rtl/m{i}.sv",
                    chunk_type="sv_module", fused_score=0.5 - i * 0.01)
            for i in range(20)
        ]
        out, _ = apply_diversity(results, top_k=4)
        assert len(out) <= 4


class TestExplainability:
    def test_signal_and_reason_recorded(self):
        results = [
            _result(f"log{i}", relative_path="sim/log/run.log",
                    chunk_type="generic_window", fused_score=0.5 - i * 0.01)
            for i in range(5)
        ]
        out, decisions = apply_diversity(results, top_k=10)
        assert all(r.signal_scores["diversity"] == 1.0 for r in out)
        assert all(decisions[r.chunk_id] == "selected" for r in out)
        rejected = next(r for r in results if r.chunk_id == "log3")
        assert rejected.signal_scores["diversity"] == 0.0
        assert decisions["log3"] == "file_limit"

    def test_candidates_beyond_top_k_never_evaluated(self):
        results = [
            _result(f"r{i}", relative_path=f"rtl/m{i}.sv",
                    chunk_type="sv_module", fused_score=0.5 - i * 0.01)
            for i in range(6)
        ]
        out, decisions = apply_diversity(results, top_k=2)
        assert [r.chunk_id for r in out] == ["r0", "r1"]
        assert set(decisions) == {"r0", "r1"}
        assert results[2].signal_scores.get("diversity") is None


class TestConfig:
    def test_disabled_returns_input_unchanged(self):
        results = [
            _result(f"log{i}", relative_path="sim/log/run.log",
                    chunk_type="generic_window", fused_score=0.5 - i * 0.01)
            for i in range(6)
        ]
        cfg = DiversityConfig(enabled=False)
        out, decisions = apply_diversity(results, top_k=3, config=cfg)
        assert [r.chunk_id for r in out] == [r.chunk_id for r in results[:3]]
        assert decisions == {}
        assert all("diversity" not in r.signal_scores for r in results)

    def test_config_validation(self):
        with pytest.raises(ValueError):
            DiversityConfig(max_per_file=0)
        with pytest.raises(ValueError):
            DiversityConfig(max_per_category=0)
        with pytest.raises(ValueError):
            DiversityConfig(pool_multiplier=0)
        with pytest.raises(ValueError):
            DiversityConfig(neighbour_window=-1)

    def test_defaults_are_sane(self):
        cfg = DEFAULT_DIVERSITY
        assert cfg.enabled is True
        assert cfg.max_per_file == 3
        assert cfg.max_per_category == 6
        assert cfg.graph_expansion_quota == 3
        assert cfg.neighbour_window == 25
        assert cfg.pool_multiplier == 3


# ── Pipeline integration (stubbed stores) ───────────────────────────────


def _stub_store_retriever(retriever, query="explain the i2c master start condition"):
    results, trace = retriever.retrieve(
        query,
        [0.0] * 8,
        project_id="p",
        top_k=5,
        filters=VectorSearchFilter(project_id="p"),
        return_trace=True,
    )
    return results, trace


def _patch_stores(monkeypatch, vector_chunks, lexical_chunks=None):
    from semrag.retrieval import pipeline as pipeline_mod

    class _StubVector:
        def __init__(self, session):
            pass

        def search(self, embedding, top_k, filters):
            return list(vector_chunks)

    class _StubLexical:
        def __init__(self, session):
            pass

        def search(self, query_text, top_k, filters):
            return list(lexical_chunks or [])

    monkeypatch.setattr(pipeline_mod, "PgvectorStore", _StubVector)
    monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
    monkeypatch.setattr(
        pipeline_mod, "to_retrieval_results",
        lambda session, raw, signal_name: raw,
    )


class TestPipelineStage65:
    def test_stage65_records_diversity_and_diversifies(self, monkeypatch):
        from semrag.retrieval.pipeline import HybridRetriever

        chunks = [
            _result(f"log{i}", relative_path="sim/log/run.log",
                    chunk_type="generic_window", fused_score=0.020 - i * 0.001,
                    line_start=i * 200, line_end=i * 200 + 79)
            for i in range(5)
        ]
        chunks.append(
            _result("rtl", relative_path="rtl/a.sv", chunk_type="sv_module",
                    fused_score=0.014, line_start=0, line_end=40)
        )
        _patch_stores(monkeypatch, chunks)

        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            reranker=None,
        )
        results, trace = _stub_store_retriever(retriever)

        by_id = {c.chunk_id: c for c in trace.chunks()}
        ids = {r.chunk_id for r in results}
        # run.log capped at 3 (file limit) even though 5 ranked above rtl.
        assert sum(c.chunk_id.startswith("log") for c in results) == 3
        assert "rtl" in ids
        assert by_id["rtl"].diversity == 1.0
        assert by_id["rtl"].diversity_reason == "selected"
        assert by_id["log3"].diversity == 0.0
        assert by_id["log3"].diversity_reason == "file_limit"

    def test_disabled_stage_preserves_previous_ranking(self, monkeypatch):
        from semrag.retrieval.pipeline import HybridRetriever

        chunks = [
            _result(f"log{i}", relative_path="sim/log/run.log",
                    chunk_type="generic_window", fused_score=0.020 - i * 0.001,
                    line_start=i * 200, line_end=i * 200 + 79)
            for i in range(5)
        ]
        chunks.append(
            _result("rtl", relative_path="rtl/a.sv", chunk_type="sv_module",
                    fused_score=0.014, line_start=0, line_end=40)
        )
        _patch_stores(monkeypatch, chunks)

        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            reranker=None,
            diversity_config=DiversityConfig(enabled=False),
        )
        results, trace = _stub_store_retriever(retriever)

        # Pre-diversity ranking: all five log chunks outrank rtl.
        assert [r.chunk_id for r in results] == [f"log{i}" for i in range(5)]
        assert all(c.diversity is None for c in trace.chunks())
        assert all(c.diversity_reason is None for c in trace.chunks())

    def test_reranker_pool_oversized_when_diversity_enabled(self, monkeypatch):
        from semrag.retrieval.pipeline import HybridRetriever

        class _RecordingReranker:
            def __init__(self):
                self.top_k = None

            def rerank(self, query, results, top_k):
                self.top_k = top_k
                return results[:top_k]

        chunks = [
            _result(f"r{i}", relative_path=f"rtl/m{i}.sv",
                    chunk_type="sv_module", fused_score=0.020 - i * 0.001)
            for i in range(8)
        ]
        _patch_stores(monkeypatch, chunks)

        rec = _RecordingReranker()
        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            reranker=rec,
        )
        _stub_store_retriever(retriever)
        assert rec.top_k == 15  # top_k=5 * pool_multiplier=3

        rec_off = _RecordingReranker()
        retriever_off = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            reranker=rec_off,
            diversity_config=DiversityConfig(enabled=False),
        )
        _stub_store_retriever(retriever_off)
        assert rec_off.top_k == 5

    def test_narrow_pool_underfills_top_k(self, monkeypatch):
        """R3D: with ``pool_multiplier=1`` the reranker returns only
        ``top_k`` candidates; once diversity rejects capped files the pool
        is exhausted and the context is short. The forward-chain candidate
        is never evaluated."""
        from semrag.retrieval.pipeline import HybridRetriever

        class _TruncatingReranker:
            def __init__(self):
                self.top_k = None

            def rerank(self, query, results, top_k):
                self.top_k = top_k
                return results[:top_k]

        chunks = [
            _result(f"log{i}", relative_path="sim/log/run.log",
                    chunk_type="generic_window", fused_score=0.020 - i * 0.001,
                    line_start=i * 200, line_end=i * 200 + 79)
            for i in range(5)
        ]
        chunks.append(
            _result("rtl", relative_path="rtl/a.sv", chunk_type="sv_module",
                    fused_score=0.014, line_start=0, line_end=40)
        )
        _patch_stores(monkeypatch, chunks)

        rec = _TruncatingReranker()
        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            reranker=rec,
            diversity_config=DiversityConfig(pool_multiplier=1),
        )
        results, _ = _stub_store_retriever(retriever)
        assert rec.top_k == 5  # top_k=5 * pool_multiplier=1
        # the 3 surviving logs occupy the whole context; rtl (rank 6) is cut.
        assert [r.chunk_id for r in results] == ["log0", "log1", "log2"]
        assert "rtl" not in {r.chunk_id for r in results}

    def test_default_pool_width_fills_top_k(self, monkeypatch):
        """R3D: the shipped default (``pool_multiplier=3``) widens the
        reranker pool so candidates that a ``pool_multiplier=1`` reranker
        would truncate still reach diversity, letting the lower-scored
        forward-chain candidate into the final context."""
        from semrag.retrieval.pipeline import HybridRetriever

        class _TruncatingReranker:
            def __init__(self):
                self.top_k = None

            def rerank(self, query, results, top_k):
                self.top_k = top_k
                return results[:top_k]

        chunks = [
            _result(f"log{i}", relative_path="sim/log/run.log",
                    chunk_type="generic_window", fused_score=0.020 - i * 0.001,
                    line_start=i * 200, line_end=i * 200 + 79)
            for i in range(5)
        ]
        chunks.append(
            _result("rtl", relative_path="rtl/a.sv", chunk_type="sv_module",
                    fused_score=0.014, line_start=0, line_end=40)
        )
        _patch_stores(monkeypatch, chunks)

        rec = _TruncatingReranker()
        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            reranker=rec,
        )
        results, _ = _stub_store_retriever(retriever)
        assert rec.top_k == 15  # top_k=5 * pool_multiplier=3
        ids = [r.chunk_id for r in results]
        assert "rtl" in ids  # forward candidate survives the wider pool
        assert sum(c.startswith("log") for c in ids) == 3  # file cap respected


class TestGraphIdentityExemption:
    """R3F: bounded identity-aware graph exemption.

    An *ordinary* retrieval copy of a node in the active graph-expansion
    source-node set (identity by ``source_node_id``) is graph-relevant and
    may use the ``graph_expansion_quota`` like a graph-marked candidate —
    so a useful structural node is no longer rejected merely because
    unrelated ordinary candidates filled the category cap. The exemption
    stays bounded: shared quota, one slot per distinct source_node_id, all
    other caps intact. Keyed on ``source_node_id`` only — never file or
    symbol names.
    """

    def _uvm(self, chunk_id: str, fused_score: float,
             source_node_id: str | None = None, graph_expanded: bool = False):
        signal_scores = {}
        if graph_expanded:
            signal_scores["graph_expansion"] = 0.5
        return RetrievalResult(
            chunk_id=chunk_id,
            project_id="proj1",
            relative_path=f"tb/i2c_master_agent/{chunk_id}.svh",
            chunk_type="uvm_class",
            text="",
            fused_score=fused_score,
            signal_scores=signal_scores,
            source_node_id=source_node_id,
        )

    def _six_normals(self):
        return [self._uvm(f"normal{i}", 0.50 - i * 0.001) for i in range(6)]

    def test_non_graph_category_limit_unchanged(self):
        """Test 1: without identity membership (empty set, or a set that
        does not contain the node), the category cap is byte-for-byte the
        pre-R3F behavior — a 7th ordinary UVM candidate is still
        rejected."""
        results = self._six_normals() + [
            self._uvm("normal6", 0.48, source_node_id="node::unrelated")
        ]
        out, decisions = apply_diversity(
            results, top_k=10, graph_source_node_ids=frozenset({"node::other"})
        )
        assert len(out) == 6
        assert "normal6" not in {r.chunk_id for r in out}
        assert decisions["normal6"] == "category_limit"

    def test_quota_three_enforced_across_identity_and_graph(self):
        """Test 2: the shared quota (3) bounds identity-matched ordinary
        candidates and graph-marked candidates together; the 4th
        graph-relevant candidate is rejected with category_limit."""
        normal = self._six_normals()
        id_a = self._uvm("id_a", 0.49, source_node_id="node::a")
        id_b = self._uvm("id_b", 0.48, source_node_id="node::b")
        graph_c = self._uvm("graph_c", 0.47, graph_expanded=True)
        id_d = self._uvm("id_d", 0.46, source_node_id="node::d")
        ids = frozenset({"node::a", "node::b", "node::d"})
        out, decisions = apply_diversity(
            normal + [id_a, id_b, graph_c, id_d], top_k=12, graph_source_node_ids=ids
        )
        selected_ids = {r.chunk_id for r in out}
        assert len(out) == 9  # 6 normal + 3 exempted
        assert {"id_a", "id_b", "graph_c"} <= selected_ids
        assert "id_d" not in selected_ids
        assert decisions["id_d"] == "category_limit"

    def test_ordinary_copy_of_graph_node_recognized(self):
        """Test 3: an ordinary candidate whose source_node_id is in the
        active graph-expansion source-node set is graph-relevant and may
        use the quota — it is not rejected merely because the UVM category
        cap is full."""
        normal = self._six_normals()
        driver = self._uvm("driver", 0.49, source_node_id="node::driver")
        out, decisions = apply_diversity(
            normal + [driver], top_k=10,
            graph_source_node_ids=frozenset({"node::driver"}),
        )
        assert "driver" in {r.chunk_id for r in out}
        assert decisions["driver"] == "selected"
        assert next(r for r in out if r.chunk_id == "driver").signal_scores["diversity"] == 1.0

    def test_same_node_ordinary_and_graph_not_two_slots(self):
        """Test 4: graph-expanded and ordinary copies of the same
        ``source_node_id`` do not consume two independent quota slots —
        the exemption grants at most one slot per distinct source_node_id."""
        normal = self._six_normals()
        ordinary = self._uvm("seq_item_o", 0.49, source_node_id="node::seq_item")
        graph = self._uvm("seq_item_g", 0.48, source_node_id="node::seq_item",
                          graph_expanded=True)
        out, decisions = apply_diversity(
            normal + [ordinary, graph], top_k=10,
            graph_source_node_ids=frozenset({"node::seq_item"}),
        )
        ids = [r.chunk_id for r in out]
        assert sum(cid.startswith("seq_item") for cid in ids) == 1
        assert "seq_item_o" in ids
        assert decisions["seq_item_g"] == "category_limit"

    def test_useful_graph_path_candidate_survives_saturation(self):
        """Test 5: a useful structural candidate (e.g. the driver on the
        query's dependency chain) survives category saturation via identity
        recognition."""
        normal = self._six_normals()
        driver = self._uvm("driver", 0.49, source_node_id="node::driver")
        out, decisions = apply_diversity(
            normal + [driver], top_k=10,
            graph_source_node_ids=frozenset({"node::driver"}),
        )
        assert "driver" in {r.chunk_id for r in out}
        assert decisions["driver"] == "selected"

    def test_unrelated_uvm_cannot_bypass_category_limit(self):
        """Test 6: being UVM is not enough — a candidate whose
        source_node_id is NOT in the graph-expansion set stays
        category-limited even though its category is saturated."""
        normal = self._six_normals()
        unrelated = self._uvm("other_uvm", 0.49, source_node_id="node::other")
        out, decisions = apply_diversity(
            normal + [unrelated], top_k=10,
            graph_source_node_ids=frozenset({"node::driver"}),
        )
        assert "other_uvm" not in {r.chunk_id for r in out}
        assert decisions["other_uvm"] == "category_limit"

    def test_off_path_graph_candidate_cannot_consume_unlimited_quota(self):
        """Test 7: off-path but graph-relevant candidates are bounded by
        the shared quota — at most 3 exempted candidates in total."""
        normal = self._six_normals()
        off = [
            self._uvm(f"off_{c}", 0.49 - i * 0.001, source_node_id=f"node::off_{c}")
            for i, c in enumerate("abcd")
        ]
        ids = frozenset(f"node::off_{c}" for c in "abcd")
        out, decisions = apply_diversity(
            normal + off, top_k=12, graph_source_node_ids=ids
        )
        admitted = [c for c in out if c.chunk_id.startswith("off")]
        assert len(admitted) == 3
        assert decisions["off_d"] == "category_limit"

    def test_identity_candidate_still_obeys_max_per_file(self):
        """Test 8: the identity exemption never overrides ``max_per_file`` —
        a graph-relevant candidate from an already-saturated file is
        rejected with file_limit."""
        results = [
            _result(f"f{i}", relative_path="tb/pkg.sv", chunk_type="uvm_package",
                    fused_score=0.50 - i * 0.001)
            for i in range(3)
        ]
        ident = _result("ident_same_file", relative_path="tb/pkg.sv",
                        chunk_type="uvm_package", fused_score=0.40,
                        source_node_id="node::x")
        out, decisions = apply_diversity(
            results + [ident], top_k=10, graph_source_node_ids=frozenset({"node::x"})
        )
        assert "ident_same_file" not in {r.chunk_id for r in out}
        assert decisions["ident_same_file"] == "file_limit"

    def test_strong_ordinary_candidate_not_arbitrarily_displaced(self):
        """Test 9: strong ordinary candidates keep their score-ordered
        slots; the exemption only applies past the category cap and never
        promotes a candidate that score order alone would not reach."""
        strong = [self._uvm(f"strong{i}", 0.60 - i * 0.001) for i in range(5)]
        ident = self._uvm("ident", 0.50, source_node_id="node::x")
        out, _ = apply_diversity(
            strong + [ident], top_k=10, graph_source_node_ids=frozenset({"node::x"})
        )
        ids = [r.chunk_id for r in out]
        assert ids[0] == "strong0"
        assert [c for c in ids if c.startswith("strong")] == [f"strong{i}" for i in range(5)]
        assert "ident" in ids

    def test_deterministic_with_identity_exemption(self):
        """Test 10: deterministic ordering is preserved under the identity
        exemption across repeated runs and deep-copied inputs."""
        results = self._six_normals() + [
            self._uvm("driver", 0.49, source_node_id="node::driver"),
        ]
        ids = frozenset({"node::driver"})
        first, first_dec = apply_diversity(results, top_k=10, graph_source_node_ids=ids)
        second, second_dec = apply_diversity(
            [r.model_copy(deep=True) for r in results], top_k=10,
            graph_source_node_ids=ids,
        )
        assert [r.chunk_id for r in first] == [r.chunk_id for r in second]
        assert first_dec == second_dec

    def test_reverse_dependents_identity_recognition(self):
        """Test 11: reverse (dependents) traversal produces the same
        source_node_id identity set; an ordinary copy of a dependent is
        recognized as graph-relevant. Direction is orthogonal — the
        exemption keys on source_node_id membership, not direction."""
        normal = self._six_normals()
        dependent = self._uvm("dependent", 0.49, source_node_id="node::dep")
        out, decisions = apply_diversity(
            normal + [dependent], top_k=10,
            graph_source_node_ids=frozenset({"node::dep"}),
        )
        assert "dependent" in {r.chunk_id for r in out}
        assert decisions["dependent"] == "selected"

    def test_forward_dependencies_identity_recognition(self):
        """Test 12: forward (dependencies) traversal feeds the same identity
        set; an ordinary copy of a dependency is recognized as
        graph-relevant and survives category saturation."""
        normal = self._six_normals()
        dependency = self._uvm("dependency", 0.49, source_node_id="node::dep")
        out, decisions = apply_diversity(
            normal + [dependency], top_k=10,
            graph_source_node_ids=frozenset({"node::dep"}),
        )
        assert "dependency" in {r.chunk_id for r in out}
        assert decisions["dependency"] == "selected"
