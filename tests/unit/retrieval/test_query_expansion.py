"""Query expansion (RI-05).

Covers the RI-05 acceptance criteria:

- the original query is always searched (never replaced by expansions)
- expansion generation adds intent-specific repository terminology
- the max expansion count is bounded by config
- duplicate formulations are removed
- expansion is deterministic (same input -> same output)
- a query with no usable expansion returns an empty list
- expansion disabled -> no formulations
- different query intents produce different expansions
- result deduplication (a chunk found by original + expansion is one result)
- the trace records expansion provenance (retrieved_by_original /
  retrieved_by_expansion / expansion_queries)
- fail-open: an expansion error never breaks retrieval
- no regression for queries that need no expansion
- latency is bounded and recorded
- end-to-end through the real pipeline stage 2.5 (stubbed stores)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semrag.retrieval.intent import QueryIntent, classify_intent
from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.query_expansion import (
    DEFAULT_QUERY_EXPANSION,
    QueryExpansionConfig,
    build_query_expansion_config,
    expand_query,
)
from semrag.storage.vector.models import VectorSearchFilter


class _FakeSettings:
    query_expansion_enabled = True
    query_expansion_max_expansions = 4
    query_expansion_max_expansion_length = 90
    query_expansion_timeout_ms = None


def _result(chunk_id, **kwargs) -> RetrievalResult:
    defaults = dict(
        project_id="p",
        relative_path="rtl/a.sv",
        chunk_type="sv_module",
        text="",
        fused_score=0.5,
    )
    defaults.update(kwargs)
    return RetrievalResult(chunk_id=chunk_id, **defaults)


class TestOriginalAlwaysSearched:
    def test_original_query_is_never_replaced(self):
        q = "why did the assertion fail"
        cfg = QueryExpansionConfig(max_expansions=3)
        out = expand_query(q, intent=QueryIntent.VERIFICATION, config=cfg)
        assert out, "expected at least one expansion"
        for formulation in out:
            # Every formulation *starts* with the original query text;
            # expansion only appends terminology, never rewrites.
            assert formulation.startswith(q)

    def test_pipeline_always_searches_original_query(self, monkeypatch):
        from semrag.retrieval import pipeline as pipeline_mod

        searched = []

        class _StubLexical:
            def __init__(self, session):
                pass

            def search(self, query_text, top_k, filters):
                searched.append(query_text)
                return []

        monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
        monkeypatch.setattr(
            pipeline_mod,
            "to_retrieval_results",
            lambda session, raw, signal_name: [],
        )

        from semrag.retrieval.pipeline import HybridRetriever

        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            query_expansion_config=QueryExpansionConfig(max_expansions=2),
        )
        retriever.retrieve(
            "why did the assertion fail",
            [0.0] * 8,
            project_id="p",
            filters=VectorSearchFilter(project_id="p"),
        )
        assert searched[0] == "why did the assertion fail"
        assert len(searched) == 1 + 2  # original + max_expansions


class TestExpansionGeneration:
    def test_adds_intent_terminology(self):
        cfg = QueryExpansionConfig(max_expansions=2)
        out = expand_query("why did the assertion fail", intent=QueryIntent.VERIFICATION, config=cfg)
        joined = " ".join(out)
        # Assertion is VERIFICATION vocabulary; the query only mentioned
        # "assertion", so the expansion must surface uvm/sequence terms.
        assert "uvm" in joined
        assert "sequencer" in joined

    def test_skips_terms_already_in_query(self):
        cfg = QueryExpansionConfig(max_expansions=3)
        base = "uvm sequencer assertion fail"
        out = expand_query(base, intent=QueryIntent.VERIFICATION, config=cfg)
        assert out, "expected at least one expansion"
        for formulation in out:
            # The appended tail (after the base query) must not re-add terms
            # the query already covers.
            tail = formulation[len(base) :]
            assert "uvm" not in tail
            assert "sequencer" not in tail
            assert "assertion" not in tail


class TestBounded:
    def test_max_expansions_is_a_bound(self):
        cfg = QueryExpansionConfig(max_expansions=2)
        out = expand_query("how does the driver drive the sequencer", intent=QueryIntent.VERIFICATION, config=cfg)
        assert len(out) <= 2

    def test_max_expansion_length_is_a_bound(self):
        cfg = QueryExpansionConfig(max_expansions=5, max_expansion_length=40)
        out = expand_query("why did the assertion fail", intent=QueryIntent.VERIFICATION, config=cfg)
        assert out
        for formulation in out:
            assert len(formulation) <= 40

    def test_zero_max_expansions_is_rejected(self):
        with pytest.raises(ValueError):
            QueryExpansionConfig(max_expansions=0)


class TestDeduplicationAndDeterminism:
    def test_duplicate_formulations_removed(self):
        # With a long max_expansion_length the same truncated formulation
        # can surface from two groups; the dedupe keeps only distinct ones.
        cfg = QueryExpansionConfig(max_expansions=3)
        out = expand_query("uvm sequencer driver monitor scoreboard", intent=QueryIntent.VERIFICATION, config=cfg)
        seen = {f.lower() for f in out}
        assert len(out) == len(seen)

    def test_deterministic_across_calls(self):
        cfg = QueryExpansionConfig(max_expansions=3)
        q = "how does the sequencer drive the driver"
        first = expand_query(q, intent=QueryIntent.VERIFICATION, config=cfg)
        second = expand_query(q, intent=QueryIntent.VERIFICATION, config=cfg)
        assert first == second


class TestEmptyResults:
    def test_general_intent_yields_nothing(self):
        out = expand_query("what's the weather", intent=QueryIntent.GENERAL, config=DEFAULT_QUERY_EXPANSION)
        assert out == []

    def test_structural_query_yields_nothing(self):
        out = expand_query(
            "summarize the repository", intent=QueryIntent.ARCHITECTURE, config=DEFAULT_QUERY_EXPANSION
        )
        assert out == []

    def test_blank_query_yields_nothing(self):
        out = expand_query("", intent=QueryIntent.VERIFICATION, config=DEFAULT_QUERY_EXPANSION)
        assert out == []

    def test_query_covering_all_terms_yields_nothing(self):
        from semrag.retrieval.intent import intent_keywords

        q = " ".join(intent_keywords(QueryIntent.VERIFICATION))
        out = expand_query(q, intent=QueryIntent.VERIFICATION, config=DEFAULT_QUERY_EXPANSION)
        assert out == []

    def test_disabled_config_yields_nothing(self):
        out = expand_query(
            "why did the assertion fail",
            intent=QueryIntent.VERIFICATION,
            config=QueryExpansionConfig(enabled=False),
        )
        assert out == []


class TestIntentSpecificity:
    def test_different_intents_give_different_expansions(self):
        cfg = QueryExpansionConfig(max_expansions=3, max_expansion_length=200)
        debug_q = "why did the test fail with an error"
        arch_q = "what module drives the sda signal"
        debug_out = expand_query(debug_q, intent=QueryIntent.DEBUG, config=cfg)
        arch_out = expand_query(arch_q, intent=QueryIntent.ARCHITECTURE, config=cfg)
        assert debug_out != arch_out
        debug_joined = " ".join(debug_out)
        arch_joined = " ".join(arch_out)
        assert "uvm_error" in debug_joined
        assert "waveform" in debug_joined
        assert "datapath" in arch_joined

    def test_intent_detected_from_query_when_pipeline_runs(self):
        # The pipeline classifies the intent itself; verify classification
        # routes a debug-flavored query to DEBUG vocabulary.
        cfg = QueryExpansionConfig(max_expansions=3, max_expansion_length=200)
        q = "why did the uvm_error happen in the waveform"
        assert classify_intent(q) == QueryIntent.DEBUG
        out = expand_query(q, intent=classify_intent(q), config=cfg)
        assert "mismatch" in " ".join(out)


class TestConfig:
    def test_defaults_are_sane(self):
        cfg = DEFAULT_QUERY_EXPANSION
        assert cfg.enabled is True
        assert cfg.max_expansions == 3
        assert cfg.max_expansion_length == 80
        assert cfg.timeout_ms is None

    def test_builds_from_settings(self):
        cfg = build_query_expansion_config(_FakeSettings())
        assert cfg.enabled is True
        assert cfg.max_expansions == 4
        assert cfg.max_expansion_length == 90

    def test_config_validation(self):
        with pytest.raises(ValueError):
            QueryExpansionConfig(max_expansions=0)
        with pytest.raises(ValueError):
            QueryExpansionConfig(max_expansion_length=0)


# ── Pipeline integration (stubbed stores) ───────────────────────────────


class TestPipelineStage25:
    def test_expansion_results_fused_and_traced(self, monkeypatch):
        from semrag.retrieval import pipeline as pipeline_mod
        from semrag.retrieval.pipeline import HybridRetriever

        original = _result("orig", relative_path="rtl/a.sv")
        via_expansion = _result("expanded", relative_path="tb/tb.svh")

        class _StubVector:
            def __init__(self, session):
                pass

            def search(self, embedding, top_k, filters):
                return []

        class _StubLexical:
            def __init__(self, session):
                pass

            def search(self, query_text, top_k, filters):
                if query_text == "why did the assertion fail":
                    return [original]
                return [via_expansion]  # any expansion formulation matches it

        monkeypatch.setattr(pipeline_mod, "PgvectorStore", _StubVector)
        monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
        monkeypatch.setattr(
            pipeline_mod, "to_retrieval_results", lambda session, raw, signal_name: raw
        )

        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            query_expansion_config=QueryExpansionConfig(max_expansions=2),
        )
        results, trace = retriever.retrieve(
            "why did the assertion fail",
            [0.0] * 8,
            project_id="p",
            top_k=5,
            filters=VectorSearchFilter(project_id="p"),
            return_trace=True,
        )

        ids = {r.chunk_id for r in results}
        assert "orig" in ids
        assert "expanded" in ids

        by_id = {c.chunk_id: c for c in trace.chunks()}
        assert by_id["orig"].retrieved_by_original is True
        assert by_id["expanded"].retrieved_by_expansion is True
        assert by_id["expanded"].expansion_queries
        assert trace.expansions, "expansions recorded on the ranking trace"
        assert trace.expansion_enabled is True
        assert trace.expansion_latency_ms is not None

    def test_original_and_expansion_collide_deduplicated(self, monkeypatch):
        from semrag.retrieval import pipeline as pipeline_mod
        from semrag.retrieval.pipeline import HybridRetriever

        hit = _result("same", relative_path="rtl/a.sv")

        class _StubVector:
            def __init__(self, session):
                pass

            def search(self, embedding, top_k, filters):
                return [hit]

        class _StubLexical:
            def __init__(self, session):
                pass

            def search(self, query_text, top_k, filters):
                return [hit]  # both the original AND expansions return it

        monkeypatch.setattr(pipeline_mod, "PgvectorStore", _StubVector)
        monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
        monkeypatch.setattr(
            pipeline_mod, "to_retrieval_results", lambda session, raw, signal_name: raw
        )

        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            query_expansion_config=QueryExpansionConfig(max_expansions=2),
        )
        results, trace = retriever.retrieve(
            "why did the assertion fail",
            [0.0] * 8,
            project_id="p",
            top_k=5,
            filters=VectorSearchFilter(project_id="p"),
            return_trace=True,
        )
        # RRF dedups by chunk_id — one result, both provenances set.
        assert len([r for r in results if r.chunk_id == "same"]) == 1
        by_id = {c.chunk_id: c for c in trace.chunks()}
        assert by_id["same"].retrieved_by_original is True
        assert by_id["same"].retrieved_by_expansion is True

    def test_disabled_preserves_pre_expansion_behavior(self, monkeypatch):
        from semrag.retrieval import pipeline as pipeline_mod
        from semrag.retrieval.pipeline import HybridRetriever

        a = _result("a", relative_path="rtl/a.sv")
        b = _result("b", relative_path="rtl/b.sv")

        class _StubVector:
            def __init__(self, session):
                pass

            def search(self, embedding, top_k, filters):
                return [b]

        class _StubLexical:
            def __init__(self, session):
                pass

            def search(self, query_text, top_k, filters):
                return [a]

        monkeypatch.setattr(pipeline_mod, "PgvectorStore", _StubVector)
        monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
        monkeypatch.setattr(
            pipeline_mod, "to_retrieval_results", lambda session, raw, signal_name: raw
        )

        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            query_expansion_config=QueryExpansionConfig(enabled=False),
        )
        results, trace = retriever.retrieve(
            "why did the assertion fail",
            [0.0] * 8,
            project_id="p",
            top_k=5,
            filters=VectorSearchFilter(project_id="p"),
            return_trace=True,
        )
        assert [r.chunk_id for r in results] == ["b", "a"]
        assert trace.expansions == []
        assert trace.expansion_enabled is False
        assert all(not c.retrieved_by_expansion for c in trace.chunks())

    def test_no_usable_expansion_leaves_results_unchanged(self, monkeypatch):
        from semrag.retrieval import pipeline as pipeline_mod
        from semrag.retrieval.pipeline import HybridRetriever

        a = _result("a", relative_path="rtl/a.sv")

        class _StubVector:
            def __init__(self, session):
                pass

            def search(self, embedding, top_k, filters):
                return []

        class _StubLexical:
            def __init__(self, session):
                pass

            def search(self, query_text, top_k, filters):
                return [a]

        monkeypatch.setattr(pipeline_mod, "PgvectorStore", _StubVector)
        monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
        monkeypatch.setattr(
            pipeline_mod, "to_retrieval_results", lambda session, raw, signal_name: raw
        )

        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            query_expansion_config=DEFAULT_QUERY_EXPANSION,
        )
        # GENERAL-intent query -> expand_query returns [] -> identical.
        results, trace = retriever.retrieve(
            "give me the weather",
            [0.0] * 8,
            project_id="p",
            top_k=5,
            filters=VectorSearchFilter(project_id="p"),
            return_trace=True,
        )
        assert [r.chunk_id for r in results] == ["a"]
        assert trace.expansions == []

    def test_fail_open_on_expansion_error(self, monkeypatch):
        from semrag.retrieval import pipeline as pipeline_mod
        from semrag.retrieval.pipeline import HybridRetriever

        a = _result("a", relative_path="rtl/a.sv")

        class _StubVector:
            def __init__(self, session):
                pass

            def search(self, embedding, top_k, filters):
                return []

        class _StubLexical:
            def __init__(self, session):
                pass

            def search(self, query_text, top_k, filters):
                return [a]

        monkeypatch.setattr(pipeline_mod, "PgvectorStore", _StubVector)
        monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
        monkeypatch.setattr(
            pipeline_mod, "to_retrieval_results", lambda session, raw, signal_name: raw
        )

        def _boom(query_text, *, intent, config):
            raise RuntimeError("expansion exploded")

        monkeypatch.setattr("semrag.retrieval.pipeline.expand_query", _boom)

        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            query_expansion_config=DEFAULT_QUERY_EXPANSION,
        )
        results, trace = retriever.retrieve(
            "why did the assertion fail",
            [0.0] * 8,
            project_id="p",
            top_k=5,
            filters=VectorSearchFilter(project_id="p"),
            return_trace=True,
        )
        # Fail-open: retrieval still returns the original-query results.
        assert [r.chunk_id for r in results] == ["a"]
        assert trace.expansion_failed is True
        assert trace.expansions == []

    def test_pool_is_bounded(self, monkeypatch):
        from semrag.retrieval import pipeline as pipeline_mod
        from semrag.retrieval.pipeline import HybridRetriever

        class _StubVector:
            def __init__(self, session):
                pass

            def search(self, embedding, top_k, filters):
                return []

        class _StubLexical:
            def __init__(self, session):
                pass

            def search(self, query_text, top_k, filters):
                # Each formulation returns its own unique 20-hit list.
                base = len(searched_queries)
                searched_queries.append(query_text)
                return [
                    _result(f"{base}-{i}", relative_path=f"rtl/m{base}.sv")
                    for i in range(20)
                ]

        searched_queries: list[str] = []
        monkeypatch.setattr(pipeline_mod, "PgvectorStore", _StubVector)
        monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
        monkeypatch.setattr(
            pipeline_mod, "to_retrieval_results", lambda session, raw, signal_name: raw
        )

        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            enable_intent_boost=False,
            enable_doc_type_prior=False,
            query_expansion_config=QueryExpansionConfig(max_expansions=3),
        )
        results, trace = retriever.retrieve(
            "why did the assertion fail",
            [0.0] * 8,
            project_id="p",
            top_k=5,
            filters=VectorSearchFilter(project_id="p"),
            return_trace=True,
        )
        # 4 queries (original + 3 expansions) x 20 hits = 80 candidates,
        # but final top_k is 5.
        assert len(searched_queries) == 4
        assert len(results) == 5
