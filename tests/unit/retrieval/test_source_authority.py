"""Source Authority (RI-02) — document-aware ranking adjustment.

Covers the RI-02 acceptance criteria:

- Authority ordering: design / verification material outranks verbose
  runtime artifacts at equal fused score.
- Configuration loading: DEFAULT_SOURCE_AUTHORITY weights, ``with_max_boost``
  and ``with_overrides`` merging, validation of categories referenced by
  rules.
- Explainability: the authority is recorded under both the canonical
  ``source_authority`` and legacy ``doc_type_prior`` signal keys, and is
  additive to ``ChunkTrace``.
- Logs are never filtered: a log chunk with strong upstream evidence
  stays in the result list, just lower-scored.
- Backward compatibility: ``apply_doc_type_prior`` and
  ``apply_source_ranking`` keep their public signatures and behavior.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from semrag.retrieval.doc_type_prior import DEFAULT_MAX_BOOST, apply_doc_type_prior
from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.ranking.source import apply_source_ranking
from semrag.retrieval.source_authority import (
    DEFAULT_SOURCE_AUTHORITY,
    SourceAuthorityConfig,
    SourceAuthorityRule,
    apply_source_authority,
    classify_document,
)


def _result(
    chunk_id: str,
    *,
    relative_path: str = "RTL/x.sv",
    chunk_type: str = "sv_module",
    fused_score: float = 0.5,
    text: str = "module x; endmodule",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="proj1",
        relative_path=relative_path,
        chunk_type=chunk_type,
        text=text,
        fused_score=fused_score,
        signal_scores={},
    )


# ── Authority ordering ───────────────────────────────────────────────────


class TestAuthorityOrdering:
    """Design / verification material must outrank verbose runtime
    artifacts at equal fused score — the headline RI-02 behavior."""

    def test_rtl_above_log_at_equal_fused_score(self):
        rtl = _result("rtl", relative_path="RTL/i2c_master.sv", fused_score=0.020)
        log = _result("log", relative_path="sim/log/run.log",
                      chunk_type="log", fused_score=0.020)
        out = apply_source_authority([rtl, log])
        assert out[0].chunk_id == "rtl"

    def test_spec_above_rtl(self):
        spec = _result("spec", relative_path="docs/architecture/design.md",
                       chunk_type="markdown_section", fused_score=0.020)
        rtl = _result("rtl", relative_path="rtl/a.sv", fused_score=0.020)
        out = apply_source_authority([spec, rtl])
        assert out[0].chunk_id == "spec"

    def test_uvm_above_generic_prose(self):
        uvm = _result("uvm", relative_path="tb/i2c_master_agent/agent.svh",
                      chunk_type="generic_window", fused_score=0.020)
        prose = _result("prose", relative_path="somewhere/notes.md",
                        chunk_type="markdown_section", fused_score=0.020)
        out = apply_source_authority([prose, uvm])
        assert out[0].chunk_id == "uvm"

    def test_log_penalized_but_still_present(self):
        rtl = _result("rtl", relative_path="rtl/a.sv", fused_score=0.020)
        log = _result("log", relative_path="sim/log/run.log",
                      chunk_type="log", fused_score=0.020)
        out = apply_source_authority([rtl, log])
        assert {r.chunk_id for r in out} == {"rtl", "log"}


# ── Document classification ──────────────────────────────────────────────


class TestClassifyDocument:
    def test_path_wins_over_chunk_type(self):
        # A markdown chunk under docs/architecture/ is an architecture
        # spec, not generic prose.
        assert classify_document("markdown_section", "docs/architecture/main.md") == "architecture_documentation"

    def test_generic_window_under_log_path_is_a_log(self):
        assert classify_document("generic_window", "sim/log/run.log") == "simulation_log"

    def test_chunk_type_fallback(self):
        assert classify_document("sv_module", "src/unexpected/x.sv") == "rtl"
        assert classify_document("markdown_preamble", "README.md") == "markdown"
        assert classify_document("generic_window", "README.md") == "other"

    def test_sv_assertion_is_verification(self):
        assert classify_document("sv_assertion", "rtl/x.sv") == "rtl"
        assert classify_document("sv_assertion", "tb/x.svh") == "uvm"

    def test_unknown_chunk_type_falls_back_to_other(self):
        assert classify_document("weird_new_type", "weird.txt") == "other"


# ── Configuration loading ────────────────────────────────────────────────


class TestConfigLoading:
    def test_default_authorities_present(self):
        for category in ("rtl", "specification", "simulation_log", "uvm", "other"):
            assert category in DEFAULT_SOURCE_AUTHORITY.authorities

    def test_default_weights_match_ri01(self):
        cfg = DEFAULT_SOURCE_AUTHORITY
        assert cfg.authority("rtl") == pytest.approx(0.10)
        assert cfg.authority("simulation_log") == pytest.approx(-0.10)
        assert cfg.authority("specification") == pytest.approx(0.15)
        assert cfg.authority("other") == pytest.approx(0.0)

    def test_default_max_boost_matches_constant(self):
        assert DEFAULT_SOURCE_AUTHORITY.max_boost == pytest.approx(DEFAULT_MAX_BOOST)
        assert DEFAULT_MAX_BOOST == 0.5

    def test_with_max_boost_preserves_weights(self):
        cfg = DEFAULT_SOURCE_AUTHORITY.with_max_boost(0.05)
        assert cfg.max_boost == pytest.approx(0.05)
        assert cfg.authority("rtl") == pytest.approx(0.05)  # clamped
        assert cfg.authority("other") == pytest.approx(0.0)

    def test_with_overrides_merges_over_defaults(self):
        cfg = DEFAULT_SOURCE_AUTHORITY.with_overrides({"uvm": 0.12})
        assert cfg.authority("uvm") == pytest.approx(0.12)
        assert cfg.authority("rtl") == pytest.approx(0.10)  # untouched

    def test_with_overrides_can_change_max_boost(self):
        cfg = DEFAULT_SOURCE_AUTHORITY.with_overrides({}, max_boost=0.05)
        assert cfg.max_boost == pytest.approx(0.05)

    def test_config_rejects_rule_without_authority(self):
        with pytest.raises(ValueError):
            SourceAuthorityConfig(
                authorities={"other": 0.0},
                path_rules=[SourceAuthorityRule(pattern="wave/", category="waveform")],
            )

    def test_config_round_trips_custom_categories(self):
        """Extensibility: a brand-new document type is config-only."""
        cfg = SourceAuthorityConfig(
            authorities={"waveform": 0.05, "other": 0.0},
            path_rules=[SourceAuthorityRule(pattern="wave/", category="waveform")],
        )
        assert classify_document("generic_window", "wave/sig.vcd", config=cfg) == "waveform"
        out = apply_source_authority(
            [_result("w", relative_path="wave/sig.vcd", fused_score=0.020)],
            config=cfg,
        )
        assert out[0].fused_score == pytest.approx(0.020 * 1.05)


# ── Explainability ───────────────────────────────────────────────────────


class TestExplainability:
    def test_authority_recorded_under_both_signal_keys(self):
        rtl = _result("rtl", relative_path="rtl/a.sv", fused_score=0.5)
        out = apply_source_authority([rtl])
        assert out[0].signal_scores["source_authority"] == pytest.approx(0.10)
        assert out[0].signal_scores["doc_type_prior"] == pytest.approx(0.10)

    def test_zero_authority_chunk_gets_no_signal_keys(self):
        neutral = _result("n", relative_path="somewhere/x.md",
                          chunk_type="markdown_section", fused_score=0.5)
        out = apply_source_authority([neutral])
        assert out[0].fused_score == pytest.approx(0.5)
        assert "source_authority" not in out[0].signal_scores
        assert "doc_type_prior" not in out[0].signal_scores

    def test_chunk_trace_carries_source_authority(self):
        from semrag.retrieval.pipeline import ChunkTrace

        ct = ChunkTrace(chunk_id="x", doc_type_prior=0.1, source_authority=0.1)
        assert ct.to_dict()["source_authority"] == 0.1

    def test_pipeline_constructor_exposes_config(self):
        from unittest.mock import MagicMock

        from semrag.retrieval.pipeline import HybridRetriever

        retriever = HybridRetriever(
            session=MagicMock(), source_authority_config=DEFAULT_SOURCE_AUTHORITY
        )
        assert retriever._source_authority_config is DEFAULT_SOURCE_AUTHORITY


# ── Logs are never filtered ──────────────────────────────────────────────


class TestLogsNotFiltered:
    def test_log_with_strong_upstream_evidence_stays_on_top(self):
        log = _result("log", relative_path="sim/log/run.log",
                      chunk_type="log", fused_score=0.5)
        rtl = _result("rtl", relative_path="rtl/a.sv", fused_score=0.3)
        out = apply_source_authority([rtl, log])
        assert out[0].chunk_id == "log"  # 0.5*0.9 = 0.45 > 0.3*1.1 = 0.33

    def test_log_survives_in_result_set(self):
        results = [
            _result("a", relative_path="rtl/a.sv", fused_score=0.5),
            _result("log", relative_path="sim/log/run.log",
                    chunk_type="log", fused_score=0.1),
        ]
        out = apply_source_authority(results)
        assert "log" in {r.chunk_id for r in out}


# ── Backward compatibility ───────────────────────────────────────────────


class TestPipelineRecordsAuthority:
    """End-to-end through the real pipeline stage 5.5 (stubbed vector /
    lexical stores): the authority adjustment must land in both the
    canonical ``source_authority`` and legacy ``doc_type_prior`` trace
    columns, with no document type filtered out."""

    def test_stage55_records_source_authority_in_trace(self, monkeypatch):
        from semrag.retrieval import pipeline as pipeline_mod
        from semrag.retrieval.pipeline import HybridRetriever
        from semrag.storage.vector.models import VectorSearchFilter

        class _StubVector:
            def __init__(self, session):
                pass

            def search(self, embedding, top_k, filters):
                return [
                    _result("rtl", relative_path="rtl/a.sv", fused_score=0.020),
                    _result("log", relative_path="sim/log/run.log",
                            chunk_type="log", fused_score=0.020),
                ]

        class _StubLexical:
            def __init__(self, session):
                pass

            def search(self, query_text, top_k, filters):
                return []

        monkeypatch.setattr(pipeline_mod, "PgvectorStore", _StubVector)
        monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
        monkeypatch.setattr(
            pipeline_mod, "to_retrieval_results",
            lambda session, raw, signal_name: raw,
        )

        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            reranker=None,
        )
        results, trace = retriever.retrieve(
            "explain the i2c master start condition",
            [0.0] * 8,
            project_id="p",
            top_k=5,
            filters=VectorSearchFilter(project_id="p"),
            return_trace=True,
        )

        by_id = {c.chunk_id: c for c in trace.chunks()}
        # rtl got +0.10, log got -0.10; both signals recorded; nothing filtered.
        assert by_id["rtl"].source_authority == pytest.approx(0.10)
        assert by_id["rtl"].doc_type_prior == pytest.approx(0.10)
        assert by_id["log"].source_authority == pytest.approx(-0.10)
        assert by_id["log"].doc_type_prior == pytest.approx(-0.10)
        assert {r.chunk_id for r in results} == {"rtl", "log"}
        assert results[0].chunk_id == "rtl"

    def test_stage55_uses_configured_authority_config(self, monkeypatch):
        from semrag.retrieval import pipeline as pipeline_mod
        from semrag.retrieval.pipeline import HybridRetriever
        from semrag.storage.vector.models import VectorSearchFilter

        class _StubVector:
            def __init__(self, session):
                pass

            def search(self, embedding, top_k, filters):
                return [_result("rtl", relative_path="rtl/a.sv", fused_score=0.020)]

        class _StubLexical:
            def __init__(self, session):
                pass

            def search(self, query_text, top_k, filters):
                return []

        monkeypatch.setattr(pipeline_mod, "PgvectorStore", _StubVector)
        monkeypatch.setattr(pipeline_mod, "PostgresLexicalSearch", _StubLexical)
        monkeypatch.setattr(
            pipeline_mod, "to_retrieval_results",
            lambda session, raw, signal_name: raw,
        )

        tuned = DEFAULT_SOURCE_AUTHORITY.with_overrides({"rtl": 0.20})
        retriever = HybridRetriever(
            session=MagicMock(),
            enable_graph_expansion=False,
            reranker=None,
            source_authority_config=tuned,
        )
        results, trace = retriever.retrieve(
            "explain the i2c master",
            [0.0] * 8,
            project_id="p",
            top_k=5,
            filters=VectorSearchFilter(project_id="p"),
            return_trace=True,
        )
        by_id = {c.chunk_id: c for c in trace.chunks()}
        assert by_id["rtl"].source_authority == pytest.approx(0.20)


class TestBackwardCompatibility:
    def test_apply_doc_type_prior_matches_source_authority_default(self):
        results = [
            _result("rtl", relative_path="rtl/a.sv", fused_score=0.5),
            _result("log", relative_path="logs/x.log", chunk_type="log", fused_score=0.5),
        ]
        legacy = apply_doc_type_prior([r.model_copy(deep=True) for r in results])
        direct = apply_source_authority([r.model_copy(deep=True) for r in results])
        assert [r.chunk_id for r in legacy] == [r.chunk_id for r in direct]
        assert legacy[0].signal_scores == direct[0].signal_scores

    def test_apply_doc_type_prior_still_writes_legacy_key(self):
        rtl = _result("rtl", relative_path="rtl/a.sv", fused_score=0.5)
        out = apply_doc_type_prior([rtl])
        assert out[0].signal_scores["doc_type_prior"] == pytest.approx(0.10)

    def test_apply_doc_type_prior_max_boost_cap_preserved(self):
        rtl = _result("rtl", relative_path="rtl/a.sv", fused_score=1.0)
        out = apply_doc_type_prior([rtl], max_boost=0.05)
        assert out[0].fused_score == pytest.approx(1.05)
        assert out[0].signal_scores["doc_type_prior"] == pytest.approx(0.05)

    def test_apply_source_ranking_signature_kept(self):
        """Existing callers that pass only the original keyword args still
        get the document-type prior behavior."""
        rtl = _result("rtl", relative_path="rtl/a.sv", fused_score=0.5)
        out = apply_source_ranking([rtl])
        assert out[0].fused_score == pytest.approx(0.55)
        assert out[0].signal_scores["source_authority"] == pytest.approx(0.10)
        assert out[0].signal_scores["doc_type_prior"] == pytest.approx(0.10)

    def test_apply_source_ranking_accepts_custom_config(self):
        cfg = DEFAULT_SOURCE_AUTHORITY.with_overrides({"rtl": 0.20})
        rtl = _result("rtl", relative_path="rtl/a.sv", fused_score=0.5)
        out = apply_source_ranking([rtl], source_authority_config=cfg)
        assert out[0].fused_score == pytest.approx(0.5 * 1.20)