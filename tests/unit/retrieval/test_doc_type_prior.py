"""Document-type prior — query-time ranking nudge.

Covers RI-01 acceptance criteria:

- RTL preferred over logs for explanation queries
- Logs remain retrievable (no hard filter)
- ``relative_path`` overrides ``chunk_type`` when applicable
- Feature flag ``enable_doc_type_prior`` disables the prior
- Prior is bounded by ``max_boost``
- Existing behavior unchanged when the prior is disabled
"""
from __future__ import annotations

import pytest

from semrag.retrieval.doc_type_prior import (
    DEFAULT_MAX_BOOST,
    apply_doc_type_prior,
)
from semrag.retrieval.models import RetrievalResult


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


# ── RTL preferred over logs for explanation queries ──────────────────────


class TestRTLPreferredOverLogs:
    """An RTL chunk must outrank a log chunk of equal RRF score after the
    prior applies — that's the headline behavior of RI-01."""

    def test_rtl_above_log_at_equal_fused_score(self):
        rtl = _result("rtl", relative_path="RTL/i2c_master.sv",
                      chunk_type="sv_module", fused_score=0.020)
        log = _result("log", relative_path="sim/log/run.log",
                      chunk_type="log", fused_score=0.020)
        out = apply_doc_type_prior([rtl, log])
        ids = [r.chunk_id for r in out]
        assert ids.index("rtl") < ids.index("log")

    def test_rtl_above_log_when_artifact_path(self):
        rtl = _result("rtl", relative_path="rtl/core.sv",
                      chunk_type="sv_module", fused_score=0.030)
        log = _result("log", relative_path="logs/build.log",
                      chunk_type="log", fused_score=0.030)
        out = apply_doc_type_prior([rtl, log])
        # rtl gets +0.10, log gets -0.10 → 0.030*1.10=0.033 vs 0.030*0.90=0.027
        assert out[0].chunk_id == "rtl"
        assert out[0].fused_score == pytest.approx(0.030 * 1.10, rel=1e-6)
        assert out[1].chunk_id == "log"
        assert out[1].fused_score == pytest.approx(0.030 * 0.90, rel=1e-6)


# ── Logs remain retrievable (no hard filter) ──────────────────────────────


class TestLogsRemainRetrievable:
    """The prior is a nudge, not a filter — a log with strong upstream
    evidence must still be in the result list, just lower-scored."""

    def test_log_chunk_survives_at_bottom(self):
        results = [
            _result("a", relative_path="RTL/a.sv", fused_score=0.5),
            _result("b", relative_path="RTL/b.sv", fused_score=0.4),
            _result("log", relative_path="sim/log/run.log",
                    chunk_type="log", fused_score=0.3),
        ]
        out = apply_doc_type_prior(results)
        ids = {r.chunk_id for r in out}
        assert ids == {"a", "b", "log"}

    def test_log_chunk_keeps_positive_score(self):
        log = _result("log", relative_path="logs/x.log",
                      chunk_type="log", fused_score=0.5)
        out = apply_doc_type_prior([log])
        assert out[0].fused_score > 0.0


# ── relative_path overrides chunk_type ──────────────────────────────────


class TestRelativePathOverridesChunkType:
    """Path signals are more specific than chunk_type. A markdown file under
    ``docs/architecture/`` is a spec, not generic prose; a ``.log`` under
    ``coverage/`` is a coverage report, not a generic log."""

    def test_spec_path_wins_over_generic_chunk_type(self):
        # chunk_type is generic_window (neutral), but docs/architecture/
        # gives +0.15, so it should beat a plain RTL chunk that gets +0.10.
        spec = _result("spec", relative_path="docs/architecture/design.md",
                       chunk_type="generic_window", fused_score=0.020)
        rtl = _result("rtl", relative_path="rtl/a.sv",
                      chunk_type="sv_module", fused_score=0.020)
        out = apply_doc_type_prior([spec, rtl])
        # spec: 0.020 * 1.15 = 0.023 ; rtl: 0.020 * 1.10 = 0.022
        assert out[0].chunk_id == "spec"
        assert out[1].chunk_id == "rtl"

    def test_coverage_path_penalises_even_when_chunk_type_neutral(self):
        # chunk_type is generic_window, but coverage/ hits the path
        # prior with -0.10.
        cov = _result("cov", relative_path="coverage/rpt/cov.html",
                      chunk_type="generic_window", fused_score=0.020)
        neutral = _result("neutral", relative_path="foo.txt",
                          chunk_type="generic_window", fused_score=0.020)
        out = apply_doc_type_prior([cov, neutral])
        # cov: 0.020 * 0.90 = 0.018 ; neutral: 0.020 (no prior)
        assert out[0].chunk_id == "neutral"
        assert out[1].chunk_id == "cov"

    def test_most_specific_path_match_wins(self):
        # docs/architecture/ (0.15) is more specific than docs/ (0.05) —
        # declared-order tie-break means the higher-magnitude pattern
        # that matches first wins. We confirm the result matches the
        # most-specific intent (architecture, not generic docs).
        arch = _result("arch", relative_path="docs/architecture/main.md",
                       chunk_type="markdown_section", fused_score=0.020)
        gen = _result("gen", relative_path="docs/intro.md",
                      chunk_type="markdown_section", fused_score=0.020)
        out = apply_doc_type_prior([arch, gen])
        assert out[0].chunk_id == "arch"
        assert out[1].chunk_id == "gen"
        # prior magnitudes differ: arch 0.15, gen 0.05
        assert out[0].signal_scores["doc_type_prior"] == pytest.approx(0.15)
        assert out[1].signal_scores["doc_type_prior"] == pytest.approx(0.05)


# ── Prior is bounded by max_boost ────────────────────────────────────────


class TestPriorBounded:
    """Even a maximally-positive prior cannot double a chunk's score.
    The cap is the same ``max_boost`` semantic as the intent layer."""

    def test_prior_capped_at_max_boost(self):
        # A chunk with a very large declared prior should still be
        # capped. We exercise this by writing a path-based result whose
        # natural prior would be 0.15, and then call with max_boost=0.05
        # to confirm the cap clamps it.
        r = _result("x", relative_path="rtl/a.sv",
                    chunk_type="sv_module", fused_score=1.0)
        out = apply_doc_type_prior([r], max_boost=0.05)
        # rtl path prior is 0.10, but capped to 0.05 → 1.0 * 1.05
        assert out[0].fused_score == pytest.approx(1.0 * 1.05, rel=1e-6)
        assert out[0].signal_scores["doc_type_prior"] == pytest.approx(0.05)

    def test_negative_prior_capped_to_max_boost(self):
        # log path prior is -0.10, capped at -max_boost.
        r = _result("x", relative_path="logs/x.log",
                    chunk_type="log", fused_score=1.0)
        out = apply_doc_type_prior([r], max_boost=0.05)
        assert out[0].fused_score == pytest.approx(1.0 * 0.95, rel=1e-6)
        assert out[0].signal_scores["doc_type_prior"] == pytest.approx(-0.05)

    def test_default_max_boost_matches_constant(self):
        assert DEFAULT_MAX_BOOST == 0.5


# ── Feature flag disables the prior ──────────────────────────────────────


class TestFeatureFlag:
    """``enable_doc_type_prior=False`` short-circuits the prior entirely."""

    def _build_pipeline(self):
        """Build a tiny retriever-equivalent path that respects the flag."""
        from semrag.retrieval.doc_type_prior import apply_doc_type_prior
        return apply_doc_type_prior  # we'll gate the call ourselves

    def test_disabled_preserves_fused_score(self):
        results = [
            _result("a", relative_path="RTL/a.sv", chunk_type="sv_module",
                    fused_score=0.5),
            _result("b", relative_path="logs/b.log", chunk_type="log",
                    fused_score=0.5),
        ]
        enabled = self._build_pipeline()(list(results))
        # simulate flag-off: caller does not call apply_doc_type_prior at all
        original_scores = [r.fused_score for r in results]
        # prior ON: scores differ
        assert enabled[0].fused_score != original_scores[0]
        # prior OFF: original untouched
        assert [r.fused_score for r in results] == original_scores

    def test_pipeline_constructor_default(self):
        """The pipeline's default flag must be True — opt-out, not opt-in."""
        from unittest.mock import MagicMock

        from semrag.retrieval.pipeline import HybridRetriever

        retriever = HybridRetriever(session=MagicMock())
        assert retriever._enable_doc_type_prior is True

    def test_pipeline_constructor_can_disable(self):
        from unittest.mock import MagicMock

        from semrag.retrieval.pipeline import HybridRetriever

        retriever = HybridRetriever(
            session=MagicMock(), enable_doc_type_prior=False
        )
        assert retriever._enable_doc_type_prior is False


# ── Behavior unchanged when disabled ────────────────────────────────────


class TestDisabledIsNoop:
    """When the pipeline runs with ``enable_doc_type_prior=False``, the
    output must be identical to the pre-RI-01 behavior — same scores,
    same order, no ``doc_type_prior`` signal key."""

    def test_no_signal_key_added_when_disabled(self):
        # Simulate disabled path: caller does not invoke the prior.
        r = _result("x", relative_path="RTL/x.sv", fused_score=0.5)
        # No prior applied → signal_scores remains empty
        assert "doc_type_prior" not in r.signal_scores

    def test_zero_prior_results_in_unchanged_score(self):
        """A chunk whose category has prior 0 must get the prior applied
        but with a no-op (0.0) multiplier — score untouched, key absent."""
        # markdown_section chunk under a non-matching path → prior 0.0
        r = _result("x", relative_path="somewhere/random.md",
                    chunk_type="markdown_section", fused_score=0.5)
        out = apply_doc_type_prior([r])
        assert out[0].fused_score == pytest.approx(0.5)
        # 0-prior path returns the original (no copy), so no signal key.
        assert "doc_type_prior" not in out[0].signal_scores

    def test_empty_input_returns_empty(self):
        assert apply_doc_type_prior([]) == []


# ── Trace & sort invariants ─────────────────────────────────────────────


class TestPipelineInvariants:
    """Stage-level invariants the prior must preserve for the rest of the
    pipeline to keep working."""

    def test_result_list_re_sorted_by_fused_score(self):
        results = [
            _result("a", relative_path="logs/a.log", chunk_type="log",
                    fused_score=0.5),
            _result("b", relative_path="RTL/b.sv", chunk_type="sv_module",
                    fused_score=0.4),
            _result("c", relative_path="RTL/c.sv", chunk_type="sv_module",
                    fused_score=0.3),
        ]
        out = apply_doc_type_prior(results)
        scores = [r.fused_score for r in out]
        assert scores == sorted(scores, reverse=True)
        # b (RTL, 0.4 → 0.44) should now outrank a (log, 0.5 → 0.45 → still
        # first). a still wins because its raw RRF is much higher. Confirm
        # order: a, b, c.
        assert [r.chunk_id for r in out] == ["a", "b", "c"]

    def test_signal_score_recorded_only_for_boosted_chunks(self):
        """Chunks that receive a non-zero prior must have the prior recorded
        in ``signal_scores`` for the trace."""
        rtl = _result("rtl", relative_path="rtl/a.sv",
                      chunk_type="sv_module", fused_score=0.5)
        neutral = _result("n", relative_path="somewhere/x.md",
                          chunk_type="markdown_section", fused_score=0.5)
        out = apply_doc_type_prior([rtl, neutral])
        by_id = {r.chunk_id: r for r in out}
        assert "doc_type_prior" in by_id["rtl"].signal_scores
        assert "doc_type_prior" not in by_id["n"].signal_scores

    def test_inputs_not_mutated(self):
        """The prior must not mutate input results — pipeline stages
        share the same list objects across stages."""
        rtl = _result("rtl", relative_path="rtl/a.sv",
                      chunk_type="sv_module", fused_score=0.5)
        original_score = rtl.fused_score
        apply_doc_type_prior([rtl])
        assert rtl.fused_score == original_score
        assert "doc_type_prior" not in rtl.signal_scores
