"""
Tests for the extracted source-ranking module.

The noisy-source penalty and the document-type prior stage were
pure-refactored out of ``retrieval/pipeline.py`` into
``retrieval/ranking/source.py``. These tests pin the extracted function
to the exact pre-refactor inline logic and verify its composition with
the doc-type prior rules in ``doc_type_prior.py``.
"""
from __future__ import annotations

import pytest

from semrag.retrieval.doc_type_prior import apply_doc_type_prior
from semrag.retrieval.models import RetrievalResult
from semrag.retrieval.ranking.source import (
    _is_noisy_source,
    apply_source_ranking,
)


def _result(
    chunk_id: str,
    *,
    relative_path: str = "RTL/x.sv",
    chunk_type: str = "sv_module",
    fused_score: float = 0.5,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        project_id="proj1",
        relative_path=relative_path,
        chunk_type=chunk_type,
        text="module x; endmodule",
        fused_score=fused_score,
        signal_scores={},
    )


# ── Regression: identical to the pre-refactor inline logic ───────────────


class TestMatchesPreRefactorInlineLogic:
    """The noisy-source penalty stage used to be inlined in the pipeline.
    ``apply_source_ranking`` must reproduce it exactly: same scores, same
    order, same in-place mutation."""

    def _pre_refactor_inline(self, results, terms, penalty):
        # The exact code that lived in pipeline.py stage 3.5.
        for r in results:
            if any(term in r.relative_path for term in terms):
                r.fused_score = r.fused_score * (1.0 - penalty)
        results.sort(key=lambda r: r.fused_score, reverse=True)
        return results

    def test_noisy_penalty_identical_to_inline(self):
        results = [
            _result("sv", relative_path="RTL/top.sv", fused_score=0.020),
            _result("log1", relative_path="sim/log/run.log", fused_score=0.018),
            _result("log2", relative_path="wave/log/run.log", fused_score=0.016),
        ]
        terms = ("/log/",)
        penalty = 0.5

        expected = self._pre_refactor_inline(
            [r.model_copy(deep=True) for r in results], terms, penalty
        )
        got = apply_source_ranking(
            [r.model_copy(deep=True) for r in results],
            noisy_source_path_terms=terms,
            noisy_source_penalty=penalty,
            enable_doc_type_prior=False,
        )

        assert [r.chunk_id for r in got] == [r.chunk_id for r in expected]
        assert [r.fused_score for r in got] == pytest.approx(
            [r.fused_score for r in expected]
        )

    def test_noisy_penalty_penalizes_matching_sources_only(self):
        got = apply_source_ranking(
            [
                _result("sv", relative_path="RTL/top.sv", fused_score=0.020),
                _result("log", relative_path="sim/log/run.log", fused_score=0.018),
            ],
            noisy_source_path_terms=("/log/",),
            noisy_source_penalty=0.5,
            enable_doc_type_prior=False,
        )
        by_id = {r.chunk_id: r for r in got}
        assert by_id["sv"].fused_score == pytest.approx(0.020)  # untouched
        assert by_id["log"].fused_score == pytest.approx(0.018 * 0.5)

    def test_noisy_penalty_re_sorts_descending(self):
        got = apply_source_ranking(
            [
                _result("a", relative_path="RTL/a.sv", fused_score=0.020),
                _result("log", relative_path="sim/log/run.log", fused_score=0.050),
            ],
            noisy_source_path_terms=("/log/",),
            noisy_source_penalty=0.5,
            enable_doc_type_prior=False,
        )
        # log 0.050 -> 0.025, below a 0.020? no: 0.025 > 0.020 still first.
        assert [r.chunk_id for r in got] == ["log", "a"]
        # A heavier penalty drops the log below the source chunk.
        got = apply_source_ranking(
            [
                _result("a", relative_path="RTL/a.sv", fused_score=0.020),
                _result("log", relative_path="sim/log/run.log", fused_score=0.030),
            ],
            noisy_source_path_terms=("/log/",),
            noisy_source_penalty=0.5,
            enable_doc_type_prior=False,
        )
        assert [r.chunk_id for r in got] == ["a", "log"]

    def test_zero_penalty_is_noop(self):
        results = [_result("a", fused_score=0.5), _result("b", fused_score=0.4)]
        got = apply_source_ranking(
            list(results),
            noisy_source_path_terms=("/log/",),
            noisy_source_penalty=0.0,
            enable_doc_type_prior=False,
        )
        assert [r.fused_score for r in got] == pytest.approx([r.fused_score for r in results])
        assert [r.chunk_id for r in got] == [r.chunk_id for r in results]


# ── Document-type prior orchestration ────────────────────────────────────


class TestDocTypePriorOrchestration:
    """With defaults, ``apply_source_ranking`` applies the doc-type prior
    by delegating to ``doc_type_prior.apply_doc_type_prior`` — the rules
    and weights stay in ``doc_type_prior.py``."""

    def test_default_equals_apply_doc_type_prior(self):
        results = [
            _result("rtl", relative_path="rtl/a.sv", chunk_type="sv_module", fused_score=0.5),
            _result("log", relative_path="logs/x.log", chunk_type="log", fused_score=0.5),
            _result("n", relative_path="somewhere/x.md", chunk_type="markdown_section", fused_score=0.3),
        ]
        expected = apply_doc_type_prior([r.model_copy(deep=True) for r in results])
        got = apply_source_ranking([r.model_copy(deep=True) for r in results])

        assert [r.chunk_id for r in got] == [r.chunk_id for r in expected]
        assert [r.fused_score for r in got] == pytest.approx([r.fused_score for r in expected])
        for got_r, exp_r in zip(got, expected, strict=True):
            assert got_r.signal_scores == exp_r.signal_scores

    def test_apply_doc_type_prior_false_skips_prior(self):
        results = [_result("rtl", relative_path="rtl/a.sv", fused_score=0.5)]
        got = apply_source_ranking(
            list(results),
            noisy_source_penalty=0.0,
            enable_doc_type_prior=False,
        )
        assert got[0].fused_score == pytest.approx(0.5)
        assert "doc_type_prior" not in got[0].signal_scores


# ── Helper ───────────────────────────────────────────────────────────────


class TestIsNoisySource:
    def test_matches_term_substring_case_sensitive(self):
        assert _is_noisy_source("sim/log/run.log", ("/log/",)) is True
        assert _is_noisy_source("RTL/top.sv", ("/log/",)) is False

    def test_empty_terms_match_nothing(self):
        assert _is_noisy_source("sim/log/run.log", ()) is False


# ── Composition ──────────────────────────────────────────────────────────


class TestComposition:
    def test_noisy_penalty_then_doc_type_prior(self):
        """When both adjustments are enabled in one call, the noisy-source
        penalty runs first, then the doc-type prior — matching the
        pipeline's stage order."""
        results = [
            _result("rtl", relative_path="RTL/top.sv", chunk_type="sv_module", fused_score=0.030),
            _result("log", relative_path="sim/log/run.log", chunk_type="log", fused_score=0.040),
        ]
        got = apply_source_ranking(
            [r.model_copy(deep=True) for r in results],
            noisy_source_path_terms=("/log/",),
            noisy_source_penalty=0.5,
        )
        by_id = {r.chunk_id: r for r in got}
        # log: 0.040 * 0.5 = 0.020, then doc-type prior (log path -> -0.10):
        # 0.020 * 0.9 = 0.018. rtl: 0.030, then +0.10 -> 0.033.
        assert by_id["rtl"].fused_score == pytest.approx(0.033)
        assert by_id["log"].fused_score == pytest.approx(0.018)
        assert [r.chunk_id for r in got] == ["rtl", "log"]