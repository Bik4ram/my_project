"""
Source Authority — document-aware ranking adjustment (RI-02).

Replaces the flat document-type prior tables of ``doc_type_prior.py``
(which bundled classification rules and weights into two private
dictionaries) with a single, configurable **Source Authority** model:

1. **Document classification** — a chunk's source document is assigned
   a semantic category (``rtl``, ``specification``, ``simulation_log``,
   ``coverage_report``, ``uvm``, ...) from its ``chunk_type`` and
   ``relative_path``. Path signals beat ``chunk_type`` because a markdown
   file under ``docs/architecture/`` is an architecture spec, not generic
   prose, and a ``generic_window`` chunk under ``sim/log/`` is a log.

2. **Authority assignment** — every category maps to an authority value
   (a small multiplicative weight) in ``SourceAuthorityConfig``. This is
   the *one place* an operator tunes: add a category, change a weight, or
   bound the whole adjustment with ``max_boost`` — no ranking code
   changes.

3. **Score adjustment** — ``apply_source_authority`` applies the
   multiplicative nudge ``fused_score * (1 + authority)`` (capped at
   ``±max_boost``, mirroring the intent layer) and re-sorts by
   ``fused_score`` desc, the pipeline's ordering invariant.

Why a separate module from ``ranking/source.py``: the ranking module
orchestrates the *noisy-source penalty* and the authority adjustment as
pipeline stages; this module owns the authority rules themselves.

Extensibility (config only)
---------------------------
Adding a new document type (waveforms, datasheets, verification plans,
timing reports) never touches ranking code — it is a
``SourceAuthorityConfig`` change: add a ``SourceAuthorityRule`` for the
path / ``chunk_type`` that identifies it, and an entry in
``authorities`` for its weight. ``classify_document`` and
``apply_source_authority`` are generic over the config.

Backward compatibility
----------------------
- ``apply_source_authority`` writes the authority under BOTH the
  canonical ``signal_scores["source_authority"]`` key and the legacy
  ``signal_scores["doc_type_prior"]`` key, so the pipeline trace, the
  RB-02 failure analyzer, and RB-03 analytics keep working unchanged.
- ``DEFAULT_SOURCE_AUTHORITY`` reproduces the exact weights of the
  RI-01 document-type prior (path table + chunk-type table translated
  verbatim), so default behavior is unchanged.
- ``doc_type_prior.py`` is kept as a thin delegating shim so existing
  imports of ``apply_doc_type_prior`` / ``DEFAULT_MAX_BOOST`` work.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from semrag.retrieval.models import RetrievalResult


class SourceAuthorityRule(BaseModel):
    """A single classification rule. For a path rule, ``pattern`` is a
    case-insensitive substring of ``relative_path``; for a chunk-type
    rule, a prefix of ``chunk_type`` (longest prefix wins). Matched
    chunks are assigned ``category``."""

    pattern: str
    category: str


class SourceAuthorityConfig(BaseModel):
    """The single configurable place for source-ranking authorities.

    - ``authorities``: category name -> authority value (the multiplier
      applied as ``1 + value``, e.g. ``0.10`` boosts 10% and ``-0.10``
      penalizes 10%).
    - ``path_rules``: ordered path-substring classification rules, most
      specific first (first match wins).
    - ``chunk_type_rules``: ``chunk_type`` prefix classification rules
      (longest prefix wins).
    - ``max_boost``: hard cap on the magnitude of any authority value,
      matching the intent layer's cap semantic.

    Every category referenced by a rule must have an explicit authority
    entry — a category without a weight is a configuration error, not a
    silent 0.
    """

    authorities: dict[str, float]
    path_rules: list[SourceAuthorityRule] = Field(default_factory=list)
    chunk_type_rules: list[SourceAuthorityRule] = Field(default_factory=list)
    max_boost: float = 0.5

    @model_validator(mode="after")
    def _categories_have_authority(self) -> SourceAuthorityConfig:
        referenced = {r.category for r in (*self.path_rules, *self.chunk_type_rules)}
        missing = sorted(referenced - set(self.authorities))
        if missing:
            raise ValueError(
                "SourceAuthorityConfig authorities missing weights for "
                f"categories referenced by rules: {', '.join(missing)}"
            )
        return self

    def authority(self, category: str) -> float:
        """Resolve the (capped) authority value for a category."""
        capped = max(0.0, float(self.max_boost))
        prior = float(self.authorities.get(category, 0.0))
        return max(-capped, min(capped, prior))

    def with_max_boost(self, max_boost: float) -> SourceAuthorityConfig:
        """Copy with a different magnitude cap (preserves the authority
        weights). Used by the ``doc_type_prior`` shim for ``max_boost``
        compatibility."""
        return self.model_copy(update={"max_boost": max_boost})

    def with_overrides(
        self,
        overrides: dict[str, float],
        *,
        max_boost: float | None = None,
    ) -> SourceAuthorityConfig:
        """Copy with per-category authority overrides merged over the
        current weights (keys not present are kept). Optionally overrides
        ``max_boost`` too. The single knob operators tune at runtime."""
        authorities = {**self.authorities, **overrides}
        update: dict = {"authorities": authorities}
        if max_boost is not None:
            update["max_boost"] = max_boost
        return self.model_copy(update=update)


# ---------------------------------------------------------------------------
# Default authority table
# ---------------------------------------------------------------------------
#
# Weights translated verbatim from the RI-01 document-type prior so the
# default behavior is unchanged. Positive authorities favour design /
# verification material a question is usually about; negative authorities
# nudge away from verbose runtime artifacts whose lexical content
# over-matches natural-language queries (simulation logs, coverage
# reports). Magnitudes stay small — this is a nudge, not a filter.

_SOURCE_AUTHORITY_WEIGHTS: dict[str, float] = {
    "rtl": 0.10,
    "specification": 0.15,
    "design_documentation": 0.05,
    "architecture_documentation": 0.15,
    "uvm": 0.08,
    "test": 0.05,
    "python": 0.05,
    "markdown": 0.0,
    "simulation_log": -0.10,
    "coverage_report": -0.10,
    "lint_report": -0.10,
    "timing_report": -0.10,
    "waveform": -0.15,
    "build_artifact": -0.05,
    "generated": -0.10,
    "tcl": 0.0,
    "other": 0.0,
}

# Path-substring rules, in priority order (first match wins, most
# specific first). Keep these aligned with real indexed repositories
# (rtl/, tb/, uvm/, tests/, docs/, logs/, coverage/, lint/, timing/,
# reports/, build/, generated/, scripts/) — do not invent folders that
# don't exist in real data.
_PATH_RULES: list[SourceAuthorityRule] = [
    SourceAuthorityRule(pattern="docs/architecture/", category="architecture_documentation"),
    SourceAuthorityRule(pattern="docs/spec/", category="specification"),
    SourceAuthorityRule(pattern="specs/", category="specification"),
    SourceAuthorityRule(pattern="docs/", category="design_documentation"),
    # Design source. ``rtl/`` is checked before ``tb/`` so a verification
    # file nested under ``rtl/`` is not reclassified (RI-01 behavior).
    SourceAuthorityRule(pattern="rtl/", category="rtl"),
    SourceAuthorityRule(pattern="design/", category="rtl"),
    # Verification source.
    SourceAuthorityRule(pattern="uvm/", category="uvm"),
    SourceAuthorityRule(pattern="testbench/", category="uvm"),
    SourceAuthorityRule(pattern="tb/", category="uvm"),
    SourceAuthorityRule(pattern="tests/", category="test"),
    # Debug / output / generated — slight penalties so they don't crowd
    # design material when the query is ambiguous.
    SourceAuthorityRule(pattern="logs/", category="simulation_log"),
    SourceAuthorityRule(pattern="log/", category="simulation_log"),
    SourceAuthorityRule(pattern="coverage/", category="coverage_report"),
    SourceAuthorityRule(pattern="lint/", category="lint_report"),
    SourceAuthorityRule(pattern="timing/", category="timing_report"),
    SourceAuthorityRule(pattern="reports/", category="build_artifact"),
    SourceAuthorityRule(pattern="build/", category="build_artifact"),
    SourceAuthorityRule(pattern="generated/", category="generated"),
    SourceAuthorityRule(pattern="gen/", category="generated"),
    SourceAuthorityRule(pattern="scripts/", category="other"),
]

# Chunk-type prefix rules (longest prefix wins), translated from the
# RI-01 chunk-type prior table.
_CHUNK_TYPE_RULES: list[SourceAuthorityRule] = [
    # SystemVerilog / Verilog design material.
    SourceAuthorityRule(pattern="sv_module_signature", category="rtl"),
    SourceAuthorityRule(pattern="sv_module", category="rtl"),
    SourceAuthorityRule(pattern="sv_package", category="rtl"),
    SourceAuthorityRule(pattern="sv_body_window", category="rtl"),
    # Assertions are verification intent, not design intent.
    SourceAuthorityRule(pattern="sv_assertion", category="uvm"),
    SourceAuthorityRule(pattern="v_module", category="rtl"),
    SourceAuthorityRule(pattern="v_body_window", category="rtl"),
    # Python.
    SourceAuthorityRule(pattern="py_class_signature", category="python"),
    SourceAuthorityRule(pattern="py_module_overview", category="python"),
    SourceAuthorityRule(pattern="py_function", category="python"),
    SourceAuthorityRule(pattern="py_class", category="python"),
    SourceAuthorityRule(pattern="py_method", category="python"),
    # Markdown prose.
    SourceAuthorityRule(pattern="markdown_section", category="markdown"),
    SourceAuthorityRule(pattern="markdown_preamble", category="markdown"),
    # Runtime / report artifacts.
    SourceAuthorityRule(pattern="coverage_report", category="coverage_report"),
    SourceAuthorityRule(pattern="lint_report", category="lint_report"),
    SourceAuthorityRule(pattern="timing_report", category="timing_report"),
    SourceAuthorityRule(pattern="waveform", category="waveform"),
    SourceAuthorityRule(pattern="log", category="simulation_log"),
    SourceAuthorityRule(pattern="generic_window", category="other"),
]

DEFAULT_SOURCE_AUTHORITY = SourceAuthorityConfig(
    authorities=_SOURCE_AUTHORITY_WEIGHTS,
    path_rules=_PATH_RULES,
    chunk_type_rules=_CHUNK_TYPE_RULES,
    max_boost=0.5,
)


def classify_document(
    chunk_type: str,
    relative_path: str,
    *,
    config: SourceAuthorityConfig = DEFAULT_SOURCE_AUTHORITY,
) -> str:
    """Classify a chunk's source document into an authority category.

    Path rules are checked first (case-insensitive substring, in declared
    order — first match wins, so more specific patterns must come first).
    Without a path match, ``chunk_type`` prefix rules apply (longest
    prefix wins). No rule matches -> ``"other"``.
    """
    lowered = relative_path.lower()
    for rule in config.path_rules:
        if rule.pattern in lowered:
            return rule.category

    best: SourceAuthorityRule | None = None
    for rule in config.chunk_type_rules:
        if chunk_type.startswith(rule.pattern) and (
            best is None or len(rule.pattern) > len(best.pattern)
        ):
            best = rule
    return best.category if best is not None else "other"


def apply_source_authority(
    results: list[RetrievalResult],
    *,
    config: SourceAuthorityConfig = DEFAULT_SOURCE_AUTHORITY,
) -> list[RetrievalResult]:
    """Apply the source authority to each result.

    Multiplier: ``fused_score * (1 + authority)`` where the authority is
    bounded by ``[-max_boost, +max_boost]``. The authority is recorded
    under both ``signal_scores["source_authority"]`` (canonical) and
    ``signal_scores["doc_type_prior"]`` (legacy key) so the ranking trace
    and the RB-02/RB-03 analytics keep working.

    Results are returned re-sorted by ``fused_score`` desc, matching the
    pipeline's ordering invariant. Chunks whose category has authority 0
    are left untouched (no copy, no signal key) — exactly the
    ``doc_type_prior`` behavior.
    """
    if not results:
        return results

    boosted: list[RetrievalResult] = []
    for result in results:
        category = classify_document(
            result.chunk_type, result.relative_path, config=config
        )
        prior = config.authority(category)
        if prior == 0.0:
            boosted.append(result)
            continue
        updated = result.model_copy(deep=True)
        updated.fused_score = result.fused_score * (1.0 + prior)
        updated.signal_scores["source_authority"] = prior
        updated.signal_scores["doc_type_prior"] = prior
        boosted.append(updated)

    boosted.sort(key=lambda r: r.fused_score, reverse=True)
    return boosted
