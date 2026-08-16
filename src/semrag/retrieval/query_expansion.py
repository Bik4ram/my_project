"""
Deterministic query expansion (RI-05).

Why this exists: hybrid retrieval is hybrid precisely because the vector
and lexical legs are good at different things — embeddings handle
semantic paraphrase, Postgres FTS handles exact terms and identifiers.
The failure mode this module targets is the gap between them: a user asks
"how does the design handle an unknown address?" and the repository only
ever says "X_FAIL", "UVM_ERROR", "addr mismatch". The vector leg is
already tolerant of the paraphrase; the *lexical* leg is not — FTS matches
the query's literal lexemes, so a question that uses different wording
than the codebase can silently miss chunks that a DV engineer would
immediately recognize as the answer.

Design decisions
----------------

1. **Lexical-leg only, zero network.** Expansion targets the lexical
   search only. The vector leg keeps the caller's single original
   embedding (already computed once per query), because re-embedding N
   expansion formulations would add N embedding round-trips of network
   latency per query for a leg that already handles paraphrase. The
   deterministic formulation below adds no network, no LLM, and no
   embedding calls at all — just a few bounded FTS queries.

2. **Reuse the intent vocabulary.** The existing ``retrieval/intent.py``
   already encodes the repository's domain terminology (``uvm_error``,
   ``covergroup``, ``mismatch``, ``waveform``, ...) keyed by
   ``QueryIntent``. Expanding the query with those exact terms is
   precisely "use the repository's wording," so we reuse them instead of
   inventing a parallel synonym dictionary.

3. **Original query always searched.** ``expand_query`` never replaces
   the query; the caller always searches the original text first, and the
   expansions are additional signals fused through the existing RRF (see
   ``pipeline.py``). A formulation is only generated when it actually
   *adds* domain terms the query doesn't already contain.

4. **Bounded and deterministic.** A pure function of
   ``(query_text, intent, config)``: no randomness, no I/O, no clock, so
   repeated calls produce identical formulations. Output is capped at
   ``max_expansions`` formulations, each truncated to
   ``max_expansion_length`` characters.

5. **Fail-open.** The pipeline wraps expansion in try/except; if this
   module ever raises, retrieval continues with the original query only
   and the failure is recorded in the trace.
"""
from __future__ import annotations

import math
import re

from pydantic import BaseModel, field_validator

from semrag.retrieval.intent import QueryIntent, intent_keywords, is_structural_query

_WS = re.compile(r"\s+")


class QueryExpansionConfig(BaseModel):
    """Configuration for deterministic query expansion.

    ``timeout_ms`` is reserved for a future external/LLM expansion
    mechanism (the task boundary keeps the door open); the deterministic
    implementation shipped here ignores it because it makes no network
    or LLM calls — there is nothing to time out. It is recorded in the
    trace so operators can see the configured ceiling.
    """

    enabled: bool = True
    max_expansions: int = 3
    max_expansion_length: int = 80
    timeout_ms: float | None = None

    @field_validator("max_expansions", "max_expansion_length")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v


DEFAULT_QUERY_EXPANSION = QueryExpansionConfig()


def _normalize(text: str) -> str:
    return _WS.sub(" ", text.strip().lower())


def _truncate(text: str, limit: int) -> str:
    """Cut ``text`` at the last whole-word boundary within ``limit``
    characters (preferring to keep the base query intact, which sits at
    the front of every formulation)."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut


def _bucket(items: list[str], n: int) -> list[list[str]]:
    """Partition ``items`` into up to ``n`` contiguous groups. Deterministic
    (no randomness, no set iteration — preserves tuple order from the
    intent vocabulary)."""
    if n <= 0 or not items:
        return []
    group_size = math.ceil(len(items) / n)
    return [items[i : i + group_size] for i in range(0, len(items), group_size)]


def expand_query(
    query_text: str,
    *,
    intent: QueryIntent,
    config: QueryExpansionConfig,
) -> list[str]:
    """Generate up to ``config.max_expansions`` additional lexical query
    formulations, each = the original query + a bounded group of the
    intent's domain terms.

    Returns ``[]`` when there is nothing useful to add: expansion
    disabled, a GENERAL intent (no domain vocabulary — inventing terms
    would be a guess, not an expansion), a structural/overview query
    (those want a file index, not more vocabulary), a blank query, or a
    query that already contains every domain term (a trivial expansion
    would just duplicate the original). The original query is never
    returned or replaced here — the caller always searches it.
    """
    if not config.enabled:
        return []
    if intent == QueryIntent.GENERAL:
        return []
    if is_structural_query(query_text):
        return []

    base = query_text.strip()
    if not base:
        return []

    base_norm = _normalize(base)
    q_lower = base.lower()

    keywords = intent_keywords(intent)
    # Drop domain terms the query already covers (substring match on the
    # lowercased query) — appending "fail" to "why did the test fail"
    # adds nothing; appending "uvm_error"/"mismatch" does.
    candidates = [kw for kw in keywords if kw.lower() not in q_lower]
    if not candidates:
        return []

    formulations: list[str] = []
    seen: set[str] = set()
    for group in _bucket(candidates, config.max_expansions):
        formulation = _truncate(f"{base} {' '.join(group)}", config.max_expansion_length)
        norm = _normalize(formulation)
        if not norm or norm == base_norm or norm in seen:
            continue
        seen.add(norm)
        formulations.append(formulation)
        if len(formulations) >= config.max_expansions:
            break
    return formulations


def build_query_expansion_config(settings) -> QueryExpansionConfig:
    """Build the expansion config from the app ``Settings`` (the single
    source of truth for runtime tunables; see ``core/settings.py``)."""
    return QueryExpansionConfig(
        enabled=settings.query_expansion_enabled,
        max_expansions=settings.query_expansion_max_expansions,
        max_expansion_length=settings.query_expansion_max_expansion_length,
        timeout_ms=settings.query_expansion_timeout_ms,
    )
