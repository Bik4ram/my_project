"""
Reranking.

**Same limitation class as Part 9's embedding provider, same two-track
response:** a real cross-encoder reranker model requires downloadable
weights (HuggingFace or similar), which is blocked in this sandbox (see
Part 9's design doc, Section 1 — the restriction is identical here).
Two things are built instead of a placeholder:

1. `LexicalOverlapReranker` — a real, local, dependency-free reranker
   using token-overlap (Jaccard similarity) between the query and each
   candidate's text. This is a genuine, if weak, relevance signal — exact
   term overlap between a query and a chunk is real evidence of
   relevance, just without a cross-encoder's ability to judge relevance
   from meaning beyond shared vocabulary. Fully tested (Section on tests
   in the design doc).

   The score combination here is a **hybrid-aware multiplicative boost**
   (not a replacement of fused_score). Earlier this module overwrote the
   RRF/intent-boost score with raw Jaccard, which produced two failure
   modes the trace surfaced: repetitive log chunks (high Jaccard from
   repeated query keywords) were winning the final cut over real UVM
   `sv_module` chunks (low Jaccard but high RRF + intent boost). Keeping
   fused_score as the primary signal and using Jaccard as a multiplier
   preserves the upstream ranking while still allowing reranker signal
   to break ties.

   On top of that bounded boost, the reranker applies a **deterministic,
   bounded metadata tie-breaker** (RI-07): candidates whose post-Jaccard
   scores are within ``near_tie_epsilon`` of each other may be reordered
   to prefer a chunk whose ``symbol_name`` matches a query term — but
   ONLY when the existing QueryIntent / structural-query logic
   (``retrieval/intent.py``) identifies the query as code-navigation
   (ARCHITECTURE intent or a structural query). It is a tie-break, not a
   score: a materially better relevance score (a gap larger than
   ``near_tie_epsilon``) is never displaced, and for every other query
   shape (non-navigation, no matching symbol, missing metadata) the
   ordering is exactly the pre-tie-break ordering with the existing
   deterministic ``chunk_id`` tie-break.

2. `HTTPRerankerClient` — real HTTP client code against the
   Cohere/TEI-style `POST /rerank` contract (`{"query": ..., "documents":
   [...]}`  -> `{"results": [{"index": ..., "relevance_score": ...}]}`)
   that vLLM and Text-Embeddings-Inference both implement for cross-
   encoder reranking. Logic (response reordering, error handling) is
   tested via mocked HTTP, the same honest boundary as Part 9's
   `OpenAICompatibleEmbeddingProvider` — real-server contract
   compatibility is unverified without a reachable instance.
"""
from __future__ import annotations

import re
import time
from typing import Protocol

import requests

from semrag.retrieval.intent import QueryIntent, classify_intent, is_structural_query
from semrag.retrieval.models import RetrievalResult

_TOKEN_PATTERN = re.compile(r"[\w]+")
# Symbol names are identifiers ("i2c_master", "clock_gen"); splitting on
# non-alphanumeric AND underscore boundaries turns them into their
# constituent tokens so a query term matches the symbol it genuinely
# names ("i2c master" ↔ symbol "i2c_master"), without a whole English
# word embedded mid-identifier ("the" inside "other_mod") counting as a
# match.
_SYMBOL_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


class Reranker(Protocol):
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]: ...


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_PATTERN.findall(text)}


# Multiplier coefficient for the lexical-overlap reranker. Fused into
# fused_score via ``fused_score * (1 + weight * jaccard)``. The default
# keeps the reranker's swing bounded to ~30% on a perfect Jaccard=1.0
# hit — large enough to break RRF ties, too small to demote genuinely-
# strong RRF hits purely on Jaccard. Tune via env var if a future eval
# shows the magnitude is wrong.
DEFAULT_LEXICAL_RERANK_WEIGHT = 0.3

# Near-tie window for the RI-07 metadata tie-break. Chosen to be smaller
# than the smallest single-leg RRF adjacent-rank delta: RRF contributes
# 1/(k+rank) per leg, so adjacent ranks differ by 1/((k+i)(k+i+1)) —
# ~2.6e-4 at rank 1, shrinking to ~1.5e-4 at rank 20 (k=60). An epsilon
# of 1e-4 therefore never treats two genuinely different RRF ranks as a
# near-tie; it only captures exact / near-exact score ties (identical
# Jaccard boosts on identical RRF bases, or graph-expansion chunks whose
# penalized scores are assigned by construction). Post-boost multipliers
# (intent, source authority) only widen real gaps, so the margin holds.
DEFAULT_NEAR_TIE_EPSILON = 1e-4


def _is_symbol_navigation_query(query: str) -> bool:
    """True when the EXISTING intent / structural-query logic identifies
    an identifier / code-navigation query. Reuses only shipped classifiers
    (``classify_intent``, ``is_structural_query``) — no new detector is
    invented here, so a bare identifier query with no intent or structural
    signal is deliberately NOT treated as code-navigation (conservative:
    if the codebase can't justify the preference, it isn't applied)."""
    if is_structural_query(query):
        return True
    return classify_intent(query) == QueryIntent.ARCHITECTURE


def _symbol_tokens(text: str) -> set[str]:
    """Lower-cased identifier sub-tokens of a query or a symbol name:
    ``"i2c_master"`` → ``{"i2c", "master"}``, ``"clock_gen"`` →
    ``{"clock", "gen"}``. Underscores are treated as boundaries so a
    natural-language query token lines up with the identifier it names."""
    return {t.lower() for t in _SYMBOL_TOKEN_PATTERN.findall(text.replace("_", " "))}


def _symbol_matches_query(query_symbol_tokens: set[str], symbol_name: str | None) -> bool:
    """Concrete symbol-to-query relationship: some query identifier sub-
    token exactly equals a sub-token of the chunk's ``symbol_name``
    (e.g. query "i2c master" matches symbol "i2c_master"). Only this
    relationship unlocks the tie-break — provenance (generated vs
    authored), language, and category never do."""
    if not symbol_name:
        return False
    return bool(query_symbol_tokens & _symbol_tokens(symbol_name))


def _apply_symbol_tiebreak(
    rescored: list[RetrievalResult],
    query_symbol_tokens: set[str],
    epsilon: float,
) -> None:
    """Deterministic, bounded metadata tie-break, applied in place.

    Candidates already sorted by ``(fused_score, chunk_id)`` desc are
    grouped into near-tie buckets: consecutive candidates whose score gap
    is ``<= epsilon``. A bucket whose FULL span is ``<= epsilon`` may be
    reordered so candidates whose ``symbol_name`` matches the query come
    first (stable within each side). The preference is recorded under
    ``signal_scores["rerank_metadata_prior"] = 1.0`` ONLY for the
    promoted candidates of a bucket whose order actually changed. Buckets
    whose scores span more than epsilon, or where no candidate matches,
    are left untouched — relevance is never displaced.
    """
    if not rescored or epsilon < 0:
        return

    buckets: list[list[RetrievalResult]] = []
    current: list[RetrievalResult] = [rescored[0]]
    for r in rescored[1:]:
        if current[-1].fused_score - r.fused_score <= epsilon:
            current.append(r)
        else:
            buckets.append(current)
            current = [r]
    buckets.append(current)

    output: list[RetrievalResult] = []
    for bucket in buckets:
        if len(bucket) > 1 and bucket[0].fused_score - bucket[-1].fused_score <= epsilon:
            preferred = [
                r for r in bucket if _symbol_matches_query(query_symbol_tokens, r.symbol_name)
            ]
            if preferred and len(preferred) < len(bucket):
                preferred_ids = {r.chunk_id for r in preferred}
                rest = [r for r in bucket if r.chunk_id not in preferred_ids]
                reordered = preferred + rest
                if [r.chunk_id for r in reordered] != [r.chunk_id for r in bucket]:
                    for r in preferred:
                        r.signal_scores["rerank_metadata_prior"] = 1.0
                    bucket[:] = reordered
        output.extend(bucket)

    rescored[:] = output


class LexicalOverlapReranker:
    """Jaccard similarity between query tokens and each candidate's
    tokens, used to re-score (not just re-sort within the existing fused
    score) the top candidates RRF fusion already narrowed down to — this
    runs on a small candidate set (Part 1's typical reranking position in
    a retrieval pipeline: cheap-first-pass then rerank-the-top-N), not
    the whole corpus.

    Score combination rule (see module docstring): instead of replacing
    ``fused_score`` with the raw Jaccard, multiply by
    ``1 + weight * jaccard``. The upstream RRF/intent score is preserved
    as the primary signal; the reranker contributes a multiplicative
    boost only. A chunk with strong RRF + low Jaccard still ranks ahead
    of a chunk with weak RRF + high Jaccard (e.g. a noisy log file
    that mentions the query terms in passing). A chunk with strong RRF
    AND high Jaccard gets a free bonus. A chunk with weak RRF and zero
    Jaccard doesn't change position. This is what a re-ranker actually
    *should* do — nudge within a strong candidate set, not veto.

    RI-07 adds one more, narrower rule on top: within a near-tie (post-
    Jaccard scores within ``near_tie_epsilon``), candidates whose
    ``symbol_name`` matches a query term may be promoted ahead of their
    near-tied peers — but only for queries the existing intent /
    structural-query logic identifies as code-navigation. See
    ``_apply_symbol_tiebreak`` for the exact rule and its bounds.
    """

    def __init__(
        self,
        weight: float = DEFAULT_LEXICAL_RERANK_WEIGHT,
        enable_metadata_tiebreak: bool = True,
        near_tie_epsilon: float = DEFAULT_NEAR_TIE_EPSILON,
    ) -> None:
        if weight < 0:
            raise ValueError("reranker weight must be >= 0")
        if near_tie_epsilon < 0:
            raise ValueError("near_tie_epsilon must be >= 0")
        self._weight = weight
        self._enable_metadata_tiebreak = enable_metadata_tiebreak
        self._near_tie_epsilon = near_tie_epsilon

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        query_tokens = _tokenize(query)
        if not query_tokens or self._weight == 0.0:
            # No query overlap or reranker disabled — leave the upstream
            # ordering alone (preserves RRF + intent-boost signal).
            return results[:top_k]

        # If the upstream fused_score is uniformly zero (test fixtures,
        # or a degenerate case where fusion produced no signal) fall
        # back to raw Jaccard. Otherwise apply the multiplier on the live
        # RRF/intent score - preserves the upstream ranking while letting
        # Jaccard break RRF tie-clusters near the boundary.
        any_nonzero = any(r.fused_score > 0.0 for r in results)

        rescored: list[RetrievalResult] = []
        for result in results:
            candidate_tokens = _tokenize(result.text)
            if not candidate_tokens:
                overlap_score = 0.0
            else:
                intersection = len(query_tokens & candidate_tokens)
                union = len(query_tokens | candidate_tokens)
                overlap_score = intersection / union if union else 0.0

            updated = result.model_copy(deep=True)
            updated.signal_scores["rerank_lexical_overlap"] = overlap_score
            if any_nonzero:
                # Hybrid-aware combination: fused_score stays primary,
                # Jaccard is a bounded multiplicative boost. Caps the
                # reranker's swing even at Jaccard = 1.0.
                updated.fused_score = result.fused_score * (1.0 + self._weight * overlap_score)
            else:
                # Degenerate upstream score - use raw Jaccard as the
                # sole signal so the reranker still produces a useful
                # ranking in the test-fixture / empty-fusion case.
                updated.fused_score = overlap_score
            rescored.append(updated)

        # Deterministic base ordering: relevance desc, then chunk_id desc
        # (matching the pipeline's ``_stable_top_k`` tie-break) so the
        # result is reproducible independent of input order.
        rescored.sort(key=lambda r: (r.fused_score, r.chunk_id), reverse=True)

        # RI-07 metadata tie-break: near-tied candidates may be reordered
        # to prefer a matching ``symbol_name`` — bounded by epsilon and
        # gated on the existing code-navigation query logic.
        if self._enable_metadata_tiebreak and _is_symbol_navigation_query(query):
            _apply_symbol_tiebreak(rescored, _symbol_tokens(query), self._near_tie_epsilon)

        return rescored[:top_k]


class RerankerError(RuntimeError):
    pass


class HTTPRerankerClient:
    """Status: real client code against the Cohere/TEI `/rerank` contract,
    not run against a live server in this delivery — see module docstring.

    Like the lexical-overlap reranker, this combines cross-encoder score
    with the upstream fused_score via a multiplicative boost rather than
    a replacement. A cross-encoder is high-resolution: relevance_score
    is in [0, 1] from a learned model, much closer to "true" relevance
    than Jaccard, but the rule stays the same — don't discard the
    fusion/intent signal, just nudge it. The weight is lower than
    Jaccard's because cross-encoders are more accurate, so their
    contribution can be larger without hijacking the ranking.
    """

    DEFAULT_WEIGHT = 1.0

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        weight: float = DEFAULT_WEIGHT,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        if weight < 0:
            raise ValueError("reranker weight must be >= 0")
        self._weight = weight

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not results:
            return []

        documents = [r.text for r in results]
        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    f"{self._base_url}/rerank",
                    json={"model": self._model, "query": query, "documents": documents, "top_n": top_k},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        else:
            raise RerankerError(f"Failed to reach reranker endpoint {self._base_url}: {last_error}")

        ranked = sorted(payload["results"], key=lambda item: item["relevance_score"], reverse=True)
        output = []
        for item in ranked[:top_k]:
            result = results[item["index"]].model_copy(deep=True)
            ce_score = float(item["relevance_score"])
            result.signal_scores["rerank_cross_encoder"] = ce_score
            if self._weight == 0.0:
                # Cross-encoder as pure ranker, no score change.
                result.fused_score = result.fused_score
            else:
                # Cross-encoder relevance_score is in [0, 1]; treat it as
                # a fractional additive boost so a relevance=1 hit gets
                # a +100% multiplier and a relevance=0 hit stays put.
                result.fused_score = result.fused_score * (1.0 + self._weight * ce_score)
            output.append(result)
        return output
