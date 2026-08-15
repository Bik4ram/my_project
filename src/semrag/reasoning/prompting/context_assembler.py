"""
Context assembly.

Greedy packing by fused-score order: results are already ranked by Part
10's hybrid retriever, and this takes them in that order, including each
one whole (never truncating a chunk mid-text - a half-cut assertion or
function is worse than not including it at all) until the next one
wouldn't fit the remaining budget, then stops. This is Part 1's "context
prioritization" and "token budgeting" requirements, implemented plainly:
prioritization is "trust the ranking you were already given," and
budgeting is "stop before you go over," not a more elaborate
knapsack-style optimization - a smarter packing (e.g. skipping one big
chunk to fit two smaller lower-ranked ones) would second-guess the
retrieval ranking in a way that's hard to justify without evidence it
actually improves answer quality, which is Part 15's job to measure, not
this module's to assume.
"""
from __future__ import annotations

from semrag.reasoning.prompting.citation import format_chunk_with_citation
from semrag.reasoning.prompting.token_estimator import TokenEstimator
from semrag.retrieval.models import RetrievalResult

DEFAULT_CHUNK_SEPARATOR = "\n\n---\n\n"


def assemble_context(
    results: list[RetrievalResult],
    budget_tokens: int,
    estimator: TokenEstimator,
    max_chunks: int | None = None,
) -> tuple[list[RetrievalResult], str, int]:
    """Returns (included_results, formatted_context_text, used_tokens).

    ``max_chunks`` is a hard cap on included chunks regardless of budget.
    Greedy packing against the token budget can still admit a dozen
    chunks of real code if each is small — fine for a frontier model, but
    a 7B local LLM on CPU spends 80+ s processing 6K input tokens and
    blows the 90 s timeout. Capping the count (in addition to the token
    budget) keeps prompts small enough for the slow path.
    """
    included: list[RetrievalResult] = []
    formatted_pieces: list[str] = []
    used_tokens = 0
    separator_tokens = estimator.estimate(DEFAULT_CHUNK_SEPARATOR)

    for result in results:
        piece = format_chunk_with_citation(result)
        piece_tokens = estimator.estimate(piece)
        additional_cost = piece_tokens + (separator_tokens if formatted_pieces else 0)

        # Budget gate first — never admit a chunk that doesn't fit.
        # The chunk itself may already exceed the budget (a 9K-char
        # generic_window of a log file). If we honored the cap before
        # the budget check, we'd include one giant chunk and exclude
        # everything else.
        if used_tokens + additional_cost > budget_tokens:
            continue

        if max_chunks is not None and len(included) >= max_chunks:
            break

        included.append(result)
        formatted_pieces.append(piece)
        used_tokens += additional_cost

    context_text = DEFAULT_CHUNK_SEPARATOR.join(formatted_pieces)
    return included, context_text, used_tokens
