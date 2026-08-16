"""
Query-intent detection.

Classifies what a question is *about* so the retrieval pipeline can bias
ranking toward the right category of source material (see
``ranking/intent.py`` for the ranking side that consumes the detected
intent).

Why this exists: a pure RRF fusion treats every chunk the same regardless
of what the user is actually asking. A question like "what module drives
the SDA line?" is asking about RTL, and an answer pointer pulled from a
simulation log entry that happens to mention SDA is worse than useless —
it looks like an answer but isn't. The prompt above the LLM can't tell
the difference; what we can do is rank design material *first* when the
question is about design, and debug material *first* when the question is
about debug.

Why not just LLM-classify the question once per ask? Each classification
call doubles the LLM round-trips for a question, and the LLM is the same
one we're trying to make produce better answers — adding a classification
step costs both latency and money without solving the actual retrieval-
quality problem (LLMs are bad at zero-shot ranking decisions; they're no
better than a small keyword classifier for the simple "is this asking
about modules or about logs" choice). The keyword heuristic below has
the same accuracy on this question shape and runs in microseconds.

The intent vocabulary is intentionally explicit (``QueryIntent`` enum).
The boost table that consumes an intent during ranking lives in
``ranking/intent.py``.
"""
from __future__ import annotations

import enum
import re

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


class QueryIntent(str, enum.Enum):
    """Coarse intent classes used to bias retrieval toward a category of
    source material. New intents require adding keywords **and** the
    corresponding boost table — these are coupled, not independent."""

    ARCHITECTURE = "architecture"
    VERIFICATION = "verification"
    DEBUG = "debug"
    GENERAL = "general"


_INTENT_KEYWORDS: dict[QueryIntent, tuple[str, ...]] = {
    QueryIntent.ARCHITECTURE: (
        "module", "rtl", "interface", "design", "instantiate", "instantiation",
        "block", "signal", "wire", "reg", "logic", "sda", "scl", "register",
        "fifo", "state machine", "fsm", "datapath", "architecture", "clock",
        "reset", "package", "import", "typedef", "struct",
    ),
    QueryIntent.VERIFICATION: (
        "uvm", "sequencer", "sequence", "driver", "monitor", "scoreboard",
        "testbench", "tb", "covergroup", "coverpoint", "cross coverage",
        "assertion", "sva", "property", "agent", "env", "environment",
        "config", "phase", "test", "base test", "write1", "write2",
        "sequence_item", "transaction",
    ),
    QueryIntent.DEBUG: (
        "fail", "failing", "failed", "error", "violation", "uvm_error",
        "uvm_fatal", "mismatch", "simulation log", "waveform", "vcd",
        "rpt", "report", "missed", "drop", "xprop", "x-prop",
        "trace", "stack", "dump", "timeout", "coverage gap",
        "coverage hole", "coverage missing",
    ),
}


def intent_keywords(intent: QueryIntent) -> tuple[str, ...]:
    """Public accessor for the intent's domain terminology.

    Exposed for ``query_expansion.py`` (RI-05), which reuses the intent
    vocabulary to generate bounded, deterministic expansion formulations
    so a query doesn't need the repository's exact wording to retrieve
    relevant context. Unknown / GENERAL intents have no vocabulary."""
    return _INTENT_KEYWORDS.get(intent, ())


def classify_intent(query: str) -> QueryIntent:
    """Pick the dominant intent. Ties broken by ranking of the intent
    block in the table above (ARCHITECTURE > VERIFICATION > DEBUG) so that
    a question that overlaps multiple vocabularies lands on the
    design-heavy side — an DV engineer asking about an RTL-level signal
    is more common than asking about a generic debug log entry."""
    text = query.lower()
    scores: dict[QueryIntent, int] = {intent: 0 for intent in QueryIntent}
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                scores[intent] += 1

    best_intent = max(scores, key=lambda i: (scores[i], -_intent_priority(i)))
    if scores[best_intent] == 0:
        return QueryIntent.GENERAL
    return best_intent


# Structural / overview queries: hand the model a FILE INDEX instead of
# trusting it to infer structure from whatever chunks happen to match
# "summarize" / "layout" / "overview" semantically (which — as the
# benchmark shows — usually pulls in repetitive log dumps).
_STRUCTURAL_QUERY_HINTS: tuple[str, ...] = (
    "summarize this repository", "summarize the repository",
    "summary of", "summarise", "overview", "repository structure",
    "repo structure", "project structure", "layout", "what's in this repo",
    "what is in this repo", "what's in this project", "describe the repository",
    "describe the project", "describe this", "explain the architecture",
    "complete architecture", "explain the structure", "module hierarchy",
    "file hierarchy", "directory structure", "components of this",
    "high-level overview", "high level overview",
)


def is_structural_query(query: str) -> bool:
    """True when the user is asking about the *shape* of the project,
    not a specific subsystem. These questions need a file index in the
    prompt or the LLM will hallucinate structure from log noise."""
    q = query.lower().strip()
    if not q:
        return False
    if any(hint in q for hint in _STRUCTURAL_QUERY_HINTS):
        return True
    # Phrases that start with a verb of comprehension typically want a
    # structural answer too: "summarize", "describe", "review", "explain".
    if re.search(r"^\s*(summari[sz]e|describe|review|overview|walk through)\b", q):
        return True
    # "Complete/whole/overall architecture" — broad view, not a specific
    # module's internals. Word-boundary match so a question about the
    # "architecture of i2c_master" still asks about a specific module.
    if re.search(r"\b(complete|whole|overall|high[- ]level)\b.*\barchitecture\b", q):
        return True
    return False


def _intent_priority(intent: QueryIntent) -> int:
    # Lower = preferred on ties. Intent table defines the order.
    order = (
        QueryIntent.ARCHITECTURE,
        QueryIntent.VERIFICATION,
        QueryIntent.DEBUG,
        QueryIntent.GENERAL,
    )
    return order.index(intent)


# ---------------------------------------------------------------------------
# Graph-expansion direction
# ---------------------------------------------------------------------------


class GraphExpansionDirection(str, enum.Enum):
    """Which way graph expansion should walk the IR edge graph relative to
    the query's seeds.

    ``DEPENDENTS`` (reverse: "who uses X?") is the historical default and
    is preserved for queries with no directional signal. ``DEPENDENCIES``
    (forward: "what does X use / depend on?") is needed for transaction
    tracing and path queries. ``BIDIRECTIONAL`` covers "connected to" /
    "relationship" phrasing with a bounded walk in both directions.

    Direction is a query-time retrieval signal only — it never changes the
    IR graph, chunking, or scoring weights. It selects which traversal
    ``graph_expansion.py`` runs."""

    DEPENDENTS = "dependents"
    DEPENDENCIES = "dependencies"
    BIDIRECTIONAL = "bidirectional"


# Phrase-level patterns, so ambiguous verbs ("uses", "calls", "depends on")
# resolve by their surrounding grammar instead of a bare keyword.
_FORWARD_DIRECTION_PATTERNS: tuple[str, ...] = (
    r"\btrace\b",
    r"\bflow(s|ing)?\b",
    r"\breach(es|ed)?\b",
    r"\bpath\b",
    r"\bthrough\b",
    r"\bdownstream\b",
    r"\bwhat does\b.*\b(call|use|depend|instantiate|reference|reach)\b",
    r"\bhow does\b.*\b(call|use|depend|instantiate|reference|reach)\b",
    r"\bextends\b",
    # Active-voice instantiation only — the passive "instantiated by" is a
    # reverse signal and is handled by the reverse pattern list.
    r"\binstantiat(?:es?)\b",
)

_REVERSE_DIRECTION_PATTERNS: tuple[str, ...] = (
    r"\bwho uses\b",
    r"\bwhat uses\b",
    r"\bwho depends on\b",
    r"\bwhat depends on\b",
    r"\bwho calls\b",
    r"\bwhat calls\b",
    r"\bwho references\b",
    r"\bwhat references\b",
    r"\breferenced by\b",
    r"\bused by\b",
    r"\binstantiated by\b",
    r"\bextended by\b",
    r"\bcalled by\b",
    r"\bdependents?\b",
    r"\bupstream\b",
    r"\bcallers?\b",
)

_BIDIRECTIONAL_DIRECTION_PATTERNS: tuple[str, ...] = (
    r"\bconnected? to\b",
    r"\brelationship between\b",
    r"\brelated? to\b",
    r"\brelate(s|d)?\b",
    r"\binteract(s|ion)?\b",
    r"\bboth\b",
)


def classify_graph_direction(query: str) -> GraphExpansionDirection:
    """Pick the graph-walk direction a query wants, deterministically.

    Forward signals ("trace", "reach", "path", "what does X use",
    "how does X reach Y", "extends") map to ``DEPENDENCIES``. Reverse
    signals ("who/what uses/calls/references X", "what depends on X",
    "instantiated by", "upstream") map to ``DEPENDENTS``. Relationship
    phrasing ("connected to", "relationship between", "related to") maps
    to ``BIDIRECTIONAL``. No signal keeps the historical ``DEPENDENTS``
    default so existing reverse behavior is unchanged.

    This is a deterministic keyword classifier — the same family as
    ``classify_intent`` — and never calls an LLM.
    """
    text = query.lower()
    forward = sum(1 for p in _FORWARD_DIRECTION_PATTERNS if re.search(p, text))
    reverse = sum(1 for p in _REVERSE_DIRECTION_PATTERNS if re.search(p, text))
    both = sum(1 for p in _BIDIRECTIONAL_DIRECTION_PATTERNS if re.search(p, text))

    if forward > 0 and reverse > 0:
        return GraphExpansionDirection.BIDIRECTIONAL
    if forward > 0:
        return GraphExpansionDirection.DEPENDENCIES
    if reverse > 0:
        return GraphExpansionDirection.DEPENDENTS
    if both > 0:
        return GraphExpansionDirection.BIDIRECTIONAL
    return GraphExpansionDirection.DEPENDENTS
