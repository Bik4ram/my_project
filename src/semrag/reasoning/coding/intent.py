"""Deterministic user-intent classification for coding requests.

The classifier is keyword/pattern based and fully deterministic — no LLM
is involved, so the "does this request modify the repository?" decision
is always reproducible and testable. Precedence:

1. explicit debug language (fix / failing / broken / error / lint /
   doesn't compile / bug) → DEBUG
2. explicit change verbs (add / create / implement / generate / write a
   / modify / update / insert / extend / change ... to) → CODE_CHANGE
3. review language → REVIEW
4. interrogative / explanatory language → INFORMATION
5. otherwise → AMBIGUOUS (never modify)
"""
from __future__ import annotations

import re

from semrag.reasoning.coding.models import CodingIntent

_DEBUG_PATTERNS = (
    r"\bfix\b",
    r"\bfix\s+the\b",
    r"\bdebug\b",
    r"\bfailing\b",
    r"\bbroken\b",
    r"\bsyntax\s+error\b",
    r"\bcompile\s+error\b",
    r"\bdoes\s+not\s+compile\b",
    r"\bdoesn'?t\s+compile\b",
    r"\blint\b",
    r"\bbug\b",
    r"\bcrash(es|ed)?\b",
    r"\bincorrect\b",
    r"\bwrong\b",
)

_CHANGE_PATTERNS = (
    r"\b(add|create|implement|generate|insert|extend|modify|update|replace|append|improve|optimize|rename)\b",
    r"\bname\s+the\b",
    r"\bnew\s+(test|file|module|class|sequence|driver|function|method)\b",
    r"\bchange\b.*\bto\b",
    r"\bwrite\s+(a|an|the|new)?\s*(test|file|module|function|driver|sequence|program)\b",
    r"\bread\s+test\b",
)

_REVIEW_PATTERNS = (
    r"\breview\b",
    r"\bcheck\s+(the\s+)?(code|correctness|style|logic)\b",
    r"\bis\s+this\s+correct\b",
    r"\bcode\s+review\b",
)

_INFORMATION_PATTERNS = (
    r"\bexplain\b",
    r"\bdescribe\b",
    r"\btrace\b",
    r"\bsummarize\b",
    r"\bwhat\s+(is|are|does|do)\b",
    r"\bhow\s+(does|is|are|can)\b",
    r"\bwhich\b",
    r"\bwhere\s+(is|are)\b",
    r"\bwhy\b",
    r"\bshow\s+me\b",
    r"\boverview\b",
)

_QUESTION_PATTERN = re.compile(r"^(who|what|which|where|when|why|how|does|do|is|are|can|could|should)\b")


def _has_patterns(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(re.search(p, lowered) for p in patterns)


def classify_intent(request_text: str) -> CodingIntent:
    """Classify a coding request. Never raises and never returns a
    modifying intent for non-modifying input."""
    text = (request_text or "").strip()
    if not text:
        return CodingIntent.AMBIGUOUS

    if _has_patterns(text, _DEBUG_PATTERNS):
        return CodingIntent.DEBUG
    if _has_patterns(text, _CHANGE_PATTERNS):
        return CodingIntent.CODE_CHANGE
    if _has_patterns(text, _REVIEW_PATTERNS):
        return CodingIntent.REVIEW

    head = text.lstrip("().:,-").strip()
    first_word_match = _QUESTION_PATTERN.match(head)
    if first_word_match or text.rstrip().endswith("?"):
        if not _has_patterns(text, _CHANGE_PATTERNS):
            return CodingIntent.INFORMATION

    if _has_patterns(text, _INFORMATION_PATTERNS):
        return CodingIntent.INFORMATION

    return CodingIntent.AMBIGUOUS


def clarify_reason(intent: CodingIntent, request_text: str) -> str | None:
    """Human-readable clarification ask for ambiguous requests."""
    if intent is not CodingIntent.AMBIGUOUS:
        return None
    return (
        "Your request is ambiguous: I could not determine whether you want me to "
        "change code or only answer a question. I did not modify any files. "
        "Please clarify whether you want a code change (and which file/behaviour) "
        "or an explanation."
    )


_CREATE_PATTERN = re.compile(
    r"\b(?:create|implement|generate)\b[^.\n]{0,60}"
    r"\b(?:test|file|module|sequence|seq|driver|class|function|transaction|agent)\b"
    r"|\bnew\s+(?:test|file|module|sequence|seq|driver|class)\b"
    r"|\b(?:add|write)\b[^.\n]{0,40}\bnew\s+(?:test|file|module|sequence|seq|driver)\b"
)


def is_create_request(request_text: str) -> bool:
    """Deterministic 'is this a create-new-file request?' check. Only
    used to choose the plan contract: content generation for create,
    patch-based changes otherwise. Modify phrases like 'add a
    constructor to the existing test' do NOT count as create."""
    text = (request_text or "").strip().lower()
    if not text:
        return False
    if _has_patterns(text, _DEBUG_PATTERNS):
        return False
    return bool(_CREATE_PATTERN.search(text))


_REQUEST_NOUN_PATTERN = re.compile(r"\b(read|write|smoke|base|random|long|short)\s+(test|sequence|seq|transaction|driver)\b")


def request_target_noun(request_text: str) -> str | None:
    """Deterministic 'what is being created' extraction for the new-file
    name, e.g. ``read_test`` from 'create a READ test'."""
    m = _REQUEST_NOUN_PATTERN.search((request_text or "").lower())
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}"