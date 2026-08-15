"""
Token estimation.

**Same limitation class as Parts 7/9/10, found the same way:** real BPE
tokenizers need downloadable vocabulary files. `tiktoken` (the obvious
choice, pure-Python, no model weights) fetches its encoding's merge-rank
file from `openaipublic.blob.core.windows.net` at first use — confirmed
blocked in this sandbox (`403 Forbidden`, same as every other blocked
domain in this series). HuggingFace `tokenizers` has the identical
problem one layer earlier (downloading the vocab itself). No real
tokenizer — for any model — is reachable here.

**What's built instead:** a real, documented, testable heuristic —
~4 characters per token for English-language text (including code,
which tends to run slightly denser due to identifiers/punctuation, so
this is a deliberately conservative/slight-overestimate ratio, not a
precise one). This is a commonly-cited rule of thumb for GPT-style BPE
tokenizers, not a number invented for this delivery — but it is
explicitly an approximation, not a substitute for the actual tokenizer of
whatever model Part 13 ends up calling. `TokenEstimator` is a Protocol
specifically so swapping in a real tokenizer (once one's model-specific
vocabulary is reachable, or by vendoring a vocab file directly rather
than fetching it at runtime) is a one-class change, not a rewrite of
everything that budgets against it.
"""
from __future__ import annotations

from typing import Protocol

CHARS_PER_TOKEN_ESTIMATE = 4.0


class TokenEstimator(Protocol):
    def estimate(self, text: str) -> int: ...


class HeuristicTokenEstimator:
    def __init__(self, chars_per_token: float = CHARS_PER_TOKEN_ESTIMATE):
        self._chars_per_token = chars_per_token

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        # Round up — under-budgeting (leaving a prompt that's actually
        # slightly over the real model's token limit) is a much worse
        # failure mode than over-budgeting (leaving a little headroom
        # unused), so ties/fractions round toward "this costs more."
        return max(1, -(-len(text) // int(self._chars_per_token)))
