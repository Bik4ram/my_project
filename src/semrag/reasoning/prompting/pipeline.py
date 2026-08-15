"""
Prompt builder pipeline.

Ties together token estimation, conversation history, and context
assembly into one call producing a complete, ready-to-send message list.
Budget allocation order matters and is deliberate: the system prompt and
conversation history are non-negotiable (an LLM without its grounding
instructions, or a conversation missing its own prior turns, produces
worse answers than one with less retrieved context) - so they're
subtracted from the total budget FIRST, and whatever remains goes to
retrieval context, which degrades gracefully by simply including fewer
chunks rather than failing outright when the budget is tight.

Conversation history itself is also bounded (`max_history_tokens`) rather
than included in full - an unbounded conversation eventually consumes the
entire budget and leaves nothing for retrieval context, defeating the
purpose of a RAG system. When history must be trimmed, the OLDEST turns
are dropped first (a sliding window) - the most recent turns are almost
always the most relevant to interpreting the current question.
"""
from __future__ import annotations

from semrag.reasoning.prompting.context_assembler import assemble_context
from semrag.reasoning.prompting.models import AssembledPrompt, ChatMessage, ConversationTurn
from semrag.reasoning.prompting.templates import SYSTEM_PROMPT, build_messages
from semrag.reasoning.prompting.token_estimator import HeuristicTokenEstimator, TokenEstimator
from semrag.retrieval.models import RetrievalResult

DEFAULT_MODEL_CONTEXT_WINDOW = 8192
DEFAULT_RESERVED_FOR_RESPONSE = 1024
DEFAULT_MAX_HISTORY_TOKENS = 1024


class PromptBuilder:
    def __init__(
        self,
        estimator: TokenEstimator | None = None,
        model_context_window: int = DEFAULT_MODEL_CONTEXT_WINDOW,
        reserved_for_response: int = DEFAULT_RESERVED_FOR_RESPONSE,
        max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS,
    ):
        self._estimator = estimator or HeuristicTokenEstimator()
        self._model_context_window = model_context_window
        self._reserved_for_response = reserved_for_response
        self._max_history_tokens = max_history_tokens

    def build(
        self,
        query: str,
        retrieval_results: list[RetrievalResult],
        conversation_history: list[ConversationTurn] | None = None,
        file_index: str | None = None,
        max_chunks: int | None = None,
    ) -> AssembledPrompt:
        conversation_history = conversation_history or []

        system_tokens = self._estimator.estimate(SYSTEM_PROMPT)
        query_tokens = self._estimator.estimate(query)
        file_index_tokens = self._estimator.estimate(file_index) if file_index else 0

        trimmed_history = self._trim_history(conversation_history)
        history_tokens = sum(self._estimator.estimate(t.content) for t in trimmed_history)

        context_budget = (
            self._model_context_window
            - self._reserved_for_response
            - system_tokens
            - query_tokens
            - history_tokens
            - file_index_tokens
        )
        context_budget = max(context_budget, 0)

        included, context_text, context_tokens = assemble_context(
            retrieval_results, context_budget, self._estimator, max_chunks=max_chunks
        )

        messages = build_messages(context_text, trimmed_history, query, file_index=file_index)

        return AssembledPrompt(
            messages=[ChatMessage(**m) for m in messages],
            included_chunks=included,
            excluded_chunk_count=len(retrieval_results) - len(included),
            estimated_total_tokens=(
                system_tokens + query_tokens + history_tokens + file_index_tokens + context_tokens
            ),
        )

    def _trim_history(self, history: list[ConversationTurn]) -> list[ConversationTurn]:
        """Keeps the most recent turns, dropping the oldest first, until
        the running total fits `max_history_tokens`."""
        kept: list[ConversationTurn] = []
        used = 0
        for turn in reversed(history):
            cost = self._estimator.estimate(turn.content)
            if used + cost > self._max_history_tokens:
                break
            kept.append(turn)
            used += cost
        return list(reversed(kept))
