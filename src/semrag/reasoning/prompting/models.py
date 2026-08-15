"""Data models for prompt construction."""
from __future__ import annotations

from pydantic import BaseModel, Field

from semrag.retrieval.models import RetrievalResult


class ConversationTurn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant' — matches OpenAI-compatible chat message roles.")
    content: str


class ChatMessage(BaseModel):
    role: str
    content: str


class AssembledPrompt(BaseModel):
    messages: list[ChatMessage]
    included_chunks: list[RetrievalResult] = Field(
        default_factory=list,
        description="Chunks that made it into the context window — the citation/provenance record for this prompt.",
    )
    excluded_chunk_count: int = Field(
        default=0, description="How many retrieved chunks didn't fit the token budget — a truncation signal."
    )
    estimated_total_tokens: int = 0
