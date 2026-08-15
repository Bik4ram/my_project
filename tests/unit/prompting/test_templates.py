"""Tests for `prompting.templates.build_messages` — specifically the
FILE INDEX block and structural grounding text."""
from __future__ import annotations


from semrag.reasoning.prompting.templates import SYSTEM_PROMPT, build_messages


def _basic_history():
    return []


def test_system_prompt_mentions_file_index():
    """The system prompt must tell the model to trust the FILE INDEX
    rather than log noise for structural claims. Without this,
    'summarize the repo' hallucinates from simulation-log content."""
    assert "FILE INDEX" in SYSTEM_PROMPT


def test_system_prompt_forbids_inventing_code_and_symbols():
    """R4 regression: the /answer endpoint fabricated an entire class
    (i2c_master_sequence), a method (sequencer::run_phase) and code
    statements on a multi-hop trace query while emitting zero citations.
    The prompt must forbid inventing code/symbols, not just paths."""
    assert "Never write code that you cannot reproduce verbatim from the CONTEXT" in SYSTEM_PROMPT
    assert "Do not invent" in SYSTEM_PROMPT
    assert "p_sequencer->" in SYSTEM_PROMPT


def test_build_messages_includes_file_index_when_provided():
    msgs = build_messages(
        context_text="module i2c_master; endmodule",
        conversation_history=_basic_history(),
        query="Summarize the repo.",
        file_index="rtl_verilog/\n  rtl/i2c_master.sv",
    )
    user_msg = next(m["content"] for m in msgs if m["role"] == "user")
    assert "FILE INDEX:" in user_msg
    assert "rtl/i2c_master.sv" in user_msg
    assert "CONTEXT:" in user_msg
    assert "QUESTION:" in user_msg


def test_build_messages_omits_file_index_when_none():
    msgs = build_messages(
        context_text="module x; endmodule",
        conversation_history=_basic_history(),
        query="what does x do?",
        file_index=None,
    )
    user_msg = next(m["content"] for m in msgs if m["role"] == "user")
    assert "FILE INDEX:" not in user_msg


def test_build_messages_empty_context_and_no_file_index_uses_tool_fallback():
    msgs = build_messages(
        context_text="",
        conversation_history=_basic_history(),
        query="?",
        file_index=None,
    )
    user_msg = next(m["content"] for m in msgs if m["role"] == "user")
    assert "Use the available tools" in user_msg


def test_build_messages_order_file_index_before_context():
    """FILE INDEX must come BEFORE CONTEXT so the model can decide
    to rely on the index for structural claims even when both are
    present."""
    msgs = build_messages(
        context_text="CONTENT_FROM_CONTEXT",
        conversation_history=_basic_history(),
        query="summarize",
        file_index="PATH_FROM_INDEX",
    )
    user_msg = next(m["content"] for m in msgs if m["role"] == "user")
    idx_pos = user_msg.index("FILE INDEX")
    ctx_pos = user_msg.index("CONTEXT")
    assert idx_pos < ctx_pos
    assert idx_pos < user_msg.index("PATH_FROM_INDEX")
    assert ctx_pos < user_msg.index("CONTENT_FROM_CONTEXT")