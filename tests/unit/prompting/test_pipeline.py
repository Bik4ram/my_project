from semrag.reasoning.prompting.models import ConversationTurn
from semrag.reasoning.prompting.pipeline import PromptBuilder
from semrag.reasoning.prompting.templates import SYSTEM_PROMPT
from semrag.retrieval.models import RetrievalResult


def _result(chunk_id: str, text: str, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, project_id="proj1", relative_path=f"RTL/{chunk_id}.sv",
        chunk_type="sv_module", text=text, fused_score=score, line_start=1, line_end=5,
    )


def test_system_prompt_is_always_first_message():
    builder = PromptBuilder()
    prompt = builder.build("what does this do?", [])
    assert prompt.messages[0].role == "system"
    assert prompt.messages[0].content == SYSTEM_PROMPT


def test_query_appears_in_final_user_message():
    builder = PromptBuilder()
    prompt = builder.build("why does the assertion fail?", [_result("a", "assertion text")])
    assert "why does the assertion fail?" in prompt.messages[-1].content


def test_retrieved_context_is_included_with_citation():
    builder = PromptBuilder()
    prompt = builder.build("question", [_result("a", "module i2c_master")])
    final_message = prompt.messages[-1].content
    assert "module i2c_master" in final_message
    assert "[Source: RTL/a.sv, lines 1-5]" in final_message
    assert len(prompt.included_chunks) == 1


def test_no_context_found_produces_explicit_note_not_silent_gap():
    """Empty retrieval context must surface explicitly to the LLM — not
    be silently dropped. The agent should see that it has no context
    and be pointed at the tools to gather some, not hallucinate an
    answer from training data."""
    builder = PromptBuilder()
    prompt = builder.build("question with no matches", [])
    final_message = prompt.messages[-1].content
    assert "retrieved context was empty" in final_message
    assert "search_codebase" in final_message
    assert prompt.included_chunks == []
    assert prompt.excluded_chunk_count == 0


def test_conversation_history_included_in_order():
    builder = PromptBuilder()
    history = [
        ConversationTurn(role="user", content="first question"),
        ConversationTurn(role="assistant", content="first answer"),
    ]
    prompt = builder.build("follow-up question", [], conversation_history=history)

    roles_and_content = [(m.role, m.content) for m in prompt.messages]
    assert ("user", "first question") in roles_and_content
    assert ("assistant", "first answer") in roles_and_content
    history_index = roles_and_content.index(("user", "first question"))
    final_index = len(roles_and_content) - 1
    assert history_index < final_index


def test_tight_context_budget_excludes_low_ranked_chunks():
    builder = PromptBuilder(model_context_window=200, reserved_for_response=50, max_history_tokens=0)
    results = [_result(f"c{i}", "x" * 100, score=1.0 - i * 0.01) for i in range(10)]
    prompt = builder.build("question", results)
    assert len(prompt.included_chunks) < len(results)
    assert prompt.excluded_chunk_count > 0


def test_old_history_trimmed_when_history_budget_is_tight():
    builder = PromptBuilder(max_history_tokens=10)
    history = [
        ConversationTurn(role="user", content="x" * 200),
        ConversationTurn(role="assistant", content="y" * 200),
        ConversationTurn(role="user", content="hi"),
    ]
    prompt = builder.build("question", [], conversation_history=history)
    contents = [m.content for m in prompt.messages]
    assert "hi" in contents
    assert not any("x" * 200 in c for c in contents)


def test_estimated_total_tokens_is_positive():
    builder = PromptBuilder()
    prompt = builder.build("question", [_result("a", "some context text")])
    assert prompt.estimated_total_tokens > 0
