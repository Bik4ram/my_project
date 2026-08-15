from semrag.reasoning.agents.agent_loop import AgentLoop
from semrag.reasoning.agents.tools import ToolRegistry
from semrag.reasoning.llm.interfaces import CompletionResult, ToolCall


class _ScriptedFakeLLM:
    def __init__(self, script: list[CompletionResult]):
        self._script = list(script)
        self.call_count = 0
        self.received_messages: list[list[dict]] = []

    def complete(self, messages, tools=None) -> CompletionResult:
        self.received_messages.append(messages)
        result = self._script[self.call_count]
        self.call_count += 1
        return result

    def stream(self, messages, tools=None):
        raise NotImplementedError


class _EchoTool:
    name = "echo"
    description = "Echoes back its input."

    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def execute(self, text: str) -> str:
        return f"echoed: {text}"

    def to_openai_schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                   "parameters": self.parameters_schema()}}


def test_direct_answer_with_no_tool_calls():
    llm = _ScriptedFakeLLM([CompletionResult(content="The answer is 42.", finish_reason="stop")])
    loop = AgentLoop(llm, ToolRegistry([]))

    result = loop.run([{"role": "user", "content": "what is the answer?"}])

    assert result.final_answer == "The answer is 42."
    assert result.iterations_used == 1
    assert result.hit_iteration_limit is False
    assert llm.call_count == 1


def test_single_tool_call_then_final_answer():
    llm = _ScriptedFakeLLM([
        CompletionResult(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "hello"})],
            finish_reason="tool_calls",
        ),
        CompletionResult(content="Done, I echoed hello.", finish_reason="stop"),
    ])
    loop = AgentLoop(llm, ToolRegistry([_EchoTool()]))

    result = loop.run([{"role": "user", "content": "echo hello"}])

    assert result.final_answer == "Done, I echoed hello."
    assert result.iterations_used == 2
    assert llm.call_count == 2

    second_call_messages = llm.received_messages[1]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "echoed: hello"


def test_multiple_tool_calls_in_one_turn():
    llm = _ScriptedFakeLLM([
        CompletionResult(
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="echo", arguments={"text": "a"}),
                ToolCall(id="call_2", name="echo", arguments={"text": "b"}),
            ],
            finish_reason="tool_calls",
        ),
        CompletionResult(content="Both done.", finish_reason="stop"),
    ])
    loop = AgentLoop(llm, ToolRegistry([_EchoTool()]))

    result = loop.run([{"role": "user", "content": "echo a and b"}])

    tool_messages = [m for m in result.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2
    assert {m["content"] for m in tool_messages} == {"echoed: a", "echoed: b"}


def test_unknown_tool_call_produces_error_message_not_crash():
    llm = _ScriptedFakeLLM([
        CompletionResult(
            content=None,
            tool_calls=[ToolCall(id="call_1", name="nonexistent_tool", arguments={})],
            finish_reason="tool_calls",
        ),
        CompletionResult(content="I couldn't find that tool.", finish_reason="stop"),
    ])
    loop = AgentLoop(llm, ToolRegistry([_EchoTool()]))

    result = loop.run([{"role": "user", "content": "do something"}])

    assert result.final_answer == "I couldn't find that tool."
    tool_message = next(m for m in result.messages if m.get("role") == "tool")
    assert "unknown tool" in tool_message["content"]


def test_hits_iteration_limit_if_model_never_stops_calling_tools():
    infinite_tool_call = CompletionResult(
        content=None,
        tool_calls=[ToolCall(id="call_x", name="echo", arguments={"text": "loop"})],
        finish_reason="tool_calls",
    )
    llm = _ScriptedFakeLLM([infinite_tool_call] * 10)
    loop = AgentLoop(llm, ToolRegistry([_EchoTool()]), max_iterations=3)

    result = loop.run([{"role": "user", "content": "loop forever"}])

    assert result.final_answer is None
    assert result.hit_iteration_limit is True
    assert result.iterations_used == 3
    assert llm.call_count == 3
