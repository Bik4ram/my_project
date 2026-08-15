"""
Agent loop.

Standard tool-calling ReAct-style loop: call the LLM with the current
messages and available tool schemas; if it returns tool calls, execute
each via the ToolRegistry, append the results as `tool` role messages,
and call again; stop when the LLM returns a plain answer (no tool calls)
or max_iterations is hit. max_iterations is a hard cap specifically
because a model can get stuck calling tools indefinitely (e.g. repeatedly
searching without ever synthesizing an answer) - bounding it means a
misbehaving model degrades to a clear "ran out of turns" result instead
of an unbounded loop.

Testable without a live model via any LLMProvider - including a scripted
fake that returns tool calls then a final answer, which is how this
module's own tests exercise the loop's control flow for real, the same
class of test Part 9 used for its embedding cache pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from semrag.reasoning.agents.tools import ToolRegistry
from semrag.reasoning.llm.interfaces import CompletionResult, LLMProvider

DEFAULT_MAX_ITERATIONS = 8

_logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    final_answer: str | None
    messages: list[dict] = field(default_factory=list)
    iterations_used: int = 0
    hit_iteration_limit: bool = False


class AgentLoop:
    def __init__(self, llm: LLMProvider, tools: ToolRegistry, max_iterations: int = DEFAULT_MAX_ITERATIONS):
        self._llm = llm
        self._tools = tools
        self._max_iterations = max_iterations

    def run(self, messages: list[dict]) -> AgentResult:
        from semrag.core.observability import timer, count, trace

        conversation = list(messages)
        tool_schemas = self._tools.schemas()

        with timer("agent.run.duration_ms", tools=len(tool_schemas)):
            for iteration in range(1, self._max_iterations + 1):
                _logger.debug(
                    "agent iteration start",
                    extra={"iteration": iteration, "max_iterations": self._max_iterations, "messages_so_far": len(conversation)},
                )
                result: CompletionResult = self._llm.complete(conversation, tools=tool_schemas or None)

                if not result.tool_calls:
                    trace(
                        _logger,
                        "agent iteration final answer",
                        iteration=iteration,
                        content_length=len(result.content or ""),
                        finish_reason=result.finish_reason,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                    )
                    conversation.append({"role": "assistant", "content": result.content or ""})
                    count("agent.run.iterations", iteration)
                    count("agent.run.final_answer", 1)
                    return AgentResult(
                        final_answer=result.content,
                        messages=conversation,
                        iterations_used=iteration,
                    )

                tool_names = [tc.name for tc in result.tool_calls]
                trace(
                    _logger,
                    "agent iteration tool calls",
                    iteration=iteration,
                    tool_calls=tool_names,
                    finish_reason=result.finish_reason,
                )
                # Drop the model's prose echo when it also emitted tool
                # calls — a small local model (qwen2.5:7b) often fills
                # ``content`` with garbage ("Ronaldo", random tokens)
                # alongside a valid tool call, and replaying that prose
                # back into the next turn's context derails the next
                # completion. Tool calls carry the intent; the prose
                # before them is noise in that case.
                conversation.append({
                    "role": "assistant",
                    "content": "" if not (result.content and result.content.strip()) else result.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in result.tool_calls
                    ],
                })

                for tool_call in result.tool_calls:
                    tool_result = self._tools.execute(tool_call.name, tool_call.arguments)
                    trace(
                        _logger,
                        "agent tool result",
                        iteration=iteration,
                        tool=tool_call.name,
                        result_length=len(tool_result),
                    )
                    conversation.append({"role": "tool", "tool_call_id": tool_call.id, "content": tool_result})

            count("agent.run.iterations", self._max_iterations)
            count("agent.run.iteration_limit_hit", 1)
            _logger.warning(
                "agent hit iteration limit",
                extra={"iterations": self._max_iterations, "messages_so_far": len(conversation)},
            )

        return AgentResult(
            final_answer=None, messages=conversation,
            iterations_used=self._max_iterations, hit_iteration_limit=True,
        )
