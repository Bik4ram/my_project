"""
Ollama native chat completion client.

Ollama's `/api/chat` endpoint is the only reliable path on Ollama 0.3x
builds (verified against 0.32.5, where `/v1/chat/completions` returns 404
despite `/v1/models` returning 200). It accepts the same `messages` and
`tools` shape OpenAI uses, but the response and streaming framing differ:

- Non-streaming: response is `{ message: {...}, done: true, ... }` — no
  `choices` wrapper.
- Streaming: one JSON object per line (NDJSON), each carrying a delta in
  `message.content` and any partial tool calls.

This client flattens both into the same `CompletionResult` / `StreamChunk`
shape the rest of the system consumes, so `AgentLoop` doesn't need to
care which provider it ran against.
"""
from __future__ import annotations

import json

from semrag.core.observability import count, get_logger, timer
from semrag.reasoning.llm._openai_compatible_impl import (
    LLMProviderError,
    OpenAICompatibleLLM,
    _post_with_retries_shared,
)
from semrag.reasoning.llm.interfaces import CompletionResult, StreamChunk, ToolCall

logger = get_logger(__name__)


class OllamaLLM(OpenAICompatibleLLM):
    """Ollama `/api/chat` client.

    Reuses the OpenAI-compatible base for retries, headers, and timeout
    handling — only the endpoint URL, request body, and response parsing
    differ."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 3,
        num_predict: int | None = None,
    ):
        super().__init__(base_url, model, api_key, timeout_seconds, max_retries)
        # Ollama's ``options.num_predict`` caps the number of tokens the
        # model will emit. Sending it explicitly is cheaper than relying
        # on the model's own stop tokens, which a 7B local model
        # sometimes ignores (it loops on a question, rephrases it, then
        # keeps going). ``None`` means "let the model decide" — used by
        # tests; production sets this from ``Settings.llm_num_predict``.
        self._num_predict = num_predict

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _endpoint(self, stream: bool) -> str:
        # Just the path — the shared helper joins it with base_url.
        return "/api/chat"

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> CompletionResult:
        payload = self._build_payload(messages, tools, stream=False)
        with timer("llm.complete.duration_ms", provider=self.provider_name, model=self._model):
            response = _post_with_retries_shared(
                self._base_url,
                self._endpoint(False),
                self._headers(),
                payload,
                self._timeout,
                self._max_retries,
                False,
                self.provider_name,
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMProviderError(f"Malformed Ollama response from {self._base_url}: {exc}")
        message = body.get("message") or {}
        openai_message = self._ollama_to_openai_message(message)
        result = OpenAICompatibleLLM._parse_message(
            openai_message, finish_reason=body.get("done_reason", "stop")
        )
        # Ollama reports token usage as `prompt_eval_count` / `eval_count`
        # rather than OpenAI's `usage.*` block. Map the same fields and
        # emit the same metrics so dashboards don't need provider-specific
        # tag values.
        prompt_tokens = body.get("prompt_eval_count")
        completion_tokens = body.get("eval_count")
        if prompt_tokens is not None or completion_tokens is not None:
            total = (prompt_tokens or 0) + (completion_tokens or 0)
            result.prompt_tokens = prompt_tokens
            result.completion_tokens = completion_tokens
            result.total_tokens = total if total else None
            count("llm.tokens.total", total, provider=self.provider_name, model=self._model)
            if prompt_tokens is not None:
                count("llm.tokens.prompt", prompt_tokens, provider=self.provider_name, model=self._model)
            if completion_tokens is not None:
                count("llm.tokens.completion", completion_tokens, provider=self.provider_name, model=self._model)
        return result

    # ------------------------------------------------------------------
    # Streaming (NDJSON, not SSE)
    # ------------------------------------------------------------------

    def stream(self, messages: list[dict], tools: list[dict] | None = None):
        payload = self._build_payload(messages, tools, stream=True)
        with timer("llm.stream.duration_ms", provider=self.provider_name, model=self._model):
            response = _post_with_retries_shared(
                self._base_url,
                self._endpoint(True),
                self._headers(),
                payload,
                self._timeout,
                self._max_retries,
                True,
                self.provider_name,
            )

        pending_tool_calls: dict[int, dict] = {}
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            try:
                event = json.loads(line)
            except ValueError:
                continue

            message = event.get("message") or {}
            delta_content = message.get("content")
            delta_tool_calls_out: list[ToolCall] = []

            for tc_delta in message.get("tool_calls") or []:
                idx = tc_delta.get("index", 0)
                entry = pending_tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if tc_delta.get("id"):
                    entry["id"] = tc_delta["id"]
                fn = tc_delta.get("function", {}) or {}
                if fn.get("name"):
                    entry["name"] = fn["name"]
                if fn.get("arguments") is not None:
                    args = fn["arguments"]
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    entry["arguments"] += str(args)

            finish_reason = event.get("done_reason")
            if event.get("done") and pending_tool_calls:
                for entry in pending_tool_calls.values():
                    try:
                        parsed_args = json.loads(entry["arguments"]) if entry["arguments"] else {}
                    except ValueError:
                        parsed_args = {}
                    delta_tool_calls_out.append(
                        ToolCall(id=entry["id"] or "", name=entry["name"] or "", arguments=parsed_args)
                    )

            yield StreamChunk(
                delta_content=delta_content,
                delta_tool_calls=delta_tool_calls_out,
                finish_reason=finish_reason,
            )
            if event.get("done"):
                return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_payload(self, messages: list[dict], tools: list[dict] | None, *, stream: bool) -> dict:
        """Ollama accepts the same `messages`/`tools` shape as OpenAI; pass
        through only the fields it understands."""
        payload: dict = {"model": self._model, "messages": messages, "stream": stream}
        if tools:
            payload["tools"] = tools
        if self._num_predict is not None:
            # ``options.num_predict`` is the Ollama-native spelling of
            # OpenAI's ``max_tokens``. Capping the output length is the
            # single biggest latency lever against qwen2.5:7b on CPU —
            # an uncapped response can run 60+ seconds generating
            # bullet points the user never asked for.
            payload["options"] = {"num_predict": self._num_predict}
        return payload

    @staticmethod
    def _ollama_to_openai_message(ollama_message: dict) -> dict:
        """Flatten Ollama's `message` into the OpenAI-shaped dict the
        shared parser expects (`role`, `content`, `tool_calls`)."""
        tool_calls: list[dict] = []
        for tc in ollama_message.get("tool_calls") or []:
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments", {})
            if isinstance(args, dict):
                args_str = json.dumps(args)
            else:
                args_str = str(args)
            tool_calls.append(
                {
                    "id": tc.get("id", f"call_{len(tool_calls)}"),
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args_str},
                }
            )
        return {
            "role": ollama_message.get("role", "assistant"),
            "content": ollama_message.get("content"),
            "tool_calls": tool_calls,
        }
