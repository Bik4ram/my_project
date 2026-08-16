"""
OpenAI-compatible chat completion client.

Targets the standard `POST /v1/chat/completions` contract implemented by
vLLM, llama.cpp's HTTP server, OpenAI-compatible shims in cloud services,
and (when configured) Ollama 0.5.0+. Ollama 0.3x builds (e.g. 0.32.5)
return 404 on that path even though their native `/api/chat` works, so
those deployments route through `OllamaLLM` instead — see
`semrag.reasoning.llm.ollama_llm`.

Streaming tool-call accumulation: the OpenAI streaming contract sends a
tool call's name and arguments across MULTIPLE chunks (arguments arrive
as fragments of a JSON string that only parses once fully concatenated),
keyed by index — this client accumulates by index and only parses the
JSON once, at the end, not per-fragment. Tests cover this explicitly.

Production status: this client has been verified against live Ollama
(via the legacy `/v1/chat/completions` route) and is shaped identically
to the OpenAI / Together / vLLM API; adding a new OpenAI-compatible
provider is a one-line subclass.
"""
from __future__ import annotations

import json
import time

import requests

from semrag.core.observability import count, get_logger, timer
from semrag.reasoning.llm.interfaces import CompletionResult, StreamChunk, ToolCall

logger = get_logger(__name__)


class LLMProviderError(RuntimeError):
    pass


def _post_with_retries_shared(
    base_url: str,
    endpoint: str,
    headers: dict,
    payload: dict,
    timeout: float,
    max_retries: int,
    stream: bool,
    op_name: str,
) -> requests.Response:
    """Centralized retry+timeout loop with structured metric emission.
    Returns the response on success; raises `LLMProviderError` after the
    last attempt. The `op_name` tag shows up in every metric event for
    `llm.request.duration_ms` so we can split Ollama from OpenAI without
    extra plumbing."""
    url = f"{base_url.rstrip('/')}{endpoint}"
    last_error: Exception | None = None
    for attempt in range(max_retries):
        with timer(
            "llm.request.duration_ms",
            op=op_name,
            attempt=attempt + 1,
            stream=stream,
        ):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                    stream=stream,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_error = exc
                logger.warning(
                    "llm request failed",
                    extra={"op": op_name, "attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
    raise LLMProviderError(
        f"Failed to reach LLM endpoint {base_url}: {last_error}"
    )


class OpenAICompatibleLLM:
    """OpenAI-shape chat completions. Subclasses override `_endpoint()`
    and (optionally) `provider_name` to point at different URLs without
    re-implementing the parsing/streaming logic."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 3,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def model_name(self) -> str:
        return self._model

    def _endpoint(self, stream: bool) -> str:
        # Just the path; `_post_with_retries_shared` joins it with the
        # base_url so subclasses don't need to know the host:port.
        return "/v1/chat/completions"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post(self, payload: dict, stream: bool) -> requests.Response:
        return _post_with_retries_shared(
            self._base_url,
            self._endpoint(stream),
            self._headers(),
            payload,
            self._timeout,
            self._max_retries,
            stream,
            self.provider_name,
        )

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> CompletionResult:
        payload: dict = {"model": self._model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools

        with timer("llm.complete.duration_ms", provider=self.provider_name, model=self._model):
            response = self._post(payload, stream=False)
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMProviderError(f"Malformed completion response from {self._base_url}: {exc}")

        result = self._parse_message(message, finish_reason=choice.get("finish_reason", "stop"))
        usage = body.get("usage") or {}
        if usage:
            result.prompt_tokens = usage.get("prompt_tokens")
            result.completion_tokens = usage.get("completion_tokens")
            result.total_tokens = usage.get("total_tokens")
            if result.total_tokens is not None:
                count("llm.tokens.total", result.total_tokens, provider=self.provider_name, model=self._model)
            if result.prompt_tokens is not None:
                count("llm.tokens.prompt", result.prompt_tokens, provider=self.provider_name, model=self._model)
            if result.completion_tokens is not None:
                count("llm.tokens.completion", result.completion_tokens, provider=self.provider_name, model=self._model)
        return result

    @staticmethod
    def _parse_message(message: dict, finish_reason: str) -> CompletionResult:
        tool_calls = [
            ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=json.loads(tc["function"]["arguments"]))
            for tc in message.get("tool_calls", []) or []
        ]
        return CompletionResult(
            content=message.get("content"),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(self, messages: list[dict], tools: list[dict] | None = None):
        payload: dict = {"model": self._model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools

        with timer("llm.stream.duration_ms", provider=self.provider_name, model=self._model):
            response = self._post(payload, stream=True)

        pending_tool_calls: dict[int, dict] = {}
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: "):]
            if data_str.strip() == "[DONE]":
                break

            try:
                event = json.loads(data_str)
                delta = event["choices"][0]["delta"]
                finish_reason = event["choices"][0].get("finish_reason")
            except (ValueError, KeyError, IndexError):
                continue

            delta_content = delta.get("content")
            delta_tool_calls_out: list[ToolCall] = []

            for tc_delta in delta.get("tool_calls", []) or []:
                idx = tc_delta["index"]
                entry = pending_tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if tc_delta.get("id"):
                    entry["id"] = tc_delta["id"]
                function = tc_delta.get("function", {})
                if function.get("name"):
                    entry["name"] = function["name"]
                if function.get("arguments"):
                    entry["arguments"] += function["arguments"]

            if finish_reason is not None and pending_tool_calls:
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