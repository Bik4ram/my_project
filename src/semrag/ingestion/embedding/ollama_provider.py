"""
Ollama-native embedding client.

Ollama's native `/api/embeddings` endpoint (and its successor `/api/embed`)
serves embeddings. The contract is single-input per request by default
(no native batching), so we issue one request per text and let the
caller batch into a `pool.imap_unordered` if it needs concurrency.

Why a separate provider instead of routing through
`OpenAICompatibleEmbeddingProvider`: Ollama 0.3x's OpenAI-compat shim may
not expose `/v1/embeddings` reliably even when the native endpoint
responds, and the request shape (`{"model": ..., "prompt": ...}` vs the
OpenAI `{"model": ..., "input": [...]}`) differs enough that a single
client with a branch in every method would be harder to reason about
than two small classes.
"""
from __future__ import annotations

import time

import requests

from semrag.core import metrics as metrics_module
from semrag.core.observability import get_logger, timer
from semrag.ingestion.embedding.interfaces import EmbeddingProvider

logger = get_logger(__name__)


class EmbeddingProviderError(RuntimeError):
    pass


class EmbeddingContextExceeded(EmbeddingProviderError):
    """Raised when the model rejects the input as exceeding its context.

    Ollama returns ``500 "the input length exceeds the context length"``
    for over-long prompts instead of silently truncating. This is a
    recoverable condition: the caller can retry with a shorter prefix."""


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        min_input_chars: int = 128,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._min_input_chars = min_input_chars

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(len(texts), 8)) as pool:
            futures = {pool.submit(self._embed_one, t): i for i, t in enumerate(texts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:  # noqa: BLE001 — surface to caller with index context
                    raise EmbeddingProviderError(f"embed_texts[{idx}] failed: {exc}") from exc
        return [r for r in results if r is not None]  # type: ignore[misc]

    def _request_embedding(self, text: str) -> list[float]:
        with timer("embedding.request.duration_ms", provider="ollama", model=self._model, attempt=1):
            request_start = time.perf_counter()
            response = requests.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
                timeout=self._timeout,
            )
            if response.status_code == 500 and "exceeds the context length" in response.text:
                raise EmbeddingContextExceeded(
                    f"ollama rejected input exceeding its context length ({len(text)} chars)"
                )
            response.raise_for_status()
            payload = response.json()
            emb = payload.get("embedding") or []
            # Wave 3: also feed the Prometheus embedding
            # latency histogram. The existing `timer` call
            # covers the structured log path; the histogram is
            # the Grafana-visible one.
            metrics_module.observe_embedding(
                provider="ollama",
                duration_s=time.perf_counter() - request_start,
            )
            if len(emb) != self._dimension:
                raise EmbeddingProviderError(
                    f"Embedding dimension mismatch: configured {self._dimension}, "
                    f"server returned {len(emb)} for model {self._model!r}."
                )
            return list(emb)

    def _embed_one(self, text: str) -> list[float]:
        last_error: Exception | None = None

        # The model rejects input that exceeds its context window instead
        # of truncating it. Rather than failing the whole indexing job,
        # embed a progressively shorter prefix until one fits (bounded
        # halving). Embedding models truncate internally anyway, and the
        # full chunk text is still stored for search/grounding.
        candidate = text
        while len(candidate) > self._min_input_chars:
            try:
                return self._request_embedding(candidate)
            except EmbeddingContextExceeded as exc:
                last_error = exc
                logger.warning(
                    "embedding input truncated (context length)",
                    extra={"provider": "ollama", "length": len(candidate)},
                )
                candidate = candidate[: len(candidate) // 2]

        # At the floor, fall through to the original transient-error retry
        # loop; a context-exceeded rejection at the floor is terminal.
        for attempt in range(self._max_retries):
            try:
                return self._request_embedding(candidate)
            except EmbeddingContextExceeded as exc:
                last_error = exc
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "embedding request failed",
                    extra={"provider": "ollama", "attempt": attempt + 1, "error": str(exc)},
                )
                if attempt < self._max_retries - 1:
                    time.sleep(2**attempt)
        raise EmbeddingProviderError(f"ollama embeddings failed after {self._max_retries} attempts: {last_error}")