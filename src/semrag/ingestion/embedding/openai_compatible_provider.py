"""
OpenAI-compatible embedding provider.

**Status: real, complete HTTP client code against the standard
`POST /v1/embeddings` contract (`{"model": ..., "input": [...]}` ->
`{"data": [{"embedding": [...], "index": ...}, ...]}`) that Ollama, vLLM,
and llama.cpp's OpenAI-compatibility servers all implement — but NOT run
against a live server in this delivery.** Same honest disclosure as Part
7's `Neo4jGraphStore`: no local model server was reachable in this
sandbox (Section 1 of this part's design doc). This is the production
path — point `base_url` at any locally-running OpenAI-compatible
embeddings endpoint and it should work as written; verify with
`tests/unit/embedding/test_openai_compatible_provider.py` (currently
skipped) before relying on it.

Retries and timeouts are real, not decorative: a local model server under
load (batch embedding a large project) can legitimately be slow, and a
transient connection error on a long-running local ingestion job
shouldn't abort the whole run — exponential backoff with a bounded retry
count, same shape as Part 2's git subprocess error handling philosophy
(fail the specific operation clearly, don't crash the whole pipeline).
"""
from __future__ import annotations

import time

import requests


class EmbeddingProviderError(RuntimeError):
    pass


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        dimension: int,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._max_retries = max_retries

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                response = requests.post(
                    f"{self._base_url}/v1/embeddings",
                    json={"model": self._model, "input": texts},
                    headers=headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                payload = response.json()
                break
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
                continue
        else:
            raise EmbeddingProviderError(
                f"Failed to reach embedding endpoint {self._base_url} after "
                f"{self._max_retries} attempts: {last_error}"
            )

        # Sort by `index` — most OpenAI-compatible servers preserve input
        # order, but the API contract only guarantees it via this field,
        # not positionally, and trusting positional order silently would
        # be a real, hard-to-notice correctness bug if a server ever
        # doesn't preserve it.
        data = sorted(payload["data"], key=lambda item: item["index"])
        embeddings = [item["embedding"] for item in data]

        if len(embeddings) != len(texts):
            raise EmbeddingProviderError(
                f"Expected {len(texts)} embeddings, got {len(embeddings)} from {self._base_url}"
            )
        for emb in embeddings:
            if len(emb) != self._dimension:
                raise EmbeddingProviderError(
                    f"Embedding dimension mismatch: configured {self._dimension}, "
                    f"server returned {len(emb)}. Check EMBEDDING_DIMENSION in "
                    f"Part 6's pgvector_models.py matches this provider's model."
                )

        return embeddings
