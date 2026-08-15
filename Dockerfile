# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# SemRAG production image
# Multi-stage build: compile deps in a builder, copy only what's needed
# into a slim runtime image. Non-root user. Pinned base for reproducibility.
# ---------------------------------------------------------------------------

ARG PYTHON_VERSION=3.12
ARG DEBIAN_RELEASE=bookworm

# -------- builder -----------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build-time deps for psycopg2-binary's manylinux wheels are bundled, but
# blake3 / pgvector may need a compiler fallback. Keep the surface narrow.
RUN apt-get update \
 && apt-get install --no-install-recommends -y build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# -------- runtime -----------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_RELEASE} AS runtime

LABEL org.opencontainers.image.title="semrag" \
      org.opencontainers.image.description="Semiconductor RAG platform" \
      org.opencontainers.image.source="https://example.invalid/semrag"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# tini for clean signal handling of uvicorn workers.
RUN apt-get update \
 && apt-get install --no-install-recommends -y tini curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN groupadd --system --gid 1001 semrag \
 && useradd --system --uid 1001 --gid semrag --create-home --shell /bin/bash semrag

WORKDIR /app

# Copy installed packages from the builder.
COPY --from=builder /install /usr/local

# Copy only what's needed at runtime.
COPY pyproject.toml ./
COPY src ./src

# Install the package itself (no deps — they came from the builder).
RUN pip install --no-cache-dir --no-deps . \
 && chown -R semrag:semrag /app

USER semrag
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "uvicorn", "semrag.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
