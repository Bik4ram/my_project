#!/usr/bin/env bash
# Start the fully-offline SemRAG stack on the OFFLINE machine:
#   1. load .env into the process environment (so the validation runner sees
#      SEMRAG_UVM_ROOT and the MCP server sees its timeout vars — pydantic-
#      settings reads .env for Settings fields but does NOT re-export them),
#   2. run alembic migrations against PostgreSQL,
#   3. confirm local Ollama + required models are present,
#   4. start the API server (uvicorn) and the dedicated indexing worker.
#
#   scripts/start_offline.sh            start everything
#   scripts/start_offline.sh --stop     stop everything
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
LOG_DIR="$REPO_ROOT/logs"
API_LOG="$LOG_DIR/api.log"
WORKER_LOG="$LOG_DIR/worker.log"
API_PIDFILE="$LOG_DIR/api.pid"
WORKER_PIDFILE="$LOG_DIR/worker.pid"

stop_stack() {
    echo "==> stopping SemRAG stack"
    for pidfile in "$API_PIDFILE" "$WORKER_PIDFILE"; do
        if [ -f "$pidfile" ]; then
            pid="$(cat "$pidfile")"
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
                echo "    stopped pid $pid ($pidfile)"
            fi
            rm -f "$pidfile"
        fi
    done
    exit 0
}

if [ "${1:-}" = "--stop" ]; then
    stop_stack
fi

if [ ! -x "$PYTHON" ]; then
    echo "error: venv python not found at $PYTHON (run scripts/install_offline.sh)" >&2
    exit 1
fi

echo "==> [1/5] Loading .env into the environment"
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
    echo "    loaded .env (SEMRAG_LLM_MODEL=${SEMRAG_LLM_MODEL:-unset})"
else
    echo "    warning: no .env found; using compiled-in defaults" >&2
fi

echo "==> [2/5] Checking PostgreSQL"
if ! "$PYTHON" -c "import psycopg2; psycopg2.connect(dbname='semrag', user='semrag', password='semrag', host='localhost', port=5432).close(); print('    reachable')" 2>/dev/null; then
    echo "    warning: could not reach the configured database; migrations will fail" >&2
fi

echo "==> [3/5] Running alembic migrations (idempotent)"
"$PYTHON" -m alembic upgrade head || {
    echo "error: alembic upgrade failed. Ensure scripts/setup_postgres.sh ran." >&2
    exit 1
}

echo "==> [4/5] Checking local Ollama and required models"
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "error: Ollama is not reachable on localhost:11434 (start it first)" >&2
    exit 1
fi
tags="$(curl -s http://localhost:11434/api/tags)"
for model in "${SEMRAG_LLM_MODEL:-qwen3-coder:30b}" "${SEMRAG_EMBEDDING_MODEL:-nomic-embed-text}"; do
    if printf '%s' "$tags" | grep -qF "\"$model\""; then
        echo "    OK: $model present"
    else
        echo "    MISSING: $model not in local Ollama (run scripts/import_models_offline.sh)" >&2
    fi
done

mkdir -p "$LOG_DIR"

echo "==> [5/5] Starting API server and worker"
nohup "$PYTHON" -m uvicorn semrag.api.app:app \
    --host 127.0.0.1 --port 8000 \
    >> "$API_LOG" 2>&1 &
echo $! > "$API_PIDFILE"
echo "    API server pid $(cat "$API_PIDFILE") -> $API_LOG"

nohup "$PYTHON" -m semrag.worker \
    >> "$WORKER_LOG" 2>&1 &
echo $! > "$WORKER_PIDFILE"
echo "    worker pid $(cat "$WORKER_PIDFILE") -> $WORKER_LOG"

echo
echo "Waiting for API readiness ..."
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:8000/livez >/dev/null 2>&1; then
        echo "    API is up: http://127.0.0.1:8000"
        break
    fi
    sleep 1
done

echo
echo "Stack started. Health:   curl http://127.0.0.1:8000/livez"
echo "Stop everything with:   scripts/start_offline.sh --stop"