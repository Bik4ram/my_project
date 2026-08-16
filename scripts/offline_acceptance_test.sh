#!/usr/bin/env bash
# Offline acceptance test for the SemRAG + OpenCode + Ollama stack.
#
# Simulates a fully air-gapped machine WITHOUT root: every Python process it
# starts runs with a sitecustomize that refuses any network connect to a
# non-loopback address. The API server, the indexing worker, and the MCP
# server are exercised under that block, so if any of them tried to reach an
# external host the test fails immediately.
#
# Usage:
#   scripts/offline_acceptance_test.sh [--port 8001] [--skip-llm]
#
# Exit code 0 only when every non-optional check passes. LLM-driven checks
# that are blocked by a missing qwen3-coder:30b are reported as BLOCKED (not
# silently passed).
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
PORT=8001
SKIP_LLM=0
BASE_URL="http://127.0.0.1:$PORT"
API_KEY="${SEMRAG_API_KEY:-dev-local-key}"
AUTH="Authorization: Bearer $API_KEY"
JSON_CT="Content-Type: application/json"
BLOCK_DIR="$(mktemp -d)"
LOG_DIR="$REPO_ROOT/logs"

PASS=0
FAIL=0
BLOCKED=0

note()  { printf '  %s\n' "$*"; }
ok()    { PASS=$((PASS+1)); printf 'PASS  %s\n' "$*"; }
bad()   { FAIL=$((FAIL+1)); printf 'FAIL  %s\n' "$*"; }
block() { BLOCKED=$((BLOCKED+1)); printf 'BLOCK %s\n' "$*"; }

api() { # api <method> <path> [json-body] [timeout]
    local method="$1" path="$2" body="${3:-}" timeout="${4:-60}"
    if [ -n "$body" ]; then
        curl -s -X "$method" -H "$AUTH" -H "$JSON_CT" \
            --max-time "$timeout" "$BASE_URL$path" -d "$body"
    else
        curl -s -X "$method" -H "$AUTH" --max-time "$timeout" "$BASE_URL$path"
    fi
}

if [ ! -x "$PYTHON" ]; then
    echo "error: venv python not found at $PYTHON" >&2
    exit 2
fi

for arg in "$@"; do
    case "$arg" in
        --port) PORT="$2"; shift 2 ;;
        --skip-llm) SKIP_LLM=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# 0. Build the network-blocking sitecustomize and the offline driver env.
# ---------------------------------------------------------------------------
cat > "$BLOCK_DIR/sitecustomize.py" <<'PYEOF'
"""Refuse every non-loopback network connect. Loaded by site.py at
interpreter startup via PYTHONPATH, so ALL Python processes spawned by the
acceptance test (API server, worker, MCP driver) are air-gapped."""
import socket as _socket

_orig_connect = _socket.socket.connect
_orig_connect_ex = _socket.socket.connect_ex
_orig_create_connection = _socket.create_connection
_orig_ssl_connect = None
try:
    import ssl as _ssl
    _orig_ssl_connect = _ssl.SSLSocket.connect
except Exception:  # pragma: no cover
    pass

_LOOPBACK = ("127.0.0.1", "::1", "localhost")

def _is_loopback(host):
    if not isinstance(host, str):
        host = str(host)
    host = host.split("%")[0].lower()
    return host in _LOOPBACK

def _guard(host, name):
    if not _is_loopback(host):
        raise OSError(
            f"[offline-sim] {name} blocked: cannot reach non-loopback host "
            f"{host!r}"
        )

def _patched_connect(self, address):
    host = address[0] if isinstance(address, tuple) else address
    _guard(host, "connect")
    return _orig_connect(self, address)

def _patched_connect_ex(self, address):
    host = address[0] if isinstance(address, tuple) else address
    _guard(host, "connect_ex")
    return _orig_connect_ex(self, address)

def _patched_create_connection(address, *args, **kwargs):
    host = address[0] if isinstance(address, tuple) else address
    _guard(host, "create_connection")
    return _orig_create_connection(address, *args, **kwargs)

_socket.socket.connect = _patched_connect
_socket.socket.connect_ex = _patched_connect_ex
_socket.create_connection = _patched_create_connection
if _orig_ssl_connect is not None:
    try:
        _ssl.SSLSocket.connect = _patched_connect
    except Exception:  # pragma: no cover
        pass
PYEOF

export PYTHONPATH="$BLOCK_DIR"

# Self-check: the block must actually be active in every Python child.
if "$PYTHON" - <<'PYEOF' 2>/dev/null
import socket
try:
    socket.create_connection(("example.com", 80), 5)
    raise SystemExit(1)
except OSError as exc:
    assert "offline-sim" in str(exc), exc
print("offline block active")
PYEOF
then
    note "offline socket block verified active"
else
    echo "error: network block not active in Python subprocesses" >&2
    exit 2
fi

echo "== offline acceptance test (base=$BASE_URL, skip_llm=$SKIP_LLM) =="

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
echo "--- prereqs"
if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    ok "Ollama reachable on localhost:11434"
else
    bad "Ollama not reachable (start 'ollama serve' first)"
fi

tags="$(curl -s http://localhost:11434/api/tags || true)"
for model in qwen3-coder:30b nomic-embed-text; do
    if printf '%s' "$tags" | grep -qF "$model"; then
        ok "required model present: $model"
    else
        block "required model missing: $model (run scripts/import_models_offline.sh)"
    fi
done

# ---------------------------------------------------------------------------
# 2. Static network audit of the runtime surface (src + MCP + scripts)
# ---------------------------------------------------------------------------
echo "--- network audit"
# Only allowlist: the optional OpenAI provider's default base URL, which is
# inert on the ollama runtime path. Everything else must be loopback.
audit_allow="https://api.openai.com"
audit_bad="$(grep -rhoE 'https?://[A-Za-z0-9._-]+' \
    --include='*.py' --include='*.sh' --include='*.json' \
    src/ semrag_mcp_server.py scripts/ \
    | sort -u \
    | grep -vE '^https?://(127\.0\.0\.1|localhost|::1)' \
    | grep -vF "$audit_allow" || true)"
if [ -z "$audit_bad" ]; then
    ok "no non-loopback URLs in the runtime surface (only the inert $audit_allow default)"
else
    bad "non-loopback URLs found in runtime surface: $(echo "$audit_bad" | tr '\n' ' ')"
fi

# ---------------------------------------------------------------------------
# 3. Start the AIR-GAPPED API + worker on PORT
# ---------------------------------------------------------------------------
echo "--- starting air-gapped stack on port $PORT"
mkdir -p "$LOG_DIR"
# Load .env into the environment (so SEMRAG_UVM_ROOT etc. propagate).
set -a; [ -f .env ] && . ./.env; set +a

# shellcheck disable=SC2155
export SEMRAG_API_RUNS_WORKER=false
export SEMRAG_LLM_PROBE_AT_STARTUP=false
export SEMRAG_TRACING_ENABLED=false

nohup env PYTHONPATH="$BLOCK_DIR" "$PYTHON" -m uvicorn semrag.api.app:app \
    --host 127.0.0.1 --port "$PORT" \
    >> "$LOG_DIR/offline_test_api.log" 2>&1 &
API_PID=$!

nohup env PYTHONPATH="$BLOCK_DIR" "$PYTHON" -m semrag.worker \
    >> "$LOG_DIR/offline_test_worker.log" 2>&1 &
WORKER_PID=$!

cleanup() {
    kill "$API_PID" "$WORKER_PID" 2>/dev/null || true
    wait "$API_PID" 2>/dev/null || true
    wait "$WORKER_PID" 2>/dev/null || true
    rm -rf "$BLOCK_DIR"
}
trap cleanup EXIT

for _ in $(seq 1 40); do
    if curl -s --max-time 2 "$BASE_URL/livez" >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if curl -s --max-time 2 "$BASE_URL/livez" >/dev/null 2>&1; then
    ok "air-gapped API server up on $BASE_URL"
else
    bad "air-gapped API server failed to start (see $LOG_DIR/offline_test_api.log)"
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. API checks (all under the network block)
# ---------------------------------------------------------------------------
echo "--- API checks"

info="$(api GET /v1/projects/rtl_uvm/info)"
info_ns="$(printf '%s' "$info" | tr -d ' ')"
if printf '%s' "$info_ns" | grep -q '"project_id":"rtl_uvm"'; then
    ok "project info for rtl_uvm"
else
    bad "project info: $info"
fi

q="$(api POST /v1/projects/rtl_uvm/query '{"query":"i2c master read test","top_k":5}')"
if printf '%s' "$q" | grep -q '"messages"'; then
    ok "retrieval-only /query returns results (no LLM needed)"
else
    bad "retrieval /query failed: $(echo "$q" | head -c 200)"
fi

for target in rtl_lint sv_generated_lint; do
    r="$(api POST /v1/projects/rtl_uvm/validate "{\"target\":\"$target\"}")"
    r_ns="$(printf '%s' "$r" | tr -d ' ')"
    if printf '%s' "$r_ns" | grep -q '"status":"passed"'; then
        ok "validation target $target passed"
    else
        bad "validation target $target: $(echo "$r" | head -c 200)"
    fi
done

# Re-index end-to-end (DB + worker + embedding pipeline, fully offline).
# Use the project's registered root path so the right tree gets indexed.
project_root="$(printf '%s' "$info_ns" | grep -o '"root_path":"[^"]*"' | head -1 | cut -d'"' -f4 || true)"
if [ -z "$project_root" ]; then
    project_root="$REPO_ROOT"
    note "project /info returned no root_path; falling back to $REPO_ROOT"
fi
idx="$(api POST /v1/projects/rtl_uvm/index "{\"root_path\":\"$project_root\"}")"
idx_ns="$(printf '%s' "$idx" | tr -d ' ')"
job_id="$(printf '%s' "$idx_ns" | grep -o '"job_id":"[^"]*"' | head -1 | cut -d'"' -f4 || true)"
if [ -n "$job_id" ]; then
    ok "index job enqueued ($job_id)"
    state="queued"
    for _ in $(seq 1 120); do
        st="$(api GET "/v1/projects/rtl_uvm/jobs/$job_id")"
        state="$(printf '%s' "$st" | tr -d ' ' | grep -o '"state":"[^"]*"' | head -1 | cut -d'"' -f4 || true)"
        if [ "$state" = "completed" ] || [ "$state" = "failed" ] || [ "$state" = "cancelled" ]; then
            break
        fi
        sleep 2
    done
    if [ "$state" = "completed" ]; then
        ok "index job completed offline"
    else
        bad "index job ended state=$state"
    fi
else
    bad "index job not created: $(echo "$idx" | head -c 200)"
fi

# ---------------------------------------------------------------------------
# 5. MCP server under the block (the exact surface OpenCode uses)
# ---------------------------------------------------------------------------
echo "--- MCP server checks"
cat > "$LOG_DIR/offline_test_mcp_driver.py" <<'PYEOF'
import asyncio
import json
import sys

sys.path.insert(0, ".")
import semrag_mcp_server as srv

async def main():
    results = {}
    results["project_info"] = await srv.semrag_project_info("rtl_uvm")
    results["search"] = await srv.semrag_search("rtl_uvm", "i2c master read sequence", top_k=3)
    results["run_validation"] = await srv.semrag_run_validation("rtl_uvm", "rtl_lint")
    results["list_files"] = await srv.semrag_list_files("rtl_uvm")
    for name in ("project_info", "list_files"):
        try:
            json.loads(results[name])
            results[name] = "ok"
        except Exception:
            results[name] = "error:" + str(results[name])[:120]
    # semrag_search returns a compact TEXT summary ("Results returned: N").
    s = results["search"]
    if s.startswith("SEARCH RESULTS FOR:"):
        n = s.split("Results returned: ", 1)[-1].split("\n", 1)[0].strip()
        results["search"] = "ok" if n.isdigit() and int(n) > 0 else "empty"
    else:
        results["search"] = "error:" + s[:120]
    try:
        v = json.loads(results["run_validation"])
        results["run_validation"] = v.get("status", "unknown")
    except Exception:
        results["run_validation"] = "error:" + str(results["run_validation"])[:120]
    print(json.dumps(results))

asyncio.run(main())
PYEOF

if env PYTHONPATH="$BLOCK_DIR" SEMRAG_URL="$BASE_URL" SEMRAG_API_KEY="$API_KEY" \
    "$PYTHON" "$LOG_DIR/offline_test_mcp_driver.py" > "$LOG_DIR/offline_test_mcp_out.json" 2> "$LOG_DIR/offline_test_mcp_err.log"; then
    out="$(cat "$LOG_DIR/offline_test_mcp_out.json")"
    for key in project_info search run_validation list_files; do
        val="$(printf '%s' "$out" | "$PYTHON" -c "import json,sys; d=json.load(sys.stdin); print(d.get('$key','missing'))" 2>/dev/null || echo missing)"
        case "$val" in
            ok|passed) ok "MCP $key" ;;
            *) bad "MCP $key -> $val" ;;
        esac
    done
else
    bad "MCP driver failed: $(head -c 300 "$LOG_DIR/offline_test_mcp_err.log")"
fi

# ---------------------------------------------------------------------------
# 6. LLM-driven checks (answer + coding agent)
# ---------------------------------------------------------------------------
echo "--- LLM-driven checks"
if [ "$SKIP_LLM" = "1" ]; then
    block "LLM checks skipped (--skip-llm)"
elif printf '%s' "$tags" | grep -qF 'qwen3-coder:30b'; then
    a="$(api POST /v1/projects/rtl_uvm/answer '{"query":"How does i2c_master handle a write?","top_k":10}' 400)"
    if printf '%s' "$a" | grep -q '"answer"'; then
        ok "full RAG /answer returned a grounded answer (qwen3-coder:30b)"
    else
        bad "RAG /answer failed: $(echo "$a" | head -c 200)"
    fi
else
    block "qwen3-coder:30b not provisioned on this machine -> full RAG /answer and semrag_code cannot be verified here"
    # If a dev fallback model is present in the API's own .env config, prove
    # the offline pipeline end-to-end. The API process reads its model from
    # .env at startup, so no export here can change it.
    dev_model="$(printf '%s' "$tags" | grep -oE '"qwen3[^"]*"' | head -1 | tr -d '"' || true)"
    if [ -n "$dev_model" ]; then
        note "proving offline RAG pipeline with fallback model $dev_model (dev-only, not the production model)"
        a="$(api POST /v1/projects/rtl_uvm/answer '{"query":"How does i2c_master handle a write?","top_k":10}' 400)"
        if printf '%s' "$a" | grep -q '"answer"'; then
            ok "full RAG /answer worked offline with dev fallback $dev_model"
        else
            bad "RAG /answer failed even with fallback: $(echo "$a" | head -c 200)"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo
echo "== summary: PASS=$PASS FAIL=$FAIL BLOCKED=$BLOCKED =="
echo "logs: $LOG_DIR/offline_test_{api,worker,mcp_driver}.log"

if [ "$FAIL" -gt 0 ]; then
    echo
    echo "OFFLINE_READY: NO (failures above)"
    exit 1
fi
if [ "$BLOCKED" -gt 0 ]; then
    echo
    echo "OFFLINE_READY: NO (blocked — see BLOCK lines above; the offline"
    echo "               plumbing is verified, but a required model or check"
    echo "               is not provisioned on this machine)"
    exit 3
fi
echo
echo "OFFLINE_READY: YES"