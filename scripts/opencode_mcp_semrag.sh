#!/usr/bin/env bash
# Portable launcher for the SemRAG MCP server under OpenCode.
#
# OpenCode config references this script (opencode.json -> mcp.semrag.command)
# so only ONE absolute path — the repo root — must be rewritten per machine
# (install_offline.sh does that). The script locates the project venv and the
# MCP server module relative to itself and execs them.
#
# Requires: a Python venv at <repo>/.../.venv with semrag's dependencies
# installed (mcp, httpx) — install_offline.sh sets this up.
set -euo pipefail

# Resolve the directory of this script (follow symlinks).
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -P "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

PYTHON="${SEMRAG_MCP_PYTHON:-$REPO_ROOT/.venv/bin/python}"
SERVER="${SEMRAG_MCP_SERVER:-$REPO_ROOT/semrag_mcp_server.py}"

if [ ! -x "$PYTHON" ]; then
    echo "opencode_mcp_semrag.sh: venv python not found at $PYTHON" >&2
    echo "Run scripts/install_offline.sh or set SEMRAG_MCP_PYTHON." >&2
    exit 1
fi
if [ ! -f "$SERVER" ]; then
    echo "opencode_mcp_semrag.sh: MCP server not found at $SERVER" >&2
    exit 1
fi

# Propagate the caller's working directory so the MCP server can resolve the
# active project context (project_id) automatically when a tool is invoked
# without an explicit project_id. OpenCode launches this script with cwd = the
# project the session is rooted in.
SEMRAG_CWD="${SEMRAG_CWD:-$(pwd)}"
export SEMRAG_CWD

exec "$PYTHON" "$SERVER"