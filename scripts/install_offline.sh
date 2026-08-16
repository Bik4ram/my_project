#!/usr/bin/env bash
# Install the offline SemRAG stack from the pre-built wheelhouse. Run on the
# OFFLINE machine ("Machine B"). Requires: Python 3.12, bash, and the bundle
# copied from the build machine (offline_wheels/ present). No Internet needed.
#
# Steps: create the venv, install every wheel from the local wheelhouse,
# install the semrag wheel, write .env from the offline template, and
# regenerate opencode.json with this machine's paths.
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WHEELS="$REPO_ROOT/offline_wheels"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
PYTHON="$REPO_ROOT/.venv/bin/python"

if [ ! -d "$WHEELS" ] || [ -z "$(ls -A "$WHEELS")" ]; then
    echo "error: wheelhouse missing at $WHEELS" >&2
    echo "run scripts/build_offline_bundle.sh on the Internet-connected machine first" >&2
    exit 1
fi

echo "==> [1/4] Creating virtualenv"
if [ ! -x "$PYTHON" ]; then
    "$PYTHON_BIN" -m venv .venv
fi

echo "==> [2/4] Installing requirements from the offline wheelhouse"
"$PYTHON" -m pip install --no-index --find-links "$WHEELS" -r requirements-offline.txt

semrag_wheel=$(ls "$WHEELS"/semrag-*.whl 2>/dev/null | head -1 || true)
if [ -z "$semrag_wheel" ]; then
    echo "error: semrag wheel not found in $WHEELS" >&2
    exit 1
fi
echo "==> [3/4] Installing the semrag package wheel ($(basename "$semrag_wheel"))"
"$PYTHON" -m pip install --no-index --find-links "$WHEELS" "$semrag_wheel"

echo "==> [4/4] Writing machine-local config"
if [ ! -f .env ] && [ -f .env.offline.example ]; then
    cp .env.offline.example .env
    echo "    wrote .env from .env.offline.example (edit it: SEMRAG_DATABASE_URL, SEMRAG_API_KEYS, SEMRAG_UVM_ROOT)"
fi

# Regenerate opencode.json so the MCP server launcher path matches this repo.
if [ -f opencode.json ] && [ -f scripts/opencode_mcp_semrag.sh ]; then
    "$PYTHON" - "$REPO_ROOT" <<'EOF'
import json
import sys

repo_root = sys.argv[1]
with open("opencode.json", "r", encoding="utf-8") as fh:
    cfg = json.load(fh)

mcp = cfg.get("mcp", {})
cmd = mcp.get("semrag", {}).get("command", [])
if cmd:
    # Replace the first element (the absolute launcher path) with this
    # machine's repo-root launcher, preserving any extra arguments.
    cmd[0] = f"{repo_root}/scripts/opencode_mcp_semrag.sh"

with open("opencode.json", "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
    fh.write("\n")
print(f"    wrote opencode.json mcp.semrag.command[0] -> {cmd[0]}")
EOF
fi

echo
echo "Verifying install ..."
"$PYTHON" -c "import semrag; print('    semrag import OK')"
"$PYTHON" -m pip check
echo
echo "Done. Next steps on this machine:"
echo "    scripts/setup_postgres.sh"
echo "    scripts/import_models_offline.sh   (if the model blobs were exported)"
echo "    scripts/start_offline.sh"
echo
echo "If the models were NOT exported, you must stage them some other way;"
echo "the system is not OFFLINE_READY until qwen3-coder:30b and"
echo "nomic-embed-text are present in local Ollama."