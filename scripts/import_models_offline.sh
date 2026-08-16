#!/usr/bin/env bash
# Import the portable Ollama model bundle into the offline machine's local
# Ollama store. Run on the OFFLINE machine after copying offline_models/.
# Requires an existing local Ollama server (systemd service or 'ollama serve').
#
#   scripts/import_models_offline.sh [path-to-models]
#     (default: <repo>/offline_models)
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${1:-$REPO_ROOT/offline_models}"

if ! command -v ollama >/dev/null 2>&1; then
    echo "error: 'ollama' not found on PATH (is Ollama installed?)" >&2
    exit 1
fi

# Determine the live Ollama model store (default ~/.ollama/models).
LIVE_STORE="${OLLAMA_MODELS:-$HOME/.ollama/models}"
MANIFESTS="$LIVE_STORE/manifests"
BLOBS="$LIVE_STORE/blobs"

if [ ! -d "$SOURCE" ] || [ ! -d "$SOURCE/manifests" ]; then
    echo "error: model bundle not found at $SOURCE (expected a manifests/ dir)" >&2
    exit 1
fi

mkdir -p "$MANIFESTS" "$BLOBS"

echo "==> importing model blobs into $LIVE_STORE"
# Blobs are content-addressed and stored flat; copy is idempotent.
find "$SOURCE/blobs" -type f -print0 | while IFS= read -r -d '' blob; do
    name="$(basename "$blob")"
    if [ ! -e "$BLOBS/$name" ]; then
        cp "$blob" "$BLOBS/$name"
    fi
done

# Merge manifests (registry.ollama.ai/library/...).
cp -r "$SOURCE/manifests/." "$MANIFESTS/"

echo
echo "==> verifying models are visible to Ollama"
ollama list

for model in qwen3-coder:30b nomic-embed-text; do
    if ollama list 2>/dev/null | grep -q "^${model%%:*}[: ]${model#*:}" || \
       ollama list 2>/dev/null | grep -qF "$model"; then
        echo "    OK: $model present"
    else
        echo "    MISSING: $model — the system is NOT offline-ready without it" >&2
    fi
done

echo
echo "Restart Ollama if the import happened while it was running:"
echo "    systemctl restart ollama   (or restart 'ollama serve')"