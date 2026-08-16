#!/usr/bin/env bash
# Export the exact required Ollama models into a portable directory
# (offline_models/) on the INTERNET-CONNECTED machine. Run:
#   scripts/prepare_models.sh          # or:  build_offline_bundle.sh --with-models
#
# Exports EXACTLY: qwen3-coder:30b (the only LLM) and nomic-embed-text
# (the only embedding model). Copy offline_models/ to the offline machine
# and run scripts/import_models_offline.sh there.
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODELS="$REPO_ROOT/offline_models"
REQUIRED_MODELS="qwen3-coder:30b nomic-embed-text"

if ! command -v ollama >/dev/null 2>&1; then
    echo "error: 'ollama' not found on PATH" >&2
    exit 1
fi

mkdir -p "$MODELS"

for model in $REQUIRED_MODELS; do
    echo "==> exporting $model"
    # OLLAMA_MODELS points the local store at our portable dir, so the pull
    # lands directly in the bundle.
    OLLAMA_MODELS="$MODELS" ollama pull "$model"
    OLLAMA_MODELS="$MODELS" ollama list
done

echo
echo "Exported to $MODELS ($(du -sh "$MODELS" | cut -f1))."
echo "The exact model set is: $REQUIRED_MODELS"
echo "Copy this directory (or the whole repo) to the offline machine, then run:"
echo "    scripts/import_models_offline.sh"