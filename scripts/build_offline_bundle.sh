#!/usr/bin/env bash
# Build the offline bundle on the INTERNET-CONNECTED machine ("Machine A").
#
# Produces, inside the repo:
#   offline_wheels/        - every pinned wheel from requirements-offline.txt
#                            plus the semrag package wheel
#   offline_models/        - portable Ollama model blobs for
#                            qwen3-coder:30b + nomic-embed-text
#                            (only with --with-models; ~18 GB for the 30B)
#
# Then copy the whole repo directory to the offline machine and run
# scripts/install_offline.sh there.
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
WHEELS="$REPO_ROOT/offline_wheels"
MODELS="$REPO_ROOT/offline_models"
WITH_MODELS=0

for arg in "$@"; do
    case "$arg" in
        --with-models) WITH_MODELS=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ ! -x "$PYTHON" ]; then
    echo "error: venv python not found at $PYTHON" >&2
    echo "hint: create it first with:  python3.12 -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 1
fi

echo "==> [1/4] Regenerating requirements-offline.txt from the venv"
"$PYTHON" -m pip freeze \
    | grep -v "^-e " \
    | grep -vE "^(pip|setuptools|wheel)==" \
    | sort > requirements-offline.txt
echo "    $(wc -l < requirements-offline.txt) pinned packages"

echo "==> [2/4] Downloading all pinned wheels (binary only)"
mkdir -p "$WHEELS"
"$PYTHON" -m pip download \
    --no-cache-dir \
    --only-binary=:all: \
    -r requirements-offline.txt \
    -d "$WHEELS"

echo "==> [3/4] Building the semrag package wheel"
"$PYTHON" -m pip wheel --no-deps --no-cache-dir -w "$WHEELS" .

echo "==> [4/4] Wheelhouse summary"
total=$(ls "$WHEELS" | wc -l)
size=$(du -sh "$WHEELS" | cut -f1)
echo "    $total wheels, $size total"
semrag_wheel=$(ls "$WHEELS"/semrag-*.whl 2>/dev/null | head -1 || true)
if [ -n "$semrag_wheel" ]; then
    echo "    semrag wheel: $(basename "$semrag_wheel")"
else
    echo "    WARNING: semrag wheel not built" >&2
fi

if [ "$WITH_MODELS" = "1" ]; then
    echo "==> [optional] Exporting Ollama models into $MODELS"
    if ! command -v ollama >/dev/null 2>&1; then
        echo "    error: 'ollama' not found on PATH; cannot export models" >&2
        exit 1
    fi
    mkdir -p "$MODELS"
    for model in qwen3-coder:30b nomic-embed-text; do
        echo "    pulling/exporting $model ..."
        OLLAMA_MODELS="$MODELS" ollama pull "$model"
    done
    echo "    models exported: $(du -sh "$MODELS" | cut -f1)"
else
    echo "==> [optional] models skipped (re-run with --with-models to export qwen3-coder:30b + nomic-embed-text)"
fi

echo
echo "Bundle ready. Copy the repository to the offline machine and run:"
echo "    scripts/install_offline.sh"
echo "    scripts/setup_postgres.sh        (as a postgres superuser)"
echo "    scripts/import_models_offline.sh (if --with-models was used)"
echo "    scripts/start_offline.sh"