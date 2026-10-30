#!/usr/bin/env bash
# Run once per collaborator from any directory, on a network-enabled cluster host.
set -euo pipefail
PLAYGROUND_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PLAYGROUND_ROOT"
if ! command -v uv >/dev/null 2>&1; then
    echo 'uv is required. Load your cluster uv module or install uv, then rerun this script.' >&2
    exit 1
fi
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PLAYGROUND_ROOT/.uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PLAYGROUND_ROOT/.uv-python}"
if [[ -e .venv || -L .venv ]]; then
    if ! .venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 11)' >/dev/null 2>&1; then
        echo 'Existing .venv is broken or is not Python 3.11. Move it aside and rerun setup.' >&2
        exit 1
    fi
else
    uv venv --python 3.11 .venv
fi
uv pip sync --python .venv/bin/python experiments/requirements-ui-lock.txt
echo 'Setup complete. In a GPU allocation, run scripts/check_identity_playground.sh --smoke'
echo 'Then start the UI with scripts/start_identity_playground.sh'
