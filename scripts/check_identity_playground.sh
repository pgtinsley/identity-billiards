#!/usr/bin/env bash
set -euo pipefail
PLAYGROUND_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PLAYGROUND_ROOT"
if [[ ! -x .venv/bin/python ]]; then
    echo 'Run scripts/setup_identity_playground.sh first.' >&2
    exit 1
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PLAYGROUND_ROOT/outputs/.matplotlib}"
umask 077
exec .venv/bin/python -m experiments.check_ui "$@"
