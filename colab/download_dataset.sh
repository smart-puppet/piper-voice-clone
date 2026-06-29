#!/usr/bin/env bash
# Colab wrapper — extracts datasets under /content/piper-work/datasets/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIPER_COLAB=1
export WORK="${WORK:-/content/piper-work}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

exec python "${ROOT}/scripts/download_dataset.py" "$@"
