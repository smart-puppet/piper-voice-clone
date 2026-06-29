#!/usr/bin/env bash
# Download a Piper dataset zip and extract it to the project datasets/ directory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-/content/piper-work}"

if [[ -d "${WORK}" ]]; then
  export PIPER_COLAB=1
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
exec python "${ROOT}/scripts/download_dataset.py" "$@"
