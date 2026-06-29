#!/usr/bin/env bash
# Colab wrapper — saves to /content/piper-work/checkpoints/pretrained-finetune.ckpt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIPER_COLAB=1
export WORK="${WORK:-/content/piper-work}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

exec python "${ROOT}/scripts/download_checkpoint.py" "$@"
