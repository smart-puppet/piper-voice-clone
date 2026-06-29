#!/usr/bin/env bash
# Generate a Piper dataset on Colab local disk; publish to Drive when finished.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIPER_COLAB=1
WORK="${WORK:-/content/piper-work}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/piper-voice-clone}"
CONFIG="${ROOT}/config.colab.yaml"
SAMPLES=""
TEMPERATURE=""
FORCE_REWRITE=0
SKIP_PUBLISH=0

usage() {
  cat <<'EOF'
Usage: bash colab/generate_dataset.sh [options]

All generation runs on Colab local storage (/content/piper-work).
Google Drive publish runs at the end if Drive is already mounted.

Mount Drive first in a notebook cell (required for publish from Terminal):
  from google.colab import drive
  drive.mount('/content/drive')

Options:
  --config PATH         YAML config (default: config.colab.yaml)
  --samples N           Override utterance count (smoke test: 10)
  --temperature F       TTS sampling temperature (overrides config)
  --force-rewrite       Regenerate all utterances (disable resume)
  --work PATH           Local work directory (default: /content/piper-work)
  --drive-root PATH     Drive publish root (default: /content/drive/MyDrive/piper-voice-clone)
  --skip-publish        Do not publish to Drive (results stay on local disk only)
  -h, --help            Show this help

Local output:
  /content/piper-work/datasets/{voice}-{lang}_{version}/

Published to Drive at the end (zip only):
  archives/{voice}-{lang}_{version}.zip
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --samples) SAMPLES="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --force-rewrite) FORCE_REWRITE=1; shift ;;
    --work) WORK="$2"; shift 2 ;;
    --drive-root) DRIVE_ROOT="$2"; shift 2 ;;
    --skip-publish) SKIP_PUBLISH=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: Config not found: ${CONFIG}" >&2
  exit 1
fi

mkdir -p "${WORK}/datasets"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ARGS=(--config "${CONFIG}")
if [[ -n "${SAMPLES}" ]]; then
  ARGS+=(--samples "${SAMPLES}")
fi
if [[ -n "${TEMPERATURE}" ]]; then
  ARGS+=(--temperature "${TEMPERATURE}")
fi
if [[ "${FORCE_REWRITE}" -eq 1 ]]; then
  ARGS+=(--force-rewrite)
fi

echo "==> Generating dataset on Colab local storage"
echo "    Config : ${CONFIG}"
echo "    Work   : ${WORK}"
cd "${ROOT}"
python scripts/generate_dataset.py "${ARGS[@]}"

if [[ "${SKIP_PUBLISH}" -eq 1 ]]; then
  echo ""
  echo "Dataset generation complete (publish skipped)."
  echo "  Local: ${WORK}/datasets/"
  exit 0
fi

echo ""
echo "==> Publishing dataset to Google Drive"
PYTHONPATH="${ROOT}/src:${ROOT}/colab:${PYTHONPATH:-}" \
  DRIVE_ROOT="${DRIVE_ROOT}" CONFIG="${CONFIG}" python - <<'PY'
import os
from pathlib import Path

from voice_cloning.dataset_generator import load_config
from piper_training_utils import publish_dataset_to_drive

cfg = load_config(Path(os.environ["CONFIG"]).resolve())
result = publish_dataset_to_drive(
    dataset_dir=cfg.dataset_dir,
    drive_root=Path(os.environ["DRIVE_ROOT"]),
)
print("Published archive:", result["archive"])
PY

echo ""
echo "Dataset generation complete."
echo "  Local  : ${WORK}/datasets/"
echo "  Drive  : ${DRIVE_ROOT}/archives/"
