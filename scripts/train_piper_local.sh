#!/usr/bin/env bash
# Fine-tune Piper on a local machine with NVIDIA GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${ROOT}/.venv"

VOICE_NAME=""
LANG_CODE="de"
DATASET_VERSION="v03"
ESPEAK_VOICE=""
MAX_EPOCHS="${MAX_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PRETRAINED_CKPT="${ROOT}/checkpoints/pretrained-finetune.ckpt"
LIMIT_TRAIN_BATCHES=""

usage() {
  cat <<'EOF'
Usage: bash scripts/train_piper_local.sh --voice NAME [options]

Required:
  --voice NAME          Voice / dataset prefix

Dataset path:
  datasets/{voice}-{lang}_{version}/

Options:
  --lang CODE           Language code (default: de)
  --version VER         Dataset version (default: v03)
  --espeak VOICE        espeak-ng voice (default: same as --lang)
  --epochs N            Max epochs (default: 10)
  --batch N             Batch size (default: 4)
  --pretrained PATH     Weights-only .ckpt (default: checkpoints/pretrained-finetune.ckpt)
  --smoke               Run 3 training batches only (sanity check)
  -h, --help            Show this help

Upload your checkpoint to checkpoints/ before the first run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --voice) VOICE_NAME="$2"; shift 2 ;;
    --lang) LANG_CODE="$2"; shift 2 ;;
    --version) DATASET_VERSION="$2"; shift 2 ;;
    --espeak) ESPEAK_VOICE="$2"; shift 2 ;;
    --epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --batch) BATCH_SIZE="$2"; shift 2 ;;
    --pretrained) PRETRAINED_CKPT="$2"; shift 2 ;;
    --smoke) LIMIT_TRAIN_BATCHES="3"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${VOICE_NAME}" ]]; then
  echo "ERROR: --voice is required" >&2
  usage
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "ERROR: Virtual environment not found. Run install/install_old_gpu.sh or install_new_gpu.sh" >&2
  exit 1
fi

ESPEAK_VOICE="${ESPEAK_VOICE:-${LANG_CODE}}"
DATASET_NAME="${VOICE_NAME}-${LANG_CODE}_${DATASET_VERSION}"
DATASET_DIR="${ROOT}/datasets/${DATASET_NAME}"

if [[ ! -f "${DATASET_DIR}/metadata.csv" && ! -f "${DATASET_DIR}/metadata_train.csv" ]]; then
  echo "ERROR: Dataset not found under ${DATASET_DIR}" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "ERROR: Pretrained checkpoint not found: ${PRETRAINED_CKPT}" >&2
  echo "Upload a compatible Piper .ckpt to checkpoints/ before training." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

PIPER_DIR="${ROOT}/piper1-gpl"
if [[ ! -d "${PIPER_DIR}" ]]; then
  echo "ERROR: piper1-gpl not found. Run install/install_old_gpu.sh or install_new_gpu.sh" >&2
  exit 1
fi

echo "==> Ensuring Piper training patches are applied"
python "${ROOT}/scripts/apply_piper_patches.py" "${PIPER_DIR}"

mapfile -t _PIPER_DATA_ARGS < <(
  PYTHONPATH="${ROOT}/src" DATASET_DIR="${DATASET_DIR}" python - <<'PY'
import os
from pathlib import Path

from voice_cloning.piper_training import dataset_training_args

args = dataset_training_args(Path(os.environ["DATASET_DIR"]))
print(args["csv_path"])
print(args["validation_split"])
print(args["num_test_examples"])
print(args["num_utterances"])
PY
)
METADATA_CSV="${_PIPER_DATA_ARGS[0]}"
VAL_SPLIT="${_PIPER_DATA_ARGS[1]}"
NUM_TEST_EXAMPLES="${_PIPER_DATA_ARGS[2]}"
NUM_UTTERANCES="${_PIPER_DATA_ARGS[3]}"

echo "Metadata: ${METADATA_CSV} (${NUM_UTTERANCES} utterances)"
echo "Validation split: ${VAL_SPLIT}, test examples: ${NUM_TEST_EXAMPLES}"

CMD=(
  python -m piper.train fit
  --data.voice_name "${VOICE_NAME}"
  --data.csv_path "${METADATA_CSV}"
  --data.audio_dir "${DATASET_DIR}/wavs/"
  --model.sample_rate 22050
  --data.espeak_voice "${ESPEAK_VOICE}"
  --data.cache_dir "${DATASET_DIR}/cache/"
  --data.config_path "${DATASET_DIR}/config.json"
  --data.batch_size "${BATCH_SIZE}"
  --data.validation_split "${VAL_SPLIT}"
  --data.num_test_examples "${NUM_TEST_EXAMPLES}"
  --model.pretrained_ckpt "${PRETRAINED_CKPT}"
  --trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint
  --trainer.callbacks.monitor val_loss
  --trainer.callbacks.mode min
  --trainer.max_epochs "${MAX_EPOCHS}"
  --trainer.callbacks.save_top_k 1
  --trainer.callbacks.save_last true
  --trainer.check_val_every_n_epoch 1
  --trainer.num_sanity_val_steps 0
)

if [[ -n "${LIMIT_TRAIN_BATCHES}" ]]; then
  CMD+=(--trainer.limit_train_batches "${LIMIT_TRAIN_BATCHES}" --trainer.max_epochs 1)
fi

echo "Training ${VOICE_NAME} from ${PRETRAINED_CKPT}"
echo "Dataset: ${DATASET_DIR}"
"${CMD[@]}"

echo ""
echo "Checkpoints: ${ROOT}/lightning_logs/version_*/checkpoints/"
echo "Export ONNX (audio + phoneme duration):"
echo "  python -c \"from pathlib import Path; from voice_cloning.piper_training import export_onnx_models; export_onnx_models(checkpoint=Path('lightning_logs/version_X/checkpoints/last.ckpt'), output_onnx=Path('models/${VOICE_NAME}-${LANG_CODE}-medium.onnx'))\""
