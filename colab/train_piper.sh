#!/usr/bin/env bash
# Fine-tune Piper on Colab local disk; publish to Drive when finished.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-/content/piper-work}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/piper-voice-clone}"

VOICE_NAME=""
LANG_CODE="de"
DATASET_VERSION="v03"
ESPEAK_VOICE=""
ADD_EPOCHS="${ADD_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-0}"
PRETRAINED_CKPT=""
SKIP_PUBLISH=0
NO_DISCONNECT=0

usage() {
  cat <<'EOF'
Usage: bash colab/train_piper.sh --voice NAME [options]

Required:
  --voice NAME          Voice / dataset prefix (e.g. myvoice)

Local layout (all under /content/piper-work):
  datasets/{voice}-{lang}_{version}/
  checkpoints/pretrained-finetune.ckpt
  lightning_logs/{voice}/
  models/{voice}-{lang}-medium.onnx
  models/{voice}-{lang}-medium.phoneme_duration.onnx

Publish to Drive at the end if Drive is already mounted (checkpoints + ONNX; TensorBoard logs stay local).

Options:
  --lang CODE           Language code (default: de)
  --version VER         Dataset version (default: v03)
  --espeak VOICE        espeak-ng voice (default: same as --lang)
  --add-epochs N        Add N epochs to resumed training (default: 10)
  --epochs N            Backward-compatible alias for --add-epochs
  --batch N             Batch size, 0 = auto from GPU VRAM (default: 0)
  --pretrained PATH     Local weights-only .ckpt (default: work/checkpoints/pretrained-finetune.ckpt)
  --work PATH           Local work directory (default: /content/piper-work)
  --drive-root PATH     Drive publish root (default: /content/drive/MyDrive/piper-voice-clone)
  --skip-publish        Do not publish to Drive (results stay on local disk only)
  --no-disconnect       Keep the Colab runtime connected after training (default: disconnect)
  -h, --help            Show this help

Before training, place your pretrained checkpoint locally:
  /content/piper-work/checkpoints/pretrained-finetune.ckpt

Examples:
  bash colab/train_piper.sh --voice myvoice --lang de --version v01
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --voice) VOICE_NAME="$2"; shift 2 ;;
    --lang) LANG_CODE="$2"; shift 2 ;;
    --version) DATASET_VERSION="$2"; shift 2 ;;
    --espeak) ESPEAK_VOICE="$2"; shift 2 ;;
    --add-epochs) ADD_EPOCHS="$2"; shift 2 ;;
    --epochs) ADD_EPOCHS="$2"; shift 2 ;;
    --batch) BATCH_SIZE="$2"; shift 2 ;;
    --pretrained) PRETRAINED_CKPT="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --drive-root) DRIVE_ROOT="$2"; shift 2 ;;
    --skip-publish) SKIP_PUBLISH=1; shift ;;
    --no-disconnect) NO_DISCONNECT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "${VOICE_NAME}" ]]; then
  echo "ERROR: --voice is required" >&2
  usage
  exit 1
fi

ESPEAK_VOICE="${ESPEAK_VOICE:-${LANG_CODE}}"
DATASET_NAME="${VOICE_NAME}-${LANG_CODE}_${DATASET_VERSION}"
DATASET_DIR="${WORK}/datasets/${DATASET_NAME}"
CACHE_DIR="${WORK}/cache/${VOICE_NAME}"
LIGHTNING_DIR="${WORK}/lightning_logs/${VOICE_NAME}"
MODELS_DIR="${WORK}/models"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-${WORK}/checkpoints/pretrained-finetune.ckpt}"
LOCAL_ONNX="${MODELS_DIR}/${VOICE_NAME}-${LANG_CODE}-medium.onnx"
LOCAL_ONNX_JSON="${LOCAL_ONNX}.json"
LOCAL_ONNX_PHONEME="${LOCAL_ONNX%.onnx}.phoneme_duration.onnx"
DRIVE_CHECKPOINTS="${DRIVE_ROOT}/checkpoints/${VOICE_NAME}"
DRIVE_ONNX="${DRIVE_ROOT}/models/${VOICE_NAME}-${LANG_CODE}-medium.onnx"
DRIVE_ONNX_JSON="${DRIVE_ONNX}.json"
DRIVE_ONNX_PHONEME="${DRIVE_ONNX%.onnx}.phoneme_duration.onnx"

export PYTHONPATH="${ROOT}/colab:${ROOT}/src:${PYTHONPATH:-}"
export PIPER_COLAB=1
export SKIP_PUBLISH="${SKIP_PUBLISH}"
export NO_DISCONNECT="${NO_DISCONNECT}"
export PIPER_LIGHTNING_DIR="${LIGHTNING_DIR}"
export PIPER_DRIVE_CHECKPOINTS="${DRIVE_CHECKPOINTS}"

mkdir -p "${WORK}" "${CACHE_DIR}" "${LIGHTNING_DIR}" "${MODELS_DIR}" \
  "$(dirname "${PRETRAINED_CKPT}")"

if [[ ! -f "${DATASET_DIR}/metadata.csv" && ! -f "${DATASET_DIR}/metadata_train.csv" ]]; then
  echo "ERROR: Local dataset not found: ${DATASET_DIR}" >&2
  echo "Run dataset generation first:" >&2
  echo "  bash colab/generate_dataset.sh --config config.colab.yaml" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
  echo "ERROR: Pretrained checkpoint not found: ${PRETRAINED_CKPT}" >&2
  echo "Copy your .ckpt to that path on Colab local storage before training." >&2
  exit 1
fi

python - <<PY
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "${ROOT}/colab")
sys.path.insert(0, "${ROOT}/src")

from piper_training_utils import (
    colab_training_profile,
    disconnect_colab_runtime,
    find_resume_checkpoint,
    publish_training_to_drive,
    restore_checkpoint_from_drive,
    run_subprocess_streaming,
    training_resume_status,
    training_cli_args,
)
from voice_cloning.piper_training import dataset_training_args, export_onnx_models

dataset_dir = Path("${DATASET_DIR}")
cache_dir = Path("${CACHE_DIR}")
lightning_dir = Path("${LIGHTNING_DIR}")
work = Path("${WORK}")
pretrained = Path("${PRETRAINED_CKPT}")
local_onnx = Path("${LOCAL_ONNX}")
local_onnx_json = Path("${LOCAL_ONNX_JSON}")
local_onnx_phoneme = Path("${LOCAL_ONNX_PHONEME}")
drive_root = Path("${DRIVE_ROOT}")
drive_checkpoints = Path("${DRIVE_CHECKPOINTS}")
drive_onnx = Path("${DRIVE_ONNX}")
drive_onnx_json = Path("${DRIVE_ONNX_JSON}")
drive_onnx_phoneme = Path("${DRIVE_ONNX_PHONEME}")
add_epochs = int("${ADD_EPOCHS}")
if add_epochs <= 0:
    raise ValueError(f"ADD_EPOCHS must be > 0, got {add_epochs}")

data_args = dataset_training_args(dataset_dir)
print(
    f"Dataset: {dataset_dir} "
    f"({data_args['num_utterances']} utterances, "
    f"val_split={data_args['validation_split']}, "
    f"test={data_args['num_test_examples']})"
)

profile = colab_training_profile(batch_size=int("${BATCH_SIZE}"))
print(
    f"Profile: batch={profile['batch_size']} workers={profile['num_workers']} "
    f"precision={profile['precision']} gpu={profile['gpu_name']} ({profile['vram_gb']} GB)"
)

if not find_resume_checkpoint(lightning_dir):
    restore_checkpoint_from_drive(
        lightning_dir=lightning_dir,
        drive_checkpoints=drive_checkpoints,
    )

resume_ckpt = find_resume_checkpoint(lightning_dir)
effective_max_epochs = add_epochs
if resume_ckpt:
    resume_status = training_resume_status(
        lightning_root=lightning_dir,
        max_epochs=10**9,
    )
    completed_epoch = int(resume_status.get("completed_epoch", -1))
    if completed_epoch >= 0:
        effective_max_epochs = completed_epoch + add_epochs + 1
        print(
            f"Resume checkpoint epoch={completed_epoch}; "
            f"adding {add_epochs} epoch(s) -> trainer.max_epochs={effective_max_epochs}"
        )
    else:
        effective_max_epochs = add_epochs
        print(
            f"Could not read completed epoch from checkpoint; "
            f"using trainer.max_epochs={effective_max_epochs}"
        )

if not resume_ckpt:
    print(f"No resume checkpoint found; training for {add_epochs} epoch(s) from pretrained.")

train_cmd = [
    sys.executable, "-u", "-m", "piper.train", "fit",
    "--data.voice_name", "${VOICE_NAME}",
    "--data.csv_path", str(data_args["csv_path"]),
    "--data.audio_dir", str(dataset_dir / "wavs"),
    "--model.sample_rate", "22050",
    "--data.espeak_voice", "${ESPEAK_VOICE}",
    "--data.cache_dir", str(cache_dir),
    "--data.config_path", str(dataset_dir / "config.json"),
    "--data.validation_split", str(data_args["validation_split"]),
    "--data.num_test_examples", str(data_args["num_test_examples"]),
    "--trainer.max_epochs", str(effective_max_epochs),
    *training_cli_args(profile),
    "--trainer.callbacks+=lightning.pytorch.callbacks.ModelCheckpoint",
    "--trainer.callbacks.monitor", "val_loss",
    "--trainer.callbacks.mode", "min",
    "--trainer.callbacks.save_top_k", "1",
    "--trainer.callbacks.save_last", "true",
    "--trainer.default_root_dir", str(lightning_dir),
]

if resume_ckpt:
    print("Resuming from local checkpoint:", resume_ckpt)
    train_cmd.extend(["--ckpt_path", str(resume_ckpt)])
else:
    print("Starting from pretrained weights:", pretrained)
    train_cmd.extend(["--model.pretrained_ckpt", str(pretrained)])

print("Launching training on local Colab storage...")
run_subprocess_streaming(train_cmd, cwd=work, env=None)

ckpt = find_resume_checkpoint(lightning_dir)
config_json = dataset_dir / "config.json"
if ckpt and config_json.exists():
    print("Exporting ONNX locally:", local_onnx, "and", local_onnx_phoneme)
    export_onnx_models(
        checkpoint=ckpt,
        output_onnx=local_onnx,
        cwd=work,
    )
    import shutil
    shutil.copy2(config_json, local_onnx_json)

skip_publish = os.environ.get("SKIP_PUBLISH", "0") == "1"
if skip_publish:
    print("Skipping Drive publish (--skip-publish).")
else:
    print("Publishing checkpoints and ONNX to Google Drive...")
    publish_training_to_drive(
        drive_root=drive_root,
        local_lightning=lightning_dir,
        drive_checkpoints=drive_checkpoints,
        local_onnx=local_onnx if local_onnx.exists() else None,
        drive_onnx=drive_onnx,
        local_onnx_phoneme=local_onnx_phoneme if local_onnx_phoneme.exists() else None,
        drive_onnx_phoneme=drive_onnx_phoneme,
        local_onnx_config=local_onnx_json if local_onnx_json.exists() else None,
        drive_onnx_config=drive_onnx_json,
    )

if os.environ.get("NO_DISCONNECT", "0") != "1":
    disconnect_colab_runtime()
else:
    print("Keeping Colab runtime connected (--no-disconnect).")
PY

echo "Done."
