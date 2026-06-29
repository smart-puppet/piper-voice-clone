#!/usr/bin/env bash
# Install Piper training + dataset tools on Google Colab (GPU runtime).
# Uses the PyTorch build that ships with Colab — does NOT reinstall torch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPER_DIR="${ROOT}/piper1-gpl"
WORK="${WORK:-/content/piper-work}"

echo "==> piper-voice-clone Colab install"
echo "    Project root: ${ROOT}"
echo "    Work dir:     ${WORK}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y \
  build-essential cmake ninja-build git \
  espeak-ng libespeak-ng-dev libsndfile1

python -m pip install -U pip wheel setuptools cython scikit-build tqdm
python -m pip install \
  huggingface_hub soundfile librosa resampy numpy scipy PyYAML requests tqdm \
  lightning tensorboard tensorboardX jsonargparse onnx pysilero-vad

python -m pip install -e "${ROOT}"

if [[ ! -d "${PIPER_DIR}/.git" ]]; then
  echo "==> Cloning piper1-gpl into ${PIPER_DIR}"
  git clone https://github.com/OHF-voice/piper1-gpl.git "${PIPER_DIR}"
else
  echo "==> piper1-gpl already present at ${PIPER_DIR}"
fi

python "${ROOT}/scripts/apply_piper_patches.py" "${PIPER_DIR}"
python -m pip install -e "${PIPER_DIR}[train]"

(
  cd "${PIPER_DIR}"
  if [[ -f build_monotonic_align.sh ]]; then
    bash build_monotonic_align.sh
  fi
  python setup.py build_ext --inplace
)

# faster-qwen3-tts for dataset generation on Colab (optional but recommended)
python -m pip install "faster-qwen3-tts>=0.2.6"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    raise SystemExit(
        "CUDA not available. Set Runtime → Change runtime type → GPU, then re-run."
    )
PY

mkdir -p "${WORK}/datasets" "${WORK}/checkpoints" "${WORK}/models" "${WORK}/cache" "${WORK}/lightning_logs"
echo ""
echo "Colab install complete."
echo "Local work directory: ${WORK}"
echo ""
echo "Next steps: run notebook cells 8–11 (checkpoint, dataset, TensorBoard, train)."
