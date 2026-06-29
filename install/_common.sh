#!/usr/bin/env bash
# Shared helpers for piper-voice-clone install scripts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT}/.venv"
PIPER_DIR="${ROOT}/piper1-gpl"
QWEN_DIR="${ROOT}/qwentts.cpp"
MODEL_DIR="${ROOT}/models/Qwen3-TTS-GGUF"

activate_venv() {
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

create_venv() {
  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "==> Creating virtual environment at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
  else
    echo "==> Using existing virtual environment at ${VENV_DIR}"
  fi
  activate_venv
  python -m pip install --upgrade pip wheel setuptools
}

install_python_deps() {
  echo "==> Installing Python dependencies"
  python -m pip install -r "${ROOT}/requirements.txt"
  python -m pip install -e "${ROOT}"
}

clone_piper() {
  if [[ ! -d "${PIPER_DIR}/.git" ]]; then
    echo "==> Cloning piper1-gpl..."
    git clone https://github.com/OHF-voice/piper1-gpl.git "${PIPER_DIR}"
  else
    echo "==> piper1-gpl already present"
  fi
}

install_piper_build_deps() {
  echo "==> Installing Piper build dependencies (scikit-build, cmake, ninja, cython)"
  python -m pip install "scikit-build<1" "cmake>=3.18,<4" "ninja>=1,<2" "cython>=3,<4"
}

install_piper_train() {
  clone_piper
  install_piper_build_deps
  echo "==> Applying Piper training patches (pretrained_ckpt, etc.)"
  python "${ROOT}/scripts/apply_piper_patches.py" "${PIPER_DIR}"
  echo "==> Installing Piper (editable) with training extras"
  python -m pip install -e "${PIPER_DIR}[train]"
}

build_piper_extensions() {
  if [[ ! -d "${PIPER_DIR}" ]]; then
    echo "ERROR: piper1-gpl not found — run install_piper_train first" >&2
    exit 1
  fi
  install_piper_build_deps
  echo "==> Building Piper native extensions"
  if [[ -f "${PIPER_DIR}/build_monotonic_align.sh" ]]; then
    (cd "${PIPER_DIR}" && bash build_monotonic_align.sh)
  else
    ALIGN_DIR="${PIPER_DIR}/src/piper/train/vits/monotonic_align"
    mkdir -p "${ALIGN_DIR}/monotonic_align"
    rm -f "${ALIGN_DIR}/core.c"
    (cd "${ALIGN_DIR}" && cythonize -i core.pyx)
    mv "${ALIGN_DIR}"/core*.so "${ALIGN_DIR}/monotonic_align/" 2>/dev/null || true
  fi
  (cd "${PIPER_DIR}" && python setup.py build_ext --inplace)
}

install_pytorch_cu126() {
  echo "==> Installing PyTorch CUDA 12.6 (Pascal sm_61 compatible)"
  python -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
}

install_pytorch_cu130() {
  echo "==> Installing PyTorch CUDA 12.8 (Ampere / Ada / Blackwell)"
  python -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu130
}

build_qwentts() {
  local cuda_arch="${1:-}"
  if [[ ! -d "${QWEN_DIR}/.git" ]]; then
    echo "==> Cloning qwentts.cpp..."
    git clone --recurse-submodules https://github.com/ServeurpersoCom/qwentts.cpp.git "${QWEN_DIR}"
  else
    echo "==> qwentts.cpp already present"
  fi

  echo "==> Building qwentts.cpp with CUDA..."
  (
    cd "${QWEN_DIR}"
    rm -rf build
    mkdir build && cd build
    CMAKE_ARGS=(-DGGML_CUDA=ON)
    if command -v nvcc >/dev/null 2>&1; then
      CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER="$(command -v nvcc)")
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
      CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc)
    fi
    if [[ -n "${cuda_arch}" ]]; then
      CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}")
      echo "    CMAKE_CUDA_ARCHITECTURES=${cuda_arch}"
    fi
    cmake .. "${CMAKE_ARGS[@]}"
    cmake --build . --config Release -j "$(nproc)"
  )
}

download_qwen_models() {
  echo "==> Downloading Qwen3-TTS GGUF models"
  mkdir -p "${MODEL_DIR}"
  python - <<PY
from pathlib import Path
from huggingface_hub import hf_hub_download

repo = "Serveurperso/Qwen3-TTS-GGUF"
model_dir = Path("${MODEL_DIR}")
for name in ("qwen-talker-0.6b-base-Q8_0.gguf", "qwen-tokenizer-12hz-Q8_0.gguf"):
    hf_hub_download(repo_id=repo, filename=name, local_dir=model_dir)
    print(f"  {name}")
PY
}

print_gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true
  fi
  python -c "
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    print('capability', torch.cuda.get_device_capability(0))
" || true
}

print_next_steps() {
  cat <<EOF

Environment ready. Activate with:
  source ${VENV_DIR}/bin/activate

Quick test (voice clone):
  cd ${ROOT}
  python scripts/test_voice.py --config config.yaml

Generate dataset:
  python scripts/generate_dataset.py --config config.yaml --samples 10

Fine-tune Piper (upload checkpoint to checkpoints/ first):
  bash scripts/train_piper_local.sh --voice myvoice --lang de --version v03

EOF
}
