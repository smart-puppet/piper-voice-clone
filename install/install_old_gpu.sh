#!/usr/bin/env bash
# Install environment for older NVIDIA GPUs (Pascal sm_61 and similar).
# Uses PyTorch cu126 wheels and optional CMAKE_CUDA_ARCHITECTURES=61 for qwentts.cpp.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

CUDA_ARCH=""
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q '^6\.'; then
    CUDA_ARCH="61"
    echo "==> Detected Pascal GPU — qwentts.cpp will target sm_61"
  fi
fi

create_venv
install_python_deps
install_pytorch_cu126
install_piper_train
build_piper_extensions
build_qwentts "${CUDA_ARCH}"
download_qwen_models

echo ""
echo "==> Old-GPU install complete"
print_gpu_info
print_next_steps
