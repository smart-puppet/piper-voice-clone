#!/usr/bin/env bash
# Install environment for modern NVIDIA GPUs (Ampere, Ada, Blackwell, etc.).
# Uses PyTorch cu130 wheels; qwentts.cpp uses default CUDA architectures.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_common.sh"

create_venv
install_python_deps
install_pytorch_cu130
install_piper_train
build_piper_extensions
build_qwentts ""
download_qwen_models

echo ""
echo "==> New-GPU install complete"
print_gpu_info
print_next_steps
