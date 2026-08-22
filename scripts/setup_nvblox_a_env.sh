#!/usr/bin/env bash
set -euo pipefail

# Backend A is intentionally isolated from Isaac Sim's pinned torch environment.
# The versions below match nvblox v0.0.10's official x86_64/CUDA 12 build helper.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${NVBLOX_A_VENV:-${REPO_ROOT}/.venv-nvblox-a}"
BOOTSTRAP_PYTHON="${NVBLOX_A_BOOTSTRAP_PYTHON:-python}"

TORCH_INDEX="https://download.pytorch.org/whl/cu128"
NVBLOX_WHEEL="https://github.com/nvidia-isaac/nvblox/releases/download/v0.0.10/nvblox_torch-0.0.10+cu12ubuntu22-py3-none-linux_x86_64.whl"

if [[ -e "${VENV_DIR}" && ! -f "${VENV_DIR}/pyvenv.cfg" ]]; then
    echo "Refusing to use a non-venv path: ${VENV_DIR}" >&2
    exit 2
fi

if [[ ! -f "${VENV_DIR}/pyvenv.cfg" ]]; then
    "${BOOTSTRAP_PYTHON}" -m venv "${VENV_DIR}"
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Python was not created at ${VENV_PYTHON}" >&2
    exit 2
fi

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install \
    --index-url "${TORCH_INDEX}" \
    "torch==2.9.1" \
    "torchvision==0.24.1"
"${VENV_PYTHON}" -m pip install --no-deps \
    "nvidia-npp-cu12==12.3.3.65"
"${VENV_PYTHON}" -m pip install \
    "numpy==1.26.0" \
    "scipy==1.15.3" \
    "${NVBLOX_WHEEL}"
"${VENV_PYTHON}" -m pip install --editable "${REPO_ROOT}[mapping]"

"${VENV_PYTHON}" -m pip check
bash "${SCRIPT_DIR}/run_nvblox_a.sh" \
    "${SCRIPT_DIR}/check_conservative_backends.py"

echo "Backend A environment is ready: ${VENV_DIR}"
