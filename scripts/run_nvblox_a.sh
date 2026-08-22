#!/usr/bin/env bash
set -euo pipefail

# Run one Python entry point inside the isolated Backend A environment while
# exposing NVIDIA's pip-installed NPP runtime to nvblox's native extension.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${NVBLOX_A_VENV:-${REPO_ROOT}/.venv-nvblox-a}"
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
    echo "Backend A environment is missing: ${VENV_DIR}" >&2
    echo "Run: bash scripts/setup_nvblox_a_env.sh" >&2
    exit 2
fi
if [[ "$#" -eq 0 ]]; then
    echo "Usage: bash scripts/run_nvblox_a.sh SCRIPT.py [arguments...]" >&2
    exit 2
fi

NPP_LIB="$("${VENV_PYTHON}" - <<'PY'
from pathlib import Path
import site

matches = []
for root in site.getsitepackages():
    matches.extend(Path(root).glob("nvidia/npp/lib/libnppc.so*"))
if not matches:
    raise SystemExit("nvidia-npp-cu12 is installed incorrectly: libnppc.so not found")
print(matches[0].parent)
PY
)"

export LD_LIBRARY_PATH="${NPP_LIB}:${LD_LIBRARY_PATH:-}"
exec "${VENV_PYTHON}" "$@"
