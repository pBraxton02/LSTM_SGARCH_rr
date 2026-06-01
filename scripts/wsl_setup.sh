#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "Run this from inside the repository."
  exit 1
fi

if [[ ! -e /dev/dxg ]]; then
  echo "GPU device /dev/dxg is not available."
  echo "Ensure this distro is WSL2 and GPU support is enabled."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "Syncing dependencies for WSL GPU workflow..."
uv sync --group dev --group linux-gpu

echo "Applying official TensorFlow virtualenv GPU link fix..."
TF_DIR="$(uv run python -c "import os, tensorflow as tf; print(os.path.dirname(tf.__file__))")"
pushd "$TF_DIR" >/dev/null
ln -svf ../nvidia/*/lib/*.so* .
popd >/dev/null

PTXAS_PATH="$(uv run python - <<'PY'
import glob
import importlib.util
import os
import sysconfig

candidates = []
spec = importlib.util.find_spec("nvidia.cuda_nvcc")
if spec and spec.submodule_search_locations:
    for root in spec.submodule_search_locations:
        candidates.extend(glob.glob(os.path.join(root, "bin", "ptxas")))

for lib_root in (sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")):
    if not lib_root:
        continue
    candidates.extend(glob.glob(os.path.join(lib_root, "nvidia", "cuda_nvcc", "bin", "ptxas")))
    candidates.extend(glob.glob(os.path.join(lib_root, "nvidia", "cuda_nvcc", "*", "bin", "ptxas")))

for path in candidates:
    if os.path.isfile(path):
        print(path)
        break
PY
)"
if [[ -n "${PTXAS_PATH}" ]]; then
  ln -sf "${PTXAS_PATH}" ".venv/bin/ptxas"
fi

echo "Preparing CUDA runtime environment helper..."
source scripts/wsl_env.sh

echo "Installing Jupyter kernel: Python (LSTM WSL GPU)"
bash scripts/wsl_kernel.sh

echo "Verifying TensorFlow GPU visibility..."
source scripts/wsl_env.sh
uv run python - <<'PY'
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
PY

echo "Done. Start Jupyter with: bash scripts/wsl_jupyter.sh"
