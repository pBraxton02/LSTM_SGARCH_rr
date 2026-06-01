#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source scripts/wsl_env.sh

echo "Checking GPU before Jupyter launch..."
if ! uv run python - <<'PY'
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)
if not gpus:
    raise SystemExit(2)
PY
then
  echo "GPU preflight failed. TensorFlow cannot see CUDA devices."
  TF_SO="$(uv run python - <<'PY'
from pathlib import Path
import tensorflow as tf
print(Path(tf.__file__).resolve().parent / "python" / "_pywrap_tensorflow_internal.so")
PY
)"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
  ldd "$TF_SO" | grep "not found" || echo "No unresolved libs reported by ldd."
  exit 1
fi

exec uv run jupyter lab "$@"
