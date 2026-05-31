#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

source scripts/wsl_env.sh

KERNEL_NAME="${1:-lstm-wsl-gpu}"
DISPLAY_NAME="${2:-Python (LSTM WSL GPU)}"
export KERNEL_NAME

uv run python -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "${DISPLAY_NAME}"

uv run python - <<'PY'
import json
import os
from pathlib import Path
from jupyter_client.kernelspec import KernelSpecManager

kernel_name = os.environ["KERNEL_NAME"]
ksm = KernelSpecManager()
spec = ksm.get_kernel_spec(kernel_name)
kernel_json = Path(spec.resource_dir) / "kernel.json"
data = json.loads(kernel_json.read_text(encoding="utf-8"))
data.setdefault("env", {})
data["env"]["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
kernel_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
print(f"Updated kernelspec: {kernel_json}")
PY
