#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced:"
  echo "  source scripts/wsl_env.sh"
  exit 1
fi

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv. Run: bash scripts/wsl_setup.sh"
  exit 1
fi

source .venv/bin/activate

NVIDIA_LIB_DIRS="$(
python - <<'PY'
import glob
import os
import sysconfig

roots = [sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")]
dirs = []
for root in roots:
    if not root:
        continue
    for path in glob.glob(os.path.join(root, "nvidia", "*", "lib")):
        if os.path.isdir(path):
            dirs.append(path)

seen = set()
ordered = []
for item in dirs:
    if item in seen:
        continue
    seen.add(item)
    ordered.append(item)
print(":".join(ordered))
PY
)"

PARTS=()
if [[ -d /usr/lib/wsl/lib ]]; then
  PARTS+=("/usr/lib/wsl/lib")
fi
if [[ -n "${NVIDIA_LIB_DIRS}" ]]; then
  PARTS+=("${NVIDIA_LIB_DIRS}")
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  PARTS+=("${LD_LIBRARY_PATH}")
fi

export LD_LIBRARY_PATH="$(IFS=:; echo "${PARTS[*]}")"
