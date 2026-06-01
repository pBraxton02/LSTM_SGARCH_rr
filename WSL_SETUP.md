# WSL-First Setup (GPU)

This repository is configured for a WSL2-first workflow for TensorFlow GPU training.

## 1. Install WSL2 (one-time, Windows Admin PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\wsl_install_admin.ps1
```

Reboot, open Ubuntu once, and create your Linux user.

## 2. Bootstrap project in WSL

From normal PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\wsl_bootstrap_from_windows.ps1
```

This runs `scripts/wsl_setup.sh` inside WSL and will:

- sync dependencies with `uv` (`dev` + `linux-gpu` groups),
- install TensorFlow CUDA extras,
- apply TensorFlow's official virtualenv symlink fix,
- install GPU kernelspec `Python (LSTM WSL GPU)`,
- verify GPU visibility.

## 3. Start Jupyter (always use this)

Inside WSL:

```bash
cd /mnt/c/Users/suchy/Studia/Masters/RR/LSTM_SGARCH_rr
bash scripts/wsl_jupyter.sh
```

In Jupyter/VSCode select kernel:

- `Python (LSTM WSL GPU)`

## 4. Manual verification

Inside WSL:

```bash
cd /mnt/c/Users/suchy/Studia/Masters/RR/LSTM_SGARCH_rr
source scripts/wsl_env.sh
uv run python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

## 5. Run the hybrid notebook

Inside WSL:

```bash
cd /mnt/c/Users/suchy/Studia/Masters/RR/LSTM_SGARCH_rr
bash scripts/wsl_jupyter.sh
```

Open:

- `LSTM_SGARCH_5min_walk_forward.ipynb`

Select kernel:

- `Python (LSTM WSL GPU)`

## 6. TensorBoard (optional manual start)

The notebook tries to auto-start TensorBoard when training starts.
If you want to start it manually:

```bash
cd /mnt/c/Users/suchy/Studia/Masters/RR/LSTM_SGARCH_rr
source scripts/wsl_env.sh
uv run python -m tensorboard.main --logdir artifacts/lstm_sgarch_walk_forward/tensorboard --host 0.0.0.0 --port 6006
```

If you see `ModuleNotFoundError: No module named 'pkg_resources'`, resync:

```bash
uv sync --group dev --group linux-gpu
```
