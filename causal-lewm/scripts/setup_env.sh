#!/usr/bin/env bash
# One-shot environment bootstrap for Causal-LeWM.
# Creates a conda env with Python 3.10 and installs LeWM's dependencies
# (stable-worldmodel) plus the extras needed for slot attention and
# C-JEPA-style object-level masking.

set -euo pipefail

ENV_NAME="${ENV_NAME:-causal-lewm}"
PY_VER="3.10"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" "python=$PY_VER" -y
fi
conda activate "$ENV_NAME"

# ffmpeg for video datasets (CLEVRER, Push-T renders)
conda install -y -c conda-forge ffmpeg

python -m pip install --upgrade pip uv

# LeWM core stack
uv pip install "stable-worldmodel[train,env]"

# Extras needed beyond LeWM:
#  - einops, hydra-core, wandb already pulled by stable-worldmodel
#  - slot attention reference + ALOE/VQA bits live in ../cjepa
uv pip install \
  webdataset hickle pycocotools tensorboardX accelerate \
  seaborn swig torchcodec av wget

echo
echo "Environment '$ENV_NAME' ready."
echo "Next:"
echo "  conda activate $ENV_NAME"
echo "  export STABLEWM_HOME=\$HOME/.stable-wm   # or your data dir"
