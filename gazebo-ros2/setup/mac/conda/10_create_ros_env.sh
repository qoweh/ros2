#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-ros_env}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.12
fi

conda activate "${ENV_NAME}"
conda config --env --remove-key channels >/dev/null 2>&1 || true
conda config --env --add channels robostack-jazzy
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict