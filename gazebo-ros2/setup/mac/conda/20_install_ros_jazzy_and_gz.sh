#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-ros_env}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

conda install -y \
  -c robostack-jazzy \
  -c conda-forge \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz \
  gz-sim8 \
  ros-dev-tools