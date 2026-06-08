#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-ros_env}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

ros2 run demo_nodes_cpp talker