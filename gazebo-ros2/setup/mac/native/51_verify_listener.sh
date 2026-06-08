#!/usr/bin/env bash
set -euo pipefail

source ~/ros2_jazzy/install/setup.bash

if [ -f ~/ros2_gz_ws/install/setup.bash ]; then
  source ~/ros2_gz_ws/install/setup.bash
fi

ros2 run demo_nodes_py listener