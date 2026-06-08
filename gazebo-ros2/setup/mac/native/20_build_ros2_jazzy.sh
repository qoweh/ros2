#!/usr/bin/env bash
set -euo pipefail

export OPENSSL_ROOT_DIR="$(brew --prefix openssl@3)"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}:$(brew --prefix qt@5)"
export PATH="${PATH}:$(brew --prefix qt@5)/bin"

mkdir -p ~/ros2_jazzy/src
cd ~/ros2_jazzy

if [ -z "$(find src -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  vcs import --input https://raw.githubusercontent.com/ros2/ros2/jazzy/ros2.repos src
fi

rosdep install --from-paths src --ignore-src --rosdistro jazzy -y --skip-keys=python_qt_binding
colcon build --symlink-install --packages-skip-by-dep python_qt_binding