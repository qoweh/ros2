#!/usr/bin/env bash
set -euo pipefail

export GZ_VERSION=harmonic
export OPENSSL_ROOT_DIR="$(brew --prefix openssl@3)"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}:$(brew --prefix qt@5)"
export PATH="${PATH}:$(brew --prefix qt@5)/bin"

source ~/ros2_jazzy/install/setup.bash

mkdir -p ~/ros2_gz_ws/src
cd ~/ros2_gz_ws/src

if [ ! -d ros_gz ]; then
  git clone https://github.com/gazebosim/ros_gz.git -b jazzy
fi

cd ~/ros2_gz_ws
rosdep install -r --from-paths src -i -y --rosdistro jazzy
colcon build