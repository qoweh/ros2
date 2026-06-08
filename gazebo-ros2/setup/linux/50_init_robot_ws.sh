#!/usr/bin/env bash
set -euo pipefail

source /opt/ros/foxy/setup.bash
mkdir -p ~/robot_ws/src
cd ~/robot_ws
colcon build --symlink-install