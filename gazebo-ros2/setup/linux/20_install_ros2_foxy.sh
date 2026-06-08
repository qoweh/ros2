#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y \
  ros-foxy-desktop \
  ros-foxy-rmw-fastrtps* \
  ros-foxy-rmw-cyclonedds*