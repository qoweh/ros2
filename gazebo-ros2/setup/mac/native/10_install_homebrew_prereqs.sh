#!/usr/bin/env bash
set -euo pipefail

brew update
brew install \
  asio \
  assimp \
  bison \
  bullet \
  cmake \
  console_bridge \
  cppcheck \
  cunit \
  eigen \
  freetype \
  graphviz \
  opencv \
  openssl@3 \
  orocos-kdl \
  pcre \
  poco \
  pyqt@5 \
  python@3.12 \
  qt@5 \
  sip \
  spdlog \
  tinyxml2

PYTHON_BIN="$(brew --prefix python@3.12)/bin/python3.12"

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -U \
  --config-settings="--global-option=build_ext" \
  --config-settings="--global-option=-I$(brew --prefix graphviz)/include/" \
  --config-settings="--global-option=-L$(brew --prefix graphviz)/lib/" \
  argcomplete \
  catkin_pkg \
  colcon-common-extensions \
  coverage \
  cryptography \
  empy \
  flake8 \
  flake8-blind-except==0.1.1 \
  flake8-builtins \
  flake8-class-newline \
  flake8-comprehensions \
  flake8-deprecated \
  flake8-docstrings \
  flake8-import-order \
  flake8-quotes \
  importlib-metadata \
  jsonschema \
  lark==1.1.1 \
  lxml \
  matplotlib \
  mock \
  mypy==0.931 \
  netifaces \
  nose \
  pep8 \
  psutil \
  pydocstyle \
  pydot \
  pygraphviz \
  pyparsing==2.4.7 \
  pytest-mock \
  rosdep \
  rosdistro \
  setuptools==59.6.0 \
  vcstool

sudo rosdep init 2>/dev/null || true
rosdep update