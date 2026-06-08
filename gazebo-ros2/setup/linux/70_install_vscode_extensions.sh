#!/usr/bin/env bash
set -euo pipefail

if ! command -v code >/dev/null 2>&1; then
  echo "VS Code CLI 'code' not found. In VS Code, run: Shell Command: Install 'code' command in PATH"
  exit 1
fi

code --install-extension ms-vscode.cpptools
code --install-extension twxs.cmake
code --install-extension ms-vscode.cmake-tools
code --install-extension ms-python.python
code --install-extension ms-iot.vscode-ros
code --install-extension smilerobotics.urdf
code --install-extension deitry.colcon-helper
code --install-extension dotjoshjohnson.xml
code --install-extension redhat.vscode-yaml
code --install-extension yzhang.markdown-all-in-one
code --install-extension ybaumes.highlight-trailing-white-spaces
code --install-extension msfukui.eof-mark
code --install-extension aaron-bond.better-comments

# Optional extensions mentioned in the post.
# code --install-extension ms-azuretools.vscode-docker
# code --install-extension ms-vscode-remote.remote-ssh
# code --install-extension ms-vscode-remote.remote-ssh-edit
# code --install-extension ms-vscode-remote.remote-containers
# code --install-extension ms-python.vscode-pylance
# code --install-extension ms-toolsai.jupyter
# code --install-extension dbaeumer.vscode-eslint
# code --install-extension uctakeoff.vscode-counter
# code --install-extension vscode-icons-team.vscode-icons