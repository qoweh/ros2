# shell notes for the conda path

What to keep:

- Let `conda init zsh` manage conda activation support.
- Enter the ROS environment with `conda activate ros_env`.

What not to do:

- Do not add `source /opt/ros/...` lines to `~/.zshrc`.
- Do not install RoboStack ROS packages into `base`.

Optional aliases after `conda init` is already working:

```bash
alias rw='conda activate ros_env'
alias cw='cd ~/robot_ws'
alias cbs='colcon build --symlink-install'
```

If you already have an environment named `ros_env`, the scripts in this folder will reuse it. Override the name when needed:

```bash
ENV_NAME=ros_env_jazzy ./10_create_ros_env.sh
ENV_NAME=ros_env_jazzy ./20_install_ros_jazzy_and_gz.sh
```