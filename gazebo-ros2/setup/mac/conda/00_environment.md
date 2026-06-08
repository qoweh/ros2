# macOS conda install notes

This is the recommended macOS route for the current machine.

Observed environment details from this Mac:

- `conda` version: `26.1.1`
- platform: `osx-arm64`
- active channels in the existing ROS env: `robostack-jazzy`, `conda-forge`
- relevant installed packages observed: `ros-jazzy-desktop 0.11.0`, `ros-jazzy-ros-gz 1.0.19`, `ros-dev-tools 1.0.1`, `gz-sim8 8.10.0`

Important notes from RoboStack:

- Do not install ROS packages into the `base` environment.
- Do not source a separate system ROS installation from your shell startup files.
- Activating the conda environment also activates the ROS environment.

Recommended order:

1. Run `10_create_ros_env.sh`.
2. Run `20_install_ros_jazzy_and_gz.sh`.
3. Open two terminals and run `30_verify_talker.sh` and `31_verify_listener.sh`.
4. Optionally run `32_start_gz_server.sh`.
5. Read `40_shell_notes.md`.