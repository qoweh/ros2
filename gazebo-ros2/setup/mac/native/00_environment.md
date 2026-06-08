# macOS native install notes

This path is a host-native macOS install using Homebrew plus source builds.

Important caveats:

- The official ROS 2 Jazzy macOS source page is still based on an older macOS support story and explicitly references Mojave.
- Gazebo documents Jazzy + Harmonic as the recommended ROS/Gazebo pairing, but the strongest official recommendation is still Ubuntu for Jazzy.
- On current macOS, especially Apple Silicon, this route is best-effort and may require extra troubleshooting.
- If you want the lower-friction route on a Mac, use `../conda/` instead.

What this path installs:

- ROS 2 Jazzy from source into `~/ros2_jazzy`
- Gazebo Harmonic from Homebrew
- `ros_gz` Jazzy from source into `~/ros2_gz_ws`

Recommended order:

1. Run `10_install_homebrew_prereqs.sh`.
2. Run `20_build_ros2_jazzy.sh`.
3. Run `30_install_gazebo_harmonic.sh`.
4. Run `40_build_ros_gz_jazzy.sh`.
5. Open two terminals and run `50_verify_talker.sh` and `51_verify_listener.sh`.
6. Optionally run `52_start_gz_server.sh`.
7. Append `60_zsh_snippet.txt` to `~/.zshrc` only if you want automatic shell setup.

If build memory usage becomes a problem during `colcon build`, retry with:

```bash
colcon build --parallel-workers 1 --executor sequential
```